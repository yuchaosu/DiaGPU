/* ============================================================
 * prep_driver.cu — preprocessing-vs-kernel cost breakdown for the paper's
 * amortization analysis: how expensive is each variant's plan build (host)
 * and upload (H2D) relative to one kernel iteration.
 *
 * Variants (same construction code as spmv_driver / spmspm_driver):
 *   spmv  : didx, stream(gather), gopt, csr_zskip, cusparse_csr
 *   spmspm: ours_flat, hm_atomic, ours_sym (L4)
 *
 * Output row:
 *   PREPCSV,file,n,D,stage,variant,prep_host_ms,upload_ms,kernel_ms,amort_applies
 * where amort_applies = (prep_host+upload)/kernel — the number of kernel
 * applies after which preprocessing is amortized.  kernel_ms=-1 marks a
 * prep-only row.
 *
 * ours_sym runs LAST and prints its prep-only row BEFORE the launch:
 * gather_flat_sym is known to hit an illegal access on wide-band H
 * (B2_14, BeH_12, O2_16, flowmeter0), which would poison the context —
 * this way the plan cost still lands in the CSV.  Downstream analysis
 * must take the LAST row per (file,variant).
 *
 * usage: prep_driver <dia_file> [iters=50]
 * ============================================================ */
#include "../dia_io.hpp"
#include "../../spmv/src/spmv_gather.cuh"
#include "../../spmv/src/spmv_zeroskip_kernels.cuh"
#include "gather_opt.cuh"
#include "../../spmspm/gather_flat.cuh"
#include "../../spmspm/gather_flat_sym.cuh"
#include "../../spmspm/paper_hm.cuh"
#include "common.cuh"

#include <cusparse.h>
#include <chrono>
#include <unordered_map>
#include <cstring>

#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

using clk = std::chrono::steady_clock;
static double ms_between(clk::time_point a, clk::time_point b){
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/* same C-structure builder as spmspm_driver */
struct Cstruct { std::vector<int> offsets, lengths; std::vector<size_t> starts; size_t nnz; };
static Cstruct make_c(const DiaHost& A, const DiaHost& B, bool upper=false){
    int n = A.n; std::vector<char> present(2*(size_t)n-1, 0);
    for (int da : A.offsets) for (int db : B.offsets){ int dc = da+db; if (dc > -n && dc < n) present[dc+n-1] = 1; }
    Cstruct C; size_t off = 0;
    for (int d = (upper ? 0 : -(n-1)); d <= n-1; ++d){
        if (!present[d+n-1]) continue;
        int len = n - std::abs(d);
        C.offsets.push_back(d); C.starts.push_back(off); C.lengths.push_back(len); off += len;
    }
    C.nnz = off; return C;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s <dia_file> [iters=50]\n", argv[0]); return 1; }
    int iters = argc > 2 ? atoi(argv[2]) : 50;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();
    const char* file = argv[1];
    const int TPB = 256;

    auto row = [&](const char* stage, const char* var, double prep, double up, double kms){
        printf("PREPCSV,%s,%d,%d,%s,%s,%.3f,%.3f,%.6f,%.0f\n",
               file, n, nd, stage, var, prep, up, kms, kms > 0 ? (prep+up)/kms : -1.0);
        fflush(stdout);
    };

    /* shared input vector (not counted as preprocessing) */
    std::vector<float> X(2*(size_t)n);
    srand(42);
    for (size_t i = 0; i < X.size(); ++i) X[i] = (float)rand()/RAND_MAX - 0.5f;
    float *dX = dupload(X), *dY;
    CUDA_CHECK(cudaMalloc(&dY, 2*(size_t)n*4));
    int SMS = 1; cudaDeviceGetAttribute(&SMS, cudaDevAttrMultiProcessorCount, 0);

    /* ==================== SpMV ==================== */

    /* ---- didx ---- */
    {
        auto t0 = clk::now();
        DidxPlan P = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
        auto t1 = clk::now();
        int *drp = dupload(P.rp), *doff = dupload(std::vector<int>(H.offsets));
        float *dv = dupload(P.val);
        unsigned char* d8 = nullptr; unsigned short* d16 = nullptr;
        if (P.wide) d16 = dupload(P.d16); else d8 = dupload(P.d8);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        int blocks = (n + TPB - 1)/TPB; size_t smem = (size_t)nd * 4;
        float kms;
        if (!P.wide){
            auto l=[&](){ spmv_didx_kernel<unsigned char,1><<<blocks,TPB,smem>>>(n,nd,drp,d8,dv,doff,dX,dY); };
            l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            kms = tms(l,10,iters);
        } else {
            auto l=[&](){ spmv_didx_kernel<unsigned short,1><<<blocks,TPB,smem>>>(n,nd,drp,d16,dv,doff,dX,dY); };
            l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            kms = tms(l,10,iters);
        }
        row("spmv","didx", ms_between(t0,t1), ms_between(t1,t2), kms);
        cudaFree(drp); cudaFree(doff); cudaFree(dv); if(d8)cudaFree(d8); if(d16)cudaFree(d16);
    }

    /* ---- didx_sym (MAINLINE symmetric mode, 2026-08-01): upper diagonals
     * stored once, dual-position reads, 1B slot|side meta.  Runs only when
     * H is numerically symmetric and Dup fits the 7-bit slot. ---- */
    {
        bool symok = true; int Dup = 0;
        std::unordered_map<int,int> ix; for (int i=0;i<nd;++i) ix[H.offsets[i]]=i;
        for (int i=0;i<nd && symok;++i){ int d=H.offsets[i]; if (d<0) continue;
            if (d>0){ auto it=ix.find(-d); if (it==ix.end()){ symok=false; break; }
                const float* a=&H.values[H.starts[i]]; const float* b=&H.values[H.starts[it->second]];
                for (int p=0;p<H.lengths[i];++p) if (a[p]!=b[p]){ symok=false; break; } }
            ++Dup; }
        if (symok && Dup > 128) symok = false;
        if (!symok) fprintf(stderr, "# didx_sym skipped: non-symmetric or Dup>128\n");
        else {
            auto t0 = clk::now();
            std::vector<int> upOff; std::vector<long long> upSt; std::vector<int> upLen;
            for (int i=0;i<nd;++i) if (H.offsets[i]>=0){ upOff.push_back(H.offsets[i]);
                upSt.push_back((long long)H.starts[i]); upLen.push_back(H.lengths[i]); }
            std::vector<int> rp2(n+1, 0);
            for (int s=0;s<Dup;++s){ const int d=upOff[s]; const size_t st=(size_t)upSt[s]; const int len=upLen[s];
                for (int p=0;p<len;++p){ if (H.values[st+p]==0.f) continue;
                    ++rp2[p+1]; if (d>0) ++rp2[p+d+1]; } }
            for (int r=0;r<n;++r) rp2[r+1]+=rp2[r];
            std::vector<unsigned char> smeta((size_t)rp2[n]);
            { std::vector<int> cur(rp2.begin(), rp2.end()-1);
              for (int s=0;s<Dup;++s){ const int d=upOff[s]; const size_t st=(size_t)upSt[s]; const int len=upLen[s];
                for (int p=0;p<len;++p){ if (H.values[st+p]==0.f) continue;
                    smeta[cur[p]++]=(unsigned char)s;
                    if (d>0) smeta[cur[p+d]++]=(unsigned char)(s|128); } } }
            auto t1 = clk::now();
            int* drp2=dupload(rp2); unsigned char* dm=dupload(smeta);
            int* dou=dupload(upOff); long long* dsu=dupload(upSt);
            float* dvf=dupload(H.values);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t2 = clk::now();
            const size_t smem = (size_t)Dup*12;
            int blocks2 = (n + TPB - 1)/TPB;
            auto l=[&](){ spmv_didx_sym_kernel<<<blocks2,TPB,smem>>>(n,Dup,drp2,dm,dou,dsu,dvf,dX,dY); };
            l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            row("spmv","didx_sym", ms_between(t0,t1), ms_between(t1,t2), tms(l,10,iters));
            cudaFree(drp2);cudaFree(dm);cudaFree(dou);cudaFree(dsu);cudaFree(dvf);
        }
    }

    /* ---- stream (gather) ---- (off by default 2026-08-01: dropped from the
     * paper's variant set; PREP_ALL_VARIANTS=1 re-enables for old ablations) */
    if (getenv("PREP_ALL_VARIANTS")) {
        auto t0 = clk::now();
        auto plan = build_dense_plan(H.offsets, H.starts, H.lengths);
        auto t1 = clk::now();
        float* dAv = dupload(H.values);
        DSeg* dS = dupload(plan);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        int nseg = (int)plan.size();
        const int ILP = (n / (TPB*4) >= 2*SMS) ? 4 : 1;
        int b1 = (n + TPB - 1)/TPB, b4 = (n + TPB*4 - 1)/(TPB*4);
        auto l=[&](){ if (ILP==4) spmv_gather_kernel<128,4,1><<<b4,TPB>>>(dAv,dS,nseg,dX,n,dY);
                      else        spmv_gather_kernel<128,1,1><<<b1,TPB>>>(dAv,dS,nseg,dX,n,dY); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("spmv","stream", ms_between(t0,t1), ms_between(t1,t2), tms(l,10,iters));
        cudaFree(dAv); cudaFree(dS);
    }

    /* ---- gopt (coarse zero-preproc + tiles + vec4) ---- */
    {
        auto t0 = clk::now();
        TilePlan T = build_coarse_plan(H, TPB*4, 1024);
        auto t1 = clk::now();
        float* dAv = dupload(T.packed); DSeg* dS = dupload(T.segs);
        int *dfp=dupload(T.fullPtr), *dfi=dupload(T.fullIdx),
            *dpp=dupload(T.partPtr), *dpi=dupload(T.partIdx);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        auto l=[&](){ spmv_tile_vec4_kernel<1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("spmv","gopt", ms_between(t0,t1), ms_between(t1,t2), tms(l,10,iters));
        cudaFree(dAv);cudaFree(dS);cudaFree(dfp);cudaFree(dfi);cudaFree(dpp);cudaFree(dpi);
    }

    /* ---- CSR zero-skip ---- (off by default 2026-08-01, same as stream) */
    if (getenv("PREP_ALL_VARIANTS")) {
        auto t0 = clk::now();
        std::vector<int> rp(n+1,0), ci; std::vector<float> cv;
        std::vector<std::vector<std::pair<int,float>>> rows(n);
        for (int k = 0; k < nd; ++k){
            int off = H.offsets[k]; size_t s = H.starts[k]; int len = H.lengths[k];
            for (int p = 0; p < len; ++p){
                float v = H.values[s + p]; if (v == 0.f) continue;
                int r = off >= 0 ? p : p - off, c = off >= 0 ? p + off : p;
                rows[r].push_back({c, v});
            }
        }
        for (int r = 0; r < n; ++r){ rp[r+1] = rp[r] + (int)rows[r].size();
            for (auto& e : rows[r]){ ci.push_back(e.first); cv.push_back(e.second); } }
        if (ci.size() > (size_t)INT32_MAX) {
            /* int32 row_ptr just overflowed above -> timings would be garbage */
            fprintf(stderr, "# csr_zskip skipped: true nnz %zu > INT32_MAX\n", ci.size());
        } else {
        auto t1 = clk::now();
        int *drp = dupload(rp), *dci = dupload(ci); float *dcv = dupload(cv);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        CsrView A{ n, drp, dci, dcv };
        int blocks = (n + TPB - 1)/TPB;
        auto l=[&](){ spmv_csr_scalar<<<blocks,TPB>>>(A, dX, dY); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("spmv","csr_zskip", ms_between(t0,t1), ms_between(t1,t2), tms(l,10,iters));
        cudaFree(drp); cudaFree(dci); cudaFree(dcv);
        }
    }

    /* ---- cuSPARSE CSR (prep = DIA->CSR convert; upload incl. handle/descr/buffer) ---- */
    if ((size_t)H.nnz > (size_t)INT32_MAX) {
        /* 32I impossible at this nnz -> run the 64I configuration instead
         * (variant "cusparse_csr_64i": the honest cuSPARSE baseline at scale;
         * 8B col_idx + 8B row_ptr entries, i.e. double index bytes vs 32I) */
        fprintf(stderr, "# cusparse_csr(32I) impossible (H nnz %zu); running 64I\n", (size_t)H.nnz);
        auto t0 = clk::now();
        CsrHost csr = dia_to_csr(H);
        std::vector<int64_t> ci64(csr.col_idx.begin(), csr.col_idx.end());
        auto t1 = clk::now();
        int64_t *drp = dupload(csr.row_ptr64), *dci = dupload(ci64);
        float* dv = dupload(csr.vals);
        cusparseHandle_t h; CUSP_CHECK(cusparseCreate(&h));
        cusparseSpMatDescr_t mH; cusparseDnVecDescr_t vIn, vOut;
        CUSP_CHECK(cusparseCreateCsr(&mH,n,n,csr.nnz,drp,dci,dv,
            CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateDnVec(&vIn,n,dX,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateDnVec(&vOut,n,dY,CUDA_R_32F));
        const float a1=1.f,b0=0.f; size_t bsz=0; void* dbuf=nullptr;
        CUSP_CHECK(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
            CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,&bsz));
        if (bsz) CUDA_CHECK(cudaMalloc(&dbuf,bsz));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        auto l=[&](){ CUSP_CHECK(cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
            CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,dbuf)); };
        l(); CUDA_CHECK(cudaDeviceSynchronize());
        row("spmv","cusparse_csr_64i", ms_between(t0,t1), ms_between(t1,t2), tms(l,10,iters));
        if (dbuf) cudaFree(dbuf);
        cusparseDestroySpMat(mH); cusparseDestroyDnVec(vIn); cusparseDestroyDnVec(vOut); cusparseDestroy(h);
        cudaFree(drp); cudaFree(dci); cudaFree(dv);
    } else {
        auto t0 = clk::now();
        CsrHost csr = dia_to_csr(H);
        auto t1 = clk::now();
        int *drp = dupload(csr.row_ptr), *dci = dupload(csr.col_idx);
        float *dv = dupload(csr.vals);
        cusparseHandle_t h; CUSP_CHECK(cusparseCreate(&h));
        cusparseSpMatDescr_t mH; cusparseDnVecDescr_t vIn, vOut;
        CUSP_CHECK(cusparseCreateCsr(&mH,n,n,csr.nnz,drp,dci,dv,
            CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateDnVec(&vIn,n,dX,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateDnVec(&vOut,n,dY,CUDA_R_32F));
        const float a1=1.f,b0=0.f; size_t bsz=0; void* dbuf=nullptr;
        CUSP_CHECK(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
            CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,&bsz));
        if (bsz) CUDA_CHECK(cudaMalloc(&dbuf,bsz));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        auto l=[&](){ CUSP_CHECK(cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
            CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,dbuf)); };
        l(); CUDA_CHECK(cudaDeviceSynchronize());
        row("spmv","cusparse_csr", ms_between(t0,t1), ms_between(t1,t2), tms(l,10,iters));
        if (dbuf) cudaFree(dbuf);
        cusparseDestroySpMat(mH); cusparseDestroyDnVec(vIn); cusparseDestroyDnVec(vOut); cusparseDestroy(h);
        cudaFree(drp); cudaFree(dci); cudaFree(dv);
    }
    cudaFree(dX); cudaFree(dY);

    /* ==================== SpMSpM ==================== */
    float* dHv = dupload(H.values);

    /* ---- ours_flat: C structure + pair plan + tiles ---- */
    {
        auto t0 = clk::now();
        Cstruct C = make_c(H, H);
        const int Cn = (int)C.offsets.size();
        std::unordered_map<int,int> bIdx; for (int i = 0; i < nd; ++i) bIdx[H.offsets[i]] = i;
        std::vector<int> pairPtr(Cn+1, 0); std::vector<GPair> pairs;
        for (int k = 0; k < Cn; ++k){
            int dc = C.offsets[k], minc = dc < 0 ? dc : 0;
            pairPtr[k] = (int)pairs.size();
            for (int ai = 0; ai < nd; ++ai){
                int da = H.offsets[ai], db = dc - da; if (db <= -n || db >= n) continue;
                auto it = bIdx.find(db); if (it == bIdx.end()) continue;
                GPair g; g.ab = H.starts[ai]; g.bb = H.starts[it->second];
                g.ash = (da < 0 ? da : 0) - minc;
                g.bsh = da + (db < 0 ? db : 0) - minc;
                g.al = H.lengths[ai]; g.bl = H.lengths[it->second];
                pairs.push_back(g);
            }
        }
        pairPtr[Cn] = (int)pairs.size();
        const int TILE = 256;
        const int ILP = n >= 65536 ? 4 : 1;
        const int POS = TILE * ILP;
        std::vector<int2> tiles;
        for (int k = 0; k < Cn; ++k) for (int ts = 0; ts < C.lengths[k]; ts += POS) tiles.push_back(make_int2(k, ts));
        auto t1 = clk::now();
        GPair* dPairs = dupload(pairs);
        int* dPairPtr = dupload(pairPtr);
        int2* dTiles = dupload(tiles);
        size_t* dCs = dupload(C.starts);
        int* dClen = dupload(std::vector<int>(C.lengths));
        float* dCv; CUDA_CHECK(cudaMalloc(&dCv, C.nnz*4));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        int nT = (int)tiles.size();
        auto l = [&](){
            if (ILP == 4) gather_flat_kernel<64,4><<<nT,TILE>>>(dHv,dHv,dTiles,dPairPtr,dPairs,dCv,dCs,dClen);
            else          gather_flat_kernel<64,1><<<nT,TILE>>>(dHv,dHv,dTiles,dPairPtr,dPairs,dCv,dCs,dClen);
        };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("spmspm","ours_flat", ms_between(t0,t1), ms_between(t1,t2), tms(l,10,iters));
        cudaFree(dPairs); cudaFree(dPairPtr); cudaFree(dTiles); cudaFree(dCv); cudaFree(dCs); cudaFree(dClen);
    }

    /* ---- HM atomic-scatter: C structure + diag lookup ---- */
    /* pre-guard (untimed): HM's layout is int32 (total_nz, diag_starts) — compute
     * the exact C nnz in size_t first; beyond INT32_MAX the row is N/A */
    size_t hm_nnz = 0;
    {
        std::vector<char> pres(2*(size_t)n - 1, 0);
        for (int ai = 0; ai < nd; ++ai) for (int bi = 0; bi < nd; ++bi) {
            long long dc = (long long)H.offsets[ai] + H.offsets[bi];
            if (dc > -n && dc < n) pres[dc + n - 1] = 1;
        }
        for (long long d = -(n-1); d <= n-1; ++d)
            if (pres[d + n - 1]) hm_nnz += (size_t)(n - llabs(d));
    }
    if (hm_nnz > (size_t)INT32_MAX) {
        fprintf(stderr, "# hm_atomic skipped: C nnz %zu exceeds HM int32 layout\n", hm_nnz);
    } else {
        auto t0 = clk::now();
        HMMatrix hA; hA.n=n; hA.num_diags=nd; hA.diag_offsets=H.offsets; hA.diag_lengths=H.lengths;
        hA.values=H.values; hA.total_nz=(int)H.nnz;
        hA.diag_starts.resize(nd); for (int i=0;i<nd;++i) hA.diag_starts[i]=(int)H.starts[i];
        HMMatrix hC = compute_c_hm_structure(hA,hA,n);
        std::vector<int> cLk = build_c_diag_lookup(hC,n);
        auto t1 = clk::now();
        float* hAv; int *hAo,*hAs,*hAl;
        auto up=[&](void**p,const void*s,size_t b){CUDA_CHECK(cudaMalloc(p,b));CUDA_CHECK(cudaMemcpy(*p,s,b,cudaMemcpyHostToDevice));};
        up((void**)&hAv,hA.values.data(),hA.values.size()*4); up((void**)&hAo,hA.diag_offsets.data(),nd*4);
        up((void**)&hAs,hA.diag_starts.data(),nd*4); up((void**)&hAl,hA.diag_lengths.data(),nd*4);
        float* hCv; int *hCo,*hCs,*hCl,*hClk; CUDA_CHECK(cudaMalloc(&hCv,(size_t)hC.total_nz*4));
        up((void**)&hCo,hC.diag_offsets.data(),hC.num_diags*4); up((void**)&hCs,hC.diag_starts.data(),hC.num_diags*4);
        up((void**)&hCl,hC.diag_lengths.data(),hC.num_diags*4); up((void**)&hClk,cLk.data(),cLk.size()*4);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        int nzA = hA.total_nz, blk = (nzA+255)/256;
        auto l=[&](){ CUDA_CHECK(cudaMemset(hCv,0,(size_t)hC.total_nz*4));
            hm_structured_sparse_matmul_kernel<<<blk,256>>>(hAv,hAo,hAs,hAl,nd, hAv,hAo,hAs,hAl,nd,
                hCv,hCo,hCs,hCl,hC.num_diags,hClk, nzA,n); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("spmspm","hm_atomic", ms_between(t0,t1), ms_between(t1,t2), tms(l,10,iters));
        cudaFree(hAv);cudaFree(hAo);cudaFree(hAs);cudaFree(hAl);
        cudaFree(hCv);cudaFree(hCo);cudaFree(hCs);cudaFree(hCl);cudaFree(hClk);
    }

    /* ---- ours_sym (L4) — LAST: kernel may crash on wide-band H ---- */
    {
        auto t0 = clk::now();
        Cstruct Cs = make_c(H, H, /*upper=*/true);
        std::vector<int> Aoff, Alen; std::vector<size_t> Ast; std::vector<int> Babs(n, -1);
        for (int i=0;i<nd;++i) if (H.offsets[i]>=0){ int a=H.offsets[i]; Babs[a]=(int)Aoff.size();
            Aoff.push_back(a); Alen.push_back(H.lengths[i]); Ast.push_back(H.starts[i]); }
        FlatSymPlan sp = build_flat_sym_plan(n, Aoff,Alen,Ast, Alen,Ast, Babs, Cs.offsets, Cs.lengths, 256*4);
        auto t1 = clk::now();
        float* dCsv; CUDA_CHECK(cudaMalloc(&dCsv, Cs.nnz*4));
        size_t* dCss = dupload(Cs.starts); int* dCsl = dupload(std::vector<int>(Cs.lengths));
        GPair* dSP = dupload(sp.pairs); int* dSPP = dupload(sp.pairPtr); int2* dST = dupload(sp.tiles);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk::now();
        double prep = ms_between(t0,t1), up = ms_between(t1,t2);
        row("spmspm","ours_sym", prep, up, -1.0);      /* prep-only row, survives a kernel crash */
        auto l = [&](){ gather_flat_sym_kernel<256,4><<<(int)sp.tiles.size(),256>>>(dHv,dHv,dST,dSPP,dSP,dCsv,dCss,dCsl); };
        CUDA_CHECK(cudaMemset(dCsv,0,Cs.nnz*4)); l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("spmspm","ours_sym", prep, up, tms(l,10,iters));
        cudaFree(dCsv); cudaFree(dCss); cudaFree(dCsl); cudaFree(dSP); cudaFree(dSPP); cudaFree(dST);
    }
    cudaFree(dHv);
    return 0;
}
