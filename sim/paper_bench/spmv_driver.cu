/* ============================================================
 * spmv_driver.cu — SpMV paper benchmark: ours + all linkable baselines.
 *
 * Kernels (kernel-only, device-resident, verified vs fp64 CPU):
 *   ours_didx_nv1 / _nv2 : locked kernel — per-nonzero diagonal-index
 *                          plan, fused NV=2 real/imag apply
 *   ours_gather_nv1/_nv2 : streaming plan (packed diagonals, position
 *                          tiles) — the SpMSpM-unified layout path
 *   row_dense_nv1        : padded dense-DIA row kernel (ablation base)
 *   cusparse_csr         : cuSPARSE CSR fp32, SPMV_CSR_ALG2
 *   diaq_fp64 / diaq_fp32: HamSim/libdiaq fused-path row kernel
 *                          (their published double + a same-precision
 *                          float build; complex arithmetic as shipped)
 * Drawloom is an external binary (.mtx input) — added by run_all.sh.
 *
 * Output: one CSV row per kernel:
 *   SPMVCSV,file,n,D,stored,nnz_true,kernel,ms,relerr
 *
 * usage: spmv_driver <dia_file> [iters=200]
 * ============================================================ */
#include "../dia_io.hpp"
#include "../../spmv/src/cuda_dia_kernels.cuh"
#include "../../spmv/src/spmv_gather.cuh"
#include "../../spmv/src/spmv_zeroskip_kernels.cuh"
#include "gather_opt.cuh"
#include "common.cuh"

#include <cusparse.h>
#include <string>

#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s <dia_file> [iters=200]\n", argv[0]); return 1; }
    int iters = argc > 2 ? atoi(argv[2]) : 200;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();
    const char* file = argv[1];

    size_t nnz_true = 0;
    for (float v : H.values) if (v != 0.f) ++nnz_true;

    auto row = [&](const char* kern, float ms, double err){
        printf("SPMVCSV,%s,%d,%d,%zu,%zu,%s,%.6f,%.3e\n",
               file, n, nd, H.nnz, nnz_true, kern, ms, err);
    };

    /* x (two components) + fp64 reference */
    std::vector<float> X(2*(size_t)n);
    srand(42);
    for (size_t i = 0; i < X.size(); ++i) X[i] = (float)rand()/RAND_MAX - 0.5f;
    std::vector<double> yr(n, 0.0), yi(n, 0.0);
    for (int k = 0; k < nd; ++k){
        int off = H.offsets[k]; size_t s = H.starts[k]; int len = H.lengths[k];
        for (int p = 0; p < len; ++p){
            int r = off >= 0 ? p : p - off, c = off >= 0 ? p + off : p;
            double v = (double)H.values[s + p];
            yr[r] += v * (double)X[c];
            yi[r] += v * (double)X[n + c];
        }
    }
    float *dX = dupload(X), *dY;
    CUDA_CHECK(cudaMalloc(&dY, 2*(size_t)n*4));
    std::vector<float> out(2*(size_t)n);
    const int TPB = 256;
    auto check1 = [&](){ CUDA_CHECK(cudaMemcpy(out.data(), dY, (size_t)n*4, cudaMemcpyDeviceToHost));
                         return rel_l2(yr, out.data(), n); };
    auto check2 = [&](){ CUDA_CHECK(cudaMemcpy(out.data(), dY, 2*(size_t)n*4, cudaMemcpyDeviceToHost));
                         return std::max(rel_l2(yr,out.data(),n), rel_l2(yi,out.data()+n,n)); };

    /* ---- ours: didx plan (locked) ---- */
    {
        DidxPlan P = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
        int *drp = dupload(P.rp), *doff = dupload(std::vector<int>(H.offsets));
        float *dv = dupload(P.val);
        int blocks = (n + TPB - 1)/TPB; size_t smem = (size_t)nd * 4;
        if (!P.wide){
            unsigned char* dx = dupload(P.d8);
            auto l1=[&](){ spmv_didx_kernel<unsigned char,1><<<blocks,TPB,smem>>>(n,nd,drp,dx,dv,doff,dX,dY); };
            auto l2=[&](){ spmv_didx_kernel<unsigned char,2><<<blocks,TPB,smem>>>(n,nd,drp,dx,dv,doff,dX,dY); };
            l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            row("ours_didx_nv1", tms(l1,10,iters), check1());
            l2(); CUDA_CHECK(cudaDeviceSynchronize());
            row("ours_didx_nv2", tms(l2,10,iters), check2());
            cudaFree(dx);
        } else {
            unsigned short* dx = dupload(P.d16);
            auto l1=[&](){ spmv_didx_kernel<unsigned short,1><<<blocks,TPB,smem>>>(n,nd,drp,dx,dv,doff,dX,dY); };
            auto l2=[&](){ spmv_didx_kernel<unsigned short,2><<<blocks,TPB,smem>>>(n,nd,drp,dx,dv,doff,dX,dY); };
            l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            row("ours_didx_nv1", tms(l1,10,iters), check1());
            l2(); CUDA_CHECK(cudaDeviceSynchronize());
            row("ours_didx_nv2", tms(l2,10,iters), check2());
            cudaFree(dx);
        }
        cudaFree(drp); cudaFree(doff); cudaFree(dv);
    }

    int SMS = 1; cudaDeviceGetAttribute(&SMS, cudaDevAttrMultiProcessorCount, 0);

    /* ---- ours: gather streaming plan (SpMSpM-unified layout) ---- */
    {
        float* dAv = dupload(H.values);
        auto plan = build_dense_plan(H.offsets, H.starts, H.lengths);
        DSeg* dS = dupload(plan);
        int nseg = (int)plan.size();
        /* ILP=4 only when the shrunken grid still fills the GPU (SM-count
         * based, not n-based: the n>=65536 heuristic tuned on 34 SMs starved
         * the H100 at n=65k-262k). */
        const int ILP = (n / (TPB*4) >= 2*SMS) ? 4 : 1;
        int b1 = (n + TPB - 1)/TPB, b4 = (n + TPB*4 - 1)/(TPB*4);
        auto l1=[&](){ if (ILP==4) spmv_gather_kernel<128,4,1><<<b4,TPB>>>(dAv,dS,nseg,dX,n,dY);
                       else        spmv_gather_kernel<128,1,1><<<b1,TPB>>>(dAv,dS,nseg,dX,n,dY); };
        auto l2=[&](){ if (ILP==4) spmv_gather_kernel<128,4,2><<<b4,TPB>>>(dAv,dS,nseg,dX,n,dY);
                       else        spmv_gather_kernel<128,1,2><<<b1,TPB>>>(dAv,dS,nseg,dX,n,dY); };
        l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("ours_gather_nv1", tms(l1,10,iters), check1());
        l2(); CUDA_CHECK(cudaDeviceSynchronize());
        row("ours_gather_nv2", tms(l2,10,iters), check2());
        cudaFree(dAv); cudaFree(dS);
    }

    /* ---- ours: optimized pure-diagonal (coarse zero-preproc + tiles + vec4).
     * Both the vec4 and the scalar-tile variant are reported so the H100 run
     * settles the occupancy threshold (34 vs 132 SMs) with data. ---- */
    {
        const int POS = TPB*4;
        TilePlan T = build_coarse_plan(H, POS, 1024);
        double reads = 0; for (auto& s : T.segs) reads += s.len;
        float* dAv = dupload(T.packed); DSeg* dS = dupload(T.segs);
        int *dfp=dupload(T.fullPtr), *dfi=dupload(T.fullIdx),
            *dpp=dupload(T.partPtr), *dpi=dupload(T.partIdx);
        auto lv1=[&](){ spmv_tile_vec4_kernel<1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        auto lv2=[&](){ spmv_tile_vec4_kernel<2><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        auto ls1=[&](){ spmv_tile_kernel<4,1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        lv1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("ours_gopt_nv1", tms(lv1,10,iters), check1());
        /* NV=2 reads plane v=1 as float4 from X+n: needs n%4==0 */
        if (n % 4 == 0) {
            lv2(); CUDA_CHECK(cudaDeviceSynchronize());
            row("ours_gopt_nv2", tms(lv2,10,iters), check2());
        } else fprintf(stderr, "# gopt_nv2 skipped: n%%4!=0 (float4 plane stride)\n");
        ls1(); CUDA_CHECK(cudaDeviceSynchronize());
        row("ours_gopt_s_nv1", tms(ls1,10,iters), check1());
        fprintf(stderr, "# gopt plan: %zu segs, reads %.1f%% of stored band\n",
                T.segs.size(), 100.0*reads/H.nnz);
        cudaFree(dAv);cudaFree(dS);cudaFree(dfp);cudaFree(dfi);cudaFree(dpp);cudaFree(dpi);
    }

    /* ---- ours_auto: the hybrid kernel — plan selected at build time from
     * (D, fill, rho).  Thresholds from the H100 sweep: streaming when D<=4
     * or fill >= 0.8*rho (byte-crossover law); gopt when D > 4096 (didx's
     * smem offset table kills occupancy); didx otherwise. ---- */
    {
        TilePlan T = build_coarse_plan(H, TPB*4, 1024);
        double reads = 0; for (auto& s : T.segs) reads += s.len;
        double rho  = reads / (double)H.nnz;
        double fill = (double)nnz_true / (double)H.nnz;
        /* D-threshold 3072, bracketed by four measurements on two GPUs:
         * work list wins at D=1243 (O2_16, A100, 4.0x) and D=2859 (vib_16,
         * H100, 4.9x); streaming wins at D=3359 (O2_20, A100, 1.68x) and
         * D=5367 (vib_18, H100, 1.27x).  Mechanism: the 4*D-byte smem
         * offset table erodes the work list's occupancy as D grows; the
         * exact crossover is hardware-dependent within (2859, 3359]. */
        const char* pick = (nd <= 4 || fill >= 0.8*rho) ? "stream"
                         : (nd > 3072 ? "gopt" : "didx");
        fprintf(stderr, "# auto pick=%s (D=%d fill=%.2f rho=%.2f)\n", pick, nd, fill, rho);

        if (!strcmp(pick, "stream")) {
            float* dAv = dupload(H.values);
            auto plan = build_dense_plan(H.offsets, H.starts, H.lengths);
            DSeg* dS = dupload(plan); int nseg = (int)plan.size();
            const int ILP = (n / (TPB*4) >= 2*SMS) ? 4 : 1;
            int b1 = (n + TPB - 1)/TPB, b4 = (n + TPB*4 - 1)/(TPB*4);
            auto l1=[&](){ if (ILP==4) spmv_gather_kernel<128,4,1><<<b4,TPB>>>(dAv,dS,nseg,dX,n,dY);
                           else        spmv_gather_kernel<128,1,1><<<b1,TPB>>>(dAv,dS,nseg,dX,n,dY); };
            auto l2=[&](){ if (ILP==4) spmv_gather_kernel<128,4,2><<<b4,TPB>>>(dAv,dS,nseg,dX,n,dY);
                           else        spmv_gather_kernel<128,1,2><<<b1,TPB>>>(dAv,dS,nseg,dX,n,dY); };
            l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            row("ours_auto_nv1", tms(l1,10,iters), check1());
            l2(); CUDA_CHECK(cudaDeviceSynchronize());
            row("ours_auto_nv2", tms(l2,10,iters), check2());
            cudaFree(dAv); cudaFree(dS);
        } else if (!strcmp(pick, "gopt")) {
            float* dAv = dupload(T.packed); DSeg* dS = dupload(T.segs);
            int *dfp=dupload(T.fullPtr), *dfi=dupload(T.fullIdx),
                *dpp=dupload(T.partPtr), *dpi=dupload(T.partIdx);
            auto l1=[&](){ spmv_tile_vec4_kernel<1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
            auto l2=[&](){ spmv_tile_vec4_kernel<2><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
            l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            row("ours_auto_nv1", tms(l1,10,iters), check1());
            if (n % 4 == 0) {
                l2(); CUDA_CHECK(cudaDeviceSynchronize());
                row("ours_auto_nv2", tms(l2,10,iters), check2());
            } else fprintf(stderr, "# auto_nv2 skipped: n%%4!=0 (float4 plane stride)\n");
            cudaFree(dAv);cudaFree(dS);cudaFree(dfp);cudaFree(dfi);cudaFree(dpp);cudaFree(dpi);
        } else {
            DidxPlan P = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
            int *drp = dupload(P.rp), *doff = dupload(std::vector<int>(H.offsets));
            float *dv = dupload(P.val);
            int blocks = (n + TPB - 1)/TPB; size_t smem = (size_t)nd * 4;
            auto go=[&](auto* dx){
                auto l1=[&](){ spmv_didx_kernel<std::remove_pointer_t<decltype(dx)>,1><<<blocks,TPB,smem>>>(n,nd,drp,dx,dv,doff,dX,dY); };
                auto l2=[&](){ spmv_didx_kernel<std::remove_pointer_t<decltype(dx)>,2><<<blocks,TPB,smem>>>(n,nd,drp,dx,dv,doff,dX,dY); };
                l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
                row("ours_auto_nv1", tms(l1,10,iters), check1());
                l2(); CUDA_CHECK(cudaDeviceSynchronize());
                row("ours_auto_nv2", tms(l2,10,iters), check2());
            };
            if (!P.wide){ unsigned char*  d8 = dupload(P.d8);  go(d8);  cudaFree(d8); }
            else        { unsigned short* d16= dupload(P.d16); go(d16); cudaFree(d16); }
            cudaFree(drp); cudaFree(doff); cudaFree(dv);
        }
    }

    /* ---- CSR scalar zero-skip (reference for the didx-vs-CSR delta) ---- */
    {
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
        int *drp = dupload(rp), *dci = dupload(ci); float *dcv = dupload(cv);
        CsrView A{ n, drp, dci, dcv };
        int blocks = (n + TPB - 1)/TPB;
        auto l=[&](){ spmv_csr_scalar<<<blocks,TPB>>>(A, dX, dY); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("csr_zskip_nv1", tms(l,10,iters), check1());
        cudaFree(drp); cudaFree(dci); cudaFree(dcv);
    }

    /* ---- padded dense-DIA row kernel (ablation base) ---- */
    {
        std::vector<float> pad((size_t)nd * n, 0.f);
        for (int k = 0; k < nd; ++k){
            int off = H.offsets[k]; size_t s = H.starts[k]; int len = H.lengths[k];
            int c0 = off >= 0 ? off : 0;
            for (int p = 0; p < len; ++p) pad[(size_t)k * n + c0 + p] = H.values[s + p];
        }
        float* dV = dupload(pad);
        int* dOff = dupload(std::vector<int>(H.offsets));
        ReconView R{ n, n, nd, dOff, dV };
        int blocks = (n + TPB - 1)/TPB;
        auto l1=[&](){ cuda_spmv_dia<<<blocks,TPB>>>(R, dX, n, dY); };
        l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        row("row_dense_nv1", tms(l1,10,iters), check1());
        cudaFree(dV); cudaFree(dOff);
    }

    /* ---- cuSPARSE CSR fp32 (ALG2), single vector ---- */
    if (!std::getenv("OURS_ONLY")) {
        CsrHost csr = dia_to_csr(H);
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
        auto l=[&](){ CUSP_CHECK(cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
            CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,dbuf)); };
        l(); CUDA_CHECK(cudaDeviceSynchronize());
        row("cusparse_csr", tms(l,10,iters), check1());
        if (dbuf) cudaFree(dbuf);
        cusparseDestroySpMat(mH); cusparseDestroyDnVec(vIn); cusparseDestroyDnVec(vOut); cusparseDestroy(h);
        cudaFree(drp); cudaFree(dci); cudaFree(dv);
    }

    /* ---- HamSim/libdiaq fused-path row kernel, fp64 + fp32 ---- */
    auto run_diaq = [&](auto vt_tag, const char* kern){
        using VT = decltype(vt_tag);
        std::vector<VT> Ar(H.values.begin(), H.values.end()), Ai(H.nnz, (VT)0);
        std::vector<unsigned int> offs(nd), lens(nd);
        for (int k = 0; k < nd; ++k){ offs[k]=(unsigned)H.starts[k]; lens[k]=(unsigned)H.lengths[k]; }
        std::vector<VT> hxr(n), hxi(n);
        for (int i = 0; i < n; ++i){ hxr[i]=(VT)X[i]; hxi[i]=(VT)X[n+i]; }
        VT *dAr=dupload(Ar), *dAi=dupload(Ai), *dxr=dupload(hxr), *dxi=dupload(hxi);
        int* dIdx = dupload(std::vector<int>(H.offsets));
        unsigned *dOf=dupload(offs), *dLe=dupload(lens);
        VT *dyr,*dyi; CUDA_CHECK(cudaMalloc(&dyr,n*sizeof(VT))); CUDA_CHECK(cudaMalloc(&dyi,n*sizeof(VT)));
        int blocks = (n + TPB - 1)/TPB;
        auto l=[&](){ diaq_spmv_row_kernel<VT><<<blocks,TPB>>>((unsigned)n,(unsigned)n,(unsigned)nd,
            dIdx,dOf,dLe,dAr,dAi,dxr,dxi,dyr,dyi); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<VT> gr(n), gi(n);
        CUDA_CHECK(cudaMemcpy(gr.data(),dyr,n*sizeof(VT),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gi.data(),dyi,n*sizeof(VT),cudaMemcpyDeviceToHost));
        double num=0,den=0,num2=0;
        for (int i=0;i<n;++i){ double d=yr[i]-(double)gr[i]; num+=d*d; den+=yr[i]*yr[i];
                               double e2=yi[i]-(double)gi[i]; num2+=e2*e2; }
        double err = std::sqrt(std::max(num,num2)/std::max(den,1e-300));
        row(kern, tms(l,10,iters), err);
        cudaFree(dAr);cudaFree(dAi);cudaFree(dxr);cudaFree(dxi);
        cudaFree(dIdx);cudaFree(dOf);cudaFree(dLe);cudaFree(dyr);cudaFree(dyi);
    };
    if (!std::getenv("OURS_ONLY")) {
        run_diaq(double(0), "diaq_fp64");
        run_diaq(float(0),  "diaq_fp32");
    }

    return 0;
}
