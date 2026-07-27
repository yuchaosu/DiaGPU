/* ============================================================
 * spmspm_driver.cu — SpMSpM (C = H·H) paper benchmark: ours + baselines.
 *
 * Kernels (kernel-only, device-resident):
 *   ours_flat       : gather_flat — precomputed pairs, uniform position
 *                     tiles, register accumulation, atomic-free
 *   hm_atomic       : Haque et al. atomic-scatter diagonal kernel
 *                     (paper_hm_kernel.cu), + the C memset it requires
 *   diaq_fp64/_fp32 : HamSim/libdiaq product kernel (block per result
 *                     diagonal, pair loop outer), + its required memset
 *   cusparse_spgemm : cuSPARSE fp32 SpGEMM (time-only; own CSR output)
 * Cross-checks: hm and diaq vs ours_flat (max |diff|), reported in the
 * CSV check column; ours_flat itself is validated in sim/beat_hm.
 *
 * Output: SPMSPMCSV,file,n,D,Cd,nnzC,kernel,ms,check
 * usage: spmspm_driver <dia_file> [iters=100] [--no-cusparse]
 * ============================================================ */
#include "../dia_io.hpp"
#include "../../spmspm/gather_flat.cuh"
#include "../../spmspm/paper_hm.cuh"
#include "common.cuh"

#include <cusparse.h>
#include <unordered_map>
#include <string>
#include <cstring>

#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

struct Cstruct { std::vector<int> offsets, lengths; std::vector<size_t> starts; size_t nnz; };
static Cstruct make_c(const DiaHost& A, const DiaHost& B){
    int n = A.n; std::vector<char> present(2*(size_t)n-1, 0);
    for (int da : A.offsets) for (int db : B.offsets){ int dc = da+db; if (dc > -n && dc < n) present[dc+n-1] = 1; }
    Cstruct C; size_t off = 0;
    for (int d = -(n-1); d <= n-1; ++d){
        if (!present[d+n-1]) continue;
        int len = n - std::abs(d);
        C.offsets.push_back(d); C.starts.push_back(off); C.lengths.push_back(len); off += len;
    }
    C.nnz = off; return C;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s <dia_file> [iters=100] [--no-cusparse]\n", argv[0]); return 1; }
    int iters = argc > 2 ? atoi(argv[2]) : 100;
    bool do_cusp = true;
    for (int a = 2; a < argc; ++a) if (!strcmp(argv[a], "--no-cusparse")) do_cusp = false;

    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();
    const char* file = argv[1];
    Cstruct C = make_c(H, H);
    const int Cn = (int)C.offsets.size();

    auto row = [&](const char* kern, float ms, double chk){
        printf("SPMSPMCSV,%s,%d,%d,%d,%zu,%s,%.6f,%.3e\n",
               file, n, nd, Cn, C.nnz, kern, ms, chk);
    };

    /* ---------------- ours: gather_flat ---------------- */
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

    float* dHv = dupload(H.values);
    GPair* dPairs = dupload(pairs);
    int* dPairPtr = dupload(pairPtr);
    int2* dTiles = dupload(tiles);
    size_t* dCs = dupload(C.starts);
    int* dClen = dupload(std::vector<int>(C.lengths));
    float* dCv; CUDA_CHECK(cudaMalloc(&dCv, C.nnz*4));
    int nT = (int)tiles.size();
    auto launch_flat = [&](){
        if (ILP == 4) gather_flat_kernel<64,4><<<nT,TILE>>>(dHv,dHv,dTiles,dPairPtr,dPairs,dCv,dCs,dClen);
        else          gather_flat_kernel<64,1><<<nT,TILE>>>(dHv,dHv,dTiles,dPairPtr,dPairs,dCv,dCs,dClen);
    };
    launch_flat(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> C_flat(C.nnz);
    CUDA_CHECK(cudaMemcpy(C_flat.data(), dCv, C.nnz*4, cudaMemcpyDeviceToHost));
    row("ours_flat", tms(launch_flat,10,iters), 0.0);
    cudaFree(dPairs); cudaFree(dPairPtr); cudaFree(dTiles); cudaFree(dCv);

    auto maxdiff_flat = [&](const std::vector<float>& g){
        double m = 0; for (size_t i = 0; i < C.nnz; ++i) m = std::max(m, (double)std::fabs((double)g[i] - (double)C_flat[i])); return m; };

    /* ---------------- HM atomic-scatter (Haque et al.) ---------------- */
    {
        HMMatrix hA; hA.n=n; hA.num_diags=nd; hA.diag_offsets=H.offsets; hA.diag_lengths=H.lengths;
        hA.values=H.values; hA.total_nz=(int)H.nnz;
        hA.diag_starts.resize(nd); for (int i=0;i<nd;++i) hA.diag_starts[i]=(int)H.starts[i];
        HMMatrix hC = compute_c_hm_structure(hA,hA,n);
        std::vector<int> cLk = build_c_diag_lookup(hC,n);
        float* hAv; int *hAo,*hAs,*hAl;
        auto up=[&](void**p,const void*s,size_t b){CUDA_CHECK(cudaMalloc(p,b));CUDA_CHECK(cudaMemcpy(*p,s,b,cudaMemcpyHostToDevice));};
        up((void**)&hAv,hA.values.data(),hA.values.size()*4); up((void**)&hAo,hA.diag_offsets.data(),nd*4);
        up((void**)&hAs,hA.diag_starts.data(),nd*4); up((void**)&hAl,hA.diag_lengths.data(),nd*4);
        float* hCv; int *hCo,*hCs,*hCl,*hClk; CUDA_CHECK(cudaMalloc(&hCv,hC.total_nz*4));
        up((void**)&hCo,hC.diag_offsets.data(),hC.num_diags*4); up((void**)&hCs,hC.diag_starts.data(),hC.num_diags*4);
        up((void**)&hCl,hC.diag_lengths.data(),hC.num_diags*4); up((void**)&hClk,cLk.data(),cLk.size()*4);
        int nzA = hA.total_nz, blk = (nzA+255)/256;
        auto launch_hm=[&](){ CUDA_CHECK(cudaMemset(hCv,0,hC.total_nz*4));
            hm_structured_sparse_matmul_kernel<<<blk,256>>>(hAv,hAo,hAs,hAl,nd, hAv,hAo,hAs,hAl,nd,
                hCv,hCo,hCs,hCl,hC.num_diags,hClk, nzA,n); };
        launch_hm(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> gC(hC.total_nz);
        CUDA_CHECK(cudaMemcpy(gC.data(),hCv,hC.total_nz*4,cudaMemcpyDeviceToHost));
        row("hm_atomic", tms(launch_hm,10,iters), maxdiff_flat(gC));
        cudaFree(hAv);cudaFree(hAo);cudaFree(hAs);cudaFree(hAl);
        cudaFree(hCv);cudaFree(hCo);cudaFree(hCs);cudaFree(hCl);cudaFree(hClk);
    }

    /* ---------------- HamSim/libdiaq product kernel, fp64 + fp32 -------- */
    auto run_diaq = [&](auto vt_tag, const char* kern){
        using VT = decltype(vt_tag);
        std::vector<unsigned> pairOff(Cn+1,0); std::vector<int> pdA,pdB; std::vector<unsigned> psA,psB;
        for (int r = 0; r < Cn; ++r){
            pairOff[r] = (unsigned)pdA.size();
            for (int k = 0; k < nd; ++k){
                int dA = H.offsets[k], dB = C.offsets[r]-dA;
                auto it = bIdx.find(dB); if (it == bIdx.end()) continue;
                pdA.push_back(dA); pdB.push_back(dB);
                psA.push_back((unsigned)k); psB.push_back((unsigned)it->second);
            }
        }
        pairOff[Cn] = (unsigned)pdA.size();
        std::vector<VT> Ar(H.values.begin(), H.values.end()), Ai(H.nnz,(VT)0);
        std::vector<unsigned> aO(nd),aL(nd),cO(Cn),cL(Cn);
        for (int k=0;k<nd;++k){ aO[k]=(unsigned)H.starts[k]; aL[k]=(unsigned)H.lengths[k]; }
        for (int r=0;r<Cn;++r){ cO[r]=(unsigned)C.starts[r]; cL[r]=(unsigned)C.lengths[r]; }
        VT *dAr=dupload(Ar), *dAi=dupload(Ai);
        int *dCi=dupload(std::vector<int>(C.offsets)), *dpA=dupload(pdA), *dpB=dupload(pdB);
        unsigned *dcO=dupload(cO), *dcL=dupload(cL), *dpO=dupload(pairOff);
        unsigned *dsA=dupload(psA), *dsB=dupload(psB), *daO=dupload(aO), *daL=dupload(aL);
        VT *dCr,*dCim;
        CUDA_CHECK(cudaMalloc(&dCr, C.nnz*sizeof(VT)));
        CUDA_CHECK(cudaMalloc(&dCim,C.nnz*sizeof(VT)));
        auto l=[&](){
            cudaMemset(dCr,0,C.nnz*sizeof(VT)); cudaMemset(dCim,0,C.nnz*sizeof(VT));
            diaq_product_kernel<VT><<<Cn,256>>>(Cn,dCi,dcO,dcL,dpO,dpA,dpB,dsA,dsB,
                daO,daL,daO,daL,dAr,dAi,dAr,dAi,dCr,dCim);
        };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<VT> gC(C.nnz);
        CUDA_CHECK(cudaMemcpy(gC.data(),dCr,C.nnz*sizeof(VT),cudaMemcpyDeviceToHost));
        double md = 0; for (size_t i=0;i<C.nnz;++i) md = std::max(md,(double)std::fabs((double)gC[i]-(double)C_flat[i]));
        row(kern, tms(l,10,iters), md);
        cudaFree(dAr);cudaFree(dAi);cudaFree(dCi);cudaFree(dpA);cudaFree(dpB);
        cudaFree(dcO);cudaFree(dcL);cudaFree(dpO);cudaFree(dsA);cudaFree(dsB);
        cudaFree(daO);cudaFree(daL);cudaFree(dCr);cudaFree(dCim);
    };
    run_diaq(double(0), "diaq_fp64");
    run_diaq(float(0),  "diaq_fp32");
    cudaFree(dHv); cudaFree(dCs); cudaFree(dClen);

    /* ---------------- cuSPARSE SpGEMM fp32 (time-only) ---------------- */
    if (do_cusp) {
        CsrHost csr = dia_to_csr(H);
        int *drp=dupload(csr.row_ptr), *dci=dupload(csr.col_idx); float* dv=dupload(csr.vals);
        cusparseHandle_t hd; CUSP_CHECK(cusparseCreate(&hd));
        const float al=1,be=0; auto OP=CUSPARSE_OPERATION_NON_TRANSPOSE; auto AL=CUSPARSE_SPGEMM_DEFAULT;
        bool ok = true; float t_sum = 0; int t_cnt = 0;
        auto run_csp=[&](float* outms){
            cusparseSpMatDescr_t mA,mB,mC;
            CUSP_CHECK(cusparseCreateCsr(&mA,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
            CUSP_CHECK(cusparseCreateCsr(&mB,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
            int* Crp; CUDA_CHECK(cudaMalloc(&Crp,(n+1)*4));
            CUSP_CHECK(cusparseCreateCsr(&mC,n,n,0,Crp,nullptr,nullptr,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
            cusparseSpGEMMDescr_t sd; cusparseSpGEMM_createDescr(&sd);
            cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0); size_t b1=0; void* bu1=nullptr;
            cusparseStatus_t st=cusparseSpGEMM_workEstimation(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b1,nullptr);
            if (st==CUSPARSE_STATUS_INSUFFICIENT_RESOURCES) ok=false;
            else {
                CUDA_CHECK(cudaMalloc(&bu1,b1)); cusparseSpGEMM_workEstimation(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b1,bu1);
                size_t b2=0; void* bu2=nullptr;
                st=cusparseSpGEMM_compute(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b2,nullptr);
                if (st==CUSPARSE_STATUS_INSUFFICIENT_RESOURCES) ok=false;
                else { CUDA_CHECK(cudaMalloc(&bu2,b2)); cusparseSpGEMM_compute(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b2,bu2); cudaFree(bu2); }
                cudaFree(bu1);
            }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            if (outms) cudaEventElapsedTime(outms,e0,e1);
            cudaFree(Crp); cusparseSpGEMM_destroyDescr(sd);
            cusparseDestroySpMat(mA); cusparseDestroySpMat(mB); cusparseDestroySpMat(mC);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        };
        int reps = std::max(3, iters/10);
        for (int i = 0; i < reps && ok; ++i){ float a=0; run_csp(&a); t_sum += a; ++t_cnt; }
        if (ok && t_cnt) row("cusparse_spgemm", t_sum/t_cnt, -1.0);
        else             row("cusparse_spgemm", -1.0, -1.0);
        cudaFree(drp); cudaFree(dci); cudaFree(dv); cusparseDestroy(hd);
    }
    return 0;
}
