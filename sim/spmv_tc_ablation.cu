/* spmv_tc_ablation.cu — TC SpMV ablation (paper): base -> all optimizations.
 *   T0 full TC (tc_spmv_regdirect)         : whole padded band, no skip
 *   T1 +symmetric  (tc_spmv_sym)           : half recon, dual gather
 *   T2 +zero-tile-skip (tc_spmv_ztile)     : drop pure-zero 16-row tiles
 *   T3 +both (tc_spmv_symztile)            : half recon AND zero-tile-skip [best TC]
 * All verified bit-... (TF32) vs full dense-CUDA reference; reports %tiles_kept.
 * build: nvcc -O3 -std=c++17 -arch=sm_80 spmv_tc_ablation.cu \
 *        ../spmv/src/tc_spmv_regdirect_kernel.cu -o spmv_tc_ablation
 * run:   ./spmv_tc_ablation <dia.txt> [iters] [--csv] */
#include "dia_io.hpp"
#include "../spmv/src/dia_reconstruct.cuh"
#include "../spmv/src/cuda_dia_kernels.cuh"
#include "../spmv/src/tc_spmv_sym.cuh"
#include "../spmv/src/tc_spmv_ztile.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
template<class F> static float tms(F f,int wu,int it){ for(int i=0;i<wu;++i)f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for(int i=0;i<it;++i)f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it; }

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s <dia.txt> [iters] [--csv]\n",argv[0]);return 1;}
    bool csv=false; int iters=200;
    for(int i=2;i<argc;++i){ if(!strcmp(argv[i],"--csv"))csv=true; else iters=atoi(argv[i]); }
    DiaHost H=load_dia(argv[1]); int n=H.n,D=(int)H.offsets.size();

    // full recon (descending)
    std::vector<int> offd(H.offsets); std::sort(offd.begin(),offd.end(),std::greater<int>());
    std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<D;++k) idxOf[offd[k]+n]=k;
    std::vector<float> full((size_t)D*n,0.f);
    for(int i=0;i<D;++i){int off=H.offsets[i],sc=(off>=0)?off:0,len=H.lengths[i];size_t base=H.starts[i];int k=idxOf[off+n];
        for(int j=0;j<len;++j) full[(size_t)k*n+sc+j]=H.values[base+j];}
    ReconSym S=build_recon_sym(n,H.offsets,H.lengths,H.starts,H.values);

    // tile plans (host)
    TilePlan Pf=build_tile_plan(n,n,D,offd,full);
    TilePlanSym Ps=build_tile_plan_sym(n,n,S.num_sym,S.offsets,S.values);

    // device
    std::vector<float> hx(n); for(int i=0;i<n;++i) hx[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f;
    float *dx,*dy,*dFull,*dHalf; int *dOffF,*dOffS,*dPtrF,*dDiF,*dPtrSU,*dDiSU,*dPtrSL,*dDiSL;
    CK(cudaMalloc(&dx,(size_t)n*4)); CK(cudaMemcpy(dx,hx.data(),(size_t)n*4,cudaMemcpyHostToDevice)); CK(cudaMalloc(&dy,(size_t)n*4));
    CK(cudaMalloc(&dFull,full.size()*4)); CK(cudaMemcpy(dFull,full.data(),full.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dHalf,S.values.size()*4)); CK(cudaMemcpy(dHalf,S.values.data(),S.values.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dOffF,D*4)); CK(cudaMemcpy(dOffF,offd.data(),D*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dOffS,S.num_sym*4)); CK(cudaMemcpy(dOffS,S.offsets.data(),S.num_sym*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dPtrF,Pf.ptr.size()*4)); CK(cudaMemcpy(dPtrF,Pf.ptr.data(),Pf.ptr.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dDiF,Pf.diags.size()*4)); CK(cudaMemcpy(dDiF,Pf.diags.data(),Pf.diags.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dPtrSU,Ps.ptrU.size()*4)); CK(cudaMemcpy(dPtrSU,Ps.ptrU.data(),Ps.ptrU.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dDiSU,Ps.diagsU.size()*4)); CK(cudaMemcpy(dDiSU,Ps.diagsU.data(),Ps.diagsU.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dPtrSL,Ps.ptrL.size()*4)); CK(cudaMemcpy(dPtrSL,Ps.ptrL.data(),Ps.ptrL.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dDiSL,Ps.diagsL.size()*4)); CK(cudaMemcpy(dDiSL,Ps.diagsL.data(),Ps.diagsL.size()*4,cudaMemcpyHostToDevice));

    ReconView Rf{n,n,D,dOffF,dFull}, Rs{n,n,S.num_sym,dOffS,dHalf};
    auto k_dense=[&](){ cuda_spmv_dia<<<(n+255)/256,256>>>(Rf,dx,n,dy); };
    auto k_t0=[&](){ launch_tc_spmv_regdirect(Rf,dx,n,dy); };
    auto k_t1=[&](){ launch_tc_spmv_regdirect_sym(Rs,dx,n,dy); };
    auto k_t2=[&](){ launch_tc_spmv_ztile(Rf,dPtrF,dDiF,dx,n,dy); };
    auto k_t3=[&](){ launch_tc_spmv_symztile(Rs,dPtrSU,dDiSU,dPtrSL,dDiSL,dx,n,dy); };

    std::vector<float> yref(n),tmp(n); float ynorm=0,e0=0,e1=0,e2=0,e3=0;
    auto run=[&](std::function<void()> k,float&e){ CK(cudaMemset(dy,0,(size_t)n*4)); k(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(tmp.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost)); for(int i=0;i<n;++i) e=std::max(e,std::fabs(tmp[i]-yref[i])); };
    CK(cudaMemset(dy,0,(size_t)n*4)); k_dense(); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yref.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i) ynorm=std::max(ynorm,std::fabs(yref[i]));
    run(k_t0,e0); run(k_t1,e1); run(k_t2,e2); run(k_t3,e3);

    float t0=tms(k_t0,10,iters),t1=tms(k_t1,10,iters),t2=tms(k_t2,10,iters),t3=tms(k_t3,10,iters);
    if(!csv){
        printf("=== TC ablation  file=%s  n=%d  D=%d  Dsym=%d ===\n",argv[1],n,D,S.num_sym);
        printf("tiles kept: ztile %.1f%%   sym+ztile %.1f%%   | recon full %.1f MB, half %.1f MB\n",
               Pf.kept_pct,Ps.kept_pct, full.size()*4/1e6, S.values.size()*4/1e6);
        printf("[verify vs dense] T0=%.1e T1=%.1e T2=%.1e T3=%.1e (rel, |y|=%.2e)\n",e0/ynorm,e1/ynorm,e2/ynorm,e3/ynorm,ynorm);
        printf("  T0 full TC        : %8.5f ms\n",t0);
        printf("  T1 +symmetric     : %8.5f ms   (%.2fx vs T0)\n",t1,t0/t1);
        printf("  T2 +zero-tile-skip: %8.5f ms   (%.2fx vs T0)\n",t2,t0/t2);
        printf("  T3 +both [bestTC] : %8.5f ms   (%.2fx vs T0)\n",t3,t0/t3);
    } else {
        // TCABL,file,n,D,Dsym,kept_ztile,kept_symztile,t0,t1,t2,t3,s1,s2,s3,relerr_max
        float re=std::max(std::max(e0,e1),std::max(e2,e3))/(ynorm>0?ynorm:1);
        printf("TCABL,%s,%d,%d,%d,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.2e\n",
               argv[1],n,D,S.num_sym,Pf.kept_pct,Ps.kept_pct,t0,t1,t2,t3,t0/t1,t0/t2,t0/t3,re);
    }
    return 0;
}
