/* spmv_sym_test.cu — verify + measure the symmetric (half-recon) SpMV.
 * Compares full dense-CUDA (reference), full TC, symmetric CUDA-core, and
 * symmetric TC — all must agree; reports timing + recon storage (full vs half).
 * build: nvcc -O3 -std=c++17 -arch=sm_80 spmv_sym_test.cu \
 *        ../spmv/src/tc_spmv_regdirect_kernel.cu -o spmv_sym_test
 * run:   ./spmv_sym_test <dia.txt> [iters] */
#include "dia_io.hpp"
#include "../spmv/src/dia_reconstruct.cuh"
#include "../spmv/src/cuda_dia_kernels.cuh"
#include "../spmv/src/tc_spmv_sym.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
template<class F> static float tms(F f,int wu,int it){ for(int i=0;i<wu;++i)f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for(int i=0;i<it;++i)f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it; }

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s <dia.txt> [iters]\n",argv[0]);return 1;}
    int iters=argc>2?atoi(argv[2]):200;
    DiaHost H=load_dia(argv[1]); int n=H.n; int D=(int)H.offsets.size();

    // full recon (descending) for the dense-CUDA reference + full TC
    std::vector<int> offd(H.offsets); std::sort(offd.begin(),offd.end(),std::greater<int>());
    std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<D;++k) idxOf[offd[k]+n]=k;
    std::vector<float> full((size_t)D*n,0.f);
    for(int i=0;i<D;++i){int off=H.offsets[i],sc=(off>=0)?off:0,len=H.lengths[i];size_t base=H.starts[i];int k=idxOf[off+n];
        for(int j=0;j<len;++j) full[(size_t)k*n+sc+j]=H.values[base+j];}

    // symmetric half recon
    ReconSym S=build_recon_sym(n,H.offsets,H.lengths,H.starts,H.values);

    // device
    std::vector<float> hx(n); for(int i=0;i<n;++i) hx[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f;
    float *dx,*dy,*dFull,*dHalf; int *dOffFull,*dOffHalf;
    CK(cudaMalloc(&dx,(size_t)n*4)); CK(cudaMemcpy(dx,hx.data(),(size_t)n*4,cudaMemcpyHostToDevice)); CK(cudaMalloc(&dy,(size_t)n*4));
    CK(cudaMalloc(&dFull,full.size()*4)); CK(cudaMemcpy(dFull,full.data(),full.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dOffFull,D*4)); CK(cudaMemcpy(dOffFull,offd.data(),D*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dHalf,S.values.size()*4)); CK(cudaMemcpy(dHalf,S.values.data(),S.values.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dOffHalf,S.num_sym*4)); CK(cudaMemcpy(dOffHalf,S.offsets.data(),S.num_sym*4,cudaMemcpyHostToDevice));

    ReconView Rf{n,n,D,dOffFull,dFull};
    ReconView Rs{n,n,S.num_sym,dOffHalf,dHalf};
    int shb=D*sizeof(int);
    auto k_dense=[&](){ cuda_spmv_dia<<<(n+255)/256,256>>>(Rf,dx,n,dy); };
    auto k_tc   =[&](){ launch_tc_spmv_regdirect(Rf,dx,n,dy); };
    auto k_scuda=[&](){ sym_spmv_cuda<<<(n+255)/256,256>>>(n,S.num_sym,dOffHalf,dHalf,dx,dy); };
    auto k_stc  =[&](){ launch_tc_spmv_regdirect_sym(Rs,dx,n,dy); };
    (void)shb;

    // reference = full dense-CUDA
    std::vector<float> yref(n),yt(n),ysc(n),ystc(n);
    CK(cudaMemset(dy,0,(size_t)n*4)); k_dense(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yref.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy,0,(size_t)n*4)); k_tc();    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yt.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy,0,(size_t)n*4)); k_scuda(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ysc.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy,0,(size_t)n*4)); k_stc();   CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ystc.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));

    float ynorm=0,et=0,esc=0,estc=0;
    for(int i=0;i<n;++i){ ynorm=std::max(ynorm,std::fabs(yref[i]));
        et=std::max(et,std::fabs(yt[i]-yref[i])); esc=std::max(esc,std::fabs(ysc[i]-yref[i])); estc=std::max(estc,std::fabs(ystc[i]-yref[i])); }

    float td=tms(k_dense,10,iters), tt=tms(k_tc,10,iters), tsc=tms(k_scuda,10,iters), tstc=tms(k_stc,10,iters);
    printf("=== SpMV symmetric test  file=%s  n=%d  D=%d  Dsym=%d ===\n",argv[1],n,D,S.num_sym);
    printf("recon storage: full %.1f MB   half %.1f MB   (%.2fx smaller)\n",
           full.size()*4/1e6, S.values.size()*4/1e6, (double)full.size()/S.values.size());
    printf("[verify vs dense]  TC=%.2e  sym-CUDA=%.2e  sym-TC=%.2e   (rel, |y|=%.2e)\n",
           et/ynorm,esc/ynorm,estc/ynorm,ynorm);
    printf("=== timings (ms, avg %d) ===\n",iters);
    printf("  full dense-CUDA : %8.5f\n",td);
    printf("  full TC         : %8.5f   (%.2fx vs dense)\n",tt,td/tt);
    printf("  sym  CUDA-core  : %8.5f   (%.2fx vs full dense, %.2fx vs full TC)\n",tsc,td/tsc,tt/tsc);
    printf("  sym  TC         : %8.5f   (%.2fx vs full TC)\n",tstc,tt/tstc);
    return 0;
}
