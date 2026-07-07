/* ============================================================
 * spmv_zeroskip_bench.cu — A/B: dense-DIA CUDA SpMV vs zero-skipped CSR SpMV.
 *
 * Tests whether skipping the interior structural zeros of the band (storing only
 * nonzeros) speeds up the memory-bound diagonal SpMV, or whether the col-index +
 * scattered-x-gather overhead eats the traffic saving. Same H, same y, verified.
 *
 * build: nvcc -O3 -std=c++17 -arch=sm_80 spmv_zeroskip_bench.cu -o spmv_zeroskip_bench
 * run:   ./spmv_zeroskip_bench <dia.txt> [iters]
 * ============================================================ */
#include "dia_io.hpp"
#include "../spmv/src/dia_reconstruct.cuh"     // ReconView
#include "../spmv/src/cuda_dia_kernels.cuh"    // cuda_spmv_dia (dense baseline)
#include "../spmv/src/spmv_zeroskip_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

template<class F> static float tms(F f,int wu,int it){
    for(int i=0;i<wu;++i)f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for(int i=0;i<it;++i)f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <dia.txt> [iters]\n",argv[0]); return 1; }
    int iters = argc>2 ? atoi(argv[2]) : 200;
    DiaHost H=load_dia(argv[1]); int n=H.n; const int D=H.offsets.size();

    // ---- dense recon (baseline): values[k*cols+col], offsets DESCENDING ----
    std::vector<int> off_desc(H.offsets); std::sort(off_desc.begin(),off_desc.end(),std::greater<int>());
    std::vector<float> recon((size_t)D*n,0.f);
    for(int k=0;k<D;++k){ int d=off_desc[k];
        int di=-1; for(int i=0;i<D;++i) if(H.offsets[i]==d){di=i;break;} if(di<0)continue;
        int sc=(d>=0)?d:0, len=H.lengths[di]; size_t base=H.starts[di];
        for(int j=0;j<len;++j) recon[(size_t)k*n + sc+j]=H.values[base+j]; }

    // ---- CSR (zero-skip): drop stored zeros. Entry mapping p=min(row,col):
    //   off>=0: val at (row=j, col=j+off);  off<0: val at (row=j+|off|, col=j).
    std::vector<std::vector<std::pair<int,float>>> rows(n);
    size_t stored=0, kept=0;
    for(int i=0;i<D;++i){ int off=H.offsets[i], len=H.lengths[i]; size_t base=H.starts[i];
        for(int j=0;j<len;++j){ float v=H.values[base+j]; ++stored; if(v==0.f) continue; ++kept;
            int r,c; if(off>=0){ r=j; c=j+off; } else { c=j; r=j-off; }
            rows[r].push_back({c,v}); } }
    std::vector<int> rp(n+1,0), ci; std::vector<float> cv; ci.reserve(kept); cv.reserve(kept);
    for(int r=0;r<n;++r){ for(auto&pc:rows[r]){ ci.push_back(pc.first); cv.push_back(pc.second); } rp[r+1]=(int)ci.size(); }
    size_t nnz=ci.size();
    double recon_mb=(double)D*n*4/1e6, csr_mb=((double)nnz*8+(double)(n+1)*4)/1e6;
    printf("=== SpMV zero-skip  file=%s  n=%d  D=%d ===\n",argv[1],n,D);
    printf("stored band=%zu  nonzeros=%zu  (fill %.1f%%)\n",stored,nnz,100.0*kept/stored);
    printf("matrix traffic: dense recon = %.1f MB   CSR(val+col) = %.1f MB   (%.2fx less)\n",
           recon_mb,csr_mb,recon_mb/csr_mb);

    // ---- device ----
    float *dRec,*dx,*dy; int *dRp,*dCi; float* dCv;
    CK(cudaMalloc(&dRec,recon.size()*4)); CK(cudaMemcpy(dRec,recon.data(),recon.size()*4,cudaMemcpyHostToDevice));
    std::vector<int> doff(off_desc); int* dOff; CK(cudaMalloc(&dOff,D*4)); CK(cudaMemcpy(dOff,doff.data(),D*4,cudaMemcpyHostToDevice));
    std::vector<float> hx(n); for(int i=0;i<n;++i) hx[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f;  // deterministic
    CK(cudaMalloc(&dx,n*4)); CK(cudaMemcpy(dx,hx.data(),n*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy,n*4));
    CK(cudaMalloc(&dRp,(n+1)*4)); CK(cudaMemcpy(dRp,rp.data(),(n+1)*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dCi,nnz*4)); CK(cudaMemcpy(dCi,ci.data(),nnz*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dCv,nnz*4)); CK(cudaMemcpy(dCv,cv.data(),nnz*4,cudaMemcpyHostToDevice));

    ReconView R{n,n,D,dOff,dRec};
    CsrView A{n,dRp,dCi,dCv};
    auto k_dense =[&](){ cuda_spmv_dia<<<(n+255)/256,256>>>(R,dx,n,dy); };
    auto k_scalar=[&](){ spmv_csr_scalar<<<(n+255)/256,256>>>(A,dx,dy); };
    auto k_warp  =[&](){ spmv_csr_warp<<<(n*32+255)/256,256>>>(A,dx,dy); };

    // ---- verify (csr vs dense) ----
    std::vector<float> yd(n),ys(n),yw(n);
    CK(cudaMemset(dy,0,n*4)); k_dense();  CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yd.data(),dy,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy,0,n*4)); k_scalar(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ys.data(),dy,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy,0,n*4)); k_warp();   CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yw.data(),dy,n*4,cudaMemcpyDeviceToHost));
    float ms=0,mw=0; for(int i=0;i<n;++i){ ms=std::max(ms,std::fabs(ys[i]-yd[i])); mw=std::max(mw,std::fabs(yw[i]-yd[i])); }
    printf("[verify] max|scalar-dense| = %.3e   max|warp-dense| = %.3e\n",ms,mw);

    // ---- time ----
    float td=tms(k_dense,10,iters), tsc=tms(k_scalar,10,iters), tw=tms(k_warp,10,iters);
    printf("=== timings (ms, avg of %d) ===\n",iters);
    printf("  dense-DIA  cuda_spmv_dia : %8.5f ms   (baseline)\n",td);
    printf("  zero-skip  csr_scalar    : %8.5f ms   (%.3fx vs dense)\n",tsc,td/tsc);
    printf("  zero-skip  csr_warp(bal) : %8.5f ms   (%.3fx vs dense)\n",tw,td/tw);
    return 0;
}
