/* ============================================================
 * spmv_dia_vs_tc.cu — SAME reconstruction (build_recon, as the TC kernel uses),
 * three ways: tensor-core MMA, dense CUDA-core, and DIA-native zero-skip CUDA.
 *
 * Answers: staying in the diagonal representation, does skipping the interior
 * zeros on CUDA cores beat the tensor-core kernel (and the dense CUDA kernel)?
 * All fed identical H; y cross-checked bit-for-bit.
 *
 * build: nvcc -O3 -std=c++17 -arch=sm_80 spmv_dia_vs_tc.cu \
 *        ../spmv/src/tc_spmv_regdirect_kernel.cu -o spmv_dia_vs_tc
 * run:   ./spmv_dia_vs_tc <dia.txt> [iters]
 * ============================================================ */
#include "dia_io.hpp"
#include "../spmv/src/dia_reconstruct.cuh"      // ReconView + launch_tc_spmv_regdirect
#include "../spmv/src/cuda_dia_kernels.cuh"     // cuda_spmv_dia (dense CUDA)
#include "../spmv/src/spmv_zeroskip_kernels.cuh"// spmv_dia_zeroskip
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
    if(D>256){ fprintf(stderr,"D=%d >256, 1-byte didx won't fit\n",D); return 1; }

    // ---- dense recon: values[k*cols+col], offsets DESCENDING (as build_recon) ----
    std::vector<int> off_desc(H.offsets); std::sort(off_desc.begin(),off_desc.end(),std::greater<int>());
    std::vector<int> idxOf(2*(size_t)n,-1);                    // offset -> descending index (via +n)
    for(int k=0;k<D;++k) idxOf[off_desc[k]+n]=k;
    std::vector<float> recon((size_t)D*n,0.f);
    for(int i=0;i<D;++i){ int off=H.offsets[i], sc=(off>=0)?off:0, len=H.lengths[i]; size_t base=H.starts[i];
        int k=idxOf[off+n]; for(int j=0;j<len;++j) recon[(size_t)k*n + sc+j]=H.values[base+j]; }

    // ---- DIA-native zero-skip: per row, (didx=descending-offset-index, val), zeros dropped ----
    std::vector<std::vector<std::pair<unsigned char,float>>> rows(n);
    size_t stored=0, kept=0;
    for(int i=0;i<D;++i){ int off=H.offsets[i], len=H.lengths[i]; size_t base=H.starts[i]; unsigned char k=(unsigned char)idxOf[off+n];
        for(int j=0;j<len;++j){ float v=H.values[base+j]; ++stored; if(v==0.f) continue; ++kept;
            int r = (off>=0)? j : j-off;                      // p=min(row,col): off>=0 r=j; off<0 r=j-off
            rows[r].push_back({k,v}); } }
    std::vector<int> rp(n+1,0); std::vector<unsigned char> didx; std::vector<float> cv; didx.reserve(kept); cv.reserve(kept);
    for(int r=0;r<n;++r){ for(auto&dv:rows[r]){ didx.push_back(dv.first); cv.push_back(dv.second); } rp[r+1]=(int)didx.size(); }
    size_t nnz=didx.size();
    double recon_mb=(double)D*n*4/1e6, dia_mb=((double)nnz*5+(double)(n+1)*4)/1e6;
    printf("=== SpMV same-recon TC vs CUDA  file=%s  n=%d  D=%d ===\n",argv[1],n,D);
    printf("band=%zu  nonzeros=%zu (fill %.1f%%) | matrix traffic: dense recon %.1f MB, DIA-zeroskip(val+1B) %.1f MB (%.2fx less)\n",
           stored,nnz,100.0*kept/stored,recon_mb,dia_mb,recon_mb/dia_mb);

    // ---- device ----
    float *dRec,*dx,*dy,*dCv; int *dOff,*dRp; unsigned char* dDidx;
    CK(cudaMalloc(&dRec,recon.size()*4)); CK(cudaMemcpy(dRec,recon.data(),recon.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dOff,D*4)); CK(cudaMemcpy(dOff,off_desc.data(),D*4,cudaMemcpyHostToDevice));
    std::vector<float> hx(n); for(int i=0;i<n;++i) hx[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f;
    CK(cudaMalloc(&dx,n*4)); CK(cudaMemcpy(dx,hx.data(),n*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy,n*4));
    CK(cudaMalloc(&dRp,(n+1)*4)); CK(cudaMemcpy(dRp,rp.data(),(n+1)*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dDidx,nnz)); CK(cudaMemcpy(dDidx,didx.data(),nnz,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dCv,nnz*4)); CK(cudaMemcpy(dCv,cv.data(),nnz*4,cudaMemcpyHostToDevice));

    ReconView R{n,n,D,dOff,dRec};
    int shbytes=D*sizeof(int);
    auto k_tc    =[&](){ launch_tc_spmv_regdirect(R,dx,n,dy); };
    auto k_dense =[&](){ cuda_spmv_dia<<<(n+255)/256,256>>>(R,dx,n,dy); };
    auto k_zs    =[&](){ spmv_dia_zeroskip<<<(n+255)/256,256,shbytes>>>(n,D,dRp,dDidx,dCv,dOff,dx,dy); };
    auto k_zsw   =[&](){ spmv_dia_zeroskip_warp<<<(n*32+255)/256,256,shbytes>>>(n,D,dRp,dDidx,dCv,dOff,dx,dy); };

    // ---- verify all vs dense ----
    std::vector<float> yt(n),yd(n),yz(n),yzw(n);
    CK(cudaMemset(dy,0,n*4)); k_dense(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yd.data(),dy,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy,0,n*4)); k_tc();    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yt.data(),dy,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy,0,n*4)); k_zs();    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yz.data(),dy,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemset(dy,0,n*4)); k_zsw();   CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yzw.data(),dy,n*4,cudaMemcpyDeviceToHost));
    float mt=0,mz=0,mzw=0,ynorm=0; for(int i=0;i<n;++i){ ynorm=std::max(ynorm,std::fabs(yd[i]));
        mt=std::max(mt,std::fabs(yt[i]-yd[i])); mz=std::max(mz,std::fabs(yz[i]-yd[i])); mzw=std::max(mzw,std::fabs(yzw[i]-yd[i])); }
    printf("[verify vs dense] TC=%.3e (rel %.1e)  zeroskip=%.3e  zeroskip_warp=%.3e   (|y|max=%.2e)\n",
           mt,mt/ynorm,mz,mzw,ynorm);

    // ---- time ----
    float tt=tms(k_tc,10,iters), td=tms(k_dense,10,iters), tz=tms(k_zs,10,iters), tzw=tms(k_zsw,10,iters);
    printf("=== timings (ms, avg of %d) ===\n",iters);
    printf("  tensor-core  tc_spmv_regdirect : %8.5f ms   (baseline)\n",tt);
    printf("  dense CUDA   cuda_spmv_dia      : %8.5f ms   (%.3fx vs TC)\n",td,tt/td);
    printf("  DIA zeroskip cuda (scalar)      : %8.5f ms   (%.3fx vs TC)\n",tz,tt/tz);
    printf("  DIA zeroskip cuda (warp/bal)    : %8.5f ms   (%.3fx vs TC)\n",tzw,tt/tzw);
    return 0;
}
