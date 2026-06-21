// test_fused.cu — validate + time the fused real+imag TC SpMV against two
// separate single-vector launches. Same matrix H (real), two RHS (xr, xi).
//   build: nvcc -O3 -arch=sm_90 test_fused.cu ../spmv/src/tc_spmv_regdirect_kernel.cu -o /tmp/test_fused
//   run:   /tmp/test_fused <dia_file>
#include "dia_io.hpp"
#include "../spmv/src/dia_reconstruct.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s <dia_file>\n",argv[0]);return 1;}
    DiaHost H=load_dia(argv[1]); int n=H.n;
    DiaMatrix DM; DM.rows=n; DM.cols=n; DM.offsets=H.offsets; DM.diag_lengths=H.lengths; DM.values=H.values;
    { std::vector<int> st(H.starts.size()); for(size_t i=0;i<H.starts.size();++i) st[i]=(int)H.starts[i]; DM.diag_starts=st; }
    ReconMatrix R=build_recon(DM);
    int* d_off; float* d_rv;
    CK(cudaMalloc(&d_off,R.num_diags*sizeof(int))); CK(cudaMalloc(&d_rv,R.values.size()*sizeof(float)));
    CK(cudaMemcpy(d_off,R.diag_offsets.data(),R.num_diags*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rv,R.values.data(),R.values.size()*sizeof(float),cudaMemcpyHostToDevice));
    ReconView RV{n,n,R.num_diags,d_off,d_rv};

    std::vector<float> hxr(n),hxi(n); srand(1);
    for(int i=0;i<n;++i){ hxr[i]=(float)rand()/RAND_MAX-0.5f; hxi[i]=(float)rand()/RAND_MAX-0.5f; }
    float *xr,*xi,*yr_s,*yi_s,*yr_f,*yi_f;
    CK(cudaMalloc(&xr,n*4)); CK(cudaMalloc(&xi,n*4));
    CK(cudaMalloc(&yr_s,n*4)); CK(cudaMalloc(&yi_s,n*4)); CK(cudaMalloc(&yr_f,n*4)); CK(cudaMalloc(&yi_f,n*4));
    CK(cudaMemcpy(xr,hxr.data(),n*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xi,hxi.data(),n*4,cudaMemcpyHostToDevice));

    // separate (baseline) and fused
    launch_tc_spmv_regdirect(RV,xr,n,yr_s); launch_tc_spmv_regdirect(RV,xi,n,yi_s);
    launch_tc_spmv_regdirect_fused(RV,xr,xi,n,yr_f,yi_f);
    CK(cudaDeviceSynchronize());

    std::vector<float> a(n),b(n),c(n),d(n);
    CK(cudaMemcpy(a.data(),yr_s,n*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(),yi_s,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.data(),yr_f,n*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(d.data(),yi_f,n*4,cudaMemcpyDeviceToHost));
    double mr=0,mi=0; for(int i=0;i<n;++i){ mr=fmax(mr,fabs(a[i]-c[i])); mi=fmax(mi,fabs(b[i]-d[i])); }
    printf("n=%d diags=%d  max|Δyr|=%.3e  max|Δyi|=%.3e  %s\n",n,R.num_diags,mr,mi,(mr<1e-5&&mi<1e-5)?"[MATCH]":"[MISMATCH]");

    // timing
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int IT=2000;
    for(int i=0;i<50;++i){ launch_tc_spmv_regdirect(RV,xr,n,yr_s); launch_tc_spmv_regdirect(RV,xi,n,yi_s); }
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
    for(int i=0;i<IT;++i){ launch_tc_spmv_regdirect(RV,xr,n,yr_s); launch_tc_spmv_regdirect(RV,xi,n,yi_s); }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ts; CK(cudaEventElapsedTime(&ts,e0,e1));
    for(int i=0;i<50;++i) launch_tc_spmv_regdirect_fused(RV,xr,xi,n,yr_f,yi_f);
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
    for(int i=0;i<IT;++i) launch_tc_spmv_regdirect_fused(RV,xr,xi,n,yr_f,yi_f);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float tf; CK(cudaEventElapsedTime(&tf,e0,e1));
    printf("separate 2x: %.4f ms/apply   fused: %.4f ms/apply   speedup = %.2fx\n",
           ts/IT, tf/IT, ts/tf);
    return 0;
}
