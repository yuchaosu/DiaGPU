/* spmv_ablation.cu — CUDA-core SpMV optimization ablation for the COMPLEX apply
 * (yr = H*xr, yi = H*xi) performed every evolution step. Parallel to
 * sim/spmspm_ablation.cu (L0->L4). All levels produce identical (yr,yi);
 * bit-exact vs the S0 dense reference. Metric = one complex apply.
 *
 *   S0 base       = dense-DIA (full padded band, zeros included)   cuda_spmv_dia x2
 *   S1 +zero-skip = drop structural zeros (byte minimization)      spmv_dia_zeroskip x2
 *   S2 +symmetric = store/compute d>=0 half (real-symmetric H)     sym_spmv_cuda x2
 *   S3 +fused r/i = one operand read for BOTH components           zeroskip_fused x1
 * S1 and S2 are two byte-reduction levers vs S0; S3 (zero-skip + fused) is the
 * best for the complex apply. Reports time[S0..S3] + speedup-vs-S0 + relerr.
 *
 * build: nvcc -O3 -std=c++17 -arch=sm_80 spmv_ablation.cu \
 *        ../spmv/src/tc_spmv_regdirect_kernel.cu -o spmv_ablation
 * run:   ./spmv_ablation <dia.txt> [iters] [--csv]
 */
#include "dia_io.hpp"
#include "../spmv/src/dia_reconstruct.cuh"
#include "../spmv/src/cuda_dia_kernels.cuh"
#include "../spmv/src/spmv_zeroskip_kernels.cuh"
#include "../spmv/src/tc_spmv_sym.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

template<class F> static float tms(F f,int wu,int it){
    for(int i=0;i<wu;++i)f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for(int i=0;i<it;++i)f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it; }

int main(int argc,char**argv){
    if(argc<2){ std::fprintf(stderr,"usage: %s <dia.txt> [iters] [--csv]\n",argv[0]); return 1; }
    bool csv=false; int iters=200;
    for(int i=2;i<argc;++i){ std::string a=argv[i]; if(a=="--csv")csv=true; else iters=atoi(argv[i]); }
    DiaHost H=load_dia(argv[1]); const int n=H.n; const int D=(int)H.offsets.size();
    const bool use_u16 = D>256;

    // deterministic complex state
    std::vector<float> hxr(n),hxi(n); for(int i=0;i<n;++i){ hxr[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f; hxi[i]=(float)(((i+n)*2654435761u)&1023)/1024.f-0.5f; }
    float *dxr,*dxi,*dyr,*dyi; for(float**p:{&dxr,&dxi,&dyr,&dyi}) CK(cudaMalloc(p,(size_t)n*4));
    CK(cudaMemcpy(dxr,hxr.data(),(size_t)n*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dxi,hxi.data(),(size_t)n*4,cudaMemcpyHostToDevice));
    const int blk=(n+255)/256;

    // ---- S0: dense recon ----
    DiaMatrix DM; DM.rows=n; DM.cols=n; DM.offsets=H.offsets; DM.diag_lengths=H.lengths; DM.values=H.values;
    { std::vector<int> st(H.starts.size()); for(size_t i=0;i<H.starts.size();++i) st[i]=(int)H.starts[i]; DM.diag_starts=st; }
    ReconMatrix R=build_recon(DM);
    int* dOff; float* dRec; CK(cudaMalloc(&dOff,R.num_diags*4)); CK(cudaMalloc(&dRec,R.values.size()*4));
    CK(cudaMemcpy(dOff,R.diag_offsets.data(),R.num_diags*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dRec,R.values.data(),R.values.size()*4,cudaMemcpyHostToDevice));
    ReconView RV{n,n,R.num_diags,dOff,dRec};
    auto s0=[&](){ cuda_spmv_dia<<<blk,256>>>(RV,dxr,n,dyr); cuda_spmv_dia<<<blk,256>>>(RV,dxi,n,dyi); };

    // ---- S1: zero-skip operands ----
    std::vector<int> off_desc(H.offsets); std::sort(off_desc.begin(),off_desc.end(),std::greater<int>());
    std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<D;++k) idxOf[off_desc[k]+n]=k;
    std::vector<int64_t> row64(n+1,0);
    for(int i=0;i<D;++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i];
        for(int j=0;j<len;++j){ if(H.values[base+j]==0.f)continue; int r=(off>=0)?j:j-off; row64[r+1]++; } }
    for(int r=0;r<n;++r) row64[r+1]+=row64[r]; const int64_t nnz=row64[n];
    std::vector<unsigned char> d8; std::vector<unsigned short> d16; if(use_u16)d16.resize(nnz); else d8.resize(nnz);
    std::vector<float> cv(nnz);
    { std::vector<int64_t> cur(row64.begin(),row64.end()-1);
      for(int i=0;i<D;++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i]; int k=idxOf[off+n];
        for(int j=0;j<len;++j){ float v=H.values[base+j]; if(v==0.f)continue; int r=(off>=0)?j:j-off; int64_t p=cur[r]++;
            if(use_u16)d16[p]=(unsigned short)k; else d8[p]=(unsigned char)k; cv[p]=v; } } }
    int64_t* dRp; CK(cudaMalloc(&dRp,(size_t)(n+1)*8)); CK(cudaMemcpy(dRp,row64.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
    float* dCv; CK(cudaMalloc(&dCv,(size_t)nnz*4)); CK(cudaMemcpy(dCv,cv.data(),(size_t)nnz*4,cudaMemcpyHostToDevice));
    int* dZoff; CK(cudaMalloc(&dZoff,D*4)); CK(cudaMemcpy(dZoff,off_desc.data(),D*4,cudaMemcpyHostToDevice));
    unsigned char* dD8=nullptr; unsigned short* dD16=nullptr;
    if(use_u16){ CK(cudaMalloc(&dD16,(size_t)nnz*2)); CK(cudaMemcpy(dD16,d16.data(),(size_t)nnz*2,cudaMemcpyHostToDevice)); }
    else       { CK(cudaMalloc(&dD8 ,(size_t)nnz  )); CK(cudaMemcpy(dD8 ,d8.data() ,(size_t)nnz  ,cudaMemcpyHostToDevice)); }
    const int shb=D*(int)sizeof(int); const long long* dRpll=(const long long*)dRp;
    auto s1=[&](){
      if(use_u16){ spmv_dia_zeroskip_u16_i64<<<blk,256,shb>>>((long long)n,D,dRpll,dD16,dCv,dZoff,dxr,dyr);
                   spmv_dia_zeroskip_u16_i64<<<blk,256,shb>>>((long long)n,D,dRpll,dD16,dCv,dZoff,dxi,dyi); }
      else       { spmv_dia_zeroskip_i64    <<<blk,256,shb>>>((long long)n,D,dRpll,dD8 ,dCv,dZoff,dxr,dyr);
                   spmv_dia_zeroskip_i64    <<<blk,256,shb>>>((long long)n,D,dRpll,dD8 ,dCv,dZoff,dxi,dyi); } };

    // ---- S2: symmetric half recon ----
    ReconSym RS=build_recon_sym(n,H.offsets,H.lengths,H.starts,H.values);
    int* dSoff; float* dSh; CK(cudaMalloc(&dSoff,RS.num_sym*4)); CK(cudaMalloc(&dSh,RS.values.size()*4));
    CK(cudaMemcpy(dSoff,RS.offsets.data(),RS.num_sym*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSh,RS.values.data(),RS.values.size()*4,cudaMemcpyHostToDevice));
    auto s2=[&](){ sym_spmv_cuda<<<blk,256>>>(n,RS.num_sym,dSoff,dSh,dxr,dyr);
                   sym_spmv_cuda<<<blk,256>>>(n,RS.num_sym,dSoff,dSh,dxi,dyi); };

    // ---- S3: fused zero-skip (one pass, both components) ----
    auto s3=[&](){
      if(use_u16) spmv_dia_zeroskip_u16_i64_fused<<<blk,256,shb>>>((long long)n,D,dRpll,dD16,dCv,dZoff,dxr,dxi,dyr,dyi);
      else        spmv_dia_zeroskip_i64_fused    <<<blk,256,shb>>>((long long)n,D,dRpll,dD8 ,dCv,dZoff,dxr,dxi,dyr,dyi); };

    // ---- verify each level vs S0 (dense), on (yr,yi) ----
    auto run_get=[&](std::function<void()> f,std::vector<float>&yr,std::vector<float>&yi){
        CK(cudaMemset(dyr,0,(size_t)n*4)); CK(cudaMemset(dyi,0,(size_t)n*4)); f(); CK(cudaDeviceSynchronize());
        yr.resize(n); yi.resize(n); CK(cudaMemcpy(yr.data(),dyr,(size_t)n*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(yi.data(),dyi,(size_t)n*4,cudaMemcpyDeviceToHost)); };
    std::vector<float> r0r,r0i,rr,ri; run_get(s0,r0r,r0i);
    double ynorm=0; for(int i=0;i<n;++i){ ynorm=std::max(ynorm,(double)std::fabs(r0r[i])); ynorm=std::max(ynorm,(double)std::fabs(r0i[i])); }
    auto relerr=[&](std::function<void()> f){ run_get(f,rr,ri); double m=0; for(int i=0;i<n;++i){ m=std::max(m,(double)std::fabs(rr[i]-r0r[i])); m=std::max(m,(double)std::fabs(ri[i]-r0i[i])); } return ynorm>0?m/ynorm:m; };
    double e1=relerr(s1), e2=relerr(s2), e3=relerr(s3), emax=std::max(std::max(e1,e2),e3);

    // ---- time each (one complex apply) ----
    float t0=tms(s0,10,iters), t1=tms(s1,10,iters), t2=tms(s2,10,iters), t3=tms(s3,10,iters);
    auto sp=[&](float b){ return b>0? t0/b : -1.f; };

    if(!csv){
        std::printf("=== SpMV ablation (complex apply)  file=%s  n=%d  D=%d  Dsym=%d  nnz=%lld ===\n",argv[1],n,D,RS.num_sym,(long long)nnz);
        std::printf("  S0 dense           : %.5f ms\n",t0);
        std::printf("  S1 +zero-skip      : %.5f ms   (%.2fx vs S0)\n",t1,sp(t1));
        std::printf("  S2 +symmetric      : %.5f ms   (%.2fx vs S0)\n",t2,sp(t2));
        std::printf("  S3 +fused r/i      : %.5f ms   (%.2fx vs S0)   <- best complex apply\n",t3,sp(t3));
        std::printf("  [verify vs dense] S1=%.2e S2=%.2e S3=%.2e%s\n",e1,e2,e3, e2>1e-3?"  (S2 large -> H not symmetric?)":"");
    } else {
        std::printf("SPVABL,%s,%d,%d,%d,%lld,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.3e\n",
                    argv[1],n,D,RS.num_sym,(long long)nnz,t0,t1,t2,t3,sp(t1),sp(t2),sp(t3),emax);
    }
    return 0;
}
