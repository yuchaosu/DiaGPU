/* ============================================================
 * e2e_speedup.cu   (the payoff number)
 *
 * Runs the IDENTICAL full Krylov/Taylor state evolution to t=final_time
 * (num_steps steps, K Taylor terms) two ways and times the whole loop:
 *
 *   ours    : this project's TF32 tensor-core diagonal SpMV
 *   cuSPARSE: CSR fp32 cusparseSpMV (CSR_ALG2) — standard same-precision-class baseline
 *
 * Everything else is identical (complex state as two real vectors since H is
 * real; two SpMVs per H-apply; same caxpy accumulation; same ping-pong). So
 * the end-to-end ratio isolates the kernel swap. Reports full-loop time, the
 * SpMV-only component, and the speedup. (#0 showed SpMV is 85-98% of device
 * time, so the full-loop speedup tracks the SpMV speedup.)
 *
 * fp32 cuSPARSE vs TF32 ours is the fair SPEED comparison (both single-precision
 * class); accuracy was settled separately in fidelity.cu (#4).
 * ============================================================ */
#include "dia_io.hpp"
#include "../spmv/src/dia_reconstruct.cuh"
#include "../spmv/src/spmv_zeroskip_kernels.cuh"   // CUDA-core zero-skip (the production kernel)
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>

#define CUDA_CHECK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); std::exit(1);} }while(0)
#define CUSPARSE_CHECK(x) do{ cusparseStatus_t s_=(x); if(s_!=CUSPARSE_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s_); std::exit(1);} }while(0)

__global__ void caxpy_f(int n,float cr,float ci,const float* xr,const float* xi,float* yr,float* yi){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ yr[i]+=cr*xr[i]-ci*xi[i]; yi[i]+=cr*xi[i]+ci*xr[i]; } }
static float xhash(int i){ unsigned h=(unsigned)i*2246822519u; h^=h>>13; h*=0x85ebca6bu; h^=h>>16;
  return (float)(h&0xFFFF)/65536.0f*2.0f-1.0f; }

int main(int argc,char**argv){
  if(argc<2){ std::fprintf(stderr,"usage: %s <dia_file> [final_time=1.2] [num_steps=1000] [K=auto]\n",argv[0]); return 1; }
  const char* path=argv[1];
  const double final_time=(argc>2)?atof(argv[2]):1.2;
  const int num_steps=(argc>3)?atoi(argv[3]):1000;
  int K=(argc>4)?atoi(argv[4]):-1;
  const double dt=final_time/num_steps;
  DiaHost H=load_dia(path); const int n=H.n;
  if(K<0){ K=6; }
  // OOM-sweep isolation: MODE=ours runs only our TF32 kernels, MODE=cusp only
  // the cuSPARSE baseline, so each side's memory ceiling is measured in its own
  // process (a baseline OOM can't mask our kernel's true ceiling). Default=both.
  const char* MODE=getenv("MODE");
  const bool do_ours=!(MODE&&!strcmp(MODE,"cusp"));
  const bool do_cusp=!(MODE&&!strcmp(MODE,"ours"));
  CsrHost csr; if(do_cusp) csr=dia_to_csr(H);   // host CSR is huge at high q — skip when ours-only

  std::printf("=== end-to-end speedup: ours(TF32) vs cuSPARSE(fp32) === MODE=%s\n",MODE?MODE:"both");
  std::printf("file: %s\nn=%d diags=%zu nnz=%lld K=%d steps=%d\n",
              path,n,H.offsets.size(),(long long)H.nnz,K,num_steps);

  std::vector<std::complex<double>> ck(K+1); ck[0]=1.0; std::complex<double> X(0.0,-dt);
  for(int k=1;k<=K;++k) ck[k]=ck[k-1]*X/(double)k;
  std::vector<float> ckr(K+1),cki(K+1); for(int k=0;k<=K;++k){ ckr[k]=ck[k].real(); cki[k]=ck[k].imag(); }

  const int TPB=256, BLK=(n+TPB-1)/TPB;
  std::vector<float> psr(n),psi(n); double nrm=0;
  for(int i=0;i<n;++i){ psr[i]=xhash(i); psi[i]=xhash(i+n); nrm+=(double)psr[i]*psr[i]+(double)psi[i]*psi[i]; }
  float s=1.0f/std::sqrt((float)nrm); for(int i=0;i<n;++i){ psr[i]*=s; psi[i]*=s; }

  float *wr,*wi,*tr,*ti,*ar,*ai,*pr,*pi;
  auto af=[&](float**p){ CUDA_CHECK(cudaMalloc(p,(size_t)n*sizeof(float))); };
  af(&wr);af(&wi);af(&tr);af(&ti);af(&ar);af(&ai);af(&pr);af(&pi);
  auto reset=[&](){ CUDA_CHECK(cudaMemcpy(pr,psr.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pi,psi.data(),n*sizeof(float),cudaMemcpyHostToDevice)); };

  // ---- cuSPARSE fp32 setup (skipped when MODE=ours) ----
  // 64-bit CSR indices: at q>=20 with wide fill the cumulative nnz exceeds
  // INT32_MAX (BH q24 nnz=2.48e9, O2 q20 nnz=2.89e9), which a 32I CSR cannot
  // represent. cuSPARSE forbids mixing 64I row offsets with 32I col indices, so
  // both are 64I (col values themselves are < n, but the type must match).
  int64_t *d_rp=nullptr,*d_ci=nullptr; float* d_v=nullptr;
  cusparseHandle_t h=nullptr; cusparseSpMatDescr_t mH=nullptr;
  cusparseDnVecDescr_t vIn=nullptr,vOut=nullptr; void* dbuf=nullptr;
  const float a1=1.f,b0=0.f;
  if(do_cusp){
    std::vector<int64_t> ci64(csr.col_idx.begin(), csr.col_idx.end());
    CUDA_CHECK(cudaMalloc(&d_rp,(n+1)*sizeof(int64_t))); CUDA_CHECK(cudaMalloc(&d_ci,csr.nnz*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_v,csr.nnz*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rp,csr.row_ptr64.data(),(n+1)*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci,ci64.data(),csr.nnz*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,csr.vals.data(),csr.nnz*sizeof(float),cudaMemcpyHostToDevice));
    CUSPARSE_CHECK(cusparseCreate(&h));
    CUSPARSE_CHECK(cusparseCreateCsr(&mH,n,n,csr.nnz,d_rp,d_ci,d_v,
      CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vIn,n,wr,CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vOut,n,tr,CUDA_R_32F));
    size_t bsz=0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
      CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,&bsz)); if(bsz) CUDA_CHECK(cudaMalloc(&dbuf,bsz));
  }
  auto spmv_cusp=[&](float* in,float* out){ CUSPARSE_CHECK(cusparseDnVecSetValues(vIn,in));
    CUSPARSE_CHECK(cusparseDnVecSetValues(vOut,out));
    CUSPARSE_CHECK(cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
      CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,dbuf)); };

  // ---- TF32 ours setup (skipped when MODE=cusp) ----
  int* d_off=nullptr; float* d_rv=nullptr; ReconView RV{n,n,0,nullptr,nullptr};
  if(do_ours){
    DiaMatrix DM; DM.rows=n; DM.cols=n; DM.offsets=H.offsets; DM.diag_lengths=H.lengths; DM.values=H.values;
    { std::vector<int> st(H.starts.size()); for(size_t i=0;i<H.starts.size();++i) st[i]=(int)H.starts[i]; DM.diag_starts=st; }
    ReconMatrix R=build_recon(DM);
    CUDA_CHECK(cudaMalloc(&d_off,R.num_diags*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_rv,R.values.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_off,R.diag_offsets.data(),R.num_diags*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rv,R.values.data(),R.values.size()*sizeof(float),cudaMemcpyHostToDevice));
    RV=ReconView{n,n,R.num_diags,d_off,d_rv};
  }

  // ---- CUDA-core zero-skip ours (the PRODUCTION kernel; headline vs cuSPARSE) ----
  int64_t* d_zrp=nullptr; unsigned char* d_zd8=nullptr; unsigned short* d_zd16=nullptr;
  float* d_zcv=nullptr; int* d_zoff=nullptr; int zD=0; bool z_u16=false; int64_t z_nnz=0;
  if(do_ours){
    std::vector<int> offd(H.offsets); std::sort(offd.begin(),offd.end(),std::greater<int>());
    zD=(int)offd.size(); z_u16=zD>256;
    std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<zD;++k) idxOf[offd[k]+n]=k;
    std::vector<int64_t> zrp(n+1,0);
    for(size_t i=0;i<H.offsets.size();++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i];
      for(int j=0;j<len;++j){ if(H.values[base+j]==0.f) continue; int r=(off>=0)?j:j-off; zrp[r+1]++; } }
    for(int r=0;r<n;++r) zrp[r+1]+=zrp[r];
    z_nnz=zrp[n];
    std::vector<unsigned char> zd8; std::vector<unsigned short> zd16;
    if(z_u16) zd16.resize(z_nnz); else zd8.resize(z_nnz);
    std::vector<float> zcv(z_nnz);
    std::vector<int64_t> cur(zrp.begin(),zrp.end()-1);
    for(size_t i=0;i<H.offsets.size();++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i]; int k=idxOf[off+n];
      for(int j=0;j<len;++j){ float v=H.values[base+j]; if(v==0.f) continue; int r=(off>=0)?j:j-off; int64_t p=cur[r]++;
        if(z_u16) zd16[p]=(unsigned short)k; else zd8[p]=(unsigned char)k; zcv[p]=v; } }
    CUDA_CHECK(cudaMalloc(&d_zrp,(size_t)(n+1)*8)); CUDA_CHECK(cudaMemcpy(d_zrp,zrp.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_zcv,(size_t)z_nnz*4)); CUDA_CHECK(cudaMemcpy(d_zcv,zcv.data(),(size_t)z_nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_zoff,zD*4)); CUDA_CHECK(cudaMemcpy(d_zoff,offd.data(),zD*4,cudaMemcpyHostToDevice));
    if(z_u16){ CUDA_CHECK(cudaMalloc(&d_zd16,(size_t)z_nnz*2)); CUDA_CHECK(cudaMemcpy(d_zd16,zd16.data(),(size_t)z_nnz*2,cudaMemcpyHostToDevice)); }
    else     { CUDA_CHECK(cudaMalloc(&d_zd8 ,(size_t)z_nnz  )); CUDA_CHECK(cudaMemcpy(d_zd8 ,zd8.data() ,(size_t)z_nnz  ,cudaMemcpyHostToDevice)); }
  }
  const int z_shb=zD*(int)sizeof(int);
  auto zs=[&](float* in,float* out){
    if(z_u16) spmv_dia_zeroskip_u16_i64<<<BLK,TPB,z_shb>>>((long long)n,zD,(const long long*)d_zrp,d_zd16,d_zcv,d_zoff,in,out);
    else      spmv_dia_zeroskip_i64    <<<BLK,TPB,z_shb>>>((long long)n,zD,(const long long*)d_zrp,d_zd8 ,d_zcv,d_zoff,in,out); };

  // one evolution step parameterized by the SpMV implementation
  auto step=[&](auto spmv){
    CUDA_CHECK(cudaMemcpy(ar,pr,n*sizeof(float),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(ai,pi,n*sizeof(float),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(wr,pr,n*sizeof(float),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(wi,pi,n*sizeof(float),cudaMemcpyDeviceToDevice));
    for(int k=1;k<=K;++k){ spmv(wr,tr); spmv(wi,ti); std::swap(wr,tr); std::swap(wi,ti);
      caxpy_f<<<BLK,TPB>>>(n,ckr[k],cki[k],wr,wi,ar,ai); }
    std::swap(pr,ar); std::swap(pi,ai);
  };
  auto tf32=[&](float* in,float* out){ launch_tc_spmv_regdirect(RV,in,n,out,0); };

  // fused step: one recon read applies H to BOTH components (yr=H*wr, yi=H*wi).
  auto step_fused=[&](){
    CUDA_CHECK(cudaMemcpy(ar,pr,n*sizeof(float),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(ai,pi,n*sizeof(float),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(wr,pr,n*sizeof(float),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(wi,pi,n*sizeof(float),cudaMemcpyDeviceToDevice));
    for(int k=1;k<=K;++k){ launch_tc_spmv_regdirect_fused(RV,wr,wi,n,tr,ti,0); std::swap(wr,tr); std::swap(wi,ti);
      caxpy_f<<<BLK,TPB>>>(n,ckr[k],cki[k],wr,wi,ar,ai); }
    std::swap(pr,ar); std::swap(pi,ai);
  };

  cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  auto time_full=[&](auto spmv){ reset();
    for(int i=0;i<5;++i) step(spmv);                       // warmup
    CUDA_CHECK(cudaDeviceSynchronize()); reset();
    CUDA_CHECK(cudaEventRecord(e0));
    for(int st=0;st<num_steps;++st) step(spmv);
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); return (double)ms; };
  auto time_spmv=[&](auto spmv,long calls){
    for(int i=0;i<20;++i) spmv(wr,tr);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(e0));
    for(long i=0;i<calls;++i) spmv(wr,tr);
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); return (double)ms; };

  // fused real+imag path (one recon read per apply)
  auto time_full_fused=[&](){ reset();
    for(int i=0;i<5;++i) step_fused();
    CUDA_CHECK(cudaDeviceSynchronize()); reset();
    CUDA_CHECK(cudaEventRecord(e0));
    for(int st=0;st<num_steps;++st) step_fused();
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); return (double)ms; };
  long calls=(long)num_steps*2*K;
  double full_zs=0,full_tf=0,full_cs=0,full_tf_fused=0,sp_zs=0,sp_tf=0,sp_cs=0;
  if(do_ours){ full_zs=time_full(zs); full_tf=time_full(tf32); full_tf_fused=time_full_fused();
               sp_zs=time_spmv(zs,calls); sp_tf=time_spmv(tf32,calls); }
  if(do_cusp){ full_cs=time_full(spmv_cusp); sp_cs=time_spmv(spmv_cusp,calls); }

  std::printf("\n--- full %d-step evolution (device, ms) ---\n",num_steps);
  if(do_ours){
    std::printf("  ours (CUDA zero-skip) : %9.3f ms   <-- production kernel\n",full_zs);
    std::printf("  ours (TF32 tensor)    : %9.3f ms   (TC reference; loses)\n",full_tf);
    std::printf("  ours (TF32 fused)     : %9.3f ms   (%.2fx vs separate)\n",full_tf_fused,full_tf/full_tf_fused);
  }
  if(do_cusp) std::printf("  cuSPARSE              : %9.3f ms\n",full_cs);
  if(do_ours&&do_cusp)
    std::printf("  SPEEDUP full: zero-skip=%.2fx,  TF32=%.2fx  (vs cuSPARSE)\n",full_cs/full_zs,full_cs/full_tf);
  std::printf("--- SpMV-only (%ld calls) ---\n",calls);
  if(do_ours) std::printf("  ours (zero-skip): %9.3f ms  (%.5f ms/call)\n",sp_zs,sp_zs/calls);
  if(do_ours) std::printf("  ours (TF32)     : %9.3f ms  (%.5f ms/call)\n",sp_tf,sp_tf/calls);
  if(do_cusp) std::printf("  cuSPARSE        : %9.3f ms  (%.5f ms/call)\n",sp_cs,sp_cs/calls);
  if(do_ours&&do_cusp) std::printf("  SPEEDUP SpMV: zero-skip=%.2fx,  TF32=%.2fx\n",sp_cs/sp_zs,sp_cs/sp_tf);
  // CSV: file,n,zD,nnz,steps,K,zs_full,tf_full,cusp_full,zs_vs_cusp,tf_vs_cusp,zs_spmv,cusp_spmv
  std::printf("E2E,%s,%d,%d,%lld,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.5f,%.5f\n",
    path,n,zD,(long long)z_nnz,num_steps,K,full_zs,full_tf,full_cs,
    (full_zs>0?full_cs/full_zs:-1),(full_tf>0?full_cs/full_tf:-1),
    (calls?sp_zs/calls:0),(calls?sp_cs/calls:0));
  std::printf("OOM_OK MODE=%s q_n=%d\n",MODE?MODE:"both",n);   // sentinel: reaching here = no OOM
  if(h) cusparseDestroy(h); return 0;
}
