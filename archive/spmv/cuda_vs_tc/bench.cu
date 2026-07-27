// ============================================================================
// CUDA-core vs tensor-core diagonal SpMV / SpMM benchmark.
//
// Tests two ideas:
//   Q1  Drop the TC's wasted output elements with a flexible CUDA-core kernel.
//       -> cuda_spmv_dia (fp32, zero waste) vs tc_spmv_regdirect (tf32) vs cuSPARSE.
//   Q2  Batch vectors so the work amortizes (SpMM).
//       -> cuda_spmm_dia (NV vectors) vs cuSPARSE SpMM.  (A clean *TC* SpMM
//          needs a block-DIA reformulation; see NOTES.md.)
//
// Build: see Makefile.  Run: ./bench <dia_file> [reps=200] [NV=8]
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "../dia_io.hpp"
#include "../../spmv/src/cuda_dia_kernels.cuh"

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)
#define CUSP_CHECK(x) do{ cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s); std::exit(1);} }while(0)

// existing tensor-core kernel (compiled from spmv/src/tc_spmv_regdirect_kernel.cu)
void launch_tc_spmv_regdirect(ReconView R, const float* d_x, int x_size, float* d_y, cudaStream_t);

static float xhash(uint32_t i){ i^=i>>16; i*=0x7feb352dU; i^=i>>15; i*=0x846ca68bU; i^=i>>16;
  return (float)((int)(i&0xffff)-32768)/32768.f; }

int main(int argc,char**argv){
  if(argc<2){ std::fprintf(stderr,"usage: %s <dia_file> [reps=200] [NV=8]\n",argv[0]); return 1; }
  const char* path=argv[1];
  const int reps=(argc>2)?atoi(argv[2]):200;
  const int NV  =(argc>3)?atoi(argv[3]):8;       // only 8 wired below
  DiaHost H=load_dia(path); const int n=H.n;

  // ---- DIA -> ReconMatrix (same layout the TC kernel uses) ----
  DiaMatrix DM; DM.rows=n; DM.cols=n; DM.offsets=H.offsets;
  DM.diag_lengths=H.lengths; DM.values=H.values;
  { std::vector<int> st(H.starts.size()); for(size_t i=0;i<H.starts.size();++i) st[i]=(int)H.starts[i]; DM.diag_starts=st; }
  ReconMatrix R=build_recon(DM);
  int* d_off; float* d_rv;
  CUDA_CHECK(cudaMalloc(&d_off,R.num_diags*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_rv,R.values.size()*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_off,R.diag_offsets.data(),R.num_diags*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rv,R.values.data(),R.values.size()*sizeof(float),cudaMemcpyHostToDevice));
  ReconView RV{n,n,R.num_diags,d_off,d_rv};

  std::printf("=== CUDA-core vs tensor-core diagonal kernels ===\n");
  std::printf("file: %s\nn=%d  diags=%d  reps=%d  NV=%d\n",path,n,R.num_diags,reps,NV);

  // ---- input vector + CPU fp64 reference SpMV ----
  std::vector<float> hx(n); for(int i=0;i<n;++i) hx[i]=xhash(i);
  std::vector<double> ref(n,0.0);
  for(int i=0;i<n;++i){ double a=0;
    for(int k=0;k<R.num_diags;++k){ int c=i+R.diag_offsets[k];
      if(c>=0&&c<n) a+=(double)R.values[(size_t)k*n+c]*(double)hx[c]; }
    ref[i]=a; }
  float* d_x; float* d_y;
  CUDA_CHECK(cudaMalloc(&d_x,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_y,n*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x,hx.data(),n*sizeof(float),cudaMemcpyHostToDevice));

  auto relerr=[&](float* dy){ std::vector<float> y(n); CUDA_CHECK(cudaMemcpy(y.data(),dy,n*sizeof(float),cudaMemcpyDeviceToHost));
    double num=0,den=0; for(int i=0;i<n;++i){ double d=(double)y[i]-ref[i]; num+=d*d; den+=ref[i]*ref[i]; }
    return std::sqrt(num/(den+1e-30)); };

  cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  auto timed=[&](auto fn){ for(int i=0;i<20;++i) fn(); CUDA_CHECK(cudaDeviceSynchronize());
    double best=1e30; for(int r=0;r<5;++r){ CUDA_CHECK(cudaEventRecord(e0));
      for(int i=0;i<reps;++i) fn(); CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
      float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); best=std::min(best,(double)ms/reps); }
    return best; };

  // ---- cuSPARSE fp32 SpMV (baseline) ----
  CsrHost csr=dia_to_csr(H);
  int *d_rp,*d_ci; float* d_v;
  CUDA_CHECK(cudaMalloc(&d_rp,(n+1)*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_ci,csr.nnz*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_v,csr.nnz*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_rp,csr.row_ptr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ci,csr.col_idx.data(),csr.nnz*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v,csr.vals.data(),csr.nnz*sizeof(float),cudaMemcpyHostToDevice));
  cusparseHandle_t h; CUSP_CHECK(cusparseCreate(&h));
  cusparseSpMatDescr_t mH; CUSP_CHECK(cusparseCreateCsr(&mH,n,n,csr.nnz,d_rp,d_ci,d_v,
    CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
  cusparseDnVecDescr_t vIn,vOut; CUSP_CHECK(cusparseCreateDnVec(&vIn,n,d_x,CUDA_R_32F));
  CUSP_CHECK(cusparseCreateDnVec(&vOut,n,d_y,CUDA_R_32F));
  const float a1=1.f,b0=0.f; size_t bsz=0;
  CUSP_CHECK(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
    CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,&bsz)); void* dbuf=nullptr; if(bsz) CUDA_CHECK(cudaMalloc(&dbuf,bsz));
  auto f_cusp=[&](){ CUSP_CHECK(cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vIn,&b0,vOut,
    CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,dbuf)); };

  // ---- kernels ----
  const int TPB=256, BLK=(n+TPB-1)/TPB;
  auto f_tc  =[&](){ launch_tc_spmv_regdirect(RV,d_x,n,d_y,0); };
  auto f_cuda=[&](){ cuda_spmv_dia<<<BLK,TPB>>>(RV,d_x,n,d_y); };

  f_cusp(); CUDA_CHECK(cudaDeviceSynchronize()); double e_cusp=relerr(d_y); double t_cusp=timed(f_cusp);
  f_tc();   CUDA_CHECK(cudaDeviceSynchronize()); double e_tc  =relerr(d_y); double t_tc  =timed(f_tc);
  f_cuda(); CUDA_CHECK(cudaDeviceSynchronize()); double e_cuda=relerr(d_y); double t_cuda=timed(f_cuda);

  std::printf("\n--- SpMV (ms/call, rel-L2 vs fp64) ---\n");
  std::printf("  cuSPARSE fp32 : %8.4f ms   relerr=%.2e   (1.00x)\n",t_cusp,e_cusp);
  std::printf("  TC tf32       : %8.4f ms   relerr=%.2e   %.2fx vs cuSPARSE\n",t_tc,  e_tc,  t_cusp/t_tc);
  std::printf("  CUDA-core fp32: %8.4f ms   relerr=%.2e   %.2fx vs cuSPARSE   %.2fx vs TC\n",
              t_cuda,e_cuda,t_cusp/t_cuda,t_tc/t_cuda);

  // ---- SpMM NV vectors: CUDA-core vs cuSPARSE (per-vector cost) ----
  if(NV==8){
    std::vector<float> hX((size_t)NV*n); for(int v=0;v<NV;++v) for(int i=0;i<n;++i) hX[(size_t)v*n+i]=xhash(i+ (uint32_t)v*1000003u);
    float *dX,*dY; CUDA_CHECK(cudaMalloc(&dX,(size_t)NV*n*sizeof(float))); CUDA_CHECK(cudaMalloc(&dY,(size_t)NV*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX,hX.data(),(size_t)NV*n*sizeof(float),cudaMemcpyHostToDevice));
    auto f_cuda_mm=[&](){ cuda_spmm_dia<8><<<BLK,TPB>>>(RV,dX,n,dY); };

    // cuSPARSE SpMM (col-major dense, ld=n)
    cusparseDnMatDescr_t mX,mY; CUSP_CHECK(cusparseCreateDnMat(&mX,n,NV,n,dX,CUDA_R_32F,CUSPARSE_ORDER_COL));
    CUSP_CHECK(cusparseCreateDnMat(&mY,n,NV,n,dY,CUDA_R_32F,CUSPARSE_ORDER_COL));
    size_t mb=0; CUSP_CHECK(cusparseSpMM_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
      &a1,mH,mX,&b0,mY,CUDA_R_32F,CUSPARSE_SPMM_CSR_ALG2,&mb)); void* mbuf=nullptr; if(mb) CUDA_CHECK(cudaMalloc(&mbuf,mb));
    auto f_cusp_mm=[&](){ CUSP_CHECK(cusparseSpMM(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
      &a1,mH,mX,&b0,mY,CUDA_R_32F,CUSPARSE_SPMM_CSR_ALG2,mbuf)); };

    // cross-check CUDA SpMM vs cuSPARSE SpMM (col 0)
    f_cuda_mm(); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ya((size_t)NV*n); CUDA_CHECK(cudaMemcpy(ya.data(),dY,(size_t)NV*n*sizeof(float),cudaMemcpyDeviceToHost));
    f_cusp_mm(); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> yb((size_t)NV*n); CUDA_CHECK(cudaMemcpy(yb.data(),dY,(size_t)NV*n*sizeof(float),cudaMemcpyDeviceToHost));
    double num=0,den=0; for(size_t i=0;i<(size_t)NV*n;++i){ double d=(double)ya[i]-yb[i]; num+=d*d; den+=(double)yb[i]*yb[i]; }
    double mm_err=std::sqrt(num/(den+1e-30));

    double t_cusp_mm=timed(f_cusp_mm), t_cuda_mm=timed(f_cuda_mm);
    std::printf("\n--- SpMM NV=%d (ms/call, and ms/vector) ---\n",NV);
    std::printf("  cuSPARSE fp32 : %8.4f ms  (%.4f ms/vec)\n",t_cusp_mm,t_cusp_mm/NV);
    std::printf("  CUDA-core fp32: %8.4f ms  (%.4f ms/vec)   %.2fx vs cuSPARSE   xcheck rel=%.2e\n",
                t_cuda_mm,t_cuda_mm/NV,t_cusp_mm/t_cuda_mm,mm_err);
    std::printf("  [ref] CUDA-core SpMV ms/vec = %.4f  -> SpMM/SpMV per-vec ratio = %.2f\n",
                t_cuda, (t_cuda_mm/NV)/t_cuda);
  }
  return 0;
}
