// Compressed-DIA SpMV vs cuSPARSE vs dense-DIA CUDA-core.  q<=16 for fast runs.
//   ./bench_comp <dia_file> [reps=200]
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
#include "../../spmv/src/compressed_dia.cuh"

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)
#define CUSP_CHECK(x) do{ cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s); std::exit(1);} }while(0)

static float xhash(uint32_t i){ i^=i>>16; i*=0x7feb352dU; i^=i>>15; i*=0x846ca68bU; i^=i>>16;
  return (float)((int)(i&0xffff)-32768)/32768.f; }

template<class T> static T* up(const std::vector<T>& v){ T* d; CUDA_CHECK(cudaMalloc(&d,std::max((size_t)1,v.size())*sizeof(T)));
  if(!v.empty()) CUDA_CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return d; }

int main(int argc,char**argv){
  if(argc<2){ std::fprintf(stderr,"usage: %s <dia_file> [reps=200]\n",argv[0]); return 1; }
  const char* path=argv[1]; const int reps=(argc>2)?atoi(argv[2]):200;
  DiaHost H=load_dia(path); const int n=H.n;

  // dense-DIA recon (for the CUDA-core baseline kernel)
  DiaMatrix DM; DM.rows=n; DM.cols=n; DM.offsets=H.offsets; DM.diag_lengths=H.lengths; DM.values=H.values;
  { std::vector<int> st(H.starts.size()); for(size_t i=0;i<H.starts.size();++i) st[i]=(int)H.starts[i]; DM.diag_starts=st; }
  ReconMatrix R=build_recon(DM);
  int* d_off=up(R.diag_offsets); float* d_rv=up(R.values);
  ReconView RV{n,n,R.num_diags,d_off,d_rv};

  // compressed
  CompressedDiaHost C=build_compressed(H);
  CompressedDiaView CV{ n, C.ndiag, up(C.off), up(C.is_dense), up(C.mag), up(C.pmask),
                        up(C.wbase), up(C.dbase), up(C.posbits), up(C.signbits), up(C.dense) };
  size_t dense_dia_bytes=(size_t)R.values.size()*4;

  std::printf("=== compressed-DIA SpMV ===\nfile: %s\nn=%d diags=%d  dense_diags=%d\n",
              path,n,C.ndiag,(int)std::count(C.is_dense.begin(),C.is_dense.end(),(uint8_t)1));
  std::printf("storage: dense-DIA=%.2f MB  compressed=%.2f MB  (%.1fx smaller)\n",
              dense_dia_bytes/1e6, C.struct_bytes()/1e6, (double)dense_dia_bytes/C.struct_bytes());

  // input + fp64 ref
  std::vector<float> hx(n); for(int i=0;i<n;++i) hx[i]=xhash(i);
  std::vector<double> ref(n,0.0);
  for(int i=0;i<n;++i){ double a=0; for(int k=0;k<R.num_diags;++k){ int c=i+R.diag_offsets[k];
      if(c>=0&&c<n) a+=(double)R.values[(size_t)k*n+c]*(double)hx[c]; } ref[i]=a; }
  float* d_x=up(hx); float* d_y; CUDA_CHECK(cudaMalloc(&d_y,n*sizeof(float)));
  auto relerr=[&](){ std::vector<float> y(n); CUDA_CHECK(cudaMemcpy(y.data(),d_y,n*sizeof(float),cudaMemcpyDeviceToHost));
    double num=0,den=0; for(int i=0;i<n;++i){ double dd=(double)y[i]-ref[i]; num+=dd*dd; den+=ref[i]*ref[i]; }
    return std::sqrt(num/(den+1e-30)); };

  cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  auto timed=[&](auto fn){ for(int i=0;i<20;++i) fn(); CUDA_CHECK(cudaDeviceSynchronize());
    double best=1e30; for(int r=0;r<5;++r){ CUDA_CHECK(cudaEventRecord(e0)); for(int i=0;i<reps;++i) fn();
      CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1)); float ms=0;
      CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); best=std::min(best,(double)ms/reps); } return best; };

  // cuSPARSE baseline
  CsrHost csr=dia_to_csr(H);
  int* d_rp=up(csr.row_ptr); int* d_ci=up(csr.col_idx); float* d_v=up(csr.vals);
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

  const int TPB=256, BLK=(n+TPB-1)/TPB;
  auto f_dense=[&](){ cuda_spmv_dia<<<BLK,TPB>>>(RV,d_x,n,d_y); };
  auto f_comp =[&](){ comp_spmv<<<BLK,TPB>>>(CV,d_x,d_y); };

  f_cusp();  CUDA_CHECK(cudaDeviceSynchronize()); double e_cs=relerr(); double t_cs=timed(f_cusp);
  f_dense(); CUDA_CHECK(cudaDeviceSynchronize()); double e_de=relerr(); double t_de=timed(f_dense);
  f_comp();  CUDA_CHECK(cudaDeviceSynchronize()); double e_co=relerr(); double t_co=timed(f_comp);

  std::printf("\n--- SpMV (ms/call, rel-L2 vs fp64) ---\n");
  std::printf("  cuSPARSE fp32      : %8.4f ms  relerr=%.2e  (1.00x)\n",t_cs,e_cs);
  std::printf("  CUDA-core dense-DIA: %8.4f ms  relerr=%.2e  %.2fx vs cuSPARSE\n",t_de,e_de,t_cs/t_de);
  std::printf("  CUDA-core COMPRESSED: %7.4f ms  relerr=%.2e  %.2fx vs cuSPARSE  %.2fx vs dense-DIA\n",
              t_co,e_co,t_cs/t_co,t_de/t_co);
  return 0;
}
