/* spmv_budget.cu — KERNEL comparison UNDER the diagonal budget.
 *
 * The kernel (this paper) and the diagonal budget (HamSim) compose: our zero-skip
 * kernel is D-sensitive (occupancy, per-row loop length, the shared-memory offset
 * array), so it loses to cuSPARSE on a wide matrix (large D) but WINS once the
 * budget narrows D. This harness sweeps the budget (keep top-B conjugate-pair
 * groups by norm, Hermiticity-preserving) and, at each level, times BOTH kernels
 * on the budgeted matrix + reports the evolution fidelity, so the crossover
 * (zs_vs_cusp crossing 1.0 as D shrinks) is measured directly.
 *
 * cuSPARSE is timed the TRUSTED way -- fixed vectors bound once, NO per-call
 * SetValues, 32-bit indices when nnz fits -- matching sim/spmv_dia_vs_tc.cu
 * (SetValues/64I inflate CSR_ALG2 and would fake a win).
 *
 * build: nvcc -O3 -std=c++17 -arch=sm_80 spmv_budget.cu \
 *        ../spmv/src/tc_spmv_regdirect_kernel.cu -o spmv_budget -lcusparse
 * run:   ./spmv_budget <dia.txt> [ftime=1.2] [steps=1000] [K=6] [iters=200]
 */
#include "dia_io.hpp"
#include "../spmv/src/spmv_zeroskip_kernels.cuh"
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)
#define SK(x) do{ cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s); std::exit(1);} }while(0)

__global__ void caxpy(int n,float cr,float ci,const float* xr,const float* xi,float* yr,float* yi){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ yr[i]+=cr*xr[i]-ci*xi[i]; yi[i]+=cr*xi[i]+ci*xr[i]; } }
__global__ void cp2(int n,const float* a,const float* b,float* c,float* d){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ c[i]=a[i]; d[i]=b[i]; } }
static float xhash(int i){ unsigned h=(unsigned)i*2246822519u; h^=h>>13; h*=0x85ebca6bu; h^=h>>16;
  return (float)(h&0xFFFF)/65536.f*2.f-1.f; }

static DiaHost subset(const DiaHost& H,const std::vector<char>& keep){
  DiaHost S; S.n=H.n; size_t off=0;
  for(size_t i=0;i<H.offsets.size();++i){ if(!keep[i]) continue;
    S.offsets.push_back(H.offsets[i]); S.lengths.push_back(H.lengths[i]); S.starts.push_back(off);
    S.values.insert(S.values.end(), H.values.begin()+H.starts[i], H.values.begin()+H.starts[i]+H.lengths[i]);
    off+=H.lengths[i]; }
  S.nnz=off; return S;
}

int main(int argc,char**argv){
  if(argc<2){ std::fprintf(stderr,"usage: %s <dia.txt> [ftime] [steps] [K] [iters]\n",argv[0]); return 1; }
  const char* path=argv[1];
  double ftime=argc>2?atof(argv[2]):1.2; int steps=argc>3?atoi(argv[3]):1000; int K=argc>4?atoi(argv[4]):6;
  int iters=argc>5?atoi(argv[5]):200; double dt=ftime/steps;
  DiaHost H=load_dia(path); const int n=H.n; const int Dfull=(int)H.offsets.size();
  const int TPB=256, BLK=(n+TPB-1)/TPB;

  std::vector<float> ckr(K+1),cki(K+1); { double rr=1,ii=0; ckr[0]=1;cki[0]=0;
    for(int k=1;k<=K;++k){ double nr=(ii*dt)/k, ni=(-rr*dt)/k; rr=nr; ii=ni; ckr[k]=rr; cki[k]=ii; } }

  // conjugate-pair budget groups, ranked by norm
  std::vector<double> dn(Dfull,0); for(int i=0;i<Dfull;++i){ double s=0; size_t b=H.starts[i];
    for(int j=0;j<H.lengths[i];++j){ double v=H.values[b+j]; s+=v*v; } dn[i]=s; }
  std::map<int,std::vector<int>> gmap; for(int i=0;i<Dfull;++i) gmap[std::abs(H.offsets[i])].push_back(i);
  std::vector<int> gkey; std::vector<double> gnorm;
  for(auto&kv:gmap){ double s=0; for(int i:kv.second) s+=dn[i]; gkey.push_back(kv.first); gnorm.push_back(s); }
  const int Ng=(int)gkey.size();
  std::vector<int> grank(Ng); for(int i=0;i<Ng;++i) grank[i]=i;
  std::sort(grank.begin(),grank.end(),[&](int a,int b){ return gnorm[a]>gnorm[b]; });
  std::vector<int> budgets; for(int b:{Ng,512,384,256,192,128,96,64,48,32,24,16,12,8,6,4,2,1})
    if(b<=Ng && (budgets.empty()||budgets.back()!=b)) budgets.push_back(b);

  float *dx,*dy,*re,*im,*ar,*ai,*wr,*wi,*tr,*ti;
  for(float**p:{&dx,&dy,&re,&im,&ar,&ai,&wr,&wi,&tr,&ti}) CK(cudaMalloc(p,(size_t)n*4));
  std::vector<float> hx(n),h0r(n),h0i(n); double nx=0,n0=0;
  for(int i=0;i<n;++i){ hx[i]=xhash(i); nx+=(double)hx[i]*hx[i]; h0r[i]=xhash(i); h0i[i]=xhash(i+n); n0+=(double)h0r[i]*h0r[i]+(double)h0i[i]*h0i[i]; }
  { float s=1.f/std::sqrt((float)nx); for(int i=0;i<n;++i) hx[i]*=s; }
  { float s=1.f/std::sqrt((float)n0); for(int i=0;i<n;++i){ h0r[i]*=s; h0i[i]*=s; } }
  CK(cudaMemcpy(dx,hx.data(),(size_t)n*4,cudaMemcpyHostToDevice));

  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_tight=[&](std::function<void()> f)->double{ for(int i=0;i<20;++i) f(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); for(int i=0;i<iters;++i) f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); return ms/iters; };
  std::vector<float> curR(n),curI(n),refR(n),refI(n);

  std::printf("SPBUD,file,n,Ng,budget,Dkept,nnz,fidelity,zs_ms,cusp_ms,zs_vs_cusp\n");
  for(size_t bi=0;bi<budgets.size();++bi){ int B=budgets[bi];
    std::vector<char> keep(Dfull,0); for(int r=0;r<B;++r) for(int i:gmap[gkey[grank[r]]]) keep[i]=1;
    DiaHost Hs=subset(H,keep); int Dk=(int)Hs.offsets.size();

    // zero-skip operands
    std::vector<int> offd(Hs.offsets); std::sort(offd.begin(),offd.end(),std::greater<int>());
    bool u16=Dk>256; std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<Dk;++k) idxOf[offd[k]+n]=k;
    std::vector<int64_t> zrp(n+1,0);
    for(size_t i=0;i<Hs.offsets.size();++i){ int off=Hs.offsets[i],len=Hs.lengths[i]; size_t base=Hs.starts[i];
      for(int j=0;j<len;++j){ if(Hs.values[base+j]==0.f) continue; int r=(off>=0)?j:j-off; zrp[r+1]++; } }
    for(int r=0;r<n;++r) zrp[r+1]+=zrp[r]; int64_t nnz=zrp[n];
    std::vector<unsigned char> zd8; std::vector<unsigned short> zd16; if(u16) zd16.resize(nnz); else zd8.resize(nnz);
    std::vector<float> zcv(nnz); std::vector<int> zcol(nnz); { std::vector<int64_t> cur(zrp.begin(),zrp.end()-1);
      for(size_t i=0;i<Hs.offsets.size();++i){ int off=Hs.offsets[i],len=Hs.lengths[i]; size_t base=Hs.starts[i]; int k=idxOf[off+n];
        for(int j=0;j<len;++j){ float v=Hs.values[base+j]; if(v==0.f) continue; int r=(off>=0)?j:j-off; int64_t p=cur[r]++;
          if(u16) zd16[p]=(unsigned short)k; else zd8[p]=(unsigned char)k; zcv[p]=v; zcol[p]=r+off; } } }
    int64_t* dRp; CK(cudaMalloc(&dRp,(size_t)(n+1)*8)); CK(cudaMemcpy(dRp,zrp.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
    float* dCv; CK(cudaMalloc(&dCv,(size_t)nnz*4)); CK(cudaMemcpy(dCv,zcv.data(),(size_t)nnz*4,cudaMemcpyHostToDevice));
    int* dOff; CK(cudaMalloc(&dOff,Dk*4)); CK(cudaMemcpy(dOff,offd.data(),Dk*4,cudaMemcpyHostToDevice));
    unsigned char* dD8=nullptr; unsigned short* dD16=nullptr;
    if(u16){ CK(cudaMalloc(&dD16,(size_t)nnz*2)); CK(cudaMemcpy(dD16,zd16.data(),(size_t)nnz*2,cudaMemcpyHostToDevice)); }
    else   { CK(cudaMalloc(&dD8 ,(size_t)nnz  )); CK(cudaMemcpy(dD8 ,zd8.data() ,(size_t)nnz  ,cudaMemcpyHostToDevice)); }
    const int shb=Dk*(int)sizeof(int); const long long* dRpll=(const long long*)dRp;
    auto zs=[&](float* in,float* out){
      if(u16) spmv_dia_zeroskip_u16_i64<<<BLK,TPB,shb>>>((long long)n,Dk,dRpll,dD16,dCv,dOff,in,out);
      else    spmv_dia_zeroskip_i64    <<<BLK,TPB,shb>>>((long long)n,Dk,dRpll,dD8 ,dCv,dOff,in,out); };

    // cuSPARSE CSR from the SAME zero-dropped nonzeros (zrp/zcol/zcv) as ours --
    // NOT dia_to_csr, which keeps the padded structural zeros (~13x nnz on O2) and
    // would slow cuSPARSE ~8x. 32I when nnz fits. Fixed vectors dx->dy, no SetValues.
    const int64_t csr_nnz = nnz; const bool i32 = csr_nnz <= 2147483000LL;
    cusparseHandle_t h; SK(cusparseCreate(&h)); cusparseSpMatDescr_t mH; cusparseDnVecDescr_t vI,vO;
    void *d_rp=nullptr,*d_ci=nullptr; float* d_v=nullptr;
    CK(cudaMalloc(&d_v,csr_nnz*4)); CK(cudaMemcpy(d_v,zcv.data(),csr_nnz*4,cudaMemcpyHostToDevice));
    if(i32){ std::vector<int> rp32(n+1); for(int i=0;i<=n;++i) rp32[i]=(int)zrp[i];
      CK(cudaMalloc(&d_rp,(size_t)(n+1)*4)); CK(cudaMalloc(&d_ci,(size_t)csr_nnz*4));
      CK(cudaMemcpy(d_rp,rp32.data(),(size_t)(n+1)*4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_ci,zcol.data(),(size_t)csr_nnz*4,cudaMemcpyHostToDevice));
      SK(cusparseCreateCsr(&mH,n,n,csr_nnz,d_rp,d_ci,d_v,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
    } else { std::vector<int64_t> ci64(zcol.begin(),zcol.end());
      CK(cudaMalloc(&d_rp,(size_t)(n+1)*8)); CK(cudaMalloc(&d_ci,(size_t)csr_nnz*8));
      CK(cudaMemcpy(d_rp,zrp.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_ci,ci64.data(),(size_t)csr_nnz*8,cudaMemcpyHostToDevice));
      SK(cusparseCreateCsr(&mH,n,n,csr_nnz,d_rp,d_ci,d_v,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F)); }
    SK(cusparseCreateDnVec(&vI,n,dx,CUDA_R_32F)); SK(cusparseCreateDnVec(&vO,n,dy,CUDA_R_32F));
    const float a1=1,b0=0; size_t bsz=0; void* dbuf=nullptr;
    SK(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vI,&b0,vO,CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,&bsz)); if(bsz) CK(cudaMalloc(&dbuf,bsz));

    double zs_ms  = time_tight([&]{ zs(dx,dy); });
    double cusp_ms= time_tight([&]{ SK(cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vI,&b0,vO,CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,dbuf)); });

    // fidelity: zs evolution vs full-budget reference
    CK(cudaMemcpy(re,h0r.data(),(size_t)n*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(im,h0i.data(),(size_t)n*4,cudaMemcpyHostToDevice));
    for(int s=0;s<steps;++s){ cp2<<<BLK,TPB>>>(n,re,im,ar,ai); cp2<<<BLK,TPB>>>(n,re,im,wr,wi);
      for(int k=1;k<=K;++k){ zs(wr,tr); zs(wi,ti); std::swap(wr,tr); std::swap(wi,ti); caxpy<<<BLK,TPB>>>(n,ckr[k],cki[k],wr,wi,ar,ai); }
      std::swap(re,ar); std::swap(im,ai); }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(curR.data(),re,(size_t)n*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(curI.data(),im,(size_t)n*4,cudaMemcpyDeviceToHost));
    if(bi==0){ refR=curR; refI=curI; }
    double ipr=0,ipi=0,na=0,nb=0; for(int i=0;i<n;++i){ ipr+=(double)refR[i]*curR[i]+(double)refI[i]*curI[i]; ipi+=(double)refR[i]*curI[i]-(double)refI[i]*curR[i];
      na+=(double)refR[i]*refR[i]+(double)refI[i]*refI[i]; nb+=(double)curR[i]*curR[i]+(double)curI[i]*curI[i]; }
    double fid=(na>0&&nb>0)? std::sqrt(ipr*ipr+ipi*ipi)/std::sqrt(na*nb):0.0;

    std::printf("SPBUD,%s,%d,%d,%d,%d,%lld,%.9f,%.5f,%.5f,%.4f\n",
      path,n,Ng,B,Dk,(long long)nnz,fid,zs_ms,cusp_ms,(zs_ms>0?cusp_ms/zs_ms:-1));
    std::fflush(stdout);
    cusparseDestroySpMat(mH); cusparseDestroyDnVec(vI); cusparseDestroyDnVec(vO); cusparseDestroy(h);
    if(dbuf) CK(cudaFree(dbuf)); CK(cudaFree(d_rp)); CK(cudaFree(d_ci)); CK(cudaFree(d_v));
    if(dD8) CK(cudaFree(dD8)); if(dD16) CK(cudaFree(dD16)); CK(cudaFree(dRp)); CK(cudaFree(dCv)); CK(cudaFree(dOff));
  }
  return 0;
}
