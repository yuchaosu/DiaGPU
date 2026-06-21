// GPU Pauli e2e operator build: U_step = sum_{k=0}^K c_k H^k, c_k = (-i dt)^k/k!,
// by chaining symplectic product + simplify, TRUNCATING each step (drop |c|<thr) so the
// chain stays bounded (the enabler validated in pauli_e2e.py). Reports U term count + time.
//   ./pauli_e2e_build <pauli_file> [dt=2.4e-3] [K=6] [thr=1e-8] [out]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

struct AddF2 { __host__ __device__ double2 operator()(const double2&a,const double2&b)const{
  return make_double2(a.x+b.x,a.y+b.y);} };
struct SmallC {                                   // drop terms with |coeff| < atol
  double atol; SmallC(double a):atol(a){}
  __host__ __device__ bool operator()(const thrust::tuple<uint64_t,double2>& t)const{
    double2 v=thrust::get<1>(t); return (v.x*v.x+v.y*v.y) < atol*atol; }
};

__global__ void prod_kernel(long N,int Tb,const uint64_t*keyA,const double2*valA,
    const uint64_t*keyB,const double2*valB,uint64_t*keyO,double2*valO){
  long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; if(idx>=N) return;
  int a=idx/Tb,b=idx%Tb; uint64_t ka=keyA[a],kb=keyB[b];
  uint32_t xa=ka>>32, za=ka&0xffffffffu, xb=kb>>32, zb=kb&0xffffffffu;
  uint32_t xr=xa^xb, zr=za^zb; int sgn=(__popc(xa&zb)&1)?-1:1;
  double2 A=valA[a],B=valB[b];
  double cr=(A.x*B.x-A.y*B.y)*sgn, ci=(A.x*B.y+A.y*B.x)*sgn;
  keyO[idx]=(((uint64_t)xr)<<32)|zr; valO[idx]=make_double2(cr,ci);
}
struct ScaleC { double2 f; ScaleC(double2 ff):f(ff){}
  __host__ __device__ double2 operator()(const double2&v)const{
    return make_double2(v.x*f.x-v.y*f.y, v.x*f.y+v.y*f.x);} };

typedef thrust::device_vector<uint64_t> DK; typedef thrust::device_vector<double2> DV;

// sort + reduce_by_key + drop |c|<atol; rewrites k,v to the compacted result
static int simplify(DK& k, DV& v, double atol){
  thrust::sort_by_key(k.begin(),k.end(),v.begin());
  DK ok(k.size()); DV ov(v.size());
  auto e=thrust::reduce_by_key(k.begin(),k.end(),v.begin(),ok.begin(),ov.begin(),
                               thrust::equal_to<uint64_t>(),AddF2());
  int m=e.first-ok.begin(); ok.resize(m); ov.resize(m);
  if(atol>0){
    auto zb=thrust::make_zip_iterator(thrust::make_tuple(ok.begin(),ov.begin()));
    auto ze=thrust::make_zip_iterator(thrust::make_tuple(ok.end(),ov.end()));
    auto ne=thrust::remove_if(zb,ze,SmallC(atol)); m=ne-zb; ok.resize(m); ov.resize(m);
  }
  k.swap(ok); v.swap(ov); return m;
}

int main(int argc,char**argv){
  if(argc<2){ std::fprintf(stderr,"usage: %s <pauli_file> [dt] [K] [thr] [out]\n",argv[0]); return 1; }
  double dt=(argc>2)?atof(argv[2]):2.4e-3; int K=(argc>3)?atoi(argv[3]):6;
  double thr=(argc>4)?atof(argv[4]):1e-8; const char* outp=(argc>5)?argv[5]:nullptr;
  FILE* f=fopen(argv[1],"r"); if(!f){perror("open");return 1;}
  int T,nq; if(fscanf(f,"%d %d",&T,&nq)!=2){return 1;} if(nq>31){std::fprintf(stderr,"nq>31\n");return 1;}
  std::vector<uint64_t> hk(T); std::vector<double2> hv(T);
  for(int t=0;t<T;++t){ unsigned long long xi,zi; double cr,ci; if(fscanf(f,"%llu %llu %lf %lf",&xi,&zi,&cr,&ci)!=4) return 1;
    hk[t]=(((uint64_t)xi)<<32)|(uint32_t)zi; hv[t]=make_double2(cr,ci); }
  fclose(f);
  DK Hk=hk; DV Hv=hv;                                 // H
  DK Uk(1); DV Uv(1); Uk[0]=0ull; Uv[0]=make_double2(1.f,0.f);  // U = identity (c_0)
  DK Pk(1); DV Pv(1); Pk[0]=0ull; Pv[0]=make_double2(1.f,0.f);  // P_0 = identity

  std::printf("=== GPU Pauli e2e operator build: U = sum c_k H^k ===\n");
  std::printf("file:%s  H terms=%d nq=%d  dt=%.3e K=%d thr=%.0e\n",argv[1],T,nq,dt,K,thr);
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventRecord(e0));

  // mirror pauli_e2e.py: at step k, P_k = simplify(P_{k-1}*H, thr/|c_k|); U += c_k*P_k.
  std::complex<double> ck(1,0), X(0,-dt);
  for(int k=1;k<=K;++k){
    ck *= X/(double)k;                                  // c_k = (-i dt)^k/k!
    double atol_chain = (thr>0)? thr/std::abs(ck) : 0.0;
    long N=(long)Pk.size()*T;
    if(N==0) break;                                     // chain truncated to nothing
    DK nk(N); DV nv(N);
    int TPB=256; int BLK=(int)((N+TPB-1)/TPB);
    prod_kernel<<<BLK,TPB>>>(N,T,thrust::raw_pointer_cast(Pk.data()),thrust::raw_pointer_cast(Pv.data()),
      thrust::raw_pointer_cast(Hk.data()),thrust::raw_pointer_cast(Hv.data()),
      thrust::raw_pointer_cast(nk.data()),thrust::raw_pointer_cast(nv.data()));
    CK(cudaGetLastError());
    simplify(nk,nv,atol_chain); Pk.swap(nk); Pv.swap(nv);   // P_k = H^k (truncated)
    if(Pk.size()==0) break;
    // U <- simplify(U + c_k*P_k, thr)
    DK addk=Pk; DV addv=Pv;
    thrust::transform(addv.begin(),addv.end(),addv.begin(),ScaleC(make_double2(ck.real(),ck.imag())));
    int m=Uk.size(); Uk.resize(m+addk.size()); Uv.resize(m+addv.size());
    thrust::copy(addk.begin(),addk.end(),Uk.begin()+m); thrust::copy(addv.begin(),addv.end(),Uv.begin()+m);
    simplify(Uk,Uv,thr);
  }
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
  std::printf("U_step terms = %d   (built in %.3f ms on GPU)\n",(int)Uk.size(),ms);

  if(outp){ std::vector<uint64_t> ok(Uk.size()); std::vector<double2> ov(Uv.size());
    thrust::copy(Uk.begin(),Uk.end(),ok.begin()); thrust::copy(Uv.begin(),Uv.end(),ov.begin());
    FILE* g=fopen(outp,"w"); fprintf(g,"%d %d\n",(int)ok.size(),nq);
    for(size_t i=0;i<ok.size();++i) fprintf(g,"%u %u %.17g %.17g\n",(uint32_t)(ok[i]>>32),(uint32_t)(ok[i]&0xffffffffu),ov[i].x,ov[i].y);
    fclose(g); std::printf("wrote %s\n",outp); }
  return 0;
}
