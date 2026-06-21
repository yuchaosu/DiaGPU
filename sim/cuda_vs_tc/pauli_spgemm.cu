// GPU SpMSpM in the Pauli basis: C = H*H by symplectic product + simplify.
// Term = c * Z^z X^x. Product: x=xa^xb, z=za^zb, c=ca*cb*(-1)^popcount(xa & zb).
// simplify = sort by (x,z) key then reduce_by_key (sum coeffs), drop |c|<eps.
//   ./pauli_spgemm <pauli_file> [out_file] [eps=1e-12]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

struct AddF2 { __host__ __device__ float2 operator()(const float2&a,const float2&b)const{
  return make_float2(a.x+b.x,a.y+b.y);} };

__global__ void product_kernel(long N,int Tb,
    const uint32_t*xa,const uint32_t*za,const float*cra,const float*cia,
    const uint32_t*xb,const uint32_t*zb,const float*crb,const float*cib,
    uint64_t*key,float2*val){
  long idx=blockIdx.x*(long)blockDim.x+threadIdx.x; if(idx>=N) return;
  int a=idx/Tb, b=idx%Tb;
  uint32_t xr=xa[a]^xb[b], zr=za[a]^zb[b];
  int sgn=(__popc(xa[a]&zb[b])&1)? -1:1;
  float cr=(cra[a]*crb[b]-cia[a]*cib[b])*sgn;
  float ci=(cra[a]*cib[b]+cia[a]*crb[b])*sgn;
  key[idx]=(((uint64_t)xr)<<32)|(uint64_t)zr;
  val[idx]=make_float2(cr,ci);
}

int main(int argc,char**argv){
  if(argc<2){ std::fprintf(stderr,"usage: %s <pauli_file> [out] [eps=1e-12]\n",argv[0]); return 1; }
  const char* path=argv[1]; const char* outp=(argc>2)?argv[2]:nullptr;
  double eps=(argc>3)?atof(argv[3]):1e-12;
  FILE* f=fopen(path,"r"); if(!f){ perror("open"); return 1; }
  int T,nq; if(fscanf(f,"%d %d",&T,&nq)!=2){ std::fprintf(stderr,"bad header\n"); return 1; }
  if(nq>31){ std::fprintf(stderr,"nq>31 needs 128-bit key (not impl)\n"); return 1; }
  std::vector<uint32_t> hx(T),hz(T); std::vector<float> hcr(T),hci(T);
  for(int t=0;t<T;++t){ unsigned long long xi,zi; double cr,ci;
    if(fscanf(f,"%llu %llu %lf %lf",&xi,&zi,&cr,&ci)!=4){ std::fprintf(stderr,"bad line %d\n",t); return 1; }
    hx[t]=(uint32_t)xi; hz[t]=(uint32_t)zi; hcr[t]=(float)cr; hci[t]=(float)ci; }
  fclose(f);
  std::printf("=== Pauli-basis SpMSpM C = H*H ===\nfile: %s\nH: %d terms, nq=%d  (candidates = %lld)\n",
              path,T,nq,(long long)T*T);

  uint32_t *dxa,*dza,*dxb,*dzb; float *dcra,*dcia,*dcrb,*dcib;
  auto upU=[&](std::vector<uint32_t>&v){ uint32_t*d; CK(cudaMalloc(&d,T*4)); CK(cudaMemcpy(d,v.data(),T*4,cudaMemcpyHostToDevice)); return d; };
  auto upF=[&](std::vector<float>&v){ float*d; CK(cudaMalloc(&d,T*4)); CK(cudaMemcpy(d,v.data(),T*4,cudaMemcpyHostToDevice)); return d; };
  dxa=upU(hx); dza=upU(hz); dcra=upF(hcr); dcia=upF(hci);
  dxb=dxa; dzb=dza; dcrb=dcra; dcib=dcia;   // B = A = H

  long N=(long)T*T;
  thrust::device_vector<uint64_t> key(N); thrust::device_vector<float2> val(N);
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  auto run=[&](){
    int TPB=256; long BLK=(N+TPB-1)/TPB;
    product_kernel<<<BLK,TPB>>>(N,T,dxa,dza,dcra,dcia,dxb,dzb,dcrb,dcib,
        thrust::raw_pointer_cast(key.data()), thrust::raw_pointer_cast(val.data()));
    thrust::sort_by_key(key.begin(),key.end(),val.begin());
    thrust::device_vector<uint64_t> okey(N); thrust::device_vector<float2> oval(N);
    auto end=thrust::reduce_by_key(key.begin(),key.end(),val.begin(),okey.begin(),oval.begin(),
                                   thrust::equal_to<uint64_t>(), AddF2());
    int m=end.first-okey.begin();
    return std::make_tuple(m, okey, oval);
  };

  // warmup + timing
  for(int i=0;i<3;++i){ auto r=run(); (void)r; } CK(cudaDeviceSynchronize());
  double best=1e30; int munique=0; thrust::device_vector<uint64_t> fkey; thrust::device_vector<float2> fval;
  for(int r=0;r<10;++r){ CK(cudaEventRecord(e0)); auto res=run(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms<best){ best=ms; munique=std::get<0>(res); fkey=std::get<1>(res); fval=std::get<2>(res);} }

  // count nonzero-coeff terms
  std::vector<uint64_t> hk(munique); std::vector<float2> hv(munique);
  CK(cudaMemcpy(hk.data(),thrust::raw_pointer_cast(fkey.data()),munique*8,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hv.data(),thrust::raw_pointer_cast(fval.data()),munique*sizeof(float2),cudaMemcpyDeviceToHost));
  int nz=0; for(int i=0;i<munique;++i) if(std::hypot(hv[i].x,hv[i].y)>eps) ++nz;
  std::printf("C = H*H : %d distinct (x,z) keys, %d with |c|>%.0e   (%.4f ms on GPU)\n",munique,nz,eps,best);

  if(outp){ FILE* g=fopen(outp,"w"); fprintf(g,"%d %d\n",nz,nq);
    for(int i=0;i<munique;++i){ if(std::hypot(hv[i].x,hv[i].y)<=eps) continue;
      uint32_t xr=(uint32_t)(hk[i]>>32), zr=(uint32_t)(hk[i]&0xffffffffu);
      fprintf(g,"%u %u %.17g %.17g\n",xr,zr,hv[i].x,hv[i].y); }
    fclose(g); std::printf("wrote %s\n",outp); }
  return 0;
}
