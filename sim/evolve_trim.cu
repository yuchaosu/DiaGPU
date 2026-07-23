/* evolve_trim.cu — does the SpMV magnitude-trim error COMPOUND over a full
 * time evolution? Fixed-order Taylor step  psi_{n+1}=sum_{m=0}^K (-i dt H)^m/m! psi_n
 * for `steps` steps, using the DIA zero-skip SpMV with H trimmed at |v|<=tau*max|H|.
 * Same fp32 kernel for every tau => isolates the trim (not fp32-vs-fp64).
 * Reference = tau=0 evolution. Reports final-state fidelity |<ref|trim>| and L2
 * error vs reference, plus norm drift and nnz kept.
 *
 * build: nvcc -O3 -std=c++17 -arch=sm_80 evolve_trim.cu -o evolve_trim
 * run:   ./evolve_trim <dia.txt> [steps=1000] [K=12]
 */
#include "dia_io.hpp"
#include "../spmv/src/spmv_zeroskip_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

struct Packed { std::vector<int64_t> row64; std::vector<unsigned char> d8;
                std::vector<unsigned short> d16; std::vector<float> cv; int64_t nnz; };
static Packed pack(const DiaHost& H,int n,int D,const std::vector<int>& idxOf,double tau_abs,bool u16){
    Packed P; P.row64.assign(n+1,0);
    for(int i=0;i<D;++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i];
        for(int j=0;j<len;++j){ if(std::fabs((double)H.values[base+j])<=tau_abs) continue; int r=(off>=0)?j:j-off; P.row64[r+1]++; } }
    for(int r=0;r<n;++r) P.row64[r+1]+=P.row64[r];
    P.nnz=P.row64[n]; if(u16)P.d16.resize(P.nnz); else P.d8.resize(P.nnz); P.cv.resize(P.nnz);
    std::vector<int64_t> cur(P.row64.begin(),P.row64.end()-1);
    for(int i=0;i<D;++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i]; int k=idxOf[off+n];
        for(int j=0;j<len;++j){ float v=H.values[base+j]; if(std::fabs((double)v)<=tau_abs) continue;
            int r=(off>=0)?j:j-off; int64_t p=cur[r]++; if(u16)P.d16[p]=(unsigned short)k; else P.d8[p]=(unsigned char)k; P.cv[p]=v; } }
    return P;
}

// t_new = (-i s)(H t) ; acc += t_new.  s=dt/m.  H t already in (htre,htim).
__global__ void taylor_term(int n,float s,const float* htre,const float* htim,
                            float* tre,float* tim,float* are,float* aim){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float nr = s*htim[i], ni = -s*htre[i];
    tre[i]=nr; tim[i]=ni; are[i]+=nr; aim[i]+=ni;
}
__global__ void copy2(int n,const float* a,const float* b,float* c,float* d){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; c[i]=a[i]; d[i]=b[i];
}

int main(int argc,char**argv){
    if(argc<2){ std::fprintf(stderr,"usage: %s <dia.txt> [steps] [K]\n",argv[0]); return 1; }
    int steps=argc>2?atoi(argv[2]):1000; int K=argc>3?atoi(argv[3]):12;
    double ftime=1.2, dt=ftime/steps;

    DiaHost H=load_dia(argv[1]); const int n=H.n; const int D=(int)H.offsets.size();
    std::vector<int> off_desc(H.offsets); std::sort(off_desc.begin(),off_desc.end(),std::greater<int>());
    std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<D;++k) idxOf[off_desc[k]+n]=k;
    const bool u16=D>256;
    double maxabs=0; for(float v:H.values) maxabs=std::max(maxabs,std::fabs((double)v));

    // normalized deterministic initial state (real)
    std::vector<float> h0(n); double nrm=0; for(int i=0;i<n;++i){ h0[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f; nrm+=(double)h0[i]*h0[i]; }
    nrm=std::sqrt(nrm); for(int i=0;i<n;++i) h0[i]/=(float)nrm;

    int* dOff; CK(cudaMalloc(&dOff,D*4)); CK(cudaMemcpy(dOff,off_desc.data(),D*4,cudaMemcpyHostToDevice));
    const int shb=D*(int)sizeof(int), blk=(n+255)/256;
    float *re,*im,*are,*aim,*tre,*tim,*htre,*htim;
    for(float** p:{&re,&im,&are,&aim,&tre,&tim,&htre,&htim}) CK(cudaMalloc(p,(size_t)n*4));

    const double taus[]={0,1e-8,1e-6,1e-4,1e-3,1e-2};
    std::vector<float> refR(n),refI(n),curR(n),curI(n);
    int64_t nnz0=1;

    std::printf("EVO,file,n,D,steps,K,tau,nnz,nnz_pct,fidelity,l2_relerr,norm_final\n");
    for(double tau:taus){
        Packed P=pack(H,n,D,idxOf,tau*maxabs,u16);
        int64_t *dRp; CK(cudaMalloc(&dRp,(size_t)(n+1)*8)); CK(cudaMemcpy(dRp,P.row64.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
        const long long* dRpll=(const long long*)dRp;
        float* dCv; CK(cudaMalloc(&dCv,(size_t)P.nnz*4)); CK(cudaMemcpy(dCv,P.cv.data(),(size_t)P.nnz*4,cudaMemcpyHostToDevice));
        unsigned char* dD8=nullptr; unsigned short* dD16=nullptr;
        std::function<void(float*,float*)> spmv;   // y = H x
        if(u16){ CK(cudaMalloc(&dD16,(size_t)P.nnz*2)); CK(cudaMemcpy(dD16,P.d16.data(),(size_t)P.nnz*2,cudaMemcpyHostToDevice));
                 spmv=[=](float* x,float* y){ spmv_dia_zeroskip_u16_i64<<<blk,256,shb>>>((long long)n,D,dRpll,dD16,dCv,dOff,x,y); }; }
        else   { CK(cudaMalloc(&dD8 ,(size_t)P.nnz  )); CK(cudaMemcpy(dD8 ,P.d8.data() ,(size_t)P.nnz  ,cudaMemcpyHostToDevice));
                 spmv=[=](float* x,float* y){ spmv_dia_zeroskip_i64<<<blk,256,shb>>>((long long)n,D,dRpll,dD8,dCv,dOff,x,y); }; }

        // init psi = (h0, 0)
        CK(cudaMemcpy(re,h0.data(),(size_t)n*4,cudaMemcpyHostToDevice)); CK(cudaMemset(im,0,(size_t)n*4));
        for(int s=0;s<steps;++s){
            copy2<<<blk,256>>>(n,re,im,are,aim);   // acc = psi
            copy2<<<blk,256>>>(n,re,im,tre,tim);    // t   = psi
            for(int m=1;m<=K;++m){
                spmv(tre,htre); spmv(tim,htim);      // H t
                taylor_term<<<blk,256>>>(n,(float)(dt/m),htre,htim,tre,tim,are,aim);
            }
            copy2<<<blk,256>>>(n,are,aim,re,im);    // psi = acc
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(curR.data(),re,(size_t)n*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(curI.data(),im,(size_t)n*4,cudaMemcpyDeviceToHost));

        if(tau==0){ refR=curR; refI=curI; nnz0=P.nnz; }
        // <ref|cur>, norms, l2
        double ipr=0,ipi=0,nr2=0,nc2=0,l2d=0;
        for(int i=0;i<n;++i){ double Rr=refR[i],Ri=refI[i],Tr=curR[i],Ti=curI[i];
            ipr+=Rr*Tr+Ri*Ti; ipi+=Rr*Ti-Ri*Tr; nr2+=Rr*Rr+Ri*Ri; nc2+=Tr*Tr+Ti*Ti;
            l2d+=(Tr-Rr)*(Tr-Rr)+(Ti-Ri)*(Ti-Ri); }
        double fid = (nr2>0&&nc2>0)? std::sqrt(ipr*ipr+ipi*ipi)/std::sqrt(nr2*nc2) : 0.0;
        double l2  = nr2>0? std::sqrt(l2d/nr2) : 0.0;
        std::printf("EVO,%s,%d,%d,%d,%d,%.0e,%lld,%.4f,%.9f,%.3e,%.6f\n",
            argv[1],n,D,steps,K,tau,(long long)P.nnz,100.0*(double)P.nnz/(double)nnz0,fid,l2,std::sqrt(nc2));
        std::fflush(stdout);
        if(dD8)CK(cudaFree(dD8)); if(dD16)CK(cudaFree(dD16)); CK(cudaFree(dRp)); CK(cudaFree(dCv));
    }
    return 0;
}
