/* trim_sweep.cu — magnitude-trim sweep for the DIA zero-skip SpMV.
 *
 * Reuses the SAME zero-skip kernel + packing as sim/spmv_dia_vs_tc.cu, but drops
 * every entry with |v| <= tau * max|H| (RELATIVE threshold). tau=0 reproduces the
 * current exact-zero-drop (the reference). For each tau it reports: nnz kept,
 * kernel time, speedup vs tau=0, and the SpMV error vs the untrimmed y
 * (max-abs relative + L2 relative). One SpMV apply (not a full evolution).
 *
 * build: nvcc -O3 -std=c++17 -arch=sm_80 trim_sweep.cu -o trim_sweep
 * run:   ./trim_sweep <dia1.txt> [dia2.txt ...]
 */
#include "dia_io.hpp"
#include "../spmv/src/spmv_zeroskip_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

struct Packed { std::vector<int64_t> row64; std::vector<unsigned char> d8;
                std::vector<unsigned short> d16; std::vector<float> cv; int64_t nnz; };

// pack nonzeros with |v| > tau_abs, into per-row CSR-like arrays (didx = diag index in off_desc order)
static Packed pack(const DiaHost& H,int n,int D,const std::vector<int>& idxOf,double tau_abs,bool u16){
    Packed P; P.row64.assign(n+1,0);
    for(int i=0;i<D;++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i];
        for(int j=0;j<len;++j){ if(std::fabs((double)H.values[base+j])<=tau_abs) continue;
            int r=(off>=0)?j:j-off; P.row64[r+1]++; } }
    for(int r=0;r<n;++r) P.row64[r+1]+=P.row64[r];
    P.nnz=P.row64[n];
    if(u16) P.d16.resize(P.nnz); else P.d8.resize(P.nnz);
    P.cv.resize(P.nnz);
    std::vector<int64_t> cur(P.row64.begin(),P.row64.end()-1);
    for(int i=0;i<D;++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i]; int k=idxOf[off+n];
        for(int j=0;j<len;++j){ float v=H.values[base+j]; if(std::fabs((double)v)<=tau_abs) continue;
            int r=(off>=0)?j:j-off; int64_t p=cur[r]++;
            if(u16) P.d16[p]=(unsigned short)k; else P.d8[p]=(unsigned char)k; P.cv[p]=v; } }
    return P;
}

static float time_kernel(std::function<void()> k,int warm,int iters){
    for(int i=0;i<warm;++i) k(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a)); for(int i=0;i<iters;++i) k(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms/iters;
}

static void run_one(const char* path){
    DiaHost H=load_dia(path); const int n=H.n; const int D=(int)H.offsets.size();
    std::vector<int> off_desc(H.offsets); std::sort(off_desc.begin(),off_desc.end(),std::greater<int>());
    std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<D;++k) idxOf[off_desc[k]+n]=k;
    const bool u16 = D>256;

    double maxabs=0; for(size_t i=0;i<H.values.size();++i) maxabs=std::max(maxabs,std::fabs((double)H.values[i]));

    // deterministic x (same hash as sim/spmv_dia_vs_tc.cu)
    std::vector<float> hx(n); for(int i=0;i<n;++i) hx[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f;
    float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*4)); CK(cudaMemcpy(dx,hx.data(),(size_t)n*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy,(size_t)n*4));
    int* dOff; CK(cudaMalloc(&dOff,D*4)); CK(cudaMemcpy(dOff,off_desc.data(),D*4,cudaMemcpyHostToDevice));
    const int shb=D*(int)sizeof(int), blk=(n+255)/256;

    const double taus[]={0,1e-10,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2};
    std::vector<float> yref(n), ytmp(n);
    int64_t nnz0=0;

    std::fprintf(stderr,"# %s  n=%d D=%d u16=%d maxabs=%.3e nnz_stored=%zu\n",path,n,D,(int)u16,maxabs,H.nnz);
    for(double tau: taus){
        double tau_abs = tau*maxabs;
        Packed P=pack(H,n,D,idxOf,tau_abs,u16);
        int64_t *dRp; CK(cudaMalloc(&dRp,(size_t)(n+1)*8)); CK(cudaMemcpy(dRp,P.row64.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
        const long long* dRpll=(const long long*)dRp;
        float* dCv; CK(cudaMalloc(&dCv,(size_t)P.nnz*4)); CK(cudaMemcpy(dCv,P.cv.data(),(size_t)P.nnz*4,cudaMemcpyHostToDevice));
        unsigned char* dD8=nullptr; unsigned short* dD16=nullptr;
        std::function<void()> k;
        if(u16){ CK(cudaMalloc(&dD16,(size_t)P.nnz*2)); CK(cudaMemcpy(dD16,P.d16.data(),(size_t)P.nnz*2,cudaMemcpyHostToDevice));
                 k=[=](){ spmv_dia_zeroskip_u16_i64<<<blk,256,shb>>>((long long)n,D,dRpll,dD16,dCv,dOff,dx,dy); }; }
        else   { CK(cudaMalloc(&dD8 ,(size_t)P.nnz  )); CK(cudaMemcpy(dD8 ,P.d8.data() ,(size_t)P.nnz  ,cudaMemcpyHostToDevice));
                 k=[=](){ spmv_dia_zeroskip_i64    <<<blk,256,shb>>>((long long)n,D,dRpll,dD8 ,dCv,dOff,dx,dy); }; }
        CK(cudaMemset(dy,0,(size_t)n*4)); k(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ytmp.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
        float ms=time_kernel(k,10,200);

        if(tau==0){ yref=ytmp; nnz0=P.nnz; }
        double maxdiff=0,ynorm=0,l2d=0,l2r=0;
        for(int i=0;i<n;++i){ double d=std::fabs((double)ytmp[i]-yref[i]);
            maxdiff=std::max(maxdiff,d); ynorm=std::max(ynorm,std::fabs((double)yref[i]));
            l2d+=d*d; l2r+=(double)yref[i]*yref[i]; }
        double maxrel = ynorm>0? maxdiff/ynorm : 0.0;
        double l2rel  = l2r>0? std::sqrt(l2d/l2r) : 0.0;
        // ref speedup: ms(tau0)/ms(tau) -- store ms0
        static float ms0=0; if(tau==0) ms0=ms;
        std::printf("TRIM,%s,%d,%d,%.3e,%.0e,%.3e,%lld,%.4f,%.4f,%.4f,%.3e,%.3e\n",
            path,n,D,maxabs,tau,tau_abs,(long long)P.nnz,
            100.0*(double)P.nnz/(double)nnz0, ms, ms0>0?ms0/ms:1.0, maxrel, l2rel);
        std::fflush(stdout);
        if(dD8)CK(cudaFree(dD8)); if(dD16)CK(cudaFree(dD16)); CK(cudaFree(dRp)); CK(cudaFree(dCv));
    }
    CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(dOff));
}

int main(int argc,char**argv){
    if(argc<2){ std::fprintf(stderr,"usage: %s <dia.txt> [more...]\n",argv[0]); return 1; }
    std::printf("TRIM,file,n,D,maxabs,tau,tau_abs,nnz,nnz_pct,ms,speedup_vs_tau0,max_relerr,l2_relerr\n");
    for(int i=1;i<argc;++i) run_one(argv[i]);
    return 0;
}
