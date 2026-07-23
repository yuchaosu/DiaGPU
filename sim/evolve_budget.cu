/* evolve_budget.cu — the DIAGONAL BUDGET in HamSim: keep only the top-B diagonals
 * of H (ranked by L2 norm), drop whole low-norm diagonals, then run the full
 * 1000-step Taylor evolution e^{-iHt}|psi> with the truncated H'. Reports, per
 * budget B: #diags kept, nnz, captured Frobenius fraction (predictor), and the
 * final-state fidelity |<psi_full|psi_B>| + L2 error vs the full-H evolution.
 * Same fp32 zero-skip kernel throughout => isolates the truncation.
 *
 * This tests "can HamSim run on a diagonal budget?" — the D-axis of the (D,W)
 * design space. Magnitude-entry trim is a separate knob (evolve_trim.cu).
 *
 * build: nvcc -O3 -std=c++17 -arch=sm_80 evolve_budget.cu -o evolve_budget
 * run:   ./evolve_budget <dia.txt> [steps=1000] [K=12]
 */
#include "dia_io.hpp"
#include "../spmv/src/spmv_zeroskip_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)

struct Packed { std::vector<int64_t> row64; std::vector<unsigned char> d8;
                std::vector<unsigned short> d16; std::vector<float> cv; int64_t nnz; };

// pack the zero-skip operands from ONLY the kept diagonals. keep[i] over H.offsets.
// off_desc/idxOf are over the KEPT offsets (descending), matching dOff below.
static Packed pack_kept(const DiaHost& H,int n,const std::vector<char>& keep,
                        const std::vector<int>& off_desc,const std::vector<int>& idxOf,bool u16){
    Packed P; P.row64.assign(n+1,0);
    for(size_t i=0;i<H.offsets.size();++i){ if(!keep[i]) continue;
        int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i];
        for(int j=0;j<len;++j){ if(H.values[base+j]==0.f) continue; int r=(off>=0)?j:j-off; P.row64[r+1]++; } }
    for(int r=0;r<n;++r) P.row64[r+1]+=P.row64[r];
    P.nnz=P.row64[n]; int Dk=(int)off_desc.size();
    if(u16)P.d16.resize(P.nnz); else P.d8.resize(P.nnz); P.cv.resize(P.nnz);
    std::vector<int64_t> cur(P.row64.begin(),P.row64.end()-1);
    for(size_t i=0;i<H.offsets.size();++i){ if(!keep[i]) continue;
        int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i]; int k=idxOf[off+n]; (void)Dk;
        for(int j=0;j<len;++j){ float v=H.values[base+j]; if(v==0.f) continue; int r=(off>=0)?j:j-off; int64_t p=cur[r]++;
            if(u16)P.d16[p]=(unsigned short)k; else P.d8[p]=(unsigned char)k; P.cv[p]=v; } }
    return P;
}

__global__ void taylor_term(int n,float s,const float* htre,const float* htim,
                            float* tre,float* tim,float* are,float* aim){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float nr=s*htim[i], ni=-s*htre[i]; tre[i]=nr; tim[i]=ni; are[i]+=nr; aim[i]+=ni;
}
__global__ void copy2(int n,const float* a,const float* b,float* c,float* d){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; c[i]=a[i]; d[i]=b[i];
}

int main(int argc,char**argv){
    if(argc<2){ std::fprintf(stderr,"usage: %s <dia.txt> [steps] [K]\n",argv[0]); return 1; }
    int steps=argc>2?atoi(argv[2]):1000; int K=argc>3?atoi(argv[3]):12;
    double ftime=1.2, dt=ftime/steps;

    DiaHost H=load_dia(argv[1]); const int n=H.n; const int Dfull=(int)H.offsets.size();

    // per-diagonal L2 norm
    std::vector<double> dn(Dfull,0);
    double totn2=0;
    for(int i=0;i<Dfull;++i){ size_t base=H.starts[i]; int len=H.lengths[i]; double s=0;
        for(int j=0;j<len;++j){ double v=H.values[base+j]; s+=v*v; } dn[i]=s; totn2+=s; }

    // HERMITICITY-PRESERVING budget: group diagonals into conjugate pairs {o,-o}
    // (key=|o|), rank GROUPS by combined norm, keep top-B whole groups. Dropping a
    // lone o without -o makes H' non-Hermitian -> e^{-iH't} non-unitary (norm blows up).
    std::map<int,std::vector<int>> gmap; for(int i=0;i<Dfull;++i) gmap[std::abs(H.offsets[i])].push_back(i);
    std::vector<int> gkey; std::vector<double> gnorm;
    for(auto&kv:gmap){ double s=0; for(int i:kv.second) s+=dn[i]; gkey.push_back(kv.first); gnorm.push_back(s); }
    const int Ng=(int)gkey.size();
    std::vector<int> grank(Ng); for(int i=0;i<Ng;++i) grank[i]=i;
    std::sort(grank.begin(),grank.end(),[&](int a,int b){ return gnorm[a]>gnorm[b]; });

    // budgets in GROUP counts (capped at Ng)
    std::vector<int> budgets;
    for(int b:{Ng,512,384,256,192,128,96,64,48,32,24,16,12,8,6,4,3,2,1})
        if(b<=Ng && (budgets.empty()||budgets.back()!=b)) budgets.push_back(b);

    float *re,*im,*are,*aim,*tre,*tim,*htre,*htim;
    for(float** p:{&re,&im,&are,&aim,&tre,&tim,&htre,&htim}) CK(cudaMalloc(p,(size_t)n*4));
    // normalized deterministic initial state
    std::vector<float> h0(n); double nrm=0; for(int i=0;i<n;++i){ h0[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f; nrm+=(double)h0[i]*h0[i]; }
    nrm=std::sqrt(nrm); for(int i=0;i<n;++i) h0[i]/=(float)nrm;

    std::vector<float> refR(n),refI(n),curR(n),curI(n);
    std::printf("BUD,file,n,Dfull,steps,K,budget,Dkept,nnz,frob_captured,fidelity,l2_relerr,norm_final\n");
    for(int B:budgets){
        std::vector<char> keep(Dfull,0); double kept_n2=0;
        for(int r=0;r<B;++r){ int g=grank[r]; for(int i:gmap[gkey[g]]){ keep[i]=1; kept_n2+=dn[i]; } }
        // off_desc/idxOf over kept offsets
        std::vector<int> off_desc; for(int i=0;i<Dfull;++i) if(keep[i]) off_desc.push_back(H.offsets[i]);
        std::sort(off_desc.begin(),off_desc.end(),std::greater<int>());
        const bool u16 = (int)off_desc.size()>256;
        std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<(int)off_desc.size();++k) idxOf[off_desc[k]+n]=k;

        Packed P=pack_kept(H,n,keep,off_desc,idxOf,u16);
        int Dk=(int)off_desc.size();
        int* dOff; CK(cudaMalloc(&dOff,Dk*4)); CK(cudaMemcpy(dOff,off_desc.data(),Dk*4,cudaMemcpyHostToDevice));
        const int shb=Dk*(int)sizeof(int), blk=(n+255)/256;
        int64_t *dRp; CK(cudaMalloc(&dRp,(size_t)(n+1)*8)); CK(cudaMemcpy(dRp,P.row64.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
        const long long* dRpll=(const long long*)dRp;
        float* dCv; CK(cudaMalloc(&dCv,(size_t)P.nnz*4)); CK(cudaMemcpy(dCv,P.cv.data(),(size_t)P.nnz*4,cudaMemcpyHostToDevice));
        unsigned char* dD8=nullptr; unsigned short* dD16=nullptr;
        std::function<void(float*,float*)> spmv;
        if(u16){ CK(cudaMalloc(&dD16,(size_t)P.nnz*2)); CK(cudaMemcpy(dD16,P.d16.data(),(size_t)P.nnz*2,cudaMemcpyHostToDevice));
                 spmv=[=](float* x,float* y){ spmv_dia_zeroskip_u16_i64<<<blk,256,shb>>>((long long)n,Dk,dRpll,dD16,dCv,dOff,x,y); }; }
        else   { CK(cudaMalloc(&dD8 ,(size_t)P.nnz  )); CK(cudaMemcpy(dD8 ,P.d8.data() ,(size_t)P.nnz  ,cudaMemcpyHostToDevice));
                 spmv=[=](float* x,float* y){ spmv_dia_zeroskip_i64<<<blk,256,shb>>>((long long)n,Dk,dRpll,dD8,dCv,dOff,x,y); }; }

        CK(cudaMemcpy(re,h0.data(),(size_t)n*4,cudaMemcpyHostToDevice)); CK(cudaMemset(im,0,(size_t)n*4));
        for(int s=0;s<steps;++s){
            copy2<<<blk,256>>>(n,re,im,are,aim); copy2<<<blk,256>>>(n,re,im,tre,tim);
            for(int m=1;m<=K;++m){ spmv(tre,htre); spmv(tim,htim);
                taylor_term<<<blk,256>>>(n,(float)(dt/m),htre,htim,tre,tim,are,aim); }
            copy2<<<blk,256>>>(n,are,aim,re,im);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(curR.data(),re,(size_t)n*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(curI.data(),im,(size_t)n*4,cudaMemcpyDeviceToHost));
        if(B==budgets.front()){ refR=curR; refI=curI; }
        double ipr=0,ipi=0,nr2=0,nc2=0,l2d=0;
        for(int i=0;i<n;++i){ double Rr=refR[i],Ri=refI[i],Tr=curR[i],Ti=curI[i];
            ipr+=Rr*Tr+Ri*Ti; ipi+=Rr*Ti-Ri*Tr; nr2+=Rr*Rr+Ri*Ri; nc2+=Tr*Tr+Ti*Ti; l2d+=(Tr-Rr)*(Tr-Rr)+(Ti-Ri)*(Ti-Ri); }
        double fid=(nr2>0&&nc2>0)? std::sqrt(ipr*ipr+ipi*ipi)/std::sqrt(nr2*nc2):0.0;
        double l2 =nr2>0? std::sqrt(l2d/nr2):0.0;
        double frob=std::sqrt(kept_n2/totn2);
        std::printf("BUD,%s,%d,%d,%d,%d,%d,%d,%lld,%.9f,%.9f,%.3e,%.6f\n",
            argv[1],n,Dfull,steps,K,B,Dk,(long long)P.nnz,frob,fid,l2,std::sqrt(nc2));
        std::fflush(stdout);
        if(dD8)CK(cudaFree(dD8)); if(dD16)CK(cudaFree(dD16)); CK(cudaFree(dRp)); CK(cudaFree(dCv)); CK(cudaFree(dOff));
    }
    return 0;
}
