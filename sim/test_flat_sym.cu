/* test_flat_sym.cu — validate + time gather_flat_sym vs the gather_meta_sym
 * reference for the symmetric C = H*H (H=H^T, upper-only store + compute).
 * Must bit-match (same math, rescheduled); tolerance 1e-5.
 *   build: nvcc -O3 -arch=sm_90 sim/test_flat_sym.cu -o /tmp/test_flat_sym
 *   run:   /tmp/test_flat_sym <dia_file> [iters=2000]
 */
#include "dia_io.hpp"
#include "../spmspm/gather_sym.cuh"        // reference gather_meta_sym
#include "../spmspm/gather_flat_sym.cuh"   // new gather_flat_sym + planner
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
constexpr int A_DMAX = 128;   // smem pair cap for gather_meta_sym
constexpr int MAXP   = 256;   // smem pair cap for gather_flat_sym
constexpr int TILE   = 256;
constexpr int ILP    = 4;

struct Cs{ std::vector<int> off,len; std::vector<size_t> start; size_t nnz; };
static Cs buildC(const std::vector<int>&Ao,const std::vector<int>&Bo,int n,bool upper){
    std::vector<char> pr(2*n-1,0); for(int da:Ao)for(int db:Bo){int dc=da+db; if(dc>-n&&dc<n)pr[dc+n-1]=1;}
    Cs C; size_t off=0; for(int d=-(n-1);d<=n-1;++d){ if(!pr[d+n-1])continue; if(upper&&d<0)continue; int l=n-std::abs(d); if(l<=0)continue;
        C.off.push_back(d); C.start.push_back(off); C.len.push_back(l); off+=l; } C.nnz=off; return C; }

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s <dia_file> [iters=2000]\n",argv[0]);return 1;}
    int iters=argc>2?atoi(argv[2]):2000;
    DiaHost H=load_dia(argv[1]); int n=H.n;

    // --- upper H (store-half): offsets>=0, values by |offset| ---
    std::vector<int> uoff,ulen; std::vector<size_t> ustart; std::vector<float> uval;
    std::vector<int> Babs(n,-1); size_t off=0;
    for(size_t i=0;i<H.offsets.size();++i){ if(H.offsets[i]<0)continue; int o=H.offsets[i],l=H.lengths[i];
        uoff.push_back(o); ulen.push_back(l); ustart.push_back(off); Babs[o]=uoff.size()-1;
        uval.insert(uval.end(), H.values.begin()+H.starts[i], H.values.begin()+H.starts[i]+l); off+=l; }
    int Hund=uoff.size(); size_t Hunnz=off;
    float* uHv; size_t* uHs; int* uHoff; int* uHlen; int* uBabs;
    CK(cudaMalloc(&uHv,Hunnz*4)); CK(cudaMemcpy(uHv,uval.data(),Hunnz*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&uHs,Hund*8)); CK(cudaMemcpy(uHs,ustart.data(),Hund*8,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&uHoff,Hund*4)); CK(cudaMemcpy(uHoff,uoff.data(),Hund*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&uHlen,Hund*4)); CK(cudaMemcpy(uHlen,ulen.data(),Hund*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&uBabs,n*4)); CK(cudaMemcpy(uBabs,Babs.data(),n*4,cudaMemcpyHostToDevice));

    // --- upper C layout (dc>=0) ---
    Cs Cu=buildC(H.offsets,H.offsets,n,true);
    float *Cu_ref,*Cu_flat; size_t *Cus; int *Cuo,*Cul;
    CK(cudaMalloc(&Cu_ref,Cu.nnz*4)); CK(cudaMalloc(&Cu_flat,Cu.nnz*4));
    CK(cudaMalloc(&Cus,Cu.start.size()*8)); CK(cudaMemcpy(Cus,Cu.start.data(),Cu.start.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Cuo,Cu.off.size()*4)); CK(cudaMemcpy(Cuo,Cu.off.data(),Cu.off.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Cul,Cu.len.size()*4)); CK(cudaMemcpy(Cul,Cu.len.data(),Cu.len.size()*4,cudaMemcpyHostToDevice));

    // --- host flat-sym plan ---
    FlatSymPlan fp = build_flat_sym_plan(n, uoff, ulen, ustart, ulen, ustart, Babs, Cu.off, Cu.len, TILE*ILP);
    if(fp.maxPairs > MAXP){ fprintf(stderr,"maxPairs %d > MAXP %d\n",fp.maxPairs,MAXP); return 1; }
    int2* dTiles; int* dPairPtr; GPair* dPairs;
    CK(cudaMalloc(&dTiles,fp.tiles.size()*sizeof(int2)));   CK(cudaMemcpy(dTiles,fp.tiles.data(),fp.tiles.size()*sizeof(int2),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dPairPtr,fp.pairPtr.size()*4));          CK(cudaMemcpy(dPairPtr,fp.pairPtr.data(),fp.pairPtr.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dPairs,fp.pairs.size()*sizeof(GPair)));  CK(cudaMemcpy(dPairs,fp.pairs.data(),fp.pairs.size()*sizeof(GPair),cudaMemcpyHostToDevice));

    auto ref =[&](){ dim3 g((n+TILE*ILP-1)/(TILE*ILP),Cu.off.size());
        gather_meta_sym<A_DMAX><<<g,TILE>>>(uHv,uHs,uHoff,uHlen,Hund, uHv,uHs,uHlen,uBabs, Cu_ref,Cus,Cuo,Cul,(int)Cu.off.size(),n); };
    auto flat=[&](){ gather_flat_sym_kernel<MAXP,ILP><<<fp.nTiles,TILE>>>(uHv,uHv,dTiles,dPairPtr,dPairs,Cu_flat,Cus,Cul); };

    CK(cudaMemset(Cu_ref,0,Cu.nnz*4)); CK(cudaMemset(Cu_flat,0,Cu.nnz*4));
    ref(); flat(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<float> R(Cu.nnz), F(Cu.nnz);
    CK(cudaMemcpy(R.data(),Cu_ref,Cu.nnz*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(F.data(),Cu_flat,Cu.nnz*4,cudaMemcpyDeviceToHost));
    double md=0; for(size_t i=0;i<Cu.nnz;++i) md=fmax(md,(double)fabs(R[i]-F[i]));
    printf("n=%d  Hupper=%d diags  Cupper=%zu diags  pairs=%zu (maxPairs=%d) tiles=%d  max|Δ|=%.3e  %s\n",
           n,Hund,Cu.off.size(),fp.pairs.size(),fp.maxPairs,fp.nTiles,md,(md<1e-5)?"[MATCH]":"[MISMATCH]");

    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto tt=[&](auto f){ for(int i=0;i<20;++i)f(); CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
        for(int i=0;i<iters;++i)f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms,e0,e1); return ms/iters; };
    float tr=tt(ref), tfl=tt(flat);
    printf("  gather_meta_sym: %.5f ms   gather_flat_sym: %.5f ms   speedup = %.2fx\n", tr, tfl, tr/tfl);
    return 0;
}
