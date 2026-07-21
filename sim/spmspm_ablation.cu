/* spmspm_ablation.cu — SpMSpM (C=H*H) ablation for the paper: base -> all opts.
 *   L0 base         = HM (Haque, atomicAdd)                     [paper_hm]
 *   L1 +atomic-free = gather_meta (smem accum, in-block resolve)[spmspm.cu]
 *   L2 +flat/pairs  = gather_flat, ILP=1 (precomputed host pairs)
 *   L3 +adaptive ILP= gather_flat, ILP=4
 *   L4 +symmetric   = gather_flat_sym (upper store/compute-half)
 * All cross-checked equal (full C for L0-L3; L4 upper C vs L2's upper diags).
 * build: nvcc -O3 -std=c++17 -arch=sm_80 spmspm_ablation.cu \
 *        ../spmspm/paper_hm_kernel.cu -o spmspm_ablation -Xcompiler -fopenmp
 * run:   ./spmspm_ablation <dia.txt> [iters] [--csv] */
#include "dia_io.hpp"
#include "../spmspm/spmspm.cu"          // gather_meta_kernel
#include "../spmspm/paper_hm.cuh"       // HM
#include "../spmspm/gather_flat.cuh"    // GPair, gather_flat_kernel
#include "../spmspm/gather_flat_sym.cuh"// gather_flat_sym_kernel, build_flat_sym_plan
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
template<class F> static float tms(F f,int wu,int it){ for(int i=0;i<wu;++i)f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for(int i=0;i<it;++i)f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it; }
static double maxdiff(const std::vector<float>&a,const std::vector<float>&b){ if(a.size()!=b.size())return 1e30;
    double m=0; for(size_t i=0;i<a.size();++i) m=std::max(m,(double)std::fabs(a[i]-b[i])); return m; }

struct Cstruct { std::vector<int> offsets,lengths; std::vector<size_t> starts; size_t nnz; };
static Cstruct make_c(const DiaHost&A,const DiaHost&B,bool upper_only){
    int n=A.n; std::vector<char> present(2*n-1,0);
    for(int da:A.offsets)for(int db:B.offsets){int dc=da+db; if(dc>-n&&dc<n)present[dc+n-1]=1;}
    Cstruct C; size_t off=0;
    for(int d=(upper_only?0:-(n-1)); d<=n-1; ++d){ if(!present[d+n-1])continue; int len=n-std::abs(d); if(len<=0)continue;
        C.offsets.push_back(d); C.starts.push_back(off); C.lengths.push_back(len); off+=len; }
    C.nnz=off; return C;
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s <dia.txt> [iters] [--csv]\n",argv[0]);return 1;}
    bool csv=false; int iters=30;
    for(int i=2;i<argc;++i){ if(!strcmp(argv[i],"--csv"))csv=true; else iters=atoi(argv[i]); }
    DiaHost H=load_dia(argv[1]); int n=H.n; const int An=(int)H.offsets.size();
    Cstruct C=make_c(H,H,false), Cs=make_c(H,H,true);
    std::vector<size_t> Hs(H.starts);

    // ---- device H (shared) ----
    float* dHv; size_t* dHs; int* dHoff,*dHlen;
    std::vector<int> Hoff(H.offsets),Hlen(H.lengths);
    CK(cudaMalloc(&dHv,H.nnz*4)); CK(cudaMemcpy(dHv,H.values.data(),H.nnz*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dHs,Hs.size()*sizeof(size_t))); CK(cudaMemcpy(dHs,Hs.data(),Hs.size()*sizeof(size_t),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dHoff,An*4)); CK(cudaMemcpy(dHoff,Hoff.data(),An*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dHlen,An*4)); CK(cudaMemcpy(dHlen,Hlen.data(),An*4,cudaMemcpyHostToDevice));
    std::vector<int> Blk(2*(size_t)n-1,-1); for(int i=0;i<An;++i) Blk[H.offsets[i]+n-1]=i;
    int* dBlk; CK(cudaMalloc(&dBlk,Blk.size()*4)); CK(cudaMemcpy(dBlk,Blk.data(),Blk.size()*4,cudaMemcpyHostToDevice));
    // full C device
    float* dCv; size_t* dCs; int* dCoff,*dClen;
    CK(cudaMalloc(&dCv,C.nnz*4));
    CK(cudaMalloc(&dCs,C.starts.size()*sizeof(size_t))); CK(cudaMemcpy(dCs,C.starts.data(),C.starts.size()*sizeof(size_t),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dCoff,C.offsets.size()*4)); CK(cudaMemcpy(dCoff,C.offsets.data(),C.offsets.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dClen,C.lengths.size()*4)); CK(cudaMemcpy(dClen,C.lengths.data(),C.lengths.size()*4,cudaMemcpyHostToDevice));
    int Cn=(int)C.offsets.size();

    // ================= L1 gather_meta =================
    constexpr int A_DMAX=256;
    if(An>A_DMAX){fprintf(stderr,"An %d > A_DMAX\n",An);return 1;}
    { const int ILP=4; dim3 mblk(256), mgrid((n+256*ILP-1)/(256*ILP),Cn); }
    auto launch_meta=[&](){ dim3 mblk(256),mgrid((n+256*4-1)/(256*4),Cn);
        gather_meta_kernel<A_DMAX><<<mgrid,mblk>>>(dHv,dHs,dHoff,dHlen,An,dHv,dHs,dHlen,dBlk,dCv,dCs,dCoff,dClen,Cn,n); };
    CK(cudaMemset(dCv,0,C.nnz*4)); launch_meta(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> C_meta(C.nnz); CK(cudaMemcpy(C_meta.data(),dCv,C.nnz*4,cudaMemcpyDeviceToHost));
    float t_meta=tms(launch_meta,3,iters);

    // ================= L2/L3 gather_flat (ILP 1 / 4) =================
    constexpr int MAXP=256;
    std::unordered_map<int,int> bIdx; for(int i=0;i<An;++i) bIdx[H.offsets[i]]=i;
    auto build_flat=[&](int FILP,std::vector<GPair>&pairs,std::vector<int>&pairPtr,std::vector<int2>&tiles){
        int POS=256*FILP; pairPtr.assign(Cn+1,0);
        for(int k=0;k<Cn;++k){int dc=C.offsets[k],minc=dc<0?dc:0; pairPtr[k]=(int)pairs.size();
            for(int ai=0;ai<An;++ai){int da=H.offsets[ai],db=dc-da; if(db<=-n||db>=n)continue; auto it=bIdx.find(db); if(it==bIdx.end())continue; int bi=it->second;
                GPair g; g.ab=H.starts[ai];g.bb=H.starts[bi];g.ash=(da<0?da:0)-minc;g.bsh=da+(db<0?db:0)-minc;g.al=H.lengths[ai];g.bl=H.lengths[bi]; pairs.push_back(g);}
            for(int ts=0;ts<C.lengths[k];ts+=POS) tiles.push_back(make_int2(k,ts));}
        pairPtr[Cn]=(int)pairs.size(); };
    // ILP=1
    std::vector<GPair> p1; std::vector<int> pp1; std::vector<int2> t1; build_flat(1,p1,pp1,t1);
    GPair* dP1; int* dPP1; int2* dT1;
    CK(cudaMalloc(&dP1,p1.size()*sizeof(GPair)));CK(cudaMemcpy(dP1,p1.data(),p1.size()*sizeof(GPair),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dPP1,(Cn+1)*4));CK(cudaMemcpy(dPP1,pp1.data(),(Cn+1)*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dT1,t1.size()*sizeof(int2)));CK(cudaMemcpy(dT1,t1.data(),t1.size()*sizeof(int2),cudaMemcpyHostToDevice));
    auto launch_f1=[&](){ gather_flat_kernel<MAXP,1><<<(int)t1.size(),256>>>(dHv,dHv,dT1,dPP1,dP1,dCv,dCs,dClen); };
    CK(cudaMemset(dCv,0,C.nnz*4)); launch_f1(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> C_f1(C.nnz); CK(cudaMemcpy(C_f1.data(),dCv,C.nnz*4,cudaMemcpyDeviceToHost));
    float t_f1=tms(launch_f1,3,iters);
    // ILP=4
    std::vector<GPair> p4; std::vector<int> pp4; std::vector<int2> t4; build_flat(4,p4,pp4,t4);
    GPair* dP4; int* dPP4; int2* dT4;
    CK(cudaMalloc(&dP4,p4.size()*sizeof(GPair)));CK(cudaMemcpy(dP4,p4.data(),p4.size()*sizeof(GPair),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dPP4,(Cn+1)*4));CK(cudaMemcpy(dPP4,pp4.data(),(Cn+1)*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dT4,t4.size()*sizeof(int2)));CK(cudaMemcpy(dT4,t4.data(),t4.size()*sizeof(int2),cudaMemcpyHostToDevice));
    auto launch_f4=[&](){ gather_flat_kernel<MAXP,4><<<(int)t4.size(),256>>>(dHv,dHv,dT4,dPP4,dP4,dCv,dCs,dClen); };
    CK(cudaMemset(dCv,0,C.nnz*4)); launch_f4(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> C_f4(C.nnz); CK(cudaMemcpy(C_f4.data(),dCv,C.nnz*4,cudaMemcpyDeviceToHost));
    float t_f4=tms(launch_f4,3,iters);

    // ================= L4 gather_flat_sym (upper store/compute-half) =================
    // upper A/B = H's d>=0 diagonals (values reused from dHv by |offset|); Babs lookup.
    std::vector<int> Aoff, Alen; std::vector<size_t> Ast; std::vector<int> Babs(n,-1);
    for(int i=0;i<An;++i){ if(H.offsets[i]>=0){ int a=H.offsets[i]; Babs[a]=(int)Aoff.size(); Aoff.push_back(a); Alen.push_back(H.lengths[i]); Ast.push_back(H.starts[i]); } }
    int Csn=(int)Cs.offsets.size();
    FlatSymPlan sp=build_flat_sym_plan(n,Aoff,Alen,Ast,Alen,Ast,Babs,Cs.offsets,Cs.lengths,256*4);
    float* dCsv; size_t* dCss; int* dCsl;
    CK(cudaMalloc(&dCsv,Cs.nnz*4));
    CK(cudaMalloc(&dCss,Csn*sizeof(size_t))); CK(cudaMemcpy(dCss,Cs.starts.data(),Csn*sizeof(size_t),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dCsl,Csn*4)); CK(cudaMemcpy(dCsl,Cs.lengths.data(),Csn*4,cudaMemcpyHostToDevice));
    GPair* dSP; int* dSPP; int2* dST;
    CK(cudaMalloc(&dSP,sp.pairs.size()*sizeof(GPair)));CK(cudaMemcpy(dSP,sp.pairs.data(),sp.pairs.size()*sizeof(GPair),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dSPP,(Csn+1)*4));CK(cudaMemcpy(dSPP,sp.pairPtr.data(),(Csn+1)*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dST,sp.tiles.size()*sizeof(int2)));CK(cudaMemcpy(dST,sp.tiles.data(),sp.tiles.size()*sizeof(int2),cudaMemcpyHostToDevice));
    auto launch_sym=[&](){ gather_flat_sym_kernel<MAXP,4><<<(int)sp.tiles.size(),256>>>(dHv,dHv,dST,dSPP,dSP,dCsv,dCss,dCsl); };
    CK(cudaMemset(dCsv,0,Cs.nnz*4)); launch_sym(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> C_sym(Cs.nnz); CK(cudaMemcpy(C_sym.data(),dCsv,Cs.nnz*4,cudaMemcpyDeviceToHost));
    float t_sym=tms(launch_sym,3,iters);

    // ================= L0 HM (base) =================
    HMMatrix hA; hA.n=n; hA.num_diags=An; hA.diag_offsets=H.offsets; hA.diag_lengths=H.lengths; hA.values=H.values; hA.total_nz=H.nnz;
    hA.diag_starts.resize(Hs.size()); for(size_t i=0;i<Hs.size();++i) hA.diag_starts[i]=(int)Hs[i];
    HMMatrix hC=compute_c_hm_structure(hA,hA,n); std::vector<int> cLk=build_c_diag_lookup(hC,n);
    auto up=[&](void**p,const void*s,size_t b){CK(cudaMalloc(p,b));CK(cudaMemcpy(*p,s,b,cudaMemcpyHostToDevice));};
    float* hAv;int*hAo,*hAs,*hAl; up((void**)&hAv,hA.values.data(),hA.values.size()*4);up((void**)&hAo,hA.diag_offsets.data(),An*4);up((void**)&hAs,hA.diag_starts.data(),An*4);up((void**)&hAl,hA.diag_lengths.data(),An*4);
    float* hCv;int*hCo,*hCs,*hCl,*hClk; CK(cudaMalloc(&hCv,hC.total_nz*4));
    up((void**)&hCo,hC.diag_offsets.data(),hC.num_diags*4);up((void**)&hCs,hC.diag_starts.data(),hC.num_diags*4);up((void**)&hCl,hC.diag_lengths.data(),hC.num_diags*4);up((void**)&hClk,cLk.data(),cLk.size()*4);
    int nzA=hA.total_nz, hmblk=(nzA+255)/256;
    auto launch_hm=[&](){ hm_structured_sparse_matmul_kernel<<<hmblk,256>>>(hAv,hAo,hAs,hAl,An,hAv,hAo,hAs,hAl,An,hCv,hCo,hCs,hCl,hC.num_diags,hClk,nzA,n); };
    CK(cudaMemset(hCv,0,hC.total_nz*4)); launch_hm(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float t_hm=tms([&](){CK(cudaMemset(hCv,0,hC.total_nz*4)); launch_hm();},3,iters);

    // ---- verify (full C: meta vs flat; sym upper vs flat's dc>=0) ----
    double d_mf=maxdiff(C_meta,C_f1), d_f14=maxdiff(C_f1,C_f4);
    // sym upper: compare C_sym[dc>=0] against C_f4 same diagonals
    double d_sym=0; { std::unordered_map<int,size_t> foff; for(int k=0;k<Cn;++k) foff[C.offsets[k]]=C.starts[k];
        for(int k=0;k<Csn;++k){ int dc=Cs.offsets[k]; auto it=foff.find(dc); if(it==foff.end())continue;
            for(int p=0;p<Cs.lengths[k];++p) d_sym=std::max(d_sym,(double)std::fabs(C_sym[Cs.starts[k]+p]-C_f4[it->second+p])); } }

    auto sp_=[&](float b){ return b>0? t_hm/b : -1; };
    if(!csv){
        printf("=== SpMSpM ablation  file=%s  n=%d  Hd=%d  nnz(C)=%zu ===\n",argv[1],n,An,C.nnz);
        printf("[verify] meta-vs-flat=%.1e flat1-vs-flat4=%.1e sym-upper-vs-flat=%.1e\n",d_mf,d_f14,d_sym);
        printf("  L0 base HM         : %9.4f ms\n",t_hm);
        printf("  L1 +atomic-free    : %9.4f ms   (%.2fx vs L0)\n",t_meta,sp_(t_meta));
        printf("  L2 +flat/pairs ILP1: %9.4f ms   (%.2fx vs L0)\n",t_f1,sp_(t_f1));
        printf("  L3 +adaptive ILP4  : %9.4f ms   (%.2fx vs L0)\n",t_f4,sp_(t_f4));
        printf("  L4 +symmetric      : %9.4f ms   (%.2fx vs L0)\n",t_sym,sp_(t_sym));
    } else {
        double re=std::max(std::max(d_mf,d_f14),d_sym);
        printf("SPABL,%s,%d,%d,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.2e\n",
               argv[1],n,An,C.nnz,t_hm,t_meta,t_f1,t_f4,t_sym,sp_(t_meta),sp_(t_f1),sp_(t_f4),sp_(t_sym),re);
    }
    return 0;
}
