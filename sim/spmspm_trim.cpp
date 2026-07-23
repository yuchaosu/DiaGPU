/* spmspm_trim.cpp — magnitude-trim on the SpMSpM (operator-product) build.
 *
 * Computes the DIA power chain P_1=H, P_{k+1}=P_k·H two ways:
 *   - EXACT   (reference): Q_{k+1}=Q_k·H, no trim -> shows fill-in growth.
 *   - BUDGETED: P_{k+1}=trim(P_k·H, tau), drop |v|<=tau*max|P| + empty diagonals.
 * Per (matrix, tau, k) reports: #diagonals(exact), #diagonals(trim), nnz(exact),
 * nnz(trim), and Frobenius relative error ||P_k - Q_k||_F / ||Q_k||_F (cumulative,
 * the realistic budgeted-chain error). Host-only + exact double accumulation:
 * this is a structure/accuracy question; kernel speed scales with nnz/#diags.
 *
 * build: g++ -O3 -std=c++17 -fopenmp spmspm_trim.cpp -o spmspm_trim
 * run:   ./spmspm_trim <dia.txt> [more...]   (env KMAX=4 OFFCAP=30000 TAUS=...)
 */
#include "dia_io.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

struct Op { int N=0; std::vector<int> offs; std::vector<std::vector<float>> diag; };

// build row-indexed dense diagonals from a loaded DIA matrix (full, both signs)
static Op to_op(const DiaHost& H){
    Op O; O.N=H.n;
    for(size_t i=0;i<H.offsets.size();++i){
        int o=H.offsets[i]; std::vector<float> d(H.n,0.f); size_t base=H.starts[i]; int len=H.lengths[i];
        if(o>=0){ for(int j=0;j<len;++j) d[j]=H.values[base+j]; }        // row=j
        else    { for(int j=0;j<len;++j) d[j-o]=H.values[base+j]; }      // row=j-o=j+|o|
        O.offs.push_back(o); O.diag.push_back(std::move(d));
    }
    return O;
}

// C = A·B  (C[r,r+c] += A[r,r+a]*B[r+a,r+a+b], b=c-a). Returns empty if #offsets>offcap.
static Op matmul(const Op& A,const Op& B,int offcap){
    const int N=A.N;
    std::map<int,std::vector<std::pair<int,int>>> groups;
    for(size_t ai=0;ai<A.offs.size();++ai) for(size_t bi=0;bi<B.offs.size();++bi){
        int c=A.offs[ai]+B.offs[bi]; if(c<=-N||c>=N) continue; groups[c].push_back({(int)ai,(int)bi}); }
    Op C; C.N=N;
    if((int)groups.size()>offcap){ C.offs.clear(); return C; }   // signal overflow via empty
    std::vector<int> couts; couts.reserve(groups.size());
    for(auto&kv:groups) couts.push_back(kv.first);
    C.offs=couts; C.diag.assign(couts.size(), std::vector<float>(N,0.f));
    #pragma omp parallel for schedule(dynamic,1)
    for(int idx=0;idx<(int)couts.size();++idx){
        int c=couts[idx]; std::vector<double> tmp(N,0.0);
        for(auto&pr:groups[c]){ int ai=pr.first,bi=pr.second; int a=A.offs[ai];
            const auto& Aa=A.diag[ai]; const auto& Bb=B.diag[bi];
            int lo=std::max(0,std::max(-a,-c)); int hi=std::min(N,std::min(N-a,N-c));
            for(int r=lo;r<hi;++r) tmp[r]+=(double)Aa[r]*(double)Bb[r+a]; }
        auto& acc=C.diag[idx]; for(int r=0;r<N;++r) acc[r]=(float)tmp[r];
    }
    return C;
}

static double maxabs(const Op& O){ double m=0; for(auto&d:O.diag) for(float v:d) m=std::max(m,std::fabs((double)v)); return m; }
static long long nnz(const Op& O){ long long c=0; for(auto&d:O.diag) for(float v:d) if(v!=0.f) ++c; return c; }

// drop entries |v|<=tau*max|O|, then remove all-zero diagonals
static Op trim(const Op& O,double tau){
    double t=tau*maxabs(O); Op R; R.N=O.N;
    for(size_t i=0;i<O.offs.size();++i){ std::vector<float> d=O.diag[i]; bool any=false;
        for(float& v:d){ if(std::fabs((double)v)<=t) v=0.f; else any=true; }
        if(any){ R.offs.push_back(O.offs[i]); R.diag.push_back(std::move(d)); } }
    return R;
}

// ||P - Q||_F / ||Q||_F over the union of offsets
static double frob_rel(const Op& P,const Op& Q){
    std::map<int,const std::vector<float>*> mp,mq;
    for(size_t i=0;i<P.offs.size();++i) mp[P.offs[i]]=&P.diag[i];
    for(size_t i=0;i<Q.offs.size();++i) mq[Q.offs[i]]=&Q.diag[i];
    std::map<int,int> keys; for(auto&kv:mp)keys[kv.first]=1; for(auto&kv:mq)keys[kv.first]=1;
    double num=0,den=0; int N=Q.N;
    for(auto&kv:keys){ int o=kv.first; const std::vector<float>* pp=mp.count(o)?mp[o]:nullptr; const std::vector<float>* qq=mq.count(o)?mq[o]:nullptr;
        for(int r=0;r<N;++r){ double pv=pp?(*pp)[r]:0.0, qv=qq?(*qq)[r]:0.0; num+=(pv-qv)*(pv-qv); den+=qv*qv; } }
    return den>0? std::sqrt(num/den) : 0.0;
}

int main(int argc,char**argv){
    if(argc<2){ std::fprintf(stderr,"usage: %s <dia.txt> [more...]\n",argv[0]); return 1; }
    int KMAX=getenv("KMAX")?atoi(getenv("KMAX")):4;
    int OFFCAP=getenv("OFFCAP")?atoi(getenv("OFFCAP")):30000;
    std::vector<double> taus={0,1e-6,1e-4,1e-3,1e-2};
    std::printf("SPTRIM,file,tau,k,D_exact,D_trim,nnz_exact,nnz_trim,nnz_trim_pct,frob_relerr\n");
    for(int f=1;f<argc;++f){
        DiaHost H=load_dia(argv[f]); Op h=to_op(H);
        std::fprintf(stderr,"# %s N=%d D(H)=%zu maxabs=%.3e\n",argv[f],H.n,h.offs.size(),maxabs(h));
        // exact chain Q_k (reference), computed once
        std::vector<Op> Q; Q.push_back(h); bool overflow=false;
        for(int k=2;k<=KMAX;++k){ Op nq=matmul(Q.back(),h,OFFCAP); if(nq.offs.empty()){ overflow=true; break; } Q.push_back(nq); }
        int kexact=(int)Q.size();
        for(double tau:taus){
            Op p=h;   // budgeted chain P_k
            for(int k=1;k<=kexact;++k){
                Op ref=Q[k-1];
                double fe = (tau==0)?0.0 : frob_rel(p,ref);
                std::printf("SPTRIM,%s,%.0e,%d,%d,%d,%lld,%lld,%.4f,%.3e\n",
                    argv[f],tau,k,(int)ref.offs.size(),(int)p.offs.size(),
                    nnz(ref),nnz(p), 100.0*(double)nnz(p)/(double)std::max<long long>(1,nnz(ref)), fe);
                std::fflush(stdout);
                if(k<kexact){ Op np=matmul(p,h,OFFCAP); if(np.offs.empty()) break; if(tau>0) np=trim(np,tau); p=std::move(np); }
            }
        }
        if(overflow) std::fprintf(stderr,"#   (exact chain stopped at k=%d: >%d offsets)\n",kexact,OFFCAP);
    }
    return 0;
}
