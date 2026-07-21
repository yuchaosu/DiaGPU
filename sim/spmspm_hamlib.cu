/* ============================================================
 * spmspm_hamlib.cu  — test the diagonal SpGEMM on REAL HamLib H.
 *
 * Computes C = H * H (the first Taylor product) for a HamLib Hamiltonian in
 * diagonal format, with ours (gather_meta), the HM atomic baseline, and
 * cuSPARSE, verified against a CPU diagonal reference. Unlike the synthetic
 * bench, H here has NON-CONTIGUOUS offsets {0,±2^i,...} with large gaps — so
 * this also tests the B_lookup path (banded != contiguous).
 *
 * The smem "sym" kernel is intentionally absent: its halos = max|shift| blow
 * up for offsets spanning ±2^(q-1), so it is inapplicable to HamLib spectra.
 *
 * build: nvcc -O3 -arch=sm_90 -Xcompiler -fopenmp spmspm_hamlib.cu \
 *        ../spmspm/paper_hm_kernel.cu -lcusparse -o spmspm_hamlib
 * ============================================================ */
#include "dia_io.hpp"
#include "../spmspm/spmspm.cu"        // gather_meta_kernel (+ PairEntry)
#include "../spmspm/paper_hm.cuh"     // HMMatrix + helpers + HM kernel decl
#include "../spmspm/gather_flat.cuh"  // gather_flat_kernel + plan (new best one-shot kernel)
#include <unordered_map>
#include <cstring>
// gather_flat smem pair-list cap (max pairs per C-diagonal <= #H diagonals).
// Overridable for the OOM probe: wide bands (BH D=112) need MAXP >= #H diags,
// else the kernel would silently read a truncated pair list.
#ifndef FLAT_MAXP
#define FLAT_MAXP 64
#endif

#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

// C = A*B structure: all offsets {da+db}, ascending, with size_t starts.
struct Cstruct { std::vector<int> offsets, lengths; std::vector<size_t> starts; size_t nnz; };
static Cstruct make_c(const DiaHost& A,const DiaHost& B){
    int n=A.n; std::vector<char> present(2*n-1,0);
    for(int da:A.offsets) for(int db:B.offsets){ int dc=da+db; if(dc>-n&&dc<n) present[dc+n-1]=1; }
    Cstruct C; size_t off=0;
    for(int d=-(n-1);d<=n-1;++d){ if(!present[d+n-1])continue; int len=n-std::abs(d); if(len<=0)continue;
        C.offsets.push_back(d); C.starts.push_back(off); C.lengths.push_back(len); off+=len; }
    C.nnz=off; return C;
}
// CPU diagonal reference C=A*B (OpenMP), C-diag layout.
static std::vector<float> cpu_ref(const DiaHost&A,const DiaHost&B,const Cstruct&C){
    std::unordered_map<int,int> bIdx; for(int i=0;i<(int)B.offsets.size();++i) bIdx[B.offsets[i]]=i;
    std::vector<float> out(C.nnz,0.f); int numC=C.offsets.size(),An=A.offsets.size(),n=A.n;
    #pragma omp parallel for schedule(dynamic)
    for(int k=0;k<numC;++k){ int dc=C.offsets[k],minc=dc<0?dc:0; size_t cb=C.starts[k]; int lenC=C.lengths[k];
        for(int p=0;p<lenC;++p){ float acc=0;
            for(int ai=0;ai<An;++ai){ int da=A.offsets[ai],db=dc-da; auto it=bIdx.find(db); if(it==bIdx.end())continue; int bi=it->second;
                int pa=p+(da<0?da:0)-minc, pb=p+da+(db<0?db:0)-minc;
                if(pa<0||pa>=A.lengths[ai])continue; if(pb<0||pb>=B.lengths[bi])continue;
                acc+=A.values[A.starts[ai]+pa]*B.values[B.starts[bi]+pb]; }
            out[cb+p]=acc; } }
    return out;
}
static double maxdiff(const std::vector<float>&a,const std::vector<float>&b){ if(a.size()!=b.size())return 1e30; double m=0; for(size_t i=0;i<a.size();++i) m=std::max(m,(double)std::fabs(a[i]-b[i])); return m; }
template<class F> static float tms(F f,int wu,int it){ for(int i=0;i<wu;++i)f(); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s); for(int i=0;i<it;++i)f(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); cudaEventDestroy(s); cudaEventDestroy(e); return ms/it; }

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s <dia_file> [iters=50] [--csv]\n",argv[0]);return 1;}
    int iters=50; bool csv=false;
    for(int i=2;i<argc;++i){ if(!strcmp(argv[i],"--csv")) csv=true; else iters=atoi(argv[i]); }
    DiaHost H=load_dia(argv[1]); int n=H.n;
    Cstruct C=make_c(H,H);
    // OOM-sweep isolation: MODE=ours|hm|cusp runs a single implementation so its
    // memory ceiling is measured in its own process (a baseline OOM can't mask
    // our kernel's true ceiling). Default = all three (with verify).
    const char* MODE=getenv("MODE");
    auto isM=[&](const char*m){return MODE&&!strcmp(MODE,m);};
    const bool do_ours = !MODE || isM("ours");
    const bool do_hm   = !MODE || isM("hm");
    const bool do_cusp = !MODE || isM("cusp");
    printf("=== HamLib SpGEMM C=H*H === MODE=%s\nfile: %s\nn=%d  H diags=%zu  nnz(H)=%zu  C diags=%zu  nnz(C)=%zu\n",
           MODE?MODE:"both",argv[1],n,H.offsets.size(),H.nnz,C.offsets.size(),C.nnz);

    // results hoisted so the summary prints regardless of which MODE ran
    float t_meta=0,t_flat=0,t_hm=0,t_csp=-1; int64_t Cnnz=0; bool csp_ok=false;
    std::vector<float> C_meta,C_flat,C_hm;
    const int An=H.offsets.size();            // #H diagonals (needed by HM + cuSPARSE too)
    std::vector<size_t> Hs(H.starts);         // diagonal start offsets (shared)

    // ================= ours: gather_meta + gather_flat =================
    if(do_ours){
    // ---- device: H (shared A=B) ----
    float* dHv; size_t* dHs; int* dHoff; int* dHlen;
    std::vector<int> Hoff(H.offsets), Hlen(H.lengths);
    CUDA_CHECK(cudaMalloc(&dHv,H.nnz*sizeof(float))); CUDA_CHECK(cudaMemcpy(dHv,H.values.data(),H.nnz*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dHs,Hs.size()*sizeof(size_t))); CUDA_CHECK(cudaMemcpy(dHs,Hs.data(),Hs.size()*sizeof(size_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dHoff,Hoff.size()*sizeof(int))); CUDA_CHECK(cudaMemcpy(dHoff,Hoff.data(),Hoff.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dHlen,Hlen.size()*sizeof(int))); CUDA_CHECK(cudaMemcpy(dHlen,Hlen.data(),Hlen.size()*sizeof(int),cudaMemcpyHostToDevice));
    std::vector<int> Blk(2*(size_t)n-1,-1); for(int i=0;i<(int)H.offsets.size();++i) Blk[H.offsets[i]+n-1]=i;
    int* dBlk; CUDA_CHECK(cudaMalloc(&dBlk,Blk.size()*sizeof(int))); CUDA_CHECK(cudaMemcpy(dBlk,Blk.data(),Blk.size()*sizeof(int),cudaMemcpyHostToDevice));
    // C device
    float* dCv; size_t* dCs; int* dCoff; int* dClen;
    CUDA_CHECK(cudaMalloc(&dCv,C.nnz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCs,C.starts.size()*sizeof(size_t))); CUDA_CHECK(cudaMemcpy(dCs,C.starts.data(),C.starts.size()*sizeof(size_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dCoff,C.offsets.size()*sizeof(int))); CUDA_CHECK(cudaMemcpy(dCoff,C.offsets.data(),C.offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dClen,C.lengths.size()*sizeof(int))); CUDA_CHECK(cudaMemcpy(dClen,C.lengths.data(),C.lengths.size()*sizeof(int),cudaMemcpyHostToDevice));

    // ---- ours: gather_meta ----
    constexpr int A_DMAX=128;  // smem pair list per C-diag <= #H diagonals (<=41 here)
    int Cn=C.offsets.size();
    if(An>A_DMAX){fprintf(stderr,"A_ndiag %d > A_DMAX\n",An);return 1;}
    const int ILP=4; dim3 mblk(256), mgrid((n+256*ILP-1)/(256*ILP),Cn);
    auto launch_meta=[&](){ gather_meta_kernel<A_DMAX><<<mgrid,mblk>>>(dHv,dHs,dHoff,dHlen,An, dHv,dHs,dHlen,dBlk, dCv,dCs,dCoff,dClen,Cn,n); };
    CUDA_CHECK(cudaMemset(dCv,0,C.nnz*sizeof(float))); launch_meta(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    C_meta.resize(C.nnz); CUDA_CHECK(cudaMemcpy(C_meta.data(),dCv,C.nnz*sizeof(float),cudaMemcpyDeviceToHost));
    t_meta=tms(launch_meta,5,iters);

    // ---- ours (NEW best one-shot kernel): gather_flat (flatten + precomputed pairs) ----
    // plan built ONCE (A=B=H), adaptive ILP (small n: many thin blocks; large n: amortize).
    constexpr int MAXP=FLAT_MAXP;
    if(An>MAXP){fprintf(stderr,"A_ndiag %d > FLAT_MAXP %d — rebuild with -DFLAT_MAXP=%d\n",An,MAXP,An);return 1;}
    int FILP = (n>=65536)?4:1, TILE=256, POS=TILE*FILP;
    std::unordered_map<int,int> bIdxF; for(int i=0;i<An;++i) bIdxF[H.offsets[i]]=i;
    std::vector<int> pairPtr(Cn+1,0); std::vector<GPair> pairs; std::vector<int2> tiles;
    for(int k=0;k<Cn;++k){ int dc=C.offsets[k],minc=dc<0?dc:0; pairPtr[k]=(int)pairs.size();
      for(int ai=0;ai<An;++ai){ int da=H.offsets[ai],db=dc-da; if(db<=-n||db>=n)continue;
        auto it=bIdxF.find(db); if(it==bIdxF.end())continue; int bi=it->second;
        GPair g; g.ab=H.starts[ai]; g.bb=H.starts[bi]; g.ash=(da<0?da:0)-minc; g.bsh=da+(db<0?db:0)-minc; g.al=H.lengths[ai]; g.bl=H.lengths[bi];
        pairs.push_back(g); }
      for(int ts=0;ts<C.lengths[k];ts+=POS) tiles.push_back(make_int2(k,ts)); }
    pairPtr[Cn]=(int)pairs.size(); int nTiles=(int)tiles.size();
    GPair* dPairs; int* dPairPtr; int2* dTiles;
    CUDA_CHECK(cudaMalloc(&dPairs,pairs.size()*sizeof(GPair))); CUDA_CHECK(cudaMemcpy(dPairs,pairs.data(),pairs.size()*sizeof(GPair),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dPairPtr,(Cn+1)*4)); CUDA_CHECK(cudaMemcpy(dPairPtr,pairPtr.data(),(Cn+1)*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dTiles,nTiles*sizeof(int2))); CUDA_CHECK(cudaMemcpy(dTiles,tiles.data(),nTiles*sizeof(int2),cudaMemcpyHostToDevice));
    auto launch_flat=[&](){ if(FILP==4) gather_flat_kernel<MAXP,4><<<nTiles,TILE>>>(dHv,dHv,dTiles,dPairPtr,dPairs,dCv,dCs,dClen);
                            else         gather_flat_kernel<MAXP,1><<<nTiles,TILE>>>(dHv,dHv,dTiles,dPairPtr,dPairs,dCv,dCs,dClen); };
    CUDA_CHECK(cudaMemset(dCv,0,C.nnz*sizeof(float))); launch_flat(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    C_flat.resize(C.nnz); CUDA_CHECK(cudaMemcpy(C_flat.data(),dCv,C.nnz*sizeof(float),cudaMemcpyDeviceToHost));
    t_flat=tms(launch_flat,5,iters);
    cudaFree(dHv);cudaFree(dHs);cudaFree(dHoff);cudaFree(dHlen);cudaFree(dBlk);
    cudaFree(dCv);cudaFree(dCs);cudaFree(dCoff);cudaFree(dClen);
    cudaFree(dPairs);cudaFree(dPairPtr);cudaFree(dTiles);
    } // end do_ours

    // ================= HM atomic baseline =================
    if(do_hm){
    HMMatrix hA; hA.n=n; hA.num_diags=An; hA.diag_offsets=H.offsets; hA.diag_lengths=H.lengths; hA.values=H.values; hA.total_nz=H.nnz;
    hA.diag_starts.resize(Hs.size()); for(size_t i=0;i<Hs.size();++i) hA.diag_starts[i]=(int)Hs[i];
    HMMatrix hC=compute_c_hm_structure(hA,hA,n); std::vector<int> cLk=build_c_diag_lookup(hC,n);
    float* hAv;int*hAo,*hAs,*hAl; auto up=[&](void**p,const void*s,size_t b){CUDA_CHECK(cudaMalloc(p,b));CUDA_CHECK(cudaMemcpy(*p,s,b,cudaMemcpyHostToDevice));};
    up((void**)&hAv,hA.values.data(),hA.values.size()*4); up((void**)&hAo,hA.diag_offsets.data(),An*4); up((void**)&hAs,hA.diag_starts.data(),An*4); up((void**)&hAl,hA.diag_lengths.data(),An*4);
    float* hCv;int*hCo,*hCs,*hCl,*hClk; CUDA_CHECK(cudaMalloc(&hCv,hC.total_nz*4));
    up((void**)&hCo,hC.diag_offsets.data(),hC.num_diags*4); up((void**)&hCs,hC.diag_starts.data(),hC.num_diags*4); up((void**)&hCl,hC.diag_lengths.data(),hC.num_diags*4); up((void**)&hClk,cLk.data(),cLk.size()*4);
    int nzA=hA.total_nz, hmblk=(nzA+255)/256;
    auto launch_hm=[&](){ hm_structured_sparse_matmul_kernel<<<hmblk,256>>>(hAv,hAo,hAs,hAl,An, hAv,hAo,hAs,hAl,An, hCv,hCo,hCs,hCl,hC.num_diags,hClk, nzA,n); };
    CUDA_CHECK(cudaMemset(hCv,0,hC.total_nz*4)); launch_hm(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    C_hm.resize(hC.total_nz); CUDA_CHECK(cudaMemcpy(C_hm.data(),hCv,hC.total_nz*4,cudaMemcpyDeviceToHost));
    t_hm=tms([&](){CUDA_CHECK(cudaMemset(hCv,0,hC.total_nz*4)); launch_hm();},5,iters);
    cudaFree(hAv);cudaFree(hAo);cudaFree(hAs);cudaFree(hAl);
    cudaFree(hCv);cudaFree(hCo);cudaFree(hCs);cudaFree(hCl);cudaFree(hClk);
    } // end do_hm

    // ================= cuSPARSE SpGEMM (CSR) =================
    if(do_cusp){
    CsrHost csr=dia_to_csr(H);
    int *drp,*dci; float* dv; CUDA_CHECK(cudaMalloc(&drp,(n+1)*4));CUDA_CHECK(cudaMalloc(&dci,csr.nnz*4));CUDA_CHECK(cudaMalloc(&dv,csr.nnz*4));
    CUDA_CHECK(cudaMemcpy(drp,csr.row_ptr.data(),(n+1)*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dci,csr.col_idx.data(),csr.nnz*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dv,csr.vals.data(),csr.nnz*4,cudaMemcpyHostToDevice));
    cusparseHandle_t hd; CUSP_CHECK(cusparseCreate(&hd)); const float al=1,be=0; auto OP=CUSPARSE_OPERATION_NON_TRANSPOSE; auto AL=CUSPARSE_SPGEMM_DEFAULT;
    csp_ok=true;
    auto run_csp=[&](float* outms){
        cusparseSpMatDescr_t mA,mB,mC; CUSP_CHECK(cusparseCreateCsr(&mA,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateCsr(&mB,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        int* Crp; CUDA_CHECK(cudaMalloc(&Crp,(n+1)*4)); CUSP_CHECK(cusparseCreateCsr(&mC,n,n,0,Crp,nullptr,nullptr,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        cusparseSpGEMMDescr_t sd; cusparseSpGEMM_createDescr(&sd); cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
        cudaEventRecord(e0); size_t b1=0; void*bu1=nullptr;
        cusparseStatus_t st=cusparseSpGEMM_workEstimation(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b1,nullptr);
        if(st==CUSPARSE_STATUS_INSUFFICIENT_RESOURCES){csp_ok=false;}
        else{ CUDA_CHECK(cudaMalloc(&bu1,b1)); cusparseSpGEMM_workEstimation(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b1,bu1);
          size_t b2=0; void*bu2=nullptr; st=cusparseSpGEMM_compute(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b2,nullptr);
          if(st==CUSPARSE_STATUS_INSUFFICIENT_RESOURCES){csp_ok=false;} else{
            CUDA_CHECK(cudaMalloc(&bu2,b2)); cusparseSpGEMM_compute(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b2,bu2);
            int64_t cr,cc,cn; cusparseSpMatGetSize(mC,&cr,&cc,&cn); Cnnz=cn; cudaFree(bu2);} cudaFree(bu1); }
        cudaEventRecord(e1); cudaEventSynchronize(e1); if(outms)cudaEventElapsedTime(outms,e0,e1);
        cudaFree(Crp); cusparseSpGEMM_destroyDescr(sd); cusparseDestroySpMat(mA);cusparseDestroySpMat(mB);cusparseDestroySpMat(mC); cudaEventDestroy(e0);cudaEventDestroy(e1);
    };
    { float a=0; for(int i=0;i<std::max(3,iters/5)&&csp_ok;++i){run_csp(&a); t_csp=(t_csp<0?a:t_csp+a);} if(csp_ok) t_csp/=std::max(3,iters/5); }
    cudaFree(drp);cudaFree(dci);cudaFree(dv); cusparseDestroy(hd);
    } // end do_cusp

    // ---- verify vs CPU (only when ours+hm both ran) ----
    // CPU reference is O(C.nnz*|H|) — too slow past q16. The kernel logic is
    // size-independent, so for large n we rely on the GPU cross-check (flat==meta==hm,
    // all verified exact vs CPU at small q) instead of recomputing on the CPU.
    // Verify only runs in default (both) mode where all three result buffers exist.
    if(do_ours&&do_hm){
      if(n<=65536){
        std::vector<float> ref=cpu_ref(H,H,C);
        printf("\n[verify] max|meta - cpu| = %.3e\n", maxdiff(C_meta,ref));
        printf("[verify] max|flat - cpu| = %.3e   max|flat - meta| = %.3e\n", maxdiff(C_flat,ref), maxdiff(C_flat,C_meta));
        printf("[verify] max|hm   - cpu| = %.3e\n", maxdiff(C_hm,ref));
      } else {
        printf("\n[verify] CPU skipped (n=%d>65536); GPU cross-check: max|flat-meta| = %.3e\n",
               n, maxdiff(C_flat,C_meta));
      }
    }

    // "ours" = gather_flat (the new best one-shot kernel); run_suite.sh parses "vs ours".
    printf("\n=== timings (ms, avg of %d; n=%d) ===\n",iters,n);
    if(do_ours) printf("  ours (gather_flat) : %9.4f ms   (gather_meta was %.4f ms, flat %.2fx)\n",t_flat,t_meta,t_meta/t_flat);
    if(do_hm){ if(do_ours) printf("  HM (atomicAdd)     : %9.4f ms   (%.2fx vs ours)\n",t_hm,t_hm/t_flat);
               else        printf("  HM (atomicAdd)     : %9.4f ms\n",t_hm); }
    if(do_cusp){ if(csp_ok){ if(do_ours) printf("  cuSPARSE SpGEMM    : %9.4f ms   (%.2fx vs ours)  [C nnz=%lld]\n",t_csp,t_csp/t_flat,(long long)Cnnz);
                             else        printf("  cuSPARSE SpGEMM    : %9.4f ms   [C nnz=%lld]\n",t_csp,(long long)Cnnz); }
                 else       printf("  cuSPARSE SpGEMM    : FILL-IN WALL (insufficient resources)\n"); }
    printf("OOM_OK MODE=%s n=%d\n",MODE?MODE:"both",n);   // sentinel: reaching here = no device OOM

    if(csv){
      // SPCSV,file,n,Hdiags,nnz_H,Cdiags,nnz_C,ours_ms,hm_ms,cusp_ms,ours_vs_hm,ours_vs_cusp
      double ours=do_ours?t_flat:-1, hm=do_hm?t_hm:-1, cusp=(do_cusp&&csp_ok)?t_csp:-1;
      double ovh=(ours>0&&hm>0)?hm/ours:-1, ovc=(ours>0&&cusp>0)?cusp/ours:-1;
      printf("SPCSV,%s,%d,%zu,%zu,%zu,%zu,%.6f,%.6f,%.6f,%.4f,%.4f\n",
             argv[1],n,H.offsets.size(),H.nnz,C.offsets.size(),C.nnz,ours,hm,cusp,ovh,ovc);
    }
    return 0;
}
