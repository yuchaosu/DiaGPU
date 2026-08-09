/* ============================================================
 * amort_spmspm_bench.cu — SpMSpM amortization: N repeated products with
 * FIXED structure (time-dependent H / parameter sweeps: coefficients
 * change, Pauli structure does not).
 *   ours_flat : pair-plan built ONCE, then N kernel launches
 *   cusparse  : N full SpGEMM rounds (symbolic structure discovery is
 *               paid EVERY time — no structure reuse in the generic API)
 * out: AMORTSPMSPM,file,variant,N,total_ms      N in {1,3,10,30,100}
 * usage: amort_spmspm_bench <dia.txt>
 * ============================================================ */
#include "../dia_io.hpp"
#include "common.cuh"
#include "../../spmspm/gather_flat.cuh"
#include "../../spmspm/gather_flat_sym.cuh"
#include <cusparse.h>
#include <chrono>
#include <unordered_map>
#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)
using clk2 = std::chrono::steady_clock;
static double msb(clk2::time_point a, clk2::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }

struct Cstruct { std::vector<int> offsets, lengths; std::vector<size_t> starts; size_t nnz; };
static Cstruct make_c(const DiaHost& A, const DiaHost& B){
    int n = A.n; std::vector<char> present(2*(size_t)n-1, 0);
    for (int da : A.offsets) for (int db : B.offsets){ int dc = da+db; if (dc > -n && dc < n) present[dc+n-1] = 1; }
    Cstruct C; size_t off = 0;
    for (int d = -(n-1); d <= n-1; ++d){
        if (!present[d+n-1]) continue;
        int len = n - std::abs(d);
        C.offsets.push_back(d); C.starts.push_back(off); C.lengths.push_back(len); off += len;
    }
    C.nnz = off; return C;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr,"usage: %s dia.txt\n", argv[0]); return 1; }
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();
    const long Ns[] = {1, 3, 10, 30, 100};

    { float* w; CUDA_CHECK(cudaMalloc(&w, 1<<20));   /* context warmup */
      cudaMemset(w, 0, 1<<20); CUDA_CHECK(cudaDeviceSynchronize()); cudaFree(w); }

    if (!getenv("SYM_ONLY"))
    for (long N : Ns) {                              /* ---- ours_flat ---- */
        auto t0 = clk2::now();
        Cstruct C = make_c(H, H);
        const int Cn = (int)C.offsets.size();
        std::unordered_map<int,int> bIdx; for (int i = 0; i < nd; ++i) bIdx[H.offsets[i]] = i;
        std::vector<int> pairPtr(Cn+1, 0); std::vector<GPair> pairs;
        for (int k = 0; k < Cn; ++k){
            int dc = C.offsets[k], minc = dc < 0 ? dc : 0;
            pairPtr[k] = (int)pairs.size();
            for (int ai = 0; ai < nd; ++ai){
                int da = H.offsets[ai], db = dc - da; if (db <= -n || db >= n) continue;
                auto it = bIdx.find(db); if (it == bIdx.end()) continue;
                GPair g; g.ab = H.starts[ai]; g.bb = H.starts[it->second];
                g.ash = (da < 0 ? da : 0) - minc; g.bsh = da + (db < 0 ? db : 0) - minc;
                g.al = H.lengths[ai]; g.bl = H.lengths[it->second];
                pairs.push_back(g);
            }
        }
        pairPtr[Cn] = (int)pairs.size();
        const int TILE = 256, ILP = n >= 65536 ? 4 : 1, POS = TILE * ILP;
        std::vector<int2> tiles;
        for (int k = 0; k < Cn; ++k) for (int ts = 0; ts < C.lengths[k]; ts += POS) tiles.push_back(make_int2(k, ts));
        float* dHv = dupload(H.values);
        GPair* dP = dupload(pairs); int* dPp = dupload(pairPtr); int2* dT = dupload(tiles);
        size_t* dCs = dupload(C.starts); int* dCl = dupload(std::vector<int>(C.lengths));
        float* dCv;
        if (cudaMalloc(&dCv, C.nnz*4) != cudaSuccess){ printf("AMORTSPMSPM,%s,ours_flat,%ld,OOM\n", argv[1], N); break; }
        const int nT = (int)tiles.size();
        for (long i = 0; i < N; ++i){
            if (ILP == 4) gather_flat_kernel<64,4><<<nT,TILE>>>(dHv,dHv,dT,dPp,dP,dCv,dCs,dCl);
            else          gather_flat_kernel<64,1><<<nT,TILE>>>(dHv,dHv,dT,dPp,dP,dCv,dCs,dCl);
        }
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("AMORTSPMSPM,%s,ours_flat,%ld,%.3f\n", argv[1], N, msb(t0, clk2::now()));
        cudaFree(dHv);cudaFree(dP);cudaFree(dPp);cudaFree(dT);cudaFree(dCs);cudaFree(dCl);cudaFree(dCv);
    }

    /* ---- ours_sym (L4): plan once + N kernels; SYMMETRIC H only ---------- */
    {
        bool sym = true;
        { std::unordered_map<int,int> ix; for (int i=0;i<nd;++i) ix[H.offsets[i]]=i;
          for (int i=0;i<nd && sym;++i){ int d=H.offsets[i]; if (d<=0) continue;
              auto it=ix.find(-d); if (it==ix.end()){ sym=false; break; }
              const float* a=&H.values[H.starts[i]]; const float* b=&H.values[H.starts[it->second]];
              for (int p=0;p<H.lengths[i];++p) if (a[p]!=b[p]){ sym=false; break; } } }
        if (!sym) fprintf(stderr, "# ours_sym skipped: not numerically symmetric\n");
        else for (long N : Ns) {
            auto t0 = clk2::now();
            int nup = 0; std::vector<int> Aoff, Alen; std::vector<size_t> Ast; std::vector<int> Babs(n, -1);
            for (int i=0;i<nd;++i) if (H.offsets[i]>=0){ Babs[H.offsets[i]]=nup++;
                Aoff.push_back(H.offsets[i]); Alen.push_back(H.lengths[i]); Ast.push_back(H.starts[i]); }
            /* upper C structure */
            std::vector<char> pres(2*(size_t)n-1,0);
            for (int da : H.offsets) for (int db : H.offsets){ int dc=da+db; if (dc>-n&&dc<n) pres[dc+n-1]=1; }
            std::vector<int> Coff, Clen; std::vector<size_t> Cst; size_t o=0;
            for (int d=0; d<=n-1; ++d){ if(!pres[d+n-1]) continue; int l=n-d;
                Coff.push_back(d); Cst.push_back(o); Clen.push_back(l); o+=l; }
            FlatSymPlan sp = build_flat_sym_plan(n, Aoff,Alen,Ast, Alen,Ast, Babs, Coff, Clen, 256*4);
            float* dHv = dupload(H.values);
            float* dCv;
            if (cudaMalloc(&dCv, o*4) != cudaSuccess){ printf("AMORTSPMSPM,%s,ours_sym,%ld,OOM\n", argv[1], N); cudaFree(dHv); break; }
            size_t* dCs = dupload(Cst); int* dCl = dupload(Clen);
            GPair* dP = dupload(sp.pairs); int* dPp = dupload(sp.pairPtr); int2* dT = dupload(sp.tiles);
            for (long i = 0; i < N; ++i){
                cudaMemset(dCv, 0, o*4);
                gather_flat_sym_kernel<256,4><<<(int)sp.tiles.size(),256>>>(dHv,dHv,dT,dPp,dP,dCv,dCs,dCl);
            }
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            printf("AMORTSPMSPM,%s,ours_sym,%ld,%.3f\n", argv[1], N, msb(t0, clk2::now()));
            cudaFree(dHv);cudaFree(dCv);cudaFree(dCs);cudaFree(dCl);cudaFree(dP);cudaFree(dPp);cudaFree(dT);
        }
    }

    /* ---- cuSPARSE: N x (workEstimation + compute + copy), fresh each round */
    if (getenv("SYM_ONLY") || getenv("OURS_ONLY")) return 0;   /* cusparse below */
    CsrHost csr = dia_to_csr(H);
    int *drp = dupload(csr.row_ptr), *dci = dupload(csr.col_idx); float* dv = dupload(csr.vals);
    cusparseHandle_t hd; CUSP_CHECK(cusparseCreate(&hd));
    const float al = 1.f, be = 0.f;
    const auto OP = CUSPARSE_OPERATION_NON_TRANSPOSE; const auto AL = CUSPARSE_SPGEMM_DEFAULT;
    for (long N : Ns) {
        auto t0 = clk2::now();
        for (long i = 0; i < N; ++i) {
            cusparseSpMatDescr_t mA,mB,mC;
            CUSP_CHECK(cusparseCreateCsr(&mA,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
            CUSP_CHECK(cusparseCreateCsr(&mB,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
            CUSP_CHECK(cusparseCreateCsr(&mC,n,n,0,nullptr,nullptr,nullptr,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
            cusparseSpGEMMDescr_t sd; CUSP_CHECK(cusparseSpGEMM_createDescr(&sd));
            size_t b1=0,b2=0; void *d1=nullptr,*d2=nullptr;
            CUSP_CHECK(cusparseSpGEMM_workEstimation(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b1,nullptr));
            CUDA_CHECK(cudaMalloc(&d1,b1));
            CUSP_CHECK(cusparseSpGEMM_workEstimation(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b1,d1));
            CUSP_CHECK(cusparseSpGEMM_compute(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b2,nullptr));
            CUDA_CHECK(cudaMalloc(&d2,b2));
            CUSP_CHECK(cusparseSpGEMM_compute(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b2,d2));
            int64_t cr,cc,cn; CUSP_CHECK(cusparseSpMatGetSize(mC,&cr,&cc,&cn));
            int *crp,*cci; float* cv;
            CUDA_CHECK(cudaMalloc(&crp,(n+1)*4)); CUDA_CHECK(cudaMalloc(&cci,cn*4)); CUDA_CHECK(cudaMalloc(&cv,cn*4));
            CUSP_CHECK(cusparseCsrSetPointers(mC,crp,cci,cv));
            CUSP_CHECK(cusparseSpGEMM_copy(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd));
            CUDA_CHECK(cudaDeviceSynchronize());
            cusparseSpGEMM_destroyDescr(sd);
            cusparseDestroySpMat(mA); cusparseDestroySpMat(mB); cusparseDestroySpMat(mC);
            cudaFree(d1);cudaFree(d2);cudaFree(crp);cudaFree(cci);cudaFree(cv);
        }
        printf("AMORTSPMSPM,%s,cusparse,%ld,%.3f\n", argv[1], N, msb(t0, clk2::now()));
    }
    return 0;
}
