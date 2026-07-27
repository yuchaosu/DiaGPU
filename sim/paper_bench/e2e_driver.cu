/* ============================================================
 * e2e_driver.cu — end-to-end paper benchmark.
 *
 * Workload (a): Taylor time evolution.  S steps; each step evaluates the
 * degree-K Horner polynomial u = sum_k c_k H^k v with complex coefficients
 * c_k = (-i dt)^k / k! and REAL H, so each apply is a fused two-vector SpMV
 * (real+imag planes).  Two pipelines, identical op sequence:
 *   ours : hybrid-selected plan (stream / gopt / didx by D, fill, rho —
 *          same selection code as spmv_driver), ONE fused launch per apply
 *   cusp : cuSPARSE CSR ALG2, TWO SpMV calls per apply (real, imag)
 * Verified: final states of the two pipelines compared (rel L2).
 * Plan build and CSR build are timed separately (the amortization table).
 *
 * Workload (b): operator-power chain H^2, H^3 with the pair-plan SpMSpM
 * (gather_flat, host-built schedule per step) vs a cuSPARSE SpGEMM chain
 * (with SpGEMM_copy materialization so each output feeds the next step).
 * Guarded by a projected-nnz cap; a cuSPARSE INSUFFICIENT_RESOURCES wall
 * is reported as -1 (itself a datapoint: the fill-in wall).
 *
 * Output rows:
 *   E2EEVCSV,file,n,D,steps,K,pick,plan_ms,csr_ms,ours_ms,cusp_ms,x,relerr
 *   E2EOPCSV,file,n,D,power,Cd,nnzC,plan_ms,kernel_ms,cusp_ms
 *
 * usage: e2e_driver <dia_file> [steps=200] [K=6] [--no-op] [--no-cusparse]
 * ============================================================ */
#include "../dia_io.hpp"
#include "../../spmv/src/spmv_gather.cuh"
#include "gather_opt.cuh"
#include "common.cuh"
#include "../../spmspm/gather_flat.cuh"

#include <cusparse.h>
#include <chrono>
#include <unordered_map>
#include <cstring>
#include <cmath>

#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

static double wall_ms(){
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

/* u = t + c*v0   (complex, planes: re at [0,n), im at [n,2n)) */
__global__ void caxpy_kernel(int n, float ar, float ai,
                             const float* __restrict__ t,
                             const float* __restrict__ v0,
                             float* __restrict__ u)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vr = v0[i], vi = v0[n + i];
    u[i]     = t[i]     + ar * vr - ai * vi;
    u[n + i] = t[n + i] + ar * vi + ai * vr;
}

/* u = c*v0 */
__global__ void cset_kernel(int n, float ar, float ai,
                            const float* __restrict__ v0, float* __restrict__ u)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vr = v0[i], vi = v0[n + i];
    u[i]     = ar * vr - ai * vi;
    u[n + i] = ar * vi + ai * vr;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s <dia_file> [steps=200] [K=6] [--no-op] [--no-cusparse]\n", argv[0]); return 1; }
    int S = argc > 2 && argv[2][0] != '-' ? atoi(argv[2]) : 200;
    int K = argc > 3 && argv[3][0] != '-' ? atoi(argv[3]) : 6;
    bool do_op = true, do_cusp = true;
    for (int a = 2; a < argc; ++a){
        if (!strcmp(argv[a], "--no-op")) do_op = false;
        if (!strcmp(argv[a], "--no-cusparse")) do_cusp = false;
    }
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();
    const char* file = argv[1];
    const int TPB = 256;
    int SMS = 1; cudaDeviceGetAttribute(&SMS, cudaDevAttrMultiProcessorCount, 0);
    size_t nnz_true = 0; for (float v : H.values) if (v != 0.f) ++nnz_true;

    /* =================== (a) evolution =================== */
    CUDA_CHECK(cudaFree(0));   /* context init outside the timed plan build */

    /* hybrid plan selection + build (timed) — same rules as spmv_driver */
    double t0 = wall_ms();
    TilePlan T = build_coarse_plan(H, TPB*4, 1024);
    double reads = 0; for (auto& s : T.segs) reads += s.len;
    double rho  = reads / (double)H.nnz;
    double fill = (double)nnz_true / (double)H.nnz;
    const char* pick = (nd <= 4 || fill >= 0.8*rho) ? "stream"
                     : (nd > 3072 ? "gopt" : "didx");

    /* device operands for whichever plan is picked */
    float *dAv = nullptr, *dv = nullptr; DSeg* dS = nullptr;
    int *drp = nullptr, *doff = nullptr, *dfp = nullptr, *dfi = nullptr, *dpp = nullptr, *dpi = nullptr;
    unsigned char* dx8 = nullptr; unsigned short* dx16 = nullptr;
    int nseg = 0; bool wide = false;
    if (!strcmp(pick, "gopt")) {
        dAv = dupload(T.packed); dS = dupload(T.segs);
        dfp = dupload(T.fullPtr); dfi = dupload(T.fullIdx);
        dpp = dupload(T.partPtr); dpi = dupload(T.partIdx);
    } else if (!strcmp(pick, "stream")) {
        dAv = dupload(H.values);
        auto plan = build_dense_plan(H.offsets, H.starts, H.lengths);
        dS = dupload(plan); nseg = (int)plan.size();
    } else {
        DidxPlan P = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
        wide = P.wide;
        drp = dupload(P.rp); doff = dupload(std::vector<int>(H.offsets)); dv = dupload(P.val);
        if (wide) dx16 = dupload(P.d16); else dx8 = dupload(P.d8);
    }
    double plan_ms = wall_ms() - t0;

    const int ILP = (n / (TPB*4) >= 2*SMS) ? 4 : 1;
    const int b1 = (n + TPB - 1)/TPB, b4 = (n + TPB*4 - 1)/(TPB*4);
    size_t smem = (size_t)nd * 4;
    auto apply_ours = [&](const float* X, float* Y){   /* fused NV=2 */
        if (!strcmp(pick, "gopt"))
            spmv_tile_vec4_kernel<2><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,X,n,Y);
        else if (!strcmp(pick, "stream")) {
            if (ILP==4) spmv_gather_kernel<128,4,2><<<b4,TPB>>>(dAv,dS,nseg,X,n,Y);
            else        spmv_gather_kernel<128,1,2><<<b1,TPB>>>(dAv,dS,nseg,X,n,Y);
        } else if (wide)
            spmv_didx_kernel<unsigned short,2><<<b1,TPB,smem>>>(n,nd,drp,dx16,dv,doff,X,Y);
        else
            spmv_didx_kernel<unsigned char,2><<<b1,TPB,smem>>>(n,nd,drp,dx8,dv,doff,X,Y);
    };

    /* cuSPARSE pipeline (CSR build timed as its plan) */
    cusparseHandle_t h = nullptr; cusparseSpMatDescr_t mH = nullptr;
    cusparseDnVecDescr_t vI = nullptr, vO = nullptr; void* dbuf = nullptr;
    int *ccrp = nullptr, *ccci = nullptr; float* ccv = nullptr;
    double csr_ms = 0;
    if (do_cusp) {
        t0 = wall_ms();
        CsrHost csr = dia_to_csr(H);
        ccrp = dupload(csr.row_ptr); ccci = dupload(csr.col_idx); ccv = dupload(csr.vals);
        CUSP_CHECK(cusparseCreate(&h));
        CUSP_CHECK(cusparseCreateCsr(&mH,n,n,csr.nnz,ccrp,ccci,ccv,
            CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateDnVec(&vI,n,nullptr,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateDnVec(&vO,n,nullptr,CUDA_R_32F));
        csr_ms = wall_ms() - t0;
    }
    const float a1 = 1.f, b0 = 0.f;
    auto apply_cusp = [&](float* X, float* Y){          /* two real SpMVs */
        for (int v = 0; v < 2; ++v) {
            CUSP_CHECK(cusparseDnVecSetValues(vI, X + (size_t)v*n));
            CUSP_CHECK(cusparseDnVecSetValues(vO, Y + (size_t)v*n));
            CUSP_CHECK(cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vI,&b0,vO,
                CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,dbuf));
        }
    };
    if (do_cusp) {          /* SpMV buffer (query with real vectors bound) */
        float* tmp; CUDA_CHECK(cudaMalloc(&tmp, 2*(size_t)n*4));
        CUSP_CHECK(cusparseDnVecSetValues(vI, tmp)); CUSP_CHECK(cusparseDnVecSetValues(vO, tmp + n));
        size_t bsz = 0;
        CUSP_CHECK(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vI,&b0,vO,
            CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,&bsz));
        if (bsz) CUDA_CHECK(cudaMalloc(&dbuf,bsz));
        cudaFree(tmp);
    }

    /* Horner coefficients c_k = (-i dt)^k / k! */
    const double dt = 1.0e-3;
    std::vector<double> cr(K+1), ci(K+1);
    { double pr = 1, pi = 0;                    /* (-i dt)^k / k! */
      cr[0] = 1; ci[0] = 0;
      for (int k = 1; k <= K; ++k) {
          double nr = pr*0 - pi*(-dt), ni = pr*(-dt) + pi*0;  /* *= -i dt */
          pr = nr; pi = ni;
          double f = 1; for (int j = 2; j <= k; ++j) f *= j;
          cr[k] = pr / f; ci[k] = pi / f;
      }
    }

    /* state buffers */
    std::vector<float> v0h(2*(size_t)n);
    srand(7);
    { double nrm = 0;
      for (size_t i = 0; i < v0h.size(); ++i){ v0h[i] = (float)rand()/RAND_MAX - 0.5f; nrm += (double)v0h[i]*v0h[i]; }
      nrm = std::sqrt(nrm);
      for (auto& x : v0h) x = (float)(x / nrm); }
    float *dst, *du, *dtv;
    CUDA_CHECK(cudaMalloc(&dst, 2*(size_t)n*4));
    CUDA_CHECK(cudaMalloc(&du,  2*(size_t)n*4));
    CUDA_CHECK(cudaMalloc(&dtv, 2*(size_t)n*4));
    const int vb = (n + TPB - 1)/TPB;

    auto run_pipeline = [&](bool ours, float* out_ms)->std::vector<float>{
        CUDA_CHECK(cudaMemcpy(dst, v0h.data(), 2*(size_t)n*4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int s = 0; s < S; ++s) {
            /* u = c_K * state; then K times: t = H u ; u = t + c_k * state */
            cset_kernel<<<vb,TPB>>>(n, (float)cr[K], (float)ci[K], dst, du);
            for (int k = K - 1; k >= 0; --k) {
                if (ours) apply_ours(du, dtv); else apply_cusp(du, dtv);
                caxpy_kernel<<<vb,TPB>>>(n, (float)cr[k], (float)ci[k], dtv, dst, du);
            }
            std::swap(du, dst);          /* new state */
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(out_ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        std::vector<float> out(2*(size_t)n);
        CUDA_CHECK(cudaMemcpy(out.data(), dst, 2*(size_t)n*4, cudaMemcpyDeviceToHost));
        return out;
    };

    float ours_ms = 0, cusp_ms = -1; double rel = -1;
    auto so = run_pipeline(true, &ours_ms);
    if (do_cusp) {
        auto sc = run_pipeline(false, &cusp_ms);
        double num = 0, den = 0;
        for (size_t i = 0; i < so.size(); ++i){ double d = (double)so[i]-sc[i]; num += d*d; den += (double)sc[i]*sc[i]; }
        rel = den > 0 ? std::sqrt(num/den) : 0;
    }
    printf("E2EEVCSV,%s,%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3e\n",
           file, n, nd, S, K, pick, plan_ms, csr_ms, ours_ms, cusp_ms,
           cusp_ms > 0 ? cusp_ms/ours_ms : -1.0, rel);

    /* =================== (b) operator chain =================== */
    if (do_op) {
        /* ours: H^2 then H^3 = H^2 * H, pair plan rebuilt per power (host) */
        struct Cs { std::vector<int> off, len; std::vector<size_t> st; size_t nnz; };
        auto mkc = [&](const std::vector<int>& ao)->Cs{
            std::vector<char> pres(2*(size_t)n-1, 0);
            for (int da : ao) for (int db : H.offsets){ int dc = da+db; if (dc > -n && dc < n) pres[dc+n-1] = 1; }
            Cs C; size_t o = 0;
            for (int d = -(n-1); d <= n-1; ++d){ if (!pres[d+n-1]) continue;
                int l = n - std::abs(d); C.off.push_back(d); C.st.push_back(o); C.len.push_back(l); o += l; }
            C.nnz = o; return C;
        };
        std::unordered_map<int,int> bIdx; for (int i = 0; i < nd; ++i) bIdx[H.offsets[i]] = i;
        float* dHv = dupload(H.values);
        /* A starts as H (device values + host structure) */
        std::vector<int> Aoff = H.offsets, Alen = H.lengths;
        std::vector<size_t> Ast = H.starts;
        float* dAvv = dHv;
        const size_t CAP = (size_t)4e8;
        for (int p = 2; p <= 3; ++p) {
            Cs C = mkc(Aoff);
            if (C.nnz > CAP) { printf("E2EOPCSV,%s,%d,%d,%d,%zu,%zu,-1,-1,-1\n",
                                      file, n, nd, p, C.off.size(), C.nnz); break; }
            double tp0 = wall_ms();
            int Cn = (int)C.off.size();
            std::vector<int> pairPtr(Cn+1, 0); std::vector<GPair> pairs;
            for (int k = 0; k < Cn; ++k){
                int dc = C.off[k], minc = dc < 0 ? dc : 0;
                pairPtr[k] = (int)pairs.size();
                for (size_t ai = 0; ai < Aoff.size(); ++ai){
                    int da = Aoff[ai], db = dc - da; if (db <= -n || db >= n) continue;
                    auto it = bIdx.find(db); if (it == bIdx.end()) continue;
                    GPair g; g.ab = Ast[ai]; g.bb = H.starts[it->second];
                    g.ash = (da < 0 ? da : 0) - minc;
                    g.bsh = da + (db < 0 ? db : 0) - minc;
                    g.al = Alen[ai]; g.bl = H.lengths[it->second];
                    pairs.push_back(g);
                }
            }
            pairPtr[Cn] = (int)pairs.size();
            const int ILPo = n >= 65536 ? 4 : 1, POS = TPB*ILPo;
            std::vector<int2> tiles;
            for (int k = 0; k < Cn; ++k) for (int ts = 0; ts < C.len[k]; ts += POS) tiles.push_back(make_int2(k, ts));
            double plan_op = wall_ms() - tp0;
            GPair* dP = dupload(pairs); int* dPp = dupload(pairPtr); int2* dT = dupload(tiles);
            size_t* dCs2 = dupload(C.st); int* dCl = dupload(std::vector<int>(C.len));
            float* dCv; CUDA_CHECK(cudaMalloc(&dCv, C.nnz*4));
            int nT = (int)tiles.size();
            auto l = [&](){
                if (ILPo == 4) gather_flat_kernel<64,4><<<nT,TPB>>>(dAvv,dHv,dT,dPp,dP,dCv,dCs2,dCl);
                else           gather_flat_kernel<64,1><<<nT,TPB>>>(dAvv,dHv,dT,dPp,dP,dCv,dCs2,dCl);
            };
            l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            float kms = tms(l, 3, 10);
            printf("E2EOPCSV,%s,%d,%d,%d,%d,%zu,%.3f,%.3f,-1\n",
                   file, n, nd, p, Cn, C.nnz, plan_op, kms);
            /* C becomes A for the next power */
            if (dAvv != dHv) cudaFree(dAvv);
            dAvv = dCv; Aoff = C.off; Alen = C.len; Ast = C.st;
            cudaFree(dP); cudaFree(dPp); cudaFree(dT); cudaFree(dCs2); cudaFree(dCl);
        }
        if (dAvv != dHv) cudaFree(dAvv);
        cudaFree(dHv);
    }
    return 0;
}
