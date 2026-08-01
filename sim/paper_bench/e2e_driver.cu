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
 * No projected-nnz cap: allocations are ATTEMPTED and a cudaMalloc failure
 * is reported as -1 (the measured OOM wall, not a projection).  A chain
 * truncated mid-power logs "# opbuild truncated" on stderr.
 *
 * Output rows:
 *   E2EEVCSV,file,n,D,steps,K,pick,plan_ms,csr_ms,ours_ms,cusp_ms,x,relerr
 *   E2EEVDIAQCSV,file,n,D,steps,K,diaq_plan_ms,diaq_ms,diaq_vs_ours,relerr_vs_ours
 *       (same Taylor pipeline with the diaq/HamSim fused complex SpMV kernel,
 *        device-resident — their best case; A_imag = 0)
 *   E2EOPCSV,file,n,D,power,Cd,nnzC,plan_ms,kernel_ms,cusp_ms
 *   E2EOPBUILDCSV: full rows carry ours vs hamsim(diaq); the --sym L4 row has
 *        hamsim=-1.  The FULL build now also runs under --sym (before the sym
 *        block, which may crash on wide-band H) so HamLib gets the diaq
 *        comparison too.
 *
 * usage: e2e_driver <dia_file> [steps=200] [K=6] [--no-op] [--no-cusparse] [--no-diaq]
 * ============================================================ */
#include "../dia_io.hpp"
#include "../../spmv/src/spmv_gather.cuh"
#include "gather_opt.cuh"
#include "common.cuh"
#include "../../spmspm/gather_flat.cuh"
#include "../../spmspm/gather_flat_sym.cuh"   // L4: symmetric upper-half operator-build

#include <cusparse.h>
#include <chrono>
#include <unordered_map>
#include <set>
#include <cstring>
#include <string>
#include <functional>
#include <cmath>

#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

static double wall_ms(){
    using namespace std::chrono;
    return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();
}

/* graceful device alloc: nullptr on OOM (error cleared), so the caller can
 * report the measured wall instead of exiting mid-driver */
static float* try_dmalloc_f(size_t nfloats){
    float* p = nullptr;
    if (cudaMalloc(&p, nfloats*4) != cudaSuccess){ cudaGetLastError(); return nullptr; }
    return p;
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

/* operator-building accumulate (real fp32): U += c * P, P a REAL DIA operand.
 * Each source diagonal k maps to U start s2u[k] (same offset, same full-diagonal
 * length), so position p aligns directly.  blockIdx.y = diagonal. */
__global__ void accum_scaled_kernel(int ndiag,
        const int* __restrict__ s2u, const int* __restrict__ sstart,
        const int* __restrict__ slen, const float* __restrict__ sval,
        float c, float* __restrict__ U)
{
    int k = blockIdx.y; if (k >= ndiag) return;
    int len = slen[k], base = sstart[k], u0 = s2u[k];
    for (int p = blockIdx.x*blockDim.x + threadIdx.x; p < len; p += gridDim.x*blockDim.x)
        U[u0 + p] += c * sval[base + p];
}
/* set a length-n array to 1.0 (the identity diagonal source for c_0 * I) */
__global__ void ones_kernel(int n, float* p){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=1.f; }

/* Taylor order K from the workload filename convention <family>_<q>_<K>.txt
 * (the adaptive, per-matrix convergence order baked in at data-gen time).
 * Falls back to 6 if the trailing _<K> is absent. */
static int k_from_name(const char* path){
    const char* b = strrchr(path, '/'); b = b ? b+1 : path;
    std::string s(b);
    size_t dot = s.rfind(".txt"); if (dot != std::string::npos) s = s.substr(0, dot);
    size_t us = s.rfind('_'); if (us == std::string::npos || us+1 >= s.size()) return 6;
    int k = atoi(s.c_str()+us+1);
    return (k >= 1 && k <= 64) ? k : 6;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s <dia_file> [steps=1000] [K=<from filename>] [--no-op] [--no-cusparse]\n", argv[0]); return 1; }
    int S = argc > 2 && argv[2][0] != '-' ? atoi(argv[2]) : 1000;
    /* K: explicit arg wins; else the per-matrix convergence order from the filename */
    int K = argc > 3 && argv[3][0] != '-' ? atoi(argv[3]) : k_from_name(argv[1]);
    bool do_op = true, do_cusp = true, do_opbuild = true, do_sym = false, do_diaq = true, dump_state = false;
    for (int a = 2; a < argc; ++a){
        if (!strcmp(argv[a], "--no-op")) do_op = false;
        if (!strcmp(argv[a], "--no-cusparse")) do_cusp = false;
        if (!strcmp(argv[a], "--no-opbuild")) do_opbuild = false;
        if (!strcmp(argv[a], "--no-diaq")) do_diaq = false;
        if (!strcmp(argv[a], "--dump-state")) dump_state = true;   /* v0 + final psi -> cwd, for external verification */
        if (!strcmp(argv[a], "--sym")) do_sym = true;   /* L4 upper-half operator-build (symmetric H only) */
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
    if (do_cusp && (size_t)H.nnz > (size_t)INT32_MAX) {
        do_cusp = false;   /* cusparseCreateCsr(32I) rejects nnz > INT32_MAX; report -1 instead of dying */
        fprintf(stderr, "# cusparse arm skipped: H nnz %zu > INT32_MAX\n", (size_t)H.nnz);
    }
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

    /* diaq/HamSim pipeline: their fused complex SpMV kernel (device-resident,
     * their best case; A_imag = 0).  Setup (uploads) timed as its plan. */
    double diaq_plan_ms = 0;
    int* dqIdx = nullptr; unsigned *dqOff = nullptr, *dqLen = nullptr;
    float *dqAr = nullptr, *dqAi = nullptr;
    if (do_diaq) {
        t0 = wall_ms();
        std::vector<unsigned> qo(nd), ql(nd);
        for (int i = 0; i < nd; ++i){ qo[i]=(unsigned)H.starts[i]; ql[i]=(unsigned)H.lengths[i]; }
        dqIdx = dupload(std::vector<int>(H.offsets));
        dqOff = dupload(qo); dqLen = dupload(ql);
        dqAr  = dupload(H.values);
        CUDA_CHECK(cudaMalloc(&dqAi, H.nnz*4)); CUDA_CHECK(cudaMemset(dqAi, 0, H.nnz*4));
        CUDA_CHECK(cudaDeviceSynchronize());
        diaq_plan_ms = wall_ms() - t0;
    }
    auto apply_diaq = [&](const float* X, float* Y){
        diaq_spmv_row_kernel<float><<<b1,TPB>>>((unsigned)n,(unsigned)n,(unsigned)nd,
            dqIdx,dqOff,dqLen, dqAr,dqAi, X, X+n, Y, Y+n);
    };

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

    auto run_pipeline = [&](int mode, float* out_ms)->std::vector<float>{   /* 0=ours 1=cusp 2=diaq */
        auto step = [&](){
            /* u = c_K * state; then K times: t = H u ; u = t + c_k * state */
            cset_kernel<<<vb,TPB>>>(n, (float)cr[K], (float)ci[K], dst, du);
            for (int k = K - 1; k >= 0; --k) {
                if (mode == 0) apply_ours(du, dtv);
                else if (mode == 1) apply_cusp(du, dtv);
                else apply_diaq(du, dtv);
                caxpy_kernel<<<vb,TPB>>>(n, (float)cr[k], (float)ci[k], dtv, dst, du);
            }
            std::swap(du, dst);          /* new state */
        };
        /* warmup: each pipeline pays its own clock/cache ramp.  Without this the
         * FIRST pipeline (ours) absorbs the cold-GPU penalty — measured up to
         * 29x inflation on the H100 (B2_10_4, first matrix; heis_16/18 4.3-4.7x). */
        CUDA_CHECK(cudaMemcpy(dst, v0h.data(), 2*(size_t)n*4, cudaMemcpyHostToDevice));
        for (int s = 0; s < std::min(S, 20); ++s) step();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dst, v0h.data(), 2*(size_t)n*4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int s = 0; s < S; ++s) step();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(out_ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        std::vector<float> out(2*(size_t)n);
        CUDA_CHECK(cudaMemcpy(out.data(), dst, 2*(size_t)n*4, cudaMemcpyDeviceToHost));
        return out;
    };

    float ours_ms = 0, cusp_ms = -1; double rel = -1;
    auto so = run_pipeline(0, &ours_ms);
    if (dump_state) {   /* fp32 planes: re [0,n) then im [n,2n) — QuTiP/SciPy triangle check */
        FILE* fv = fopen("e2e_state_v0.bin", "wb");
        FILE* fp = fopen("e2e_state_final.bin", "wb");
        if (fv && fp) {
            fwrite(v0h.data(), 4, v0h.size(), fv);
            fwrite(so.data(),  4, so.size(),  fp);
            fprintf(stderr, "# dumped e2e_state_{v0,final}.bin (n=%d, S=%d, K=%d)\n", n, S, K);
        }
        if (fv) fclose(fv); if (fp) fclose(fp);
    }
    if (do_cusp) {
        auto sc = run_pipeline(1, &cusp_ms);
        double num = 0, den = 0;
        for (size_t i = 0; i < so.size(); ++i){ double d = (double)so[i]-sc[i]; num += d*d; den += (double)sc[i]*sc[i]; }
        rel = den > 0 ? std::sqrt(num/den) : 0;
    }
    printf("E2EEVCSV,%s,%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3e\n",
           file, n, nd, S, K, pick, plan_ms, csr_ms, ours_ms, cusp_ms,
           cusp_ms > 0 ? cusp_ms/ours_ms : -1.0, rel);
    if (do_diaq) {
        float dq_ms = 0;
        auto sd = run_pipeline(2, &dq_ms);
        double num = 0, den = 0;
        for (size_t i = 0; i < so.size(); ++i){ double d = (double)sd[i]-so[i]; num += d*d; den += (double)so[i]*so[i]; }
        double reld = den > 0 ? std::sqrt(num/den) : 0;
        printf("E2EEVDIAQCSV,%s,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3e\n",
               file, n, nd, S, K, diaq_plan_ms, dq_ms,
               ours_ms > 0 ? dq_ms/ours_ms : -1.0, reld);
        cudaFree(dqIdx); cudaFree(dqOff); cudaFree(dqLen); cudaFree(dqAr); cudaFree(dqAi);
    }

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
        for (int p = 2; p <= 3; ++p) {
            Cs C = mkc(Aoff);
            /* no cap: attempt the C allocation; failure = the measured OOM wall */
            float* dCv = try_dmalloc_f(C.nnz);
            if (!dCv) { printf("E2EOPCSV,%s,%d,%d,%d,%zu,%zu,-1,-1,-1\n",
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

    /* =================== (c-sym) operator-building — L4 SYMMETRIC (upper-half) ===================
     * --sym: build U = sum w_k H^k storing U and every power P_k UPPER-half (offsets>=0) via
     * gather_flat_sym. VALID ONLY for symmetric H (HamLib). SuiteSparse -> run WITHOUT --sym.
     * Runs as a FUNCTION invoked AFTER the full build below: gather_flat_sym is known to
     * crash on wide-band H, and running it last keeps the full ours-vs-hamsim row safe. */
    auto run_sym_opbuild = [&](){
        auto mkc_up = [&](const std::vector<int>& ao, std::vector<int>& coff, std::vector<size_t>& cst, std::vector<int>& clen)->size_t{
            std::vector<char> pres(2*(size_t)n-1,0);
            for (int da:ao) for (int db:H.offsets){ int dc=da+db; if(dc>-n&&dc<n) pres[dc+n-1]=1; }
            size_t o=0; coff.clear();cst.clear();clen.clear();
            for (int d=0; d<=n-1; ++d){ if(!pres[d+n-1])continue; int l=n-d; coff.push_back(d); cst.push_back(o); clen.push_back(l); o+=l; }
            return o;
        };
        std::set<int> uset; uset.insert(0);
        { std::set<int> cur{0};
          for (int k=1;k<=K;++k){ std::set<int> nx; for(int a:cur) for(int b:H.offsets){int d=a+b; if(d>-n&&d<n)nx.insert(d);} cur.swap(nx);
            for(int d:cur) if(d>=0) uset.insert(d); } }
        std::vector<int> Uo(uset.begin(),uset.end()); int Ud=(int)Uo.size();
        std::unordered_map<int,int> uslot; std::vector<size_t> Ust(Ud); std::vector<int> Ulen(Ud);
        size_t uo=0; for(int i=0;i<Ud;++i){ uslot[Uo[i]]=i; Ulen[i]=n-Uo[i]; Ust[i]=uo; uo+=Ulen[i]; }
        const size_t Unnz=uo;
        /* wall = OOM *or* int32 index limit: accum takes int starts, so any
         * U beyond INT32_MAX would wrap the (int)Ust casts below */
        float* dU = Unnz > (size_t)INT32_MAX ? nullptr : try_dmalloc_f(Unnz);
        if (!dU){ printf("E2EOPBUILDCSV,%s,%d,%d,%d,%zu,-1,-1,-1\n",file,n,K,Ud,Unnz);
                  fprintf(stderr,"# opbuild(sym) wall: Unnz=%zu (%s)\n",Unnz,
                          Unnz>(size_t)INT32_MAX?"int32 index limit":"OOM"); }
        else {
            std::vector<int> Hup_off,Hup_len; std::vector<size_t> Hup_st; std::vector<int> Babs(n,-1);
            for (int i=0;i<nd;++i) if(H.offsets[i]>=0){ int a=H.offsets[i]; Babs[a]=(int)Hup_off.size(); Hup_off.push_back(a); Hup_len.push_back(H.lengths[i]); Hup_st.push_back(H.starts[i]); }
            float* dHvals=dupload(H.values);
            std::vector<double> w(K+1); w[0]=1.0; for(int k=1;k<=K;++k) w[k]=w[k-1]*dt/k;
            float* dOnes; CUDA_CHECK(cudaMalloc(&dOnes,(size_t)n*4)); ones_kernel<<<(n+TPB-1)/TPB,TPB>>>(n,dOnes);
            auto accum=[&](int sd,int* dS2U,int* dSst,int* dSlen,float* sval,double c_,int maxl){
                dim3 g((maxl+TPB-1)/TPB,sd); accum_scaled_kernel<<<g,TPB>>>(sd,dS2U,dSst,dSlen,sval,(float)c_,dU); };
            cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
            CUDA_CHECK(cudaMemset(dU,0,Unnz*4));
            { std::vector<int> s2u{(int)Ust[uslot[0]]},sst{0},sln{n}; int* a=dupload(s2u);int* b=dupload(sst);int* c=dupload(sln);
              accum(1,a,b,c,dOnes,w[0],n); cudaFree(a);cudaFree(b);cudaFree(c); }
            { int hu=(int)Hup_off.size(); std::vector<int> s2u(hu),sst(hu); int ml=0;
              for(int i=0;i<hu;++i){ s2u[i]=(int)Ust[uslot[Hup_off[i]]]; sst[i]=(int)Hup_st[i]; ml=std::max(ml,Hup_len[i]); }
              int* a=dupload(s2u);int* b=dupload(sst);int* c=dupload(std::vector<int>(Hup_len));
              accum(hu,a,b,c,dHvals,w[1],ml); cudaFree(a);cudaFree(b);cudaFree(c); }
            std::vector<int> Aoff=Hup_off, Alen=Hup_len; std::vector<size_t> Ast=Hup_st; float* dAv=dHvals;
            for (int k=2;k<=K;++k){
                std::vector<int> Coff,Clen; std::vector<size_t> Cst; size_t Cnnz=mkc_up(Aoff,Coff,Cst,Clen); int Cn2=(int)Coff.size();
                float* dPk = Cnnz > (size_t)INT32_MAX ? nullptr : try_dmalloc_f(Cnnz);
                if (!dPk){ fprintf(stderr,"# opbuild(sym) truncated at k=%d (%s, Cnnz=%zu)\n", k,
                                   Cnnz>(size_t)INT32_MAX?"int32 index limit":"OOM", Cnnz);
                           if(dAv!=dHvals)cudaFree(dAv); dAv=dHvals; break; }
                CUDA_CHECK(cudaMemset(dPk,0,Cnnz*4));
                FlatSymPlan spn=build_flat_sym_plan(n, Aoff,Alen,Ast, Hup_len,Hup_st, Babs, Coff,Clen, TPB*4);
                size_t* dCst=dupload(Cst); int* dCl=dupload(std::vector<int>(Clen));
                GPair* dP=dupload(spn.pairs); int* dPp=dupload(spn.pairPtr); int2* dT=dupload(spn.tiles);
                gather_flat_sym_kernel<256,4><<<(int)spn.tiles.size(),256>>>(dAv,dHvals,dT,dPp,dP,dPk,dCst,dCl);
                CUDA_CHECK(cudaGetLastError());
                std::vector<int> c2u(Cn2),cst_i(Cn2); int ml=0;
                for(int q=0;q<Cn2;++q){ c2u[q]=(int)Ust[uslot[Coff[q]]]; cst_i[q]=(int)Cst[q]; ml=std::max(ml,Clen[q]); }
                int* dC2U=dupload(c2u); int* dCsti=dupload(cst_i);
                accum(Cn2,dC2U,dCsti,dCl,dPk,w[k],ml);
                cudaFree(dC2U);cudaFree(dCsti);cudaFree(dP);cudaFree(dPp);cudaFree(dT);cudaFree(dCst);cudaFree(dCl);
                if(dAv!=dHvals)cudaFree(dAv); dAv=dPk; Aoff=Coff; Alen=Clen; Ast=Cst;
            }
            if(dAv!=dHvals)cudaFree(dAv);
            CUDA_CHECK(cudaDeviceSynchronize()); float sym_ms=0; cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&sym_ms,e0,e1);
            cudaEventDestroy(e0);cudaEventDestroy(e1);
            /* E2EOPBUILDCSV,file,n,K,U_diags(upper),U_stored(upper),ours_sym_ms,-1,-1 */
            printf("E2EOPBUILDCSV,%s,%d,%d,%d,%zu,%.3f,-1,-1\n", file,n,K,Ud,Unnz,sym_ms);
            cudaFree(dU);cudaFree(dOnes);cudaFree(dHvals);
        }
    };

    /* =================== (c) operator-building Taylor: U = sum_{k=0}^{K} c_k H^k (FULL) ===================
     * Full-C build, ours(gather_flat) vs hamsim(diaq).  Runs for EVERY matrix (also under
     * --sym, so HamLib gets the hamsim comparison); the L4 sym build follows afterwards.
     * H real so each P_k real; only c_k = (-i dt)^k/k! are complex. */
    if (do_opbuild) {
        auto mkc2 = [&](const std::vector<int>& ao, std::vector<int>& coff,
                        std::vector<size_t>& cst, std::vector<int>& clen)->size_t {
            std::vector<char> pres(2*(size_t)n-1, 0);
            for (int da : ao) for (int db : H.offsets){ int dc=da+db; if (dc>-n && dc<n) pres[dc+n-1]=1; }
            size_t o=0; coff.clear(); cst.clear(); clen.clear();
            for (int d=-(n-1); d<=n-1; ++d){ if(!pres[d+n-1]) continue; int l=n-std::abs(d);
                coff.push_back(d); cst.push_back(o); clen.push_back(l); o+=l; }
            return o;
        };
        std::set<int> uset; uset.insert(0);
        { std::set<int> cur{0};
          for (int k=1;k<=K;++k){ std::set<int> nx;
            for (int a:cur) for (int b:H.offsets){ int d=a+b; if(d>-n && d<n) nx.insert(d); }
            cur.swap(nx); for(int d:cur) uset.insert(d); } }
        std::vector<int> Uo(uset.begin(), uset.end());
        const int Ud=(int)Uo.size();
        std::unordered_map<int,int> uslot;
        std::vector<size_t> Ust(Ud); std::vector<int> Ulen(Ud);
        size_t uo=0; for (int i=0;i<Ud;++i){ uslot[Uo[i]]=i; Ulen[i]=n-std::abs(Uo[i]); Ust[i]=uo; uo+=Ulen[i]; }
        const size_t Unnz=uo;
        /* wall = OOM *or* int32 index limit ((int)Ust / (int)H.starts casts below;
         * Unnz >= H.nnz since U contains every H diagonal, so one check covers both) */
        float* dU = Unnz > (size_t)INT32_MAX ? nullptr : try_dmalloc_f(Unnz);
        if (!dU) {
            printf("E2EOPBUILDCSV,%s,%d,%d,%d,%zu,-1,-1,-1\n", file,n,K,Ud,Unnz);
            fprintf(stderr,"# opbuild wall: Unnz=%zu (%s)\n",Unnz,
                    Unnz>(size_t)INT32_MAX?"int32 index limit":"OOM");
        } else {
            std::unordered_map<int,int> hIdx; for (int i=0;i<nd;++i) hIdx[H.offsets[i]]=i;
            float* dHvals = dupload(H.values);
            std::vector<double> w(K+1); w[0]=1.0; for (int k=1;k<=K;++k) w[k]=w[k-1]*dt/k;  /* |c_k| = dt^k/k! */
            float* dOnes; CUDA_CHECK(cudaMalloc(&dOnes,(size_t)n*4)); ones_kernel<<<(n+TPB-1)/TPB,TPB>>>(n,dOnes);
            std::vector<int> h2u(nd), hst_i(nd);
            for (int i=0;i<nd;++i){ h2u[i]=(int)Ust[uslot[H.offsets[i]]]; hst_i[i]=(int)H.starts[i]; }
            int* dHoffU=dupload(h2u); int* dHstart=dupload(hst_i); int* dHlen=dupload(std::vector<int>(H.lengths));
            auto accum=[&](int sd,int* dS2U,int* dSst,int* dSlen,float* sval,double c_,int maxl){
                dim3 g((maxl+TPB-1)/TPB, sd);
                accum_scaled_kernel<<<g,TPB>>>(sd,dS2U,dSst,dSlen,sval,(float)c_,dU);
            };
            auto build=[&](bool hamsim, float* out_ms){
                cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
                CUDA_CHECK(cudaMemset(dU,0,Unnz*4));
                /* w_0 * I  (offset-0 diagonal, source = ones) */
                { std::vector<int> s2u{(int)Ust[uslot[0]]}, sst{0}, sln{n};
                  int* a=dupload(s2u); int* b=dupload(sst); int* c=dupload(sln);
                  accum(1,a,b,c,dOnes,w[0],n); cudaFree(a);cudaFree(b);cudaFree(c); }
                /* w_1 * H */
                accum(nd,dHoffU,dHstart,dHlen,dHvals,w[1],n);
                /* P_2..P_K */
                std::vector<int> Aoff=H.offsets, Alen=H.lengths; std::vector<size_t> Ast_=H.starts;
                float* dAv2=dHvals;
                for (int k=2;k<=K;++k){
                    std::vector<int> Coff,Clen; std::vector<size_t> Cst;
                    size_t Cnnz=mkc2(Aoff,Coff,Cst,Clen); int Cn=(int)Coff.size();
                    float* dCv = Cnnz > (size_t)INT32_MAX ? nullptr : try_dmalloc_f(Cnnz);
                    if (!dCv){ fprintf(stderr,"# opbuild(%s) truncated at k=%d (%s, Cnnz=%zu)\n",
                                       hamsim?"hamsim":"ours", k,
                                       Cnnz>(size_t)INT32_MAX?"int32 index limit":"OOM", Cnnz);
                               if(dAv2!=dHvals) cudaFree(dAv2); dAv2=dHvals; break; }
                    size_t* dCst=dupload(Cst); int* dClen=dupload(std::vector<int>(Clen));
                    if (!hamsim) {                                   /* ours: gather_flat */
                        std::vector<int> pairPtr(Cn+1,0); std::vector<GPair> pairs;
                        for (int q=0;q<Cn;++q){ int dc=Coff[q], minc=dc<0?dc:0; pairPtr[q]=(int)pairs.size();
                            for (size_t ai=0; ai<Aoff.size(); ++ai){ int da=Aoff[ai], db=dc-da; if(db<=-n||db>=n) continue;
                                auto it=hIdx.find(db); if(it==hIdx.end()) continue;
                                GPair g; g.ab=Ast_[ai]; g.bb=H.starts[it->second];
                                g.ash=(da<0?da:0)-minc; g.bsh=da+(db<0?db:0)-minc; g.al=Alen[ai]; g.bl=H.lengths[it->second];
                                pairs.push_back(g); } }
                        pairPtr[Cn]=(int)pairs.size();
                        int ILPo=n>=65536?4:1, POS=TPB*ILPo; std::vector<int2> tiles;
                        for (int q=0;q<Cn;++q) for (int ts=0; ts<Clen[q]; ts+=POS) tiles.push_back(make_int2(q,ts));
                        GPair* dP=dupload(pairs); int* dPp=dupload(pairPtr); int2* dT=dupload(tiles); int nT=(int)tiles.size();
                        if (ILPo==4) gather_flat_kernel<64,4><<<nT,TPB>>>(dAv2,dHvals,dT,dPp,dP,dCv,dCst,dClen);
                        else         gather_flat_kernel<64,1><<<nT,TPB>>>(dAv2,dHvals,dT,dPp,dP,dCv,dCst,dClen);
                        cudaFree(dP); cudaFree(dPp); cudaFree(dT);
                    } else {                                          /* hamsim: diaq_product (imag=0) */
                        std::vector<unsigned> pairOff(Cn+1,0),psA,psB,cO(Cn),cL(Cn); std::vector<int> pdA,pdB;
                        for (int q=0;q<Cn;++q){ pairOff[q]=(unsigned)pdA.size();
                            for (size_t ai=0; ai<Aoff.size(); ++ai){ int da=Aoff[ai], db=Coff[q]-da;
                                auto it=hIdx.find(db); if(it==hIdx.end()) continue;
                                pdA.push_back(da); pdB.push_back(db); psA.push_back((unsigned)ai); psB.push_back((unsigned)it->second); } }
                        pairOff[Cn]=(unsigned)pdA.size();
                        std::vector<unsigned> aO(Aoff.size()),aL(Aoff.size()),bO(nd),bL(nd);
                        size_t Annz=0; for (size_t i=0;i<Aoff.size();++i){ aO[i]=(unsigned)Ast_[i]; aL[i]=(unsigned)Alen[i]; Annz+=Alen[i]; }
                        for (int i=0;i<nd;++i){ bO[i]=(unsigned)H.starts[i]; bL[i]=(unsigned)H.lengths[i]; }
                        for (int q=0;q<Cn;++q){ cO[q]=(unsigned)Cst[q]; cL[q]=(unsigned)Clen[q]; }
                        int* dCi=dupload(std::vector<int>(Coff)); int* dpA=dupload(pdA); int* dpB=dupload(pdB);
                        unsigned *dcO=dupload(cO),*dcL=dupload(cL),*dpO=dupload(pairOff),*dsA=dupload(psA),*dsB=dupload(psB);
                        unsigned *daO=dupload(aO),*daL=dupload(aL),*dbO=dupload(bO),*dbL=dupload(bL);
                        float *dAi,*dHi,*dCi2;
                        CUDA_CHECK(cudaMalloc(&dAi,(Annz?Annz:1)*4)); CUDA_CHECK(cudaMemset(dAi,0,(Annz?Annz:1)*4));
                        CUDA_CHECK(cudaMalloc(&dHi,H.nnz*4)); CUDA_CHECK(cudaMemset(dHi,0,H.nnz*4));
                        CUDA_CHECK(cudaMalloc(&dCi2,Cnnz*4)); CUDA_CHECK(cudaMemset(dCi2,0,Cnnz*4));
                        diaq_product_kernel<float><<<Cn,256>>>(Cn,dCi,dcO,dcL,dpO,dpA,dpB,dsA,dsB,
                            daO,daL,dbO,dbL,dAv2,dAi,dHvals,dHi,dCv,dCi2);
                        cudaFree(dCi);cudaFree(dpA);cudaFree(dpB);cudaFree(dcO);cudaFree(dcL);cudaFree(dpO);
                        cudaFree(dsA);cudaFree(dsB);cudaFree(daO);cudaFree(daL);cudaFree(dbO);cudaFree(dbL);
                        cudaFree(dAi);cudaFree(dHi);cudaFree(dCi2);
                    }
                    CUDA_CHECK(cudaGetLastError());
                    /* accumulate c_k * P_k into U */
                    std::vector<int> c2u(Cn),cst_i(Cn); int maxl=0;
                    for (int q=0;q<Cn;++q){ c2u[q]=(int)Ust[uslot[Coff[q]]]; cst_i[q]=(int)Cst[q]; maxl=std::max(maxl,Clen[q]); }
                    int* dC2U=dupload(c2u); int* dCsti=dupload(cst_i);
                    accum(Cn,dC2U,dCsti,dClen,dCv,w[k],maxl);
                    cudaFree(dC2U); cudaFree(dCsti);
                    if (dAv2!=dHvals) cudaFree(dAv2);
                    dAv2=dCv; Aoff=Coff; Alen=Clen; Ast_=Cst;
                    cudaFree(dCst); cudaFree(dClen);
                }
                if (dAv2!=dHvals) cudaFree(dAv2);
                CUDA_CHECK(cudaDeviceSynchronize());
                cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(out_ms,e0,e1);
                cudaEventDestroy(e0); cudaEventDestroy(e1);
            };
            float ours_ms=0, ham_ms=0;
            build(false,&ours_ms);
            build(true,&ham_ms);
            /* E2EOPBUILDCSV,file,n,K,U_diags,U_stored,ours_ms,hamsim_ms,hamsim_vs_ours */
            printf("E2EOPBUILDCSV,%s,%d,%d,%d,%zu,%.3f,%.3f,%.3f\n",
                   file,n,K,Ud,Unnz,ours_ms,ham_ms, ham_ms>0?ham_ms/ours_ms:-1.0);
            cudaFree(dU); cudaFree(dOnes); cudaFree(dHvals);
            cudaFree(dHoffU); cudaFree(dHstart); cudaFree(dHlen);
        }
    }
    /* L4 sym build LAST — a wide-band crash here cannot eat the rows above */
    if (do_opbuild && do_sym) run_sym_opbuild();
    return 0;
}
