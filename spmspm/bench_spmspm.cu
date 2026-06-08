/* ============================================================
 * bench_spmspm.cu
 *
 * Ablation harness for diagonal-format banded SpGEMM (C = A * B):
 *   (1) sym_spgemm_kernel        -- our smem / no-atomics kernel
 *   (2) hm_structured_..._kernel -- HM baseline (1 thread/A-nz, atomicAdd)
 *   (3) cuSPARSE generic SpGEMM  -- library baseline (CSR)
 *
 * Workload: single product, banded A and B, half-bandwidth w
 * (offsets -w..w). Reproducible (fixed seed).
 *
 * Correctness: our-kernel C is cross-checked against the HM kernel
 * (identical diagonal layout, elementwise) and, for small n, against
 * a CPU dense reference; cuSPARSE is checked vs dense at small n.
 *
 * Timing is kernel/GPU-only via cudaEvents (warmup + averaged iters).
 * NOTE ON FAIRNESS: our kernel and the HM kernel both receive C's
 * structure from host precompute (numeric-only). cuSPARSE computes
 * C's structure itself, so we report cuSPARSE both as full SpGEMM
 * (symbolic+numeric, the real library cost) AND numeric-only
 * (cusparseSpGEMM_compute) so the apples-to-apples number is visible.
 * ============================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>
#include <cusparse.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif

#include "spmspm.cu"        // our kernel + PairEntry
#include "paper_hm_kernel.cu" // HM baseline kernel + paper_hm.cuh

// ---- compile-time kernel config (override with -DCFG_T= etc.) ------------
// NOTE: distinct names from the kernel's template params (N_TILE/THREADS).
#ifndef CFG_T
#define CFG_T 8       // max C diagonals per group (== group size)
#endif
#ifndef CFG_N
#define CFG_N 256     // n-tile width per block
#endif
#ifndef CFG_TH
#define CFG_TH 256    // threads per block
#endif
constexpr int T_DIAGS = CFG_T;
constexpr int N_TILE  = CFG_N;
constexpr int THREADS = CFG_TH;

// ---- error checking ------------------------------------------------------
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e_)); exit(1);} } while(0)

#define CUSPARSE_CHECK(x) do { cusparseStatus_t s_ = (x); \
    if (s_ != CUSPARSE_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSPARSE error %s:%d: %d (%s)\n", __FILE__, __LINE__, \
            (int)s_, cusparseGetErrorString(s_)); exit(1);} } while(0)

static inline int min0(int x) { return x < 0 ? x : 0; }

// ============================================================
// Banded matrix in diagonal format (HM position layout).
// values[] is the concatenation of diagonals (ascending offset);
// within a diagonal, index = HM position = min(row,col).
// ============================================================
struct DiaMatrix {
    int n = 0;
    std::vector<int>    offsets;  // signed, ascending
    std::vector<size_t> starts;   // value offset of each diagonal
    std::vector<int>    lengths;  // n - |offset|
    std::vector<float>  values;   // flat
    size_t              nnz = 0;
};

// Build a banded matrix with offsets -w..w, random values, fixed seed.
static DiaMatrix make_banded(int n, int w, unsigned seed) {
    DiaMatrix M;
    M.n = n;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t off = 0;
    for (int d = -w; d <= w; ++d) {
        int len = n - std::abs(d);
        if (len <= 0) continue;
        M.offsets.push_back(d);
        M.starts.push_back(off);
        M.lengths.push_back(len);
        for (int p = 0; p < len; ++p) M.values.push_back(dist(rng));
        off += len;
    }
    M.nnz = off;
    return M;
}

// All C diagonals present in A*B: offsets {d_a + d_b}, ascending.
static DiaMatrix make_c_structure(const DiaMatrix& A, const DiaMatrix& B) {
    int n = A.n;
    std::vector<char> present(2 * n - 1, 0);
    for (int da : A.offsets)
        for (int db : B.offsets) {
            int dc = da + db;
            if (dc > -(n) && dc < n) present[dc + (n - 1)] = 1;
        }
    DiaMatrix C; C.n = n;
    size_t off = 0;
    for (int d = -(n - 1); d <= (n - 1); ++d) {
        if (!present[d + (n - 1)]) continue;
        int len = n - std::abs(d);
        if (len <= 0) continue;
        C.offsets.push_back(d);
        C.starts.push_back(off);
        C.lengths.push_back(len);
        off += len;
    }
    C.nnz = off;
    C.values.assign(off, 0.0f);
    return C;
}

// ============================================================
// Host scheduler: partition C diagonals into groups and build the
// per-group pair schedule consumed by sym_spgemm_kernel.
// ============================================================
struct Schedule {
    int a_halo = 0, b_halo = 0;
    int num_groups = 0;
    int max_a = 0, max_b = 0, max_pairs = 0;
    int num_ntiles = 0, num_blocks = 0;
    std::vector<int> a_count, b_count, pair_count, t_out;
    std::vector<int> a_globals, b_globals, c_globals; // padded, group-strided
    std::vector<PairEntry> pairs;                      // padded, group-strided
    std::vector<int> block_group, block_tile;
};

static Schedule build_schedule(const DiaMatrix& A, const DiaMatrix& B,
                               const DiaMatrix& C, int n) {
    Schedule S;

    // signed offset -> diagonal index, for A and B lookups
    std::unordered_map<int,int> bIdx;
    for (int i = 0; i < (int)B.offsets.size(); ++i) bIdx[B.offsets[i]] = i;

    const int numC = (int)C.offsets.size();
    S.num_groups = (numC + T_DIAGS - 1) / T_DIAGS;

    // Per-group lists (ragged); flattened later with padding.
    std::vector<std::vector<int>> gA(S.num_groups), gB(S.num_groups),
                                  gC(S.num_groups);
    std::vector<std::vector<PairEntry>> gP(S.num_groups);

    for (int g = 0; g < S.num_groups; ++g) {
        int cs = g * T_DIAGS;
        int ce = std::min(cs + T_DIAGS, numC);
        std::unordered_map<int,int> aLoc, bLoc; // global diag idx -> local

        for (int cl = 0; cl < ce - cs; ++cl) {
            int cidx = cs + cl;
            int dC = C.offsets[cidx];
            gC[g].push_back(cidx);

            for (int ai = 0; ai < (int)A.offsets.size(); ++ai) {
                int dA = A.offsets[ai];
                int dB = dC - dA;
                auto it = bIdx.find(dB);
                if (it == bIdx.end()) continue;
                int bi = it->second;

                int a_local;
                auto ia = aLoc.find(ai);
                if (ia == aLoc.end()) { a_local = (int)gA[g].size();
                    aLoc[ai] = a_local; gA[g].push_back(ai); }
                else a_local = ia->second;

                int b_local;
                auto ib = bLoc.find(bi);
                if (ib == bLoc.end()) { b_local = (int)gB[g].size();
                    bLoc[bi] = b_local; gB[g].push_back(bi); }
                else b_local = ib->second;

                PairEntry pe;
                pe.a_local = a_local;
                pe.b_local = b_local;
                pe.c_local = cl;
                pe.a_shift = min0(dA) - min0(dC);
                pe.b_shift = dA + min0(dB) - min0(dC);
                gP[g].push_back(pe);

                S.a_halo = std::max(S.a_halo, std::abs(pe.a_shift));
                S.b_halo = std::max(S.b_halo, std::abs(pe.b_shift));
            }
        }
        S.max_a     = std::max(S.max_a,     (int)gA[g].size());
        S.max_b     = std::max(S.max_b,     (int)gB[g].size());
        S.max_pairs = std::max(S.max_pairs, (int)gP[g].size());
    }

    // Flatten with padding.
    S.a_count.resize(S.num_groups);
    S.b_count.resize(S.num_groups);
    S.pair_count.resize(S.num_groups);
    S.t_out.resize(S.num_groups);
    S.a_globals.assign((size_t)S.num_groups * S.max_a, 0);
    S.b_globals.assign((size_t)S.num_groups * S.max_b, 0);
    S.c_globals.assign((size_t)S.num_groups * T_DIAGS, 0);
    S.pairs.assign((size_t)S.num_groups * S.max_pairs, PairEntry{0,0,0,0,0});
    for (int g = 0; g < S.num_groups; ++g) {
        S.a_count[g]    = (int)gA[g].size();
        S.b_count[g]    = (int)gB[g].size();
        S.pair_count[g] = (int)gP[g].size();
        S.t_out[g]      = (int)gC[g].size();
        for (int i = 0; i < (int)gA[g].size(); ++i)
            S.a_globals[(size_t)g * S.max_a + i] = gA[g][i];
        for (int i = 0; i < (int)gB[g].size(); ++i)
            S.b_globals[(size_t)g * S.max_b + i] = gB[g][i];
        for (int i = 0; i < (int)gC[g].size(); ++i)
            S.c_globals[(size_t)g * T_DIAGS + i] = gC[g][i];
        for (int i = 0; i < (int)gP[g].size(); ++i)
            S.pairs[(size_t)g * S.max_pairs + i] = gP[g][i];
    }

    // Block mapping: one block per (group, n-tile).
    S.num_ntiles = (n + N_TILE - 1) / N_TILE;
    for (int g = 0; g < S.num_groups; ++g)
        for (int t = 0; t < S.num_ntiles; ++t) {
            S.block_group.push_back(g);
            S.block_tile.push_back(t * N_TILE);
        }
    S.num_blocks = (int)S.block_group.size();
    return S;
}

// ============================================================
// Device buffers for a diagonal matrix.
// ============================================================
struct DevDia {
    float*  values = nullptr;
    size_t* starts = nullptr;
    int*    abs_off = nullptr; // |offset| (for our kernel's diag_len)
};

static DevDia upload_dia(const DiaMatrix& M) {
    DevDia d;
    std::vector<int> absoff(M.offsets.size());
    for (size_t i = 0; i < M.offsets.size(); ++i) absoff[i] = std::abs(M.offsets[i]);
    CUDA_CHECK(cudaMalloc(&d.values, M.values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.starts, M.starts.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d.abs_off, absoff.size() * sizeof(int)));
    if (!M.values.empty())
        CUDA_CHECK(cudaMemcpy(d.values, M.values.data(),
                   M.values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.starts, M.starts.data(),
               M.starts.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.abs_off, absoff.data(),
               absoff.size() * sizeof(int), cudaMemcpyHostToDevice));
    return d;
}

// ============================================================
// CSR build (for cuSPARSE) from a diagonal matrix.
// ============================================================
struct CsrMatrix {
    int n;
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
    std::vector<float> vals;
};

static CsrMatrix dia_to_csr(const DiaMatrix& M) {
    int n = M.n;
    CsrMatrix csr; csr.n = n;
    csr.row_ptr.assign(n + 1, 0);
    // offsets are ascending -> per row, columns come out ascending.
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < (int)M.offsets.size(); ++k) {
            int d = M.offsets[k];
            int col = i + d;
            if (col < 0 || col >= n) continue;
            int pos = (d >= 0) ? i : (i + d);
            csr.col_idx.push_back(col);
            csr.vals.push_back(M.values[M.starts[k] + pos]);
        }
        csr.row_ptr[i + 1] = (int)csr.col_idx.size();
    }
    return csr;
}

// ============================================================
// HMMatrix from DiaMatrix (signed offsets, int starts) -- no dense.
// ============================================================
static HMMatrix dia_to_hm(const DiaMatrix& M) {
    HMMatrix hm;
    hm.n = M.n;
    hm.num_diags = (int)M.offsets.size();
    hm.diag_offsets = M.offsets;
    hm.diag_lengths = M.lengths;
    hm.diag_starts.resize(M.starts.size());
    for (size_t i = 0; i < M.starts.size(); ++i)
        hm.diag_starts[i] = (int)M.starts[i];
    hm.values = M.values;
    hm.total_nz = (int)M.nnz;
    return hm;
}

// ============================================================
// Timing helper
// ============================================================
template <class F>
static float time_ms(F&& launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms / iters;
}

static double max_abs_diff(const std::vector<float>& a,
                           const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

// CPU dense reference (small n only). Returns C diagonal values in the
// same layout as DiaMatrix C, for elementwise comparison.
static std::vector<float> dense_ref_to_cdiag(const DiaMatrix& A,
        const DiaMatrix& B, const DiaMatrix& C) {
    int n = A.n;
    std::vector<float> dense((size_t)n * n, 0.0f);
    // dense = A*B (both small)
    std::vector<float> Ad((size_t)n*n,0), Bd((size_t)n*n,0);
    auto fill = [n](std::vector<float>& M, const DiaMatrix& X){
        for (int k=0;k<(int)X.offsets.size();++k){int d=X.offsets[k];
            for(int p=0;p<X.lengths[k];++p){int r=(d>=0)?p:p-d;int c=(d>=0)?p+d:p;
                M[(size_t)r*n+c]=X.values[X.starts[k]+p];}}};
    fill(Ad, A); fill(Bd, B);
    for (int i=0;i<n;++i) for(int j=0;j<n;++j){float s=0;
        for(int k=0;k<n;++k) s+=Ad[(size_t)i*n+k]*Bd[(size_t)k*n+j];
        dense[(size_t)i*n+j]=s;}
    std::vector<float> out(C.values.size(), 0.0f);
    for (int k=0;k<(int)C.offsets.size();++k){int d=C.offsets[k];
        for(int p=0;p<C.lengths[k];++p){int r=(d>=0)?p:p-d;int c=(d>=0)?p+d:p;
            out[C.starts[k]+p]=dense[(size_t)r*n+c];}}
    return out;
}

// ============================================================
// CPU diagonal SpGEMM (OpenMP). Same diagonal algebra as the GPU
// kernels: O(pairs * n), so it scales to large n (unlike the dense
// O(n^3) reference). Serves as BOTH the correctness reference and a
// CPU performance datapoint. Returns C in DiaMatrix-C layout.
// ============================================================
static std::vector<float> cpu_dia_spgemm(const DiaMatrix& A, const DiaMatrix& B,
                                         const DiaMatrix& C) {
    std::unordered_map<int,int> bIdx;
    for (int i = 0; i < (int)B.offsets.size(); ++i) bIdx[B.offsets[i]] = i;
    std::vector<float> out(C.nnz, 0.0f);
    const int numC = (int)C.offsets.size();
    const int Andiag = (int)A.offsets.size();

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < numC; ++k) {
        int dc = C.offsets[k];
        int minc = dc < 0 ? dc : 0;
        size_t cbase = C.starts[k];
        int lenC = C.lengths[k];
        for (int p = 0; p < lenC; ++p) {
            float acc = 0.0f;
            for (int ai = 0; ai < Andiag; ++ai) {
                int da = A.offsets[ai];
                int db = dc - da;
                auto it = bIdx.find(db);
                if (it == bIdx.end()) continue;
                int bi = it->second;
                int pa = p + (da < 0 ? da : 0) - minc;
                int pb = p + da + (db < 0 ? db : 0) - minc;
                if (pa < 0 || pa >= A.lengths[ai]) continue;
                if (pb < 0 || pb >= B.lengths[bi]) continue;
                acc += A.values[A.starts[ai] + pa] * B.values[B.starts[bi] + pb];
            }
            out[cbase + p] = acc;
        }
    }
    return out;
}

#ifdef USE_MKL
// MKL sparse-sparse SpGEMM (CSR). Returns avg ms over iters; fills C_out in
// DiaMatrix-C layout for verification.
static float run_mkl_spmm(const CsrMatrix& A, const CsrMatrix& B, int n,
                          int iters, const DiaMatrix& Cstruct,
                          std::vector<float>& C_out) {
    std::vector<int> Arp = A.row_ptr, Aci = A.col_idx; std::vector<float> Av = A.vals;
    std::vector<int> Brp = B.row_ptr, Bci = B.col_idx; std::vector<float> Bv = B.vals;
    sparse_matrix_t hA, hB;
    mkl_sparse_s_create_csr(&hA, SPARSE_INDEX_BASE_ZERO, n, n,
        Arp.data(), Arp.data()+1, Aci.data(), Av.data());
    mkl_sparse_s_create_csr(&hB, SPARSE_INDEX_BASE_ZERO, n, n,
        Brp.data(), Brp.data()+1, Bci.data(), Bv.data());

    sparse_matrix_t hC = nullptr;
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, hA, hB, &hC); // warmup
    mkl_sparse_destroy(hC); hC = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, hA, hB, &hC);
        if (i < iters - 1) { mkl_sparse_destroy(hC); hC = nullptr; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count() / iters;

    // export last C and map to diagonal layout
    sparse_index_base_t base; MKL_INT rows, cols, *rs, *re, *ci; float* vv;
    mkl_sparse_s_export_csr(hC, &base, &rows, &cols, &rs, &re, &ci, &vv);
    C_out.assign(Cstruct.nnz, 0.0f);
    std::unordered_map<int,int> coff;
    for (int k = 0; k < (int)Cstruct.offsets.size(); ++k) coff[Cstruct.offsets[k]] = k;
    for (int i = 0; i < n; ++i)
        for (MKL_INT q = rs[i]; q < re[i]; ++q) {
            int col = ci[q]; int d = col - i;
            auto it = coff.find(d); if (it == coff.end()) continue;
            int kk = it->second; int pos = (d >= 0) ? i : (i + d);
            C_out[Cstruct.starts[kk] + pos] = vv[q];
        }
    mkl_sparse_destroy(hC);
    mkl_sparse_destroy(hA); mkl_sparse_destroy(hB);
    return ms;
}
#endif

int main(int argc, char** argv) {
    int n     = (argc > 1) ? atoi(argv[1]) : 100000;
    int w     = (argc > 2) ? atoi(argv[2]) : 16;
    int iters = (argc > 3) ? atoi(argv[3]) : 50;
    const int warmup = 5;

    printf("=== Banded SpGEMM ablation  (n=%d, w=%d, iters=%d) ===\n",
           n, w, iters);

    // ---- build problem ----
    DiaMatrix A = make_banded(n, w, 1234u);
    DiaMatrix B = make_banded(n, w, 5678u);
    DiaMatrix C = make_c_structure(A, B);
    printf("nnz(A)=%zu nnz(B)=%zu  C diags=%zu nnz(C)=%zu\n",
           A.nnz, B.nnz, C.offsets.size(), C.nnz);

    Schedule S = build_schedule(A, B, C, n);
    printf("groups=%d  halos(a,b)=(%d,%d)  max(a,b,pairs)=(%d,%d,%d)  blocks=%d\n",
           S.num_groups, S.a_halo, S.b_halo, S.max_a, S.max_b, S.max_pairs,
           S.num_blocks);

    // ---- upload diagonal matrices ----
    DevDia dA = upload_dia(A), dB = upload_dia(B);
    float*  C_vals; CUDA_CHECK(cudaMalloc(&C_vals, C.nnz * sizeof(float)));
    size_t* C_starts; CUDA_CHECK(cudaMalloc(&C_starts, C.starts.size()*sizeof(size_t)));
    int*    C_absoff; CUDA_CHECK(cudaMalloc(&C_absoff, C.offsets.size()*sizeof(int)));
    {
        std::vector<int> cabs(C.offsets.size());
        for (size_t i=0;i<C.offsets.size();++i) cabs[i]=std::abs(C.offsets[i]);
        CUDA_CHECK(cudaMemcpy(C_starts, C.starts.data(),
                   C.starts.size()*sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_absoff, cabs.data(),
                   cabs.size()*sizeof(int), cudaMemcpyHostToDevice));
    }

    // ---- upload schedule ----
    auto up_i = [](const std::vector<int>& v){ int* p; size_t b=v.size()*sizeof(int);
        CUDA_CHECK(cudaMalloc(&p,b)); CUDA_CHECK(cudaMemcpy(p,v.data(),b,cudaMemcpyHostToDevice)); return p; };
    int* d_acount = up_i(S.a_count);
    int* d_bcount = up_i(S.b_count);
    int* d_pcount = up_i(S.pair_count);
    int* d_tout   = up_i(S.t_out);
    int* d_ag     = up_i(S.a_globals);
    int* d_bg     = up_i(S.b_globals);
    int* d_cg     = up_i(S.c_globals);
    int* d_bgrp   = up_i(S.block_group);
    int* d_btile  = up_i(S.block_tile);
    PairEntry* d_pairs; {
        size_t b = S.pairs.size()*sizeof(PairEntry);
        CUDA_CHECK(cudaMalloc(&d_pairs,b));
        CUDA_CHECK(cudaMemcpy(d_pairs,S.pairs.data(),b,cudaMemcpyHostToDevice));
    }

    // ---- shared memory size + opt-in ----
    int A_tile_w = N_TILE + 2*S.a_halo;
    int B_tile_w = N_TILE + 2*S.b_halo;
    size_t smem = ((size_t)S.max_a*A_tile_w + (size_t)S.max_b*B_tile_w
                   + (size_t)T_DIAGS*N_TILE) * sizeof(float);
    auto kern = sym_spgemm_kernel<T_DIAGS, N_TILE, THREADS>;
    CUDA_CHECK(cudaFuncSetAttribute(kern,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    printf("smem/block = %.1f KB\n", smem/1024.0);

    auto launch_ours = [&](){
        kern<<<S.num_blocks, THREADS, smem>>>(
            dA.values, dA.starts, dA.abs_off,
            dB.values, dB.starts, dB.abs_off,
            C_vals, C_starts, C_absoff,
            n, S.a_halo, S.b_halo,
            d_acount, d_bcount, d_pcount, d_tout,
            d_ag, d_bg, d_pairs, d_cg,
            d_bgrp, d_btile,
            S.max_a, S.max_b, S.max_pairs);
    };
    CUDA_CHECK(cudaMemset(C_vals, 0, C.nnz*sizeof(float)));
    launch_ours();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> C_ours(C.nnz);
    CUDA_CHECK(cudaMemcpy(C_ours.data(), C_vals, C.nnz*sizeof(float),
               cudaMemcpyDeviceToHost));

    float t_ours = time_ms([&](){
        CUDA_CHECK(cudaMemset(C_vals, 0, C.nnz*sizeof(float)));
        launch_ours();
    }, warmup, iters);

    // ================= gather variant (no smem, no atomics) =================
    // Upload signed offsets / lengths for A and C, and a B-offset lookup.
    auto up_signed = [](const std::vector<int>& v){ int* p; size_t b=v.size()*sizeof(int);
        CUDA_CHECK(cudaMalloc(&p,b)); CUDA_CHECK(cudaMemcpy(p,v.data(),b,cudaMemcpyHostToDevice)); return p; };
    int* gA_off = up_signed(A.offsets);
    int* gA_len = up_signed(A.lengths);
    int* gB_len = up_signed(B.lengths);
    int* gC_off = up_signed(C.offsets);
    int* gC_len = up_signed(C.lengths);
    std::vector<int> Blookup(2*(size_t)n - 1, -1);
    for (int i=0;i<(int)B.offsets.size();++i) Blookup[B.offsets[i] + (n-1)] = i;
    int* gB_lk = up_signed(Blookup);

    int A_ndiag = (int)A.offsets.size();
    int C_ndiag = (int)C.offsets.size();
    dim3 gblk(256);
    dim3 ggrid((n + 255)/256, C_ndiag);
    auto launch_gather = [&](){
        gather_spgemm_kernel<<<ggrid, gblk>>>(
            dA.values, dA.starts, gA_off, gA_len, A_ndiag,
            dB.values, dB.starts, gB_len, gB_lk,
            C_vals, C_starts, gC_off, gC_len, C_ndiag, n);
    };
    CUDA_CHECK(cudaMemset(C_vals, 0, C.nnz*sizeof(float)));
    launch_gather();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> C_gather(C.nnz);
    CUDA_CHECK(cudaMemcpy(C_gather.data(), C_vals, C.nnz*sizeof(float),
               cudaMemcpyDeviceToHost));
    float t_gather = time_ms([&](){ launch_gather(); }, warmup, iters);

    // ----- gather + smem-metadata + ILP -----
    constexpr int A_DMAX = 512;          // upper bound on A's diagonal count
    if (A_ndiag > A_DMAX) { fprintf(stderr, "A_ndiag %d > A_DMAX\n", A_ndiag); exit(1); }
    const int ILP = 4;
    dim3 mblk(256);
    dim3 mgrid((n + 256*ILP - 1)/(256*ILP), C_ndiag);
    auto launch_meta = [&](){
        gather_meta_kernel<A_DMAX><<<mgrid, mblk>>>(
            dA.values, dA.starts, gA_off, gA_len, A_ndiag,
            dB.values, dB.starts, gB_len, gB_lk,
            C_vals, C_starts, gC_off, gC_len, C_ndiag, n);
    };
    CUDA_CHECK(cudaMemset(C_vals, 0, C.nnz*sizeof(float)));
    launch_meta();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> C_meta(C.nnz);
    CUDA_CHECK(cudaMemcpy(C_meta.data(), C_vals, C.nnz*sizeof(float),
               cudaMemcpyDeviceToHost));
    float t_meta = time_ms([&](){ launch_meta(); }, warmup, iters);

    // ================= HM baseline =================
    HMMatrix hA = dia_to_hm(A), hB = dia_to_hm(B);
    HMMatrix hC = compute_c_hm_structure(hA, hB, n);
    std::vector<int> cLookup = build_c_diag_lookup(hC, n);

    // upload HM A
    float* hA_v; int *hA_o,*hA_s,*hA_l;
    CUDA_CHECK(cudaMalloc(&hA_v, hA.values.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hA_o, hA.diag_offsets.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hA_s, hA.diag_starts.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hA_l, hA.diag_lengths.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(hA_v,hA.values.data(),hA.values.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hA_o,hA.diag_offsets.data(),hA.diag_offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hA_s,hA.diag_starts.data(),hA.diag_starts.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hA_l,hA.diag_lengths.data(),hA.diag_lengths.size()*sizeof(int),cudaMemcpyHostToDevice));
    // upload HM B
    float* hB_v; int *hB_o,*hB_s,*hB_l;
    CUDA_CHECK(cudaMalloc(&hB_v, hB.values.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hB_o, hB.diag_offsets.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hB_s, hB.diag_starts.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hB_l, hB.diag_lengths.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(hB_v,hB.values.data(),hB.values.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hB_o,hB.diag_offsets.data(),hB.diag_offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hB_s,hB.diag_starts.data(),hB.diag_starts.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hB_l,hB.diag_lengths.data(),hB.diag_lengths.size()*sizeof(int),cudaMemcpyHostToDevice));
    // upload HM C structure
    float* hC_v; int *hC_o,*hC_s,*hC_l,*hC_lk;
    CUDA_CHECK(cudaMalloc(&hC_v, hC.total_nz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hC_o, hC.diag_offsets.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hC_s, hC.diag_starts.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hC_l, hC.diag_lengths.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hC_lk, cLookup.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(hC_o,hC.diag_offsets.data(),hC.diag_offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hC_s,hC.diag_starts.data(),hC.diag_starts.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hC_l,hC.diag_lengths.data(),hC.diag_lengths.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hC_lk,cLookup.data(),cLookup.size()*sizeof(int),cudaMemcpyHostToDevice));

    int nzA = hA.total_nz;
    int hm_blocks = (nzA + 255) / 256;
    auto launch_hm = [&](){
        hm_structured_sparse_matmul_kernel<<<hm_blocks, 256>>>(
            hA_v, hA_o, hA_s, hA_l, hA.num_diags,
            hB_v, hB_o, hB_s, hB_l, hB.num_diags,
            hC_v, hC_o, hC_s, hC_l, hC.num_diags, hC_lk,
            nzA, n);
    };
    CUDA_CHECK(cudaMemset(hC_v, 0, hC.total_nz*sizeof(float)));
    launch_hm();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> C_hm(hC.total_nz);
    CUDA_CHECK(cudaMemcpy(C_hm.data(), hC_v, hC.total_nz*sizeof(float),
               cudaMemcpyDeviceToHost));

    float t_hm = time_ms([&](){
        CUDA_CHECK(cudaMemset(hC_v, 0, hC.total_nz*sizeof(float)));
        launch_hm();
    }, warmup, iters);

    // ================= cuSPARSE baseline =================
    CsrMatrix csrA = dia_to_csr(A), csrB = dia_to_csr(B);
    int *dA_rp,*dA_ci; float* dA_v;
    int *dB_rp,*dB_ci; float* dB_v;
    auto up_csr = [&](const CsrMatrix& m, int** rp, int** ci, float** v){
        CUDA_CHECK(cudaMalloc(rp,(m.n+1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(ci,m.col_idx.size()*sizeof(int)));
        CUDA_CHECK(cudaMalloc(v, m.vals.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(*rp,m.row_ptr.data(),(m.n+1)*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*ci,m.col_idx.data(),m.col_idx.size()*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*v, m.vals.data(), m.vals.size()*sizeof(float),cudaMemcpyHostToDevice));
    };
    up_csr(csrA,&dA_rp,&dA_ci,&dA_v);
    up_csr(csrB,&dB_rp,&dB_ci,&dB_v);
    int64_t nnzA = (int64_t)csrA.vals.size();
    int64_t nnzB = (int64_t)csrB.vals.size();

    cusparseHandle_t handle; CUSPARSE_CHECK(cusparseCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType ctype = CUDA_R_32F;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;

    // Run one full SpGEMM; capture result + per-phase timings. Buffers are
    // re-created each call so the reported time is the true end-to-end cost.
    float t_cusparse_total = 0.f, t_cusparse_numeric = 0.f;
    std::vector<float> C_cusparse;
    int64_t Cr=0, Cc=0, Cnnz=0;

    auto run_cusparse = [&](bool keep, float* total_ms, float* numeric_ms){
        cusparseSpMatDescr_t matA, matB, matC;
        CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, n, nnzA, dA_rp, dA_ci, dA_v,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, ctype));
        CUSPARSE_CHECK(cusparseCreateCsr(&matB, n, n, nnzB, dB_rp, dB_ci, dB_v,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, ctype));
        int* dC_rp; CUDA_CHECK(cudaMalloc(&dC_rp,(n+1)*sizeof(int)));
        CUSPARSE_CHECK(cusparseCreateCsr(&matC, n, n, 0, dC_rp, nullptr, nullptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, ctype));

        cusparseSpGEMMDescr_t spd; CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spd));
        cudaEvent_t e0,e1,e2; cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);

        cudaEventRecord(e0);
        size_t bs1=0; void* buf1=nullptr;
        CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle,op,op,&alpha,matA,matB,
            &beta,matC,ctype,alg,spd,&bs1,nullptr));
        CUDA_CHECK(cudaMalloc(&buf1,bs1));
        CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle,op,op,&alpha,matA,matB,
            &beta,matC,ctype,alg,spd,&bs1,buf1));

        cudaEventRecord(e1); // after symbolic work-estimation
        size_t bs2=0; void* buf2=nullptr;
        CUSPARSE_CHECK(cusparseSpGEMM_compute(handle,op,op,&alpha,matA,matB,
            &beta,matC,ctype,alg,spd,&bs2,nullptr));
        CUDA_CHECK(cudaMalloc(&buf2,bs2));
        CUSPARSE_CHECK(cusparseSpGEMM_compute(handle,op,op,&alpha,matA,matB,
            &beta,matC,ctype,alg,spd,&bs2,buf2));

        int64_t cr,cc,cn; CUSPARSE_CHECK(cusparseSpMatGetSize(matC,&cr,&cc,&cn));
        int* dC_ci; float* dC_v;
        CUDA_CHECK(cudaMalloc(&dC_ci, cn*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dC_v,  cn*sizeof(float)));
        CUSPARSE_CHECK(cusparseCsrSetPointers(matC, dC_rp, dC_ci, dC_v));
        CUSPARSE_CHECK(cusparseSpGEMM_copy(handle,op,op,&alpha,matA,matB,
            &beta,matC,ctype,alg,spd));
        cudaEventRecord(e2);
        CUDA_CHECK(cudaEventSynchronize(e2));

        if (total_ms){ float t; cudaEventElapsedTime(&t,e0,e2); *total_ms=t; }
        if (numeric_ms){ float t; cudaEventElapsedTime(&t,e1,e2); *numeric_ms=t; }

        if (keep) {
            Cr=cr; Cc=cc; Cnnz=cn;
            // copy result to host for verification
            std::vector<int> rp(n+1), ci(cn); std::vector<float> vv(cn);
            CUDA_CHECK(cudaMemcpy(rp.data(),dC_rp,(n+1)*sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ci.data(),dC_ci,cn*sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vv.data(),dC_v, cn*sizeof(float),cudaMemcpyDeviceToHost));
            // map CSR -> C-diagonal layout for comparison
            C_cusparse.assign(C.nnz, 0.0f);
            std::unordered_map<int,int> coff; // signed offset -> C diag index
            for (int k=0;k<(int)C.offsets.size();++k) coff[C.offsets[k]]=k;
            for (int i=0;i<n;++i)
                for (int p=rp[i];p<rp[i+1];++p){
                    int col=ci[p]; int d=col-i;
                    auto it=coff.find(d); if(it==coff.end()) continue;
                    int kk=it->second; int pos=(d>=0)?i:(i+d);
                    C_cusparse[C.starts[kk]+pos]=vv[p];
                }
        }
        cudaFree(dC_ci); cudaFree(dC_v); cudaFree(dC_rp);
        cudaFree(buf1); cudaFree(buf2);
        cusparseSpGEMM_destroyDescr(spd);
        cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
        cudaEventDestroy(e0);cudaEventDestroy(e1);cudaEventDestroy(e2);
    };

    run_cusparse(true, nullptr, nullptr); // warm + keep result
    // average timing
    { float tt=0, tn=0;
      for (int i=0;i<iters;++i){ float a,b; run_cusparse(false,&a,&b); tt+=a; tn+=b; }
      t_cusparse_total = tt/iters; t_cusparse_numeric = tn/iters; }
    printf("cuSPARSE C nnz = %lld\n", (long long)Cnnz);

    // ================= CPU diagonal reference (OpenMP) =================
    int cpu_threads = 1;
#ifdef _OPENMP
    cpu_threads = omp_get_max_threads();
#endif
    const int cpu_iters = 3;
    std::vector<float> C_cpu;
    auto tc0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cpu_iters; ++i) C_cpu = cpu_dia_spgemm(A, B, C);
    auto tc1 = std::chrono::high_resolution_clock::now();
    float t_cpu = std::chrono::duration<float, std::milli>(tc1 - tc0).count() / cpu_iters;

    // ================= MKL sparse SpMM (famous CPU library) =================
    float t_mkl = -1.0f; std::vector<float> C_mkl;
#ifdef USE_MKL
    mkl_set_num_threads(cpu_threads);
    t_mkl = run_mkl_spmm(csrA, csrB, n, std::max(3, iters/5), C, C_mkl);
    printf("MKL threads = %d\n", cpu_threads);
#endif

    // ================= verification =================
    double d_ours_hm = max_abs_diff(C_ours, C_hm);
    printf("\n[verify] max|C_ours - C_hm|      = %.3e\n", d_ours_hm);
    printf("[verify] max|C_gather - C_hm|    = %.3e\n", max_abs_diff(C_gather, C_hm));
    printf("[verify] max|C_meta - C_hm|      = %.3e\n", max_abs_diff(C_meta, C_hm));
    double d_ours_csp = max_abs_diff(C_ours, C_cusparse);
    printf("[verify] max|C_ours - C_cusparse| = %.3e\n", d_ours_csp);
    printf("[verify] max|C_meta - C_cpu|      = %.3e\n", max_abs_diff(C_meta, C_cpu));
#ifdef USE_MKL
    printf("[verify] max|C_meta - C_mkl|      = %.3e\n", max_abs_diff(C_meta, C_mkl));
#endif
    if (n <= 1024) {
        std::vector<float> ref = dense_ref_to_cdiag(A, B, C);
        printf("[verify] max|C_cpu - dense|       = %.3e\n",
               max_abs_diff(C_cpu, ref));
    } else {
        printf("[verify] (dense O(n^3) reference skipped for n>1024; CPU-diag is the scalable ref)\n");
    }

    // ================= report =================
    double flops = 2.0 * (double)S.max_pairs; // not used; compute real below
    // real FMAs ~ sum over pairs of diagonal length; approx pairs*n
    size_t total_pairs = 0; for (int g=0; g<S.num_groups; ++g) total_pairs += S.pair_count[g];
    double gflop = 2.0 * (double)total_pairs * (double)n / 1e9;
    printf("\n=== timings (ms, avg of %d) ===\n", iters);
    printf("  ours-smem (no-atomic)      : %8.4f ms  (%.1f GFLOP/s)\n",
           t_ours, gflop/(t_ours/1e3));
    printf("  ours-gather (no-atomic)    : %8.4f ms  (%.1f GFLOP/s)\n",
           t_gather, gflop/(t_gather/1e3));
    printf("  ours-gather+meta+ILP       : %8.4f ms  (%.1f GFLOP/s)\n",
           t_meta, gflop/(t_meta/1e3));
    printf("  HM baseline (atomicAdd)    : %8.4f ms  (%.1f GFLOP/s)\n",
           t_hm, gflop/(t_hm/1e3));
    printf("  cuSPARSE total (sym+num)   : %8.4f ms\n", t_cusparse_total);
    printf("  cuSPARSE numeric-only      : %8.4f ms\n", t_cusparse_numeric);
    printf("  --- CPU (%d threads) ---\n", cpu_threads);
    printf("  CPU diagonal (OpenMP)      : %8.4f ms  (%.1f GFLOP/s)\n",
           t_cpu, gflop/(t_cpu/1e3));
    if (t_mkl > 0)
        printf("  MKL sparse SpMM            : %8.4f ms  (%.1f GFLOP/s)\n",
               t_mkl, gflop/(t_mkl/1e3));
    printf("\n=== speedups (best-of-ours) ===\n");
    float t_best = std::min(std::min(t_ours, t_gather), t_meta);
    printf("  meta vs gather  : %6.2fx\n", t_gather / t_meta);
    printf("  best vs HM      : %6.2fx\n", t_hm / t_best);
    printf("  best vs cuSP tot: %6.2fx\n", t_cusparse_total / t_best);
    printf("  best vs cuSP num: %6.2fx\n", t_cusparse_numeric / t_best);
    printf("  best vs CPU-diag: %6.2fx\n", t_cpu / t_best);
    if (t_mkl > 0) printf("  best vs MKL     : %6.2fx\n", t_mkl / t_best);
    (void)flops; (void)Cr; (void)Cc;

    cusparseDestroy(handle);
    return 0;
}
