/*
 * test_hybrid_simple.cu — Correctness + timing test for hybrid_kernel_simple.
 *
 * Compares hybrid_kernel_simple against:
 *   1. CPU reference (naive triple-loop)
 *   2. HM kernel (Hossain–Mahmud, atomics-based baseline) — square matrices only
 *
 * Compile:
 *   nvcc test_hybrid_simple.cu diag_hybrid_kernel.cu paper_hm_kernel.cu \
 *        -o test_hybrid_simple -std=c++17
 */

#include "diag_hybrid_kernel.cuh"
#include "paper_hm.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

/* ── Include the simple kernel directly ── */
#include "diag_hybrid_simple.cu"

/* ── CUDA error check ── */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* ── Timing ── */
static constexpr int N_WARMUP  = 3;
static constexpr int N_MEASURE = 10;

struct GpuTimer {
    cudaEvent_t ev0, ev1;
    GpuTimer()  { cudaEventCreate(&ev0); cudaEventCreate(&ev1); }
    ~GpuTimer() { cudaEventDestroy(ev0); cudaEventDestroy(ev1); }
    void start(cudaStream_t s = 0) { cudaEventRecord(ev0, s); }
    float stop(cudaStream_t s = 0) {
        cudaEventRecord(ev1, s);
        cudaEventSynchronize(ev1);
        float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1);
        return ms;
    }
};

struct Timing { float mean_ms, min_ms, max_ms; };

template <typename Fn>
static Timing measure_gpu(Fn&& fn) {
    for (int i = 0; i < N_WARMUP; ++i) { fn(); cudaDeviceSynchronize(); }
    GpuTimer t;
    std::vector<float> s(N_MEASURE);
    for (int i = 0; i < N_MEASURE; ++i) { t.start(); fn(); s[i] = t.stop(); }
    Timing r;
    r.mean_ms = std::accumulate(s.begin(), s.end(), 0.f) / N_MEASURE;
    r.min_ms  = *std::min_element(s.begin(), s.end());
    r.max_ms  = *std::max_element(s.begin(), s.end());
    return r;
}

/* ── Upload / Download ── */
template <typename T>
static T* upload(const std::vector<T>& h) {
    if (h.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

/* ── Build a DiagMatrix with deterministic values ── */
static DiagMatrix make_diag_matrix(int rows, int cols,
                                   const std::vector<int>& offsets, float seed)
{
    DiagMatrix M;
    M.rows = rows; M.cols = cols;
    M.num_diags = static_cast<int>(offsets.size());
    M.offsets = offsets;
    M.diag_starts.resize(M.num_diags);
    M.diag_lengths.resize(M.num_diags);
    int base = 0;
    for (int i = 0; i < M.num_diags; ++i) {
        int len = DiagMatrix::diag_length(rows, cols, offsets[i]);
        M.diag_starts[i] = base;
        M.diag_lengths[i] = len;
        for (int p = 0; p < len; ++p)
            M.values.push_back(
                1.0f + seed * static_cast<float>((abs(offsets[i]) * 13 + p) % 97) * 0.01f);
        base += len;
    }
    return M;
}

/* ── CPU reference ── */
static float cpu_reference(const DiagMatrix& A, const DiagMatrix& B,
                            int M, int N, std::vector<float>& C_dense)
{
    C_dense.assign(static_cast<size_t>(M) * N, 0.0f);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int ai = 0; ai < A.num_diags; ++ai) {
        int d_a = A.offsets[ai], a_sr = std::max(0, -d_a), a_sc = std::max(0, d_a);
        int a_len = A.diag_lengths[ai];
        for (int bi = 0; bi < B.num_diags; ++bi) {
            int d_b = B.offsets[bi], b_sr = std::max(0, -d_b), b_sc = std::max(0, d_b);
            int b_len = B.diag_lengths[bi];
            for (int pa = 0; pa < a_len; ++pa) {
                int row = a_sr + pa, k = a_sc + pa, pb = k - b_sr;
                if (pb < 0 || pb >= b_len) continue;
                int col = b_sc + pb;
                if (row >= M || col >= N) continue;
                C_dense[static_cast<size_t>(row) * N + col] +=
                    A.values[A.diag_starts[ai] + pa] * B.values[B.diag_starts[bi] + pb];
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

/* ── Scatter GPU diagonal output to dense and compare ── */
static float compare_result(const std::vector<float>& C_diag,
                             const HybridPlan& plan,
                             const std::vector<float>& C_ref, int M, int N)
{
    std::vector<float> C_dense(static_cast<size_t>(M) * N, 0.0f);
    for (const auto& cd : plan.c_diags) {
        int c_sc = std::max(0, cd.c_offset);
        for (int k = 0; k < cd.length; ++k) {
            int row = cd.c_sr + k, col = c_sc + k;
            if (row < M && col < N)
                C_dense[static_cast<size_t>(row) * N + col] = C_diag[cd.values_start + k];
        }
    }
    float max_err = 0;
    for (size_t i = 0; i < C_dense.size(); ++i)
        max_err = std::max(max_err, fabsf(C_dense[i] - C_ref[i]));
    return max_err;
}

/* ── Scatter HM C output to dense and compare ── */
static float compare_hm_result(const std::vector<float>& hm_vals,
                                const HMMatrix& hm_C,
                                const std::vector<float>& C_ref, int N)
{
    std::vector<float> C_dense(static_cast<size_t>(N) * N, 0.0f);
    for (int ci = 0; ci < hm_C.num_diags; ++ci) {
        int d  = hm_C.diag_offsets[ci];
        int sr = std::max(0, -d), sc = std::max(0, d);
        int len = hm_C.diag_lengths[ci], st = hm_C.diag_starts[ci];
        for (int p = 0; p < len; ++p)
            C_dense[static_cast<size_t>(sr + p) * N + (sc + p)] = hm_vals[st + p];
    }
    float max_err = 0;
    for (size_t i = 0; i < C_dense.size(); ++i)
        max_err = std::max(max_err, fabsf(C_dense[i] - C_ref[i]));
    return max_err;
}

/* ── Run one test ── */
static bool run_test(const char* name,
                     int M, int K, int N,
                     const std::vector<int>& a_offsets,
                     const std::vector<int>& b_offsets,
                     float tol = 1e-3f)
{
    const bool is_square = (M == K && K == N);

    printf("══════════════════════════════════════════\n");
    printf("[%s] M=%d K=%d N=%d  A_diags=%zu  B_diags=%zu\n",
           name, M, K, N, a_offsets.size(), b_offsets.size());

    DiagMatrix A = make_diag_matrix(M, K, a_offsets, 1.0f);
    DiagMatrix B = make_diag_matrix(K, N, b_offsets, 2.0f);
    sort_diag_matrix_by_offset(A);

    /* ── CPU reference ── */
    std::vector<float> C_ref;
    float cpu_ms = cpu_reference(A, B, M, N, C_ref);
    printf("  CPU:           %.3f ms\n", cpu_ms);

    /* ── Build hybrid plan ── */
    HybridPlan plan = build_hybrid_plan(A, B, M, K, N);
    printf("  plan: tasks=%zu  c_diags=%zu  smem=%d bytes\n",
           plan.tasks.size(), plan.c_diags.size(), plan.max_smem);

    if (plan.tasks.empty()) { printf("  SKIP (no tasks)\n\n"); return true; }

    /* ── Upload A, B ── */
    float* d_Av = upload(A.values);     int* d_Ao = upload(A.offsets);
    int*   d_As = upload(A.diag_starts); int* d_Al = upload(A.diag_lengths);
    float* d_Bv = upload(B.values);     int* d_Bo = upload(B.offsets);
    int*   d_Bs = upload(B.diag_starts); int* d_Bl = upload(B.diag_lengths);

    /* ── Upload plan ── */
    HybridTask*  d_tasks = upload(plan.tasks);
    HybridCDiag* d_cdiags = upload(plan.c_diags);
    int*         d_ac = upload(plan.a_contrib);
    int*         d_bc = upload(plan.b_contrib.empty() ? std::vector<int>{0} : plan.b_contrib);
    PartBMeta*   d_pb = upload(plan.part_b_meta);

    float* d_Cv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Cv, static_cast<size_t>(plan.total_c_values) * sizeof(float)));

    HybridKernelArgs kargs = {};
    kargs.tasks = d_tasks;  kargs.n_tasks = static_cast<int>(plan.tasks.size());
    kargs.max_smem = plan.max_smem;
    kargs.c_diags = d_cdiags; kargs.n_c_diags = static_cast<int>(plan.c_diags.size());
    kargs.a_contrib = d_ac; kargs.b_contrib = d_bc; kargs.part_b_meta = d_pb;
    kargs.A_vals = d_Av; kargs.A_offsets = d_Ao; kargs.A_starts = d_As;
    kargs.A_lengths = d_Al; kargs.A_num_diags = A.num_diags;
    kargs.B_vals = d_Bv; kargs.B_offsets = d_Bo; kargs.B_starts = d_Bs;
    kargs.B_lengths = d_Bl; kargs.C_vals = d_Cv;

    /* ── Launch simple kernel ── */
    int grid = kargs.n_tasks;
    int smem = kargs.max_smem + (HYBRID_TILE + 16) * static_cast<int>(sizeof(float)); /* extra for null slot */
    int dev_max_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&dev_max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    smem = std::min(smem, dev_max_smem);
    CUDA_CHECK(cudaFuncSetAttribute(hybrid_kernel_simple,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    CUDA_CHECK(cudaMemset(d_Cv, 0, static_cast<size_t>(plan.total_c_values) * sizeof(float)));

    Timing t_simple = measure_gpu([&] {
        CUDA_CHECK(cudaMemset(d_Cv, 0, static_cast<size_t>(plan.total_c_values) * sizeof(float)));
        hybrid_kernel_simple<<<grid, HYBRID_BLOCK, smem>>>(kargs);
    });

    /* Correctness */
    CUDA_CHECK(cudaMemset(d_Cv, 0, static_cast<size_t>(plan.total_c_values) * sizeof(float)));
    hybrid_kernel_simple<<<grid, HYBRID_BLOCK, smem>>>(kargs);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto C_simple = download(d_Cv, static_cast<size_t>(plan.total_c_values));
    float simple_err = compare_result(C_simple, plan, C_ref, M, N);

    printf("  hybrid_simple: %.3f ms (mean)  %.3f ms (min)  err=%.2e  %s\n",
           t_simple.mean_ms, t_simple.min_ms, simple_err,
           simple_err <= tol ? "PASS" : "FAIL");

    bool pass = (simple_err <= tol);

    /* ── HM kernel (square only) ── */
    if (is_square) {
        HMMatrix hm_A, hm_B;
        hm_A.n = N; hm_A.num_diags = A.num_diags;
        hm_A.diag_offsets = A.offsets; hm_A.diag_starts = A.diag_starts;
        hm_A.diag_lengths = A.diag_lengths; hm_A.values = A.values;
        hm_A.total_nz = static_cast<int>(A.values.size());

        hm_B.n = N; hm_B.num_diags = B.num_diags;
        hm_B.diag_offsets = B.offsets; hm_B.diag_starts = B.diag_starts;
        hm_B.diag_lengths = B.diag_lengths; hm_B.values = B.values;
        hm_B.total_nz = static_cast<int>(B.values.size());

        HMMatrix hm_C = compute_c_hm_structure(hm_A, hm_B, N);
        auto c_lookup = build_c_diag_lookup(hm_C, N);

        float* d_hAv = upload(hm_A.values);  int* d_hAo = upload(hm_A.diag_offsets);
        int*   d_hAs = upload(hm_A.diag_starts); int* d_hAl = upload(hm_A.diag_lengths);
        float* d_hBv = upload(hm_B.values);  int* d_hBo = upload(hm_B.diag_offsets);
        int*   d_hBs = upload(hm_B.diag_starts); int* d_hBl = upload(hm_B.diag_lengths);
        float* d_hCv = nullptr;
        CUDA_CHECK(cudaMalloc(&d_hCv, static_cast<size_t>(hm_C.total_nz) * sizeof(float)));
        int* d_hCo = upload(hm_C.diag_offsets); int* d_hCs = upload(hm_C.diag_starts);
        int* d_hCl = upload(hm_C.diag_lengths); int* d_hCk = upload(c_lookup);

        const int nzA = hm_A.total_nz;
        const int hm_block = 256;
        const int hm_grid = (nzA + hm_block - 1) / hm_block;
        const int hm_nA = hm_A.num_diags, hm_nB = hm_B.num_diags, hm_nC = hm_C.num_diags;
        const int hm_cnz = hm_C.total_nz;

        Timing t_hm = measure_gpu([=] {
            CUDA_CHECK(cudaMemset(d_hCv, 0, static_cast<size_t>(hm_cnz) * sizeof(float)));
            hm_structured_sparse_matmul_kernel<<<hm_grid, hm_block>>>(
                d_hAv, d_hAo, d_hAs, d_hAl, hm_nA,
                d_hBv, d_hBo, d_hBs, d_hBl, hm_nB,
                d_hCv, d_hCo, d_hCs, d_hCl, hm_nC,
                d_hCk, nzA, N);
        });

        /* HM correctness */
        CUDA_CHECK(cudaMemset(d_hCv, 0, static_cast<size_t>(hm_cnz) * sizeof(float)));
        hm_structured_sparse_matmul_kernel<<<hm_grid, hm_block>>>(
            d_hAv, d_hAo, d_hAs, d_hAl, hm_nA,
            d_hBv, d_hBo, d_hBs, d_hBl, hm_nB,
            d_hCv, d_hCo, d_hCs, d_hCl, hm_nC,
            d_hCk, nzA, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto hm_out = download(d_hCv, static_cast<size_t>(hm_cnz));
        float hm_err = compare_hm_result(hm_out, hm_C, C_ref, N);

        printf("  HM kernel:     %.3f ms (mean)  %.3f ms (min)  err=%.2e  %s\n",
               t_hm.mean_ms, t_hm.min_ms, hm_err,
               hm_err <= tol ? "PASS" : "FAIL");

        /* Speedup */
        float speedup = t_hm.min_ms / t_simple.min_ms;
        printf("  speedup vs HM: %.2fx (%s)\n",
               speedup, speedup > 1.0f ? "simple wins" : "HM wins");

        pass = pass && (hm_err <= tol);

        cudaFree(d_hAv); cudaFree(d_hAo); cudaFree(d_hAs); cudaFree(d_hAl);
        cudaFree(d_hBv); cudaFree(d_hBo); cudaFree(d_hBs); cudaFree(d_hBl);
        cudaFree(d_hCv); cudaFree(d_hCo); cudaFree(d_hCs); cudaFree(d_hCl);
        cudaFree(d_hCk);
    } else {
        printf("  HM kernel:     skipped (non-square)\n");
    }

    printf("  result: %s\n\n", pass ? "PASS" : "FAIL");

    /* Cleanup */
    cudaFree(d_Av); cudaFree(d_Ao); cudaFree(d_As); cudaFree(d_Al);
    cudaFree(d_Bv); cudaFree(d_Bo); cudaFree(d_Bs); cudaFree(d_Bl);
    cudaFree(d_tasks); cudaFree(d_cdiags);
    cudaFree(d_ac); cudaFree(d_bc); cudaFree(d_pb); cudaFree(d_Cv);

    return pass;
}

/* ── Generate consecutive offsets ── */
static std::vector<int> range_offsets(int half) {
    std::vector<int> v;
    for (int d = -half; d <= half; ++d) v.push_back(d);
    return v;
}

/* ── Main ── */
int main()
{
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  hybrid_kernel_simple vs HM comparison   ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    int fails = 0;

    fails += !run_test("tiny 8x8, 3 diags",
                        8, 8, 8, {-1, 0, 1}, {-1, 0, 1});

    fails += !run_test("small 16x16, 5 diags",
                        16, 16, 16, {-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2});

    fails += !run_test("medium 256x256, 11 diags",
                        256, 256, 256, range_offsets(5), range_offsets(5));

    fails += !run_test("large 512x512, 41 diags",
                        512, 512, 512, range_offsets(20), range_offsets(20));

    fails += !run_test("rect 256x128x256, 7 diags",
                        256, 128, 256,
                        {-3, -1, 0, 1, 2, 4, 5}, {-2, -1, 0, 1, 3, 4, 6});

    fails += !run_test("stress 1024x1024, 101 diags",
                        1024, 1024, 1024, range_offsets(50), range_offsets(50));

    printf("════════════════════════════════════════════\n");
    printf("TOTAL: %d failed  —  %s\n", fails, fails == 0 ? "ALL PASS" : "SOME FAILED");
    return fails;
}
