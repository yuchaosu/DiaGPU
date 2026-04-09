/* ============================================================
 * test_hybrid.cu
 *
 * Correctness + timing verification for diag_hybrid_kernel.
 *
 * For each test case:
 *   1. CPU reference (naive triple-loop, std::chrono timed)
 *   2. GPU sequential  — corner + s1 + s2 as 3 separate kernels
 *      Timed individually per kernel and as a total.
 *   3. GPU pipelined   — corner + fused(s1+s2) as 2 kernels
 *      Timed individually per kernel and as a total.
 *   All GPU timings use CUDA events (device-side, sub-microsecond
 *   resolution).  Each kernel is warmed up then measured N_MEASURE
 *   times; mean and minimum are reported.
 *
 * Output per test:
 *   - Plan summary (task counts, buffer sizes)
 *   - Timing table (per-kernel breakdown + totals + CPU)
 *   - Speedup (sequential → pipelined)
 *   - Correctness (max absolute error vs CPU reference)
 *
 * Compile:
 *   nvcc test_hybrid.cu diag_hybrid_kernel.cu paper_hm_kernel.cu -o test_hybrid -std=c++17
 * ============================================================ */

#include "diag_hybrid_kernel.cuh"
#include "paper_hm.cuh"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

/* ============================================================
 * Timing configuration
 * ============================================================ */
static constexpr int N_WARMUP  = 3;   // untimed warm-up runs per kernel
static constexpr int N_MEASURE = 10;  // timed runs (mean and min reported)

/* ============================================================
 * CUDA error checking
 * ============================================================ */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* ============================================================
 * GpuTimer — RAII wrapper around a pair of CUDA events.
 *
 * Usage:
 *   GpuTimer t;
 *   t.start(stream);
 *   kernel<<<...>>>();
 *   float ms = t.stop(stream);   // blocks until kernel finishes
 * ============================================================ */
struct GpuTimer {
    cudaEvent_t ev_start, ev_stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
    }

    void start(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(ev_start, s));
    }

    /* Records stop event, synchronises, returns elapsed ms. */
    float stop(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(ev_stop, s));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        return ms;
    }
};

/* ============================================================
 * TimingResult — statistics from N_MEASURE timed runs.
 * ============================================================ */
struct TimingResult {
    float mean_ms = 0.0f;
    float min_ms  = 0.0f;
    float max_ms  = 0.0f;
};

/* Run fn() N_WARMUP times (untimed), then N_MEASURE times with
 * CUDA events.  fn() must enqueue work on stream 0 and return. */
template <typename Fn>
static TimingResult measure_gpu(Fn&& fn)
{
    /* Warm-up: let the GPU reach steady-state clocks and populate
     * caches with kernel code / constant data. */
    for (int i = 0; i < N_WARMUP; ++i) {
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    GpuTimer t;
    std::vector<float> samples(N_MEASURE);
    for (int i = 0; i < N_MEASURE; ++i) {
        t.start();
        fn();
        samples[i] = t.stop();
    }

    TimingResult r;
    r.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0f)
                / static_cast<float>(N_MEASURE);
    r.min_ms  = *std::min_element(samples.begin(), samples.end());
    r.max_ms  = *std::max_element(samples.begin(), samples.end());
    return r;
}

/* ============================================================
 * CPU reference: naive O(nnz_A × nnz_B) multiply.
 * Returns elapsed wall-clock time in ms.
 * ============================================================ */
static float cpu_reference(const DiagMatrix& A, const DiagMatrix& B,
                            int M, int /*K*/, int N,
                            std::vector<float>& C_dense)
{
    C_dense.assign(static_cast<size_t>(M) * N, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int ai = 0; ai < A.num_diags; ++ai) {
        int d_a   = A.offsets[ai];
        int a_sr  = (d_a >= 0) ? 0 : -d_a;
        int a_sc  = (d_a >= 0) ? d_a : 0;
        int a_len = A.diag_lengths[ai];

        for (int bi = 0; bi < B.num_diags; ++bi) {
            int d_b   = B.offsets[bi];
            int b_sr  = (d_b >= 0) ? 0 : -d_b;
            int b_sc  = (d_b >= 0) ? d_b : 0;
            int b_len = B.diag_lengths[bi];

            for (int pa = 0; pa < a_len; ++pa) {
                int row = a_sr + pa;
                int k   = a_sc + pa;
                int pb  = k - b_sr;
                if (pb < 0 || pb >= b_len) continue;
                int col = b_sc + pb;
                if (row >= M || col >= N) continue;
                C_dense[static_cast<size_t>(row) * N + col] +=
                    A.values[A.diag_starts[ai] + pa] *
                    B.values[B.diag_starts[bi] + pb];
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

/* ============================================================
 * Helpers: upload / download
 * ============================================================ */
template <typename T>
static T* upload(const std::vector<T>& h)
{
    if (h.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
static std::vector<T> download(const T* d_ptr, size_t count)
{
    std::vector<T> h(count);
    CUDA_CHECK(cudaMemcpy(h.data(), d_ptr, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return h;
}

/* ============================================================
 * make_diag_matrix
 * ============================================================ */
static DiagMatrix make_diag_matrix(int rows, int cols,
                                   const std::vector<int>& offsets,
                                   float seed)
{
    DiagMatrix M;
    M.rows      = rows;
    M.cols      = cols;
    M.num_diags = static_cast<int>(offsets.size());
    M.offsets   = offsets;
    M.diag_starts.resize(M.num_diags);
    M.diag_lengths.resize(M.num_diags);

    int base = 0;
    for (int i = 0; i < M.num_diags; ++i) {
        int len = DiagMatrix::diag_length(rows, cols, offsets[i]);
        M.diag_starts[i]  = base;
        M.diag_lengths[i] = len;
        int d = offsets[i];
        for (int p = 0; p < len; ++p)
            M.values.push_back(
                1.0f + seed * static_cast<float>((abs(d) * 13 + p) % 97) * 0.01f);
        base += len;
    }
    return M;
}

/* ============================================================
 * compare_result — scatter diagonal GPU output to dense,
 * compare against CPU reference, return max absolute error.
 * ============================================================ */
static float compare_result(const std::vector<float>& C_gpu_diag,
                             const HybridPlan& plan,
                             const std::vector<float>& C_cpu_dense,
                             int M, int N)
{
    std::vector<float> C_gpu_dense(static_cast<size_t>(M) * N, 0.0f);
    for (const auto& cd : plan.c_diags) {
        int c_sc = (cd.c_offset >= 0) ? cd.c_offset : 0;
        for (int k = 0; k < cd.length; ++k) {
            int row = cd.c_sr + k;
            int col = c_sc + k;
            if (row < M && col < N)
                C_gpu_dense[static_cast<size_t>(row) * N + col] =
                    C_gpu_diag[cd.values_start + k];
        }
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < C_gpu_dense.size(); ++i)
        max_err = std::max(max_err, fabsf(C_gpu_dense[i] - C_cpu_dense[i]));
    return max_err;
}

/* ============================================================
 * Markdown output file — all results go here.
 * ============================================================ */
static FILE* g_out = nullptr;

/* ============================================================
 * Markdown table helpers
 * ============================================================ */
static void md_timing_row(const char* label,
                           float mean_ms, float min_ms, float max_ms)
{
    fprintf(g_out, "| %s | %.3f | %.3f | %.3f |\n",
            label, mean_ms, min_ms, max_ms);
}

static void md_timing_row_cpu(const char* label, float ms)
{
    fprintf(g_out, "| %s | %.3f | — | — |\n", label, ms);
}

static void md_timing_header()
{
    fprintf(g_out, "| Phase | mean (ms) | min (ms) | max (ms) |\n");
    fprintf(g_out, "|:------|----------:|---------:|---------:|\n");
}

/* ============================================================
 * run_test — full correctness + timing test for one configuration
 * ============================================================ */
static bool run_test(const char* name,
                     DiagMatrix A, DiagMatrix B,
                     int M, int K, int N,
                     float tol = 1e-3f,
                     bool skip_cpu_check = false,
                     bool profile       = false) /* wrap with cudaProfilerStart/Stop */
{
    printf("[%s] M=%d K=%d N=%d ...\n", name, M, K, N);  /* terminal progress */
    fprintf(g_out, "\n---\n\n## %s\n\n", name);
    fprintf(g_out, "**Config:** M=%d  K=%d  N=%d  |  A\\_diags=%d  B\\_diags=%d\n\n",
            M, K, N, A.num_diags, B.num_diags);

    /* ---- 0. Sort A by offset (kernel requirement). ---- */
    sort_diag_matrix_by_offset(A);

    /* ---- 1. CPU reference (single run, wall-clock timed). ---- */
    std::vector<float> C_ref;
    float cpu_ms = 0.0f;
    if (!skip_cpu_check)
        cpu_ms = cpu_reference(A, B, M, K, N, C_ref);

    /* ---- 2. Build hybrid plan. ---- */
    HybridPlan plan = build_hybrid_plan(A, B, M, K, N);
    fprintf(g_out, "**Plan:** tasks=%zu  a\\_contrib=%zu  b\\_contrib=%zu  part\\_b\\_meta=%zu  max\\_smem=%d bytes\n\n",
            plan.tasks.size(), plan.a_contrib.size(), plan.b_contrib.size(),
            plan.part_b_meta.size(), plan.max_smem);

    /* ---- 3. Upload data. ---- */
    float* d_A_vals    = upload(A.values);
    int*   d_A_offsets = upload(A.offsets);
    int*   d_A_starts  = upload(A.diag_starts);
    int*   d_A_lengths = upload(A.diag_lengths);
    float* d_B_vals    = upload(B.values);
    int*   d_B_offsets = upload(B.offsets);
    int*   d_B_starts  = upload(B.diag_starts);
    int*   d_B_lengths = upload(B.diag_lengths);

    HybridTask* d_tasks    = upload(plan.tasks);
    HybridCDiag* d_cdiags  = upload(plan.c_diags);
    int* d_acontrib        = upload(plan.a_contrib);
    int* d_bcontrib        = upload(plan.b_contrib.empty()
                                   ? std::vector<int>{0} : plan.b_contrib);
    /* upload() returns nullptr for empty vec; safe here because the kernel
     * never dereferences part_b_meta when no A diagonals contribute
     * (no tasks are emitted, so launch_hybrid is a no-op). */
    PartBMeta* d_part_b_meta = upload(plan.part_b_meta);

    float* d_C_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C_vals,
               static_cast<size_t>(plan.total_c_values) * sizeof(float)));

    /* ---- 4. Assemble KernelArgs. ---- */
    HybridKernelArgs kargs = {};
    kargs.tasks     = d_tasks;
    kargs.n_tasks   = static_cast<int>(plan.tasks.size());
    kargs.max_smem  = plan.max_smem;
    kargs.c_diags   = d_cdiags;
    kargs.n_c_diags = static_cast<int>(plan.c_diags.size());
    kargs.a_contrib = d_acontrib;
    kargs.b_contrib = d_bcontrib;
    kargs.part_b_meta = d_part_b_meta;
    kargs.A_vals    = d_A_vals;
    kargs.A_offsets = d_A_offsets;
    kargs.A_starts  = d_A_starts;
    kargs.A_lengths = d_A_lengths;
    kargs.A_num_diags = A.num_diags;
    kargs.B_vals    = d_B_vals;
    kargs.B_offsets = d_B_offsets;
    kargs.B_starts  = d_B_starts;
    kargs.B_lengths = d_B_lengths;
    kargs.C_vals    = d_C_vals;

    /* ---- 5. Launch and time. ---- */
    if (profile) cudaProfilerStart();
    TimingResult t_total = measure_gpu([&] {
        launch_hybrid(kargs);
    });
    if (profile) cudaProfilerStop();

    /* ---- 8b. paper_hm_kernel baseline (square matrices only). ---- */
    TimingResult t_hm   = {};
    float        hm_err = -1.0f;
    bool         hm_ok  = true;

    /* Device pointers for HM — kept alive until cleanup. */
    float* d_hA_vals = nullptr; int* d_hA_off = nullptr;
    int*   d_hA_st   = nullptr; int* d_hA_len = nullptr;
    float* d_hB_vals = nullptr; int* d_hB_off = nullptr;
    int*   d_hB_st   = nullptr; int* d_hB_len = nullptr;
    float* d_hC_vals = nullptr; int* d_hC_off = nullptr;
    int*   d_hC_st   = nullptr; int* d_hC_len = nullptr;
    int*   d_hC_lkp  = nullptr;
    int    hm_C_nz   = 0;
    if (M == K && K == N) {
        /* Build HMMatrix from DiagMatrix fields (same layout). */
        HMMatrix hm_A, hm_B;
        hm_A.n = N;  hm_A.num_diags = A.num_diags;
        hm_A.diag_offsets = A.offsets;
        hm_A.diag_starts  = A.diag_starts;
        hm_A.diag_lengths = A.diag_lengths;
        hm_A.values       = A.values;
        hm_A.total_nz     = static_cast<int>(A.values.size());

        hm_B.n = N;  hm_B.num_diags = B.num_diags;
        hm_B.diag_offsets = B.offsets;
        hm_B.diag_starts  = B.diag_starts;
        hm_B.diag_lengths = B.diag_lengths;
        hm_B.values       = B.values;
        hm_B.total_nz     = static_cast<int>(B.values.size());

        HMMatrix hm_C     = compute_c_hm_structure(hm_A, hm_B, N);
        auto     c_lookup = build_c_diag_lookup(hm_C, N);
        hm_C_nz           = hm_C.total_nz;

        d_hA_vals = upload(hm_A.values);
        d_hA_off  = upload(hm_A.diag_offsets);
        d_hA_st   = upload(hm_A.diag_starts);
        d_hA_len  = upload(hm_A.diag_lengths);
        d_hB_vals = upload(hm_B.values);
        d_hB_off  = upload(hm_B.diag_offsets);
        d_hB_st   = upload(hm_B.diag_starts);
        d_hB_len  = upload(hm_B.diag_lengths);
        CUDA_CHECK(cudaMalloc(&d_hC_vals,
                              static_cast<size_t>(hm_C_nz) * sizeof(float)));
        d_hC_off  = upload(hm_C.diag_offsets);
        d_hC_st   = upload(hm_C.diag_starts);
        d_hC_len  = upload(hm_C.diag_lengths);
        d_hC_lkp  = upload(c_lookup);

        const int nzA      = hm_A.total_nz;
        const int hm_block = 256;
        const int hm_grid  = (nzA + hm_block - 1) / hm_block;
        const int hm_nA    = hm_A.num_diags;
        const int hm_nB    = hm_B.num_diags;
        const int hm_nC    = hm_C.num_diags;

        /* Capture device pointers by value so the lambda is self-contained. */
        float* _hAv = d_hA_vals; int* _hAo = d_hA_off;
        int*   _hAs = d_hA_st;   int* _hAl = d_hA_len;
        float* _hBv = d_hB_vals; int* _hBo = d_hB_off;
        int*   _hBs = d_hB_st;   int* _hBl = d_hB_len;
        float* _hCv = d_hC_vals; int* _hCo = d_hC_off;
        int*   _hCs = d_hC_st;   int* _hCl = d_hC_len;
        int*   _hCk = d_hC_lkp;

        t_hm = measure_gpu([=] {
            CUDA_CHECK(cudaMemset(_hCv, 0,
                                  static_cast<size_t>(hm_C_nz) * sizeof(float)));
            hm_structured_sparse_matmul_kernel<<<hm_grid, hm_block>>>(
                _hAv, _hAo, _hAs, _hAl, hm_nA,
                _hBv, _hBo, _hBs, _hBl, hm_nB,
                _hCv, _hCo, _hCs, _hCl, hm_nC,
                _hCk, nzA, N);
        });

        /* Correctness: scatter HM C to dense and compare (only when
         * C_ref is populated, i.e. skip_cpu_check is false). */
        if (!skip_cpu_check) {
            CUDA_CHECK(cudaMemset(d_hC_vals, 0,
                                  static_cast<size_t>(hm_C_nz) * sizeof(float)));
            hm_structured_sparse_matmul_kernel<<<hm_grid, hm_block>>>(
                d_hA_vals, d_hA_off, d_hA_st, d_hA_len, hm_nA,
                d_hB_vals, d_hB_off, d_hB_st, d_hB_len, hm_nB,
                d_hC_vals, d_hC_off, d_hC_st, d_hC_len, hm_nC,
                d_hC_lkp, nzA, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            auto hm_C_host = download(d_hC_vals, static_cast<size_t>(hm_C_nz));
            std::vector<float> C_hm_dense(static_cast<size_t>(M) * N, 0.0f);
            for (int ci = 0; ci < hm_nC; ++ci) {
                int d_c = hm_C.diag_offsets[ci];
                int sr  = (d_c >= 0) ? 0 : -d_c;
                int sc  = (d_c >= 0) ? d_c : 0;
                int len = hm_C.diag_lengths[ci];
                int st  = hm_C.diag_starts[ci];
                for (int p = 0; p < len; ++p)
                    C_hm_dense[static_cast<size_t>(sr + p) * N + (sc + p)] =
                        hm_C_host[st + p];
            }
            hm_err = 0.0f;
            for (size_t i = 0; i < C_hm_dense.size(); ++i)
                hm_err = std::max(hm_err,
                                  fabsf(C_hm_dense[i] - C_ref[i]));
            hm_ok = (hm_err < tol);
        }
    }

    /* ---- 9. Correctness check. ---- */
    bool all_ok = true;
    all_ok &= hm_ok;
    float err_hybrid = -1.0f;

    if (!skip_cpu_check) {
        CUDA_CHECK(cudaMemset(d_C_vals, 0,
                   static_cast<size_t>(plan.total_c_values) * sizeof(float)));
        launch_hybrid(kargs);
        CUDA_CHECK(cudaDeviceSynchronize());
        {
            auto C_gpu = download(d_C_vals,
                                  static_cast<size_t>(plan.total_c_values));
            err_hybrid = compare_result(C_gpu, plan, C_ref, M, N);
            all_ok &= (err_hybrid < tol);
        }
    }
    {

        /* ---- 10. Write markdown results. ---- */

        /* ---- Timing table ---- */
        fprintf(g_out, "### Unified kernel  (warmup=%d, runs=%d)\n\n",
                N_WARMUP, N_MEASURE);
        md_timing_header();
        if (!skip_cpu_check)
            md_timing_row_cpu("CPU reference (single run)", cpu_ms);
        else
            fprintf(g_out, "| CPU reference | skipped | — | — |\n");
        md_timing_row("**hybrid unified (compute+reduce)**",
                      t_total.mean_ms, t_total.min_ms, t_total.max_ms);
        fprintf(g_out, "\n");

        /* ---- Baseline table ---- */
        fprintf(g_out, "### Baseline\n\n");
        md_timing_header();
        if (t_hm.mean_ms > 0.0f)
            md_timing_row("paper\\_hm\\_kernel (atomicAdd)",
                          t_hm.mean_ms, t_hm.min_ms, t_hm.max_ms);
        else
            fprintf(g_out, "| paper\\_hm\\_kernel | skipped (non-square) | — | — |\n");
        fprintf(g_out, "\n");

        /* ---- Speedups ---- */
        fprintf(g_out, "### Speedups\n\n");
        fprintf(g_out, "| Comparison | speedup | from (ms) | to (ms) |\n");
        fprintf(g_out, "|:-----------|--------:|----------:|--------:|\n");
        if (t_hm.mean_ms > 0.0f && t_total.mean_ms > 0.0f)
            fprintf(g_out, "| hybrid vs paper\\_hm | %.2fx | %.3f | %.3f |\n",
                    t_hm.mean_ms / t_total.mean_ms,
                    t_hm.mean_ms, t_total.mean_ms);
        fprintf(g_out, "\n");

        /* ---- Correctness ---- */
        fprintf(g_out, "### Correctness\n\n");
        if (skip_cpu_check) {
            fprintf(g_out, "_Skipped for large test._\n\n");
        } else {
            fprintf(g_out, "| Kernel | max\\_err | result |\n");
            fprintf(g_out, "|:-------|----------:|:------|\n");
            if (err_hybrid >= 0.0f)
                fprintf(g_out, "| hybrid unified | %.2e | %s |\n",
                        static_cast<double>(err_hybrid),
                        (err_hybrid < tol) ? "PASS" : "**FAIL**");
            if (hm_err >= 0.0f)
                fprintf(g_out, "| paper\\_hm\\_kernel | %.2e | %s |\n",
                        static_cast<double>(hm_err),
                        hm_ok ? "PASS" : "**FAIL**");
            else
                fprintf(g_out, "| paper\\_hm\\_kernel | — | skipped (non-square) |\n");
            fprintf(g_out, "\n");
        }
        fflush(g_out);
    }

    /* ---- 11. Cleanup. ---- */
    cudaFree(d_A_vals);    cudaFree(d_A_offsets);
    cudaFree(d_A_starts);  cudaFree(d_A_lengths);
    cudaFree(d_B_vals);    cudaFree(d_B_offsets);
    cudaFree(d_B_starts);  cudaFree(d_B_lengths);
    cudaFree(d_tasks);
    cudaFree(d_cdiags);    cudaFree(d_acontrib);  cudaFree(d_bcontrib);
    cudaFree(d_part_b_meta);
    cudaFree(d_C_vals);

    /* HM baseline cleanup. */
    cudaFree(d_hA_vals); cudaFree(d_hA_off);
    cudaFree(d_hA_st);   cudaFree(d_hA_len);
    cudaFree(d_hB_vals); cudaFree(d_hB_off);
    cudaFree(d_hB_st);   cudaFree(d_hB_len);
    cudaFree(d_hC_vals); cudaFree(d_hC_off);
    cudaFree(d_hC_st);   cudaFree(d_hC_len);
    cudaFree(d_hC_lkp);

    return all_ok;
}

/* ============================================================
 * main
 * ============================================================ */
int main()
{
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    /* Open markdown output file. */
    g_out = fopen("results.md", "w");
    if (!g_out) {
        fprintf(stderr, "ERROR: cannot open results.md for writing\n");
        return 1;
    }

    /* Top-level header. */
    fprintf(g_out, "# DiagSpMM Hybrid Kernel Benchmark Results\n\n");
    fprintf(g_out, "| | |\n|:---|:---|\n");
    fprintf(g_out, "| **Device** | %s |\n", prop.name);
    fprintf(g_out, "| **SM** | %d.%d (%d SMs) |\n",
            prop.major, prop.minor, prop.multiProcessorCount);
    fprintf(g_out, "| **L2** | %.1f MB |\n",
            prop.l2CacheSize / 1048576.0);
    fprintf(g_out, "| **Smem/SM** | %.0f KB |\n",
            prop.sharedMemPerMultiprocessor / 1024.0);
    fprintf(g_out, "| **Warmup runs** | %d |\n", N_WARMUP);
    fprintf(g_out, "| **Timed runs** | %d |\n\n", N_MEASURE);

    printf("Device : %s\n", prop.name);
    printf("SM     : %d.%d  (%d SMs)\n",
           prop.major, prop.minor, prop.multiProcessorCount);
    printf("Output : results.md\n\n");

    bool all_pass = true;

    /* Test 1 — corner path only, tiny */
    {
        auto A = make_diag_matrix(8, 8, {0}, 1.0f);
        auto B = make_diag_matrix(8, 8, {0}, 2.0f);
        all_pass &= run_test("tiny_8x8_single_diag", A, B, 8, 8, 8);
    }

    /* Test 2 — corner path only, 4 diagonals each */
    {
        auto A = make_diag_matrix(32, 32, {-1, 0, 1, 2}, 1.0f);
        auto B = make_diag_matrix(32, 32, {-2, 0, 1, 3}, 2.0f);
        all_pass &= run_test("small_32x32_corner_only", A, B, 32, 32, 32);
    }

    /* Test 3 — corner path only, 10 diagonals each, multi-segment */
    {
        std::vector<int> oa, ob;
        for (int d = -4; d <= 5; ++d) oa.push_back(d);
        for (int d = -5; d <= 4; ++d) ob.push_back(d);
        auto A = make_diag_matrix(256, 256, oa, 1.0f);
        auto B = make_diag_matrix(256, 256, ob, 2.0f);
        all_pass &= run_test("medium_256x256_10diags", A, B, 256, 256, 256);
    }

    /* Test 4 — heavy path (20 contributors at dC=0) */
    {
        std::vector<int> oa, ob;
        for (int d = -9;  d <= 10; ++d) oa.push_back(d);
        for (int d = -10; d <= 9;  ++d) ob.push_back(d);
        auto A = make_diag_matrix(256, 256, oa, 1.0f);
        auto B = make_diag_matrix(256, 256, ob, 2.0f);
        all_pass &= run_test("heavy_256x256_20diags", A, B, 256, 256, 256);
    }

    /* Test 5 — heavy path, larger matrix, more segments */
    {
        std::vector<int> oa, ob;
        for (int d = -14; d <= 15; ++d) oa.push_back(d);
        for (int d = -15; d <= 14; ++d) ob.push_back(d);
        auto A = make_diag_matrix(512, 512, oa, 1.5f);
        auto B = make_diag_matrix(512, 512, ob, 0.7f);
        all_pass &= run_test("heavy_512x512_30diags", A, B, 512, 512, 512);
    }

    /* Test 6 — extreme corner offsets, length-2 diagonals */
    {
        int sz = 64;
        auto A = make_diag_matrix(sz, sz, {sz-2, 0, -(sz-2)}, 1.0f);
        auto B = make_diag_matrix(sz, sz, {-(sz-2), 0, sz-2}, 2.0f);
        all_pass &= run_test("extreme_offsets_64x64", A, B, sz, sz, sz);
    }

    /* Test 7 — non-square matrices */
    {
        auto A = make_diag_matrix(128, 64,  {-3, 0, 3, 6}, 1.0f);
        auto B = make_diag_matrix(64,  256, {-2, 0, 2, 4}, 2.0f);
        all_pass &= run_test("nonsquare_128x64x256", A, B, 128, 64, 256);
    }

    /* Test 8 — all diagonals present, many length-1 elements */
    {
        int sz = 16;
        std::vector<int> offs;
        for (int d = -(sz-1); d <= (sz-1); ++d) offs.push_back(d);
        auto A = make_diag_matrix(sz, sz, offs, 1.0f);
        auto B = make_diag_matrix(sz, sz, offs, 1.0f);
        all_pass &= run_test("all_diags_16x16", A, B, sz, sz, sz);
    }

    /* Test 9 — 1024×1024, heavy path, 30 diagonals */
    {
        constexpr int sz = 1024;
        std::vector<int> oa, ob;
        for (int d = -14; d <= 15; ++d) oa.push_back(d);
        for (int d = -15; d <= 14; ++d) ob.push_back(d);
        auto A = make_diag_matrix(sz, sz, oa, 1.0f);
        auto B = make_diag_matrix(sz, sz, ob, 0.5f);
        all_pass &= run_test("heavy_1024x1024_30diags", A, B, sz, sz, sz);
    }

    /* Test 10 — 2048×2048, heavy path, 64 diagonals, skip CPU check */
    {
        constexpr int sz = 2048;
        std::vector<int> oa, ob;
        for (int d = -31; d <= 32; ++d) oa.push_back(d);
        for (int d = -31; d <= 32; ++d) ob.push_back(d);
        auto A = make_diag_matrix(sz, sz, oa, 1.0f);
        auto B = make_diag_matrix(sz, sz, ob, 0.5f);
        all_pass &= run_test("heavy_2048x2048_64diags",
                             A, B, sz, sz, sz, 1e-3f, /*skip_cpu_check=*/true);
    }

    /* Test 11 — 4096×4096, heavy path, 64 diagonals, skip CPU check */
    {
        constexpr int sz = 4096;
        std::vector<int> oa, ob;
        for (int d = -31; d <= 32; ++d) oa.push_back(d);
        for (int d = -31; d <= 32; ++d) ob.push_back(d);
        auto A = make_diag_matrix(sz, sz, oa, 1.0f);
        auto B = make_diag_matrix(sz, sz, ob, 0.5f);
        all_pass &= run_test("heavy_4096x4096_64diags",
                             A, B, sz, sz, sz, 1e-3f, /*skip_cpu_check=*/true);
    }

    /* Test 12 — 8192×8192, heavy path, 64 diagonals, skip CPU check.
     * profile=true: wraps s1 and seq-launch measure_gpu with
     * cudaProfilerStart/Stop so ncu --profile-from-start off captures
     * only these runs. */
    {
        constexpr int sz = 8192;
        std::vector<int> oa, ob;
        for (int d = -31; d <= 32; ++d) oa.push_back(d);
        for (int d = -31; d <= 32; ++d) ob.push_back(d);
        auto A = make_diag_matrix(sz, sz, oa, 1.0f);
        auto B = make_diag_matrix(sz, sz, ob, 0.5f);
        all_pass &= run_test("heavy_8192x8192_64diags",
                             A, B, sz, sz, sz, 1e-3f, /*skip_cpu_check=*/true,
                             /*profile=*/true);
    }

    /* Test 13 — 16384×16384, heavy path, 64 diagonals, skip CPU check */
    {
        constexpr int sz = 16384;
        std::vector<int> oa, ob;
        for (int d = -31; d <= 32; ++d) oa.push_back(d);
        for (int d = -31; d <= 32; ++d) ob.push_back(d);
        auto A = make_diag_matrix(sz, sz, oa, 1.0f);
        auto B = make_diag_matrix(sz, sz, ob, 0.5f);
        all_pass &= run_test("large_16384x16384_64diags",
                             A, B, sz, sz, sz, 1e-3f, /*skip_cpu_check=*/true);
    }

    const char* verdict = all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED";
    fprintf(g_out, "\n---\n\n## Summary\n\n**%s**\n", verdict);
    fclose(g_out);

    printf("\n%s\n", verdict);
    printf("Results written to results.md\n");
    return all_pass ? 0 : 1;
}
