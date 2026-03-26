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
 *   nvcc test_hybrid.cu diag_hybrid_kernel.cu -o test_hybrid -std=c++17
 * ============================================================ */

#include "diag_hybrid_kernel.cuh"

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
 * print_timing_row — fixed-width table row helper
 * ============================================================ */
static void print_timing_row(const char* label,
                              float mean_ms, float min_ms, float max_ms)
{
    printf("  | %-38s | %8.3f | %8.3f | %8.3f |\n",
           label, mean_ms, min_ms, max_ms);
}

static void print_timing_row_cpu(const char* label, float ms)
{
    printf("  | %-38s | %8.3f |    (CPU) |    (CPU) |\n", label, ms);
}

static void print_timing_sep()
{
    printf("  +----------------------------------------+----------+----------+----------+\n");
}

/* ============================================================
 * run_test — full correctness + timing test for one configuration
 * ============================================================ */
static bool run_test(const char* name,
                     DiagMatrix A, DiagMatrix B,
                     int M, int K, int N,
                     float tol = 1e-3f)
{
    printf("\n[%s]  M=%d K=%d N=%d  A_diags=%d  B_diags=%d\n",
           name, M, K, N, A.num_diags, B.num_diags);

    /* ---- 0. Sort A by offset (kernel requirement). ---- */
    sort_diag_matrix_by_offset(A);

    /* ---- 1. CPU reference (single run, wall-clock timed). ---- */
    std::vector<float> C_ref;
    float cpu_ms = cpu_reference(A, B, M, K, N, C_ref);

    /* ---- 2. Build hybrid plan. ---- */
    HybridPlan plan = build_hybrid_plan(A, B, M, K, N);
    printf("  corner_tasks=%-5zu  s1_tasks=%-5zu  s2_tasks=%-5zu  "
           "pairs=%-5zu  partial_buf=%d floats\n",
           plan.corner_tasks.size(), plan.s1_tasks.size(),
           plan.s2_tasks.size(),     plan.pairs.size(),
           plan.partial_buf_size);

    /* ---- 3. Upload read-only data. ---- */
    float* d_A_vals    = upload(A.values);
    int*   d_A_offsets = upload(A.offsets);
    int*   d_A_starts  = upload(A.diag_starts);
    int*   d_A_lengths = upload(A.diag_lengths);
    float* d_B_vals    = upload(B.values);
    int*   d_B_offsets = upload(B.offsets);
    int*   d_B_starts  = upload(B.diag_starts);
    int*   d_B_lengths = upload(B.diag_lengths);

    auto b_lookup = build_b_diag_lookup(B, N);
    int* d_B_lookup = upload(b_lookup);

    int B_off_min = *std::min_element(B.offsets.begin(), B.offsets.end());
    int B_off_max = *std::max_element(B.offsets.begin(), B.offsets.end());

    HybridCornerTask* d_corner = upload(plan.corner_tasks);
    HybridS1Task*     d_s1     = upload(plan.s1_tasks);
    HybridS2Task*     d_s2     = upload(plan.s2_tasks);
    HybridCDiag*      d_cdiags = upload(plan.c_diags);
    HybridPair*       d_pairs  = upload(plan.pairs);

    float* d_C_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C_vals,
               static_cast<size_t>(plan.total_c_values) * sizeof(float)));

    float* d_partial = nullptr;
    if (plan.partial_buf_size > 0)
        CUDA_CHECK(cudaMalloc(&d_partial,
                   static_cast<size_t>(plan.partial_buf_size) * sizeof(float)));

    /* ---- 4. Assemble KernelArgs. ---- */
    HybridKernelArgs kargs = {};
    kargs.corner_tasks = d_corner;
    kargs.n_corner     = static_cast<int>(plan.corner_tasks.size());
    kargs.s1_tasks     = d_s1;
    kargs.n_s1         = static_cast<int>(plan.s1_tasks.size());
    kargs.s2_tasks     = d_s2;
    kargs.n_s2         = static_cast<int>(plan.s2_tasks.size());
    kargs.c_diags      = d_cdiags;
    kargs.A_vals       = d_A_vals;
    kargs.A_offsets    = d_A_offsets;
    kargs.A_starts     = d_A_starts;
    kargs.A_lengths    = d_A_lengths;
    kargs.A_num_diags  = A.num_diags;
    kargs.B_vals       = d_B_vals;
    kargs.B_offsets    = d_B_offsets;
    kargs.B_starts     = d_B_starts;
    kargs.B_lengths    = d_B_lengths;
    kargs.B_lookup     = d_B_lookup;
    kargs.n            = N;
    kargs.B_offset_min = B_off_min;
    kargs.B_offset_max = B_off_max;
    kargs.pairs        = d_pairs;
    kargs.C_vals       = d_C_vals;
    kargs.partial_buf  = d_partial;

    /* ---- 5. SM count (for per-kernel grid replication). ---- */
    int sm_count = 0;
    {
        int dev; CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sm_count,
                                          cudaDevAttrMultiProcessorCount, dev));
    }

    /* ---- 6. Per-kernel timing helpers.
     *
     * We time the three sequential kernels individually so we can see
     * which phase dominates.  Grid sizes mirror launch_hybrid exactly.
     * ---------------------------------------------------------------- */
    int grid_corner = (kargs.n_corner > 0)
                    ? std::min(sm_count * 8, kargs.n_corner) : 0;
    int grid_s1     = (kargs.n_s1 > 0)
                    ? std::min(sm_count * 4, kargs.n_s1) : 0;
    int grid_s2     = (kargs.n_s2 > 0)
                    ? std::min(sm_count * 4, kargs.n_s2) : 0;
    int grid_fused  = (kargs.n_s1 + kargs.n_s2 > 0)
                    ? std::min(sm_count * 4, kargs.n_s1 + kargs.n_s2) : 0;

    /* ctrl array for pipelined kernel (allocated once, re-inited per run). */
    int  n_s2   = kargs.n_s2;
    int* d_ctrl = nullptr;
    if (kargs.n_s1 > 0 || kargs.n_s2 > 0) {
        std::vector<int> h_ctrl(2 + 2 * n_s2, 0);
        for (int i = 0; i < n_s2; ++i)
            h_ctrl[2 + n_s2 + i] = plan.s2_tasks[i].num_partials;
        d_ctrl = upload(h_ctrl);
    }

    /* ---- 7. Individual kernel timings. ---- */

    TimingResult t_corner = {};
    if (grid_corner > 0) {
        t_corner = measure_gpu([&] {
            hybrid_corner_kernel<<<grid_corner, HYBRID_BLOCK_CORNER>>>(kargs);
        });
    }

    TimingResult t_s1 = {};
    if (grid_s1 > 0) {
        t_s1 = measure_gpu([&] {
            hybrid_heavy_s1_kernel<<<grid_s1, HYBRID_BLOCK_HEAVY_S1>>>(kargs);
        });
    }

    TimingResult t_s2 = {};
    if (grid_s2 > 0) {
        t_s2 = measure_gpu([&] {
            hybrid_heavy_s2_reduce_kernel<<<grid_s2, HYBRID_BLOCK_HEAVY_S2>>>(kargs);
        });
    }

    TimingResult t_fused = {};
    if (grid_fused > 0 && d_ctrl) {
        t_fused = measure_gpu([&] {
            init_fused_ctrl(kargs, d_ctrl);
            hybrid_heavy_fused_kernel<<<grid_fused, HYBRID_BLOCK_HEAVY_S1>>>(
                kargs, d_ctrl);
        });
    }

    /* ---- 8. End-to-end launch timings
     *         (includes kernel launch API overhead for all phases).
     * ---------------------------------------------------------------- */
    TimingResult t_seq = measure_gpu([&] {
        launch_hybrid(kargs);
    });

    TimingResult t_pipe = {};
    if (d_ctrl) {
        t_pipe = measure_gpu([&] {
            launch_hybrid_pipelined(kargs, d_ctrl);
        });
    }

    /* ---- 9. Correctness check (one run each, output written once). ---- */
    bool all_ok = true;

    /* Sequential correctness. */
    CUDA_CHECK(cudaMemset(d_C_vals, 0,
               static_cast<size_t>(plan.total_c_values) * sizeof(float)));
    launch_hybrid(kargs);
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        auto C_gpu = download(d_C_vals,
                              static_cast<size_t>(plan.total_c_values));
        float err = compare_result(C_gpu, plan, C_ref, M, N);
        bool ok = (err < tol);
        all_ok &= ok;
        /* Store for printing below. */
        printf(""); /* placeholder: printed in table below */
        (void)err; (void)ok;  /* printed in summary */

        /* Re-use err for pipelined check. */
        float err_seq = err;
        float err_pipe = -1.0f;

        /* Pipelined correctness. */
        if (d_ctrl) {
            CUDA_CHECK(cudaMemset(d_C_vals, 0,
                       static_cast<size_t>(plan.total_c_values) * sizeof(float)));
            launch_hybrid_pipelined(kargs, d_ctrl);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto C_gpu2 = download(d_C_vals,
                                   static_cast<size_t>(plan.total_c_values));
            err_pipe = compare_result(C_gpu2, plan, C_ref, M, N);
            all_ok &= (err_pipe < tol);
        }

        /* ---- 10. Print results. ---- */
        printf("\n");
        printf("  Timing  (warmup=%d, runs=%d):\n", N_WARMUP, N_MEASURE);
        print_timing_sep();
        printf("  | %-38s | %8s | %8s | %8s |\n",
               "Phase", "mean(ms)", "min(ms)", "max(ms)");
        print_timing_sep();

        print_timing_row_cpu("CPU reference (single run)", cpu_ms);
        print_timing_sep();

        if (grid_corner > 0)
            print_timing_row("corner kernel",
                             t_corner.mean_ms, t_corner.min_ms, t_corner.max_ms);
        if (grid_s1 > 0)
            print_timing_row("s1 kernel (heavy partial sums)",
                             t_s1.mean_ms, t_s1.min_ms, t_s1.max_ms);
        if (grid_s2 > 0)
            print_timing_row("s2 kernel (heavy reduction)",
                             t_s2.mean_ms, t_s2.min_ms, t_s2.max_ms);
        print_timing_row("sequential total  (3 launches)",
                         t_seq.mean_ms, t_seq.min_ms, t_seq.max_ms);
        print_timing_sep();

        if (grid_corner > 0)
            print_timing_row("corner kernel (same as above)",
                             t_corner.mean_ms, t_corner.min_ms, t_corner.max_ms);
        if (grid_fused > 0)
            print_timing_row("fused kernel  (s1+s2 pipelined)",
                             t_fused.mean_ms, t_fused.min_ms, t_fused.max_ms);
        if (d_ctrl)
            print_timing_row("pipelined total (2 launches)",
                             t_pipe.mean_ms, t_pipe.min_ms, t_pipe.max_ms);
        else
            printf("  | %-38s | %8s | %8s | %8s |\n",
                   "pipelined total (2 launches)", "skipped", "", "");
        print_timing_sep();

        if (d_ctrl && t_pipe.mean_ms > 0.0f) {
            float speedup = t_seq.mean_ms / t_pipe.mean_ms;
            printf("  Speedup sequential → pipelined : %.2fx  "
                   "(mean: %.3f ms → %.3f ms)\n",
                   speedup, t_seq.mean_ms, t_pipe.mean_ms);
        }

        /* s1 / (s1+s2) ratio: fraction of sequential heavy time in s1. */
        if (grid_s1 > 0 && grid_s2 > 0) {
            float heavy_total = t_s1.mean_ms + t_s2.mean_ms;
            printf("  s1 / (s1+s2) ratio             : %.1f%%  "
                   "(s2 is %.1f%% of heavy work)\n",
                   100.0f * t_s1.mean_ms / heavy_total,
                   100.0f * t_s2.mean_ms / heavy_total);
        }

        /* Correctness. */
        printf("\n  Correctness:\n");
        printf("    %-38s  max_err=%.2e  %s\n",
               "sequential  (launch_hybrid)",
               static_cast<double>(err_seq),
               (err_seq < tol) ? "PASS" : "FAIL");
        if (err_pipe >= 0.0f) {
            printf("    %-38s  max_err=%.2e  %s\n",
                   "pipelined   (launch_hybrid_pipelined)",
                   static_cast<double>(err_pipe),
                   (err_pipe < tol) ? "PASS" : "FAIL");
        } else {
            printf("    %-38s  skipped (no heavy tasks)\n",
                   "pipelined   (launch_hybrid_pipelined)");
        }
    }

    /* ---- 11. Cleanup. ---- */
    cudaFree(d_A_vals);    cudaFree(d_A_offsets);
    cudaFree(d_A_starts);  cudaFree(d_A_lengths);
    cudaFree(d_B_vals);    cudaFree(d_B_offsets);
    cudaFree(d_B_starts);  cudaFree(d_B_lengths);
    cudaFree(d_B_lookup);
    cudaFree(d_corner);    cudaFree(d_s1);
    cudaFree(d_s2);        cudaFree(d_cdiags);
    cudaFree(d_pairs);
    cudaFree(d_C_vals);
    if (d_partial) cudaFree(d_partial);
    if (d_ctrl)    cudaFree(d_ctrl);

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
    printf("Device : %s\n", prop.name);
    printf("SM     : %d.%d  (%d SMs)\n",
           prop.major, prop.minor, prop.multiProcessorCount);
    printf("L2     : %.1f MB\n",
           prop.l2CacheSize / 1048576.0);
    printf("Smem/SM: %.0f KB\n\n",
           prop.sharedMemPerMultiprocessor / 1024.0);

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

    printf("\n============================================\n");
    printf("Result: %s\n",
           all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("============================================\n");
    return all_pass ? 0 : 1;
}
