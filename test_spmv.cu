/* ============================================================
 * test_spmv.cu
 *
 * Correctness + timing test for dia_spmv (y = A * x).
 *
 * For each test case:
 *   1. CPU reference (naive row loop, std::chrono timed)
 *   2. GPU kernel  (CUDA events, N_WARMUP + N_MEASURE runs)
 *   3. Correctness: max absolute error vs CPU
 *
 * Compile (sm_90 = H100; adjust -arch as needed):
 *   nvcc test_spmv.cu dia_spmv.cu -o test_spmv \
 *        -std=c++17 -O3 -arch=sm_90
 * ============================================================ */

#include "dia_spmv.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

/* ============================================================
 * CUDA error check
 * ============================================================ */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                     \
        if (_e != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error %s:%d  %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(_e));     \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

static constexpr int N_WARMUP  =  5;
static constexpr int N_MEASURE = 20;

/* ============================================================
 * GPU event timer
 * ============================================================ */
struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e)); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void   begin(cudaStream_t st = 0) { CUDA_CHECK(cudaEventRecord(s, st)); }
    float  end  (cudaStream_t st = 0) {
        CUDA_CHECK(cudaEventRecord(e, st));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        return ms;
    }
};

/* ============================================================
 * Build a test DiaSpmvMatrix
 * ============================================================ */
static DiaSpmvMatrix make_dia(int rows, int cols,
                               const std::vector<int>& offsets,
                               unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    DiaSpmvMatrix A;
    A.rows = rows; A.cols = cols;
    A.num_diags = static_cast<int>(offsets.size());
    A.offsets   = offsets;

    int ptr = 0;
    for (int d = 0; d < A.num_diags; ++d) {
        int len = DiaSpmvMatrix::diag_length(rows, cols, offsets[d]);
        A.diag_starts.push_back(ptr);
        A.diag_lengths.push_back(len);
        for (int p = 0; p < len; ++p) A.values.push_back(dist(rng));
        ptr += len;
    }
    return A;
}

/* ============================================================
 * CPU reference SpMV
 * ============================================================ */
static void cpu_spmv(const DiaSpmvMatrix& A,
                     const std::vector<float>& x,
                     std::vector<float>& y)
{
    std::fill(y.begin(), y.end(), 0.f);
    for (int d = 0; d < A.num_diags; ++d) {
        int off = A.offsets[d];
        int sr  = DiaSpmvMatrix::diag_start_row(off);
        int sc  = DiaSpmvMatrix::diag_start_col(off);
        int len = A.diag_lengths[d];
        int as  = A.diag_starts[d];
        for (int p = 0; p < len; ++p)
            y[sr + p] += A.values[as + p] * x[sc + p];
    }
}

/* ============================================================
 * Device buffer helper
 * ============================================================ */
struct DevDia {
    float* vals = nullptr;
    int*   offs = nullptr, *starts = nullptr, *lengths = nullptr;

    void upload(const DiaSpmvMatrix& A) {
        size_t vs = A.values.size()  * sizeof(float);
        size_t ds = A.num_diags      * sizeof(int);
        CUDA_CHECK(cudaMalloc(&vals,    vs)); CUDA_CHECK(cudaMemcpy(vals,    A.values.data(),       vs, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&offs,    ds)); CUDA_CHECK(cudaMemcpy(offs,    A.offsets.data(),      ds, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&starts,  ds)); CUDA_CHECK(cudaMemcpy(starts,  A.diag_starts.data(),  ds, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&lengths, ds)); CUDA_CHECK(cudaMemcpy(lengths, A.diag_lengths.data(), ds, cudaMemcpyHostToDevice));
    }
    void free_all() {
        cudaFree(vals); cudaFree(offs); cudaFree(starts); cudaFree(lengths);
        vals = offs = starts = lengths = nullptr;
    }
};

/* ============================================================
 * Run one test case
 * ============================================================ */
static void run_test(const char* name,
                     int rows, int cols,
                     const std::vector<int>& offsets)
{
    printf("\n=== %s  rows=%d  cols=%d  num_diags=%d ===\n",
           name, rows, cols, (int)offsets.size());

    DiaSpmvMatrix A = make_dia(rows, cols, offsets);

    std::vector<float> h_x(cols), h_y_cpu(rows, 0.f), h_y_gpu(rows, 0.f);
    { std::mt19937 rng(99); std::uniform_real_distribution<float> d(-1.f,1.f);
      for (float& v : h_x) v = d(rng); }

    /* CPU reference */
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_spmv(A, h_x, h_y_cpu);
    double cpu_ms = std::chrono::duration<double,std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();

    /* Host plan */
    SpMVPlan plan = build_spmv_plan(A);
    printf("  Tasks: %d (skipped %d empty tiles)\n",
           (int)plan.tasks.size(),
           (rows + SPMV_TILE - 1) / SPMV_TILE - (int)plan.tasks.size());

    /* Upload */
    DevDia dA;  dA.upload(A);

    float  *d_x = nullptr, *d_y = nullptr;
    SpMVTask* d_tasks = nullptr;
    int*      d_dlist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tasks, plan.tasks.size()    * sizeof(SpMVTask)));
    CUDA_CHECK(cudaMalloc(&d_dlist, plan.diag_list.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_x,     h_x.data(),              cols * sizeof(float),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tasks, plan.tasks.data(),    plan.tasks.size()     * sizeof(SpMVTask), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dlist, plan.diag_list.data(), plan.diag_list.size() * sizeof(int),    cudaMemcpyHostToDevice));

    /* Warmup */
    for (int i = 0; i < N_WARMUP; ++i) {
        CUDA_CHECK(cudaMemset(d_y, 0, rows * sizeof(float)));
        launch_dia_spmv(plan, d_tasks, d_dlist,
                        rows, cols,
                        dA.vals, dA.offs, dA.starts, dA.lengths,
                        d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Measure */
    GpuTimer timer;
    std::vector<float> run_ms(N_MEASURE);
    for (int i = 0; i < N_MEASURE; ++i) {
        CUDA_CHECK(cudaMemset(d_y, 0, rows * sizeof(float)));
        timer.begin();
        launch_dia_spmv(plan, d_tasks, d_dlist,
                        rows, cols,
                        dA.vals, dA.offs, dA.starts, dA.lengths,
                        d_x, d_y);
        run_ms[i] = timer.end();
    }

    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, rows * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* Stats */
    float max_err = 0.f;
    for (int i = 0; i < rows; ++i)
        max_err = std::max(max_err, std::abs(h_y_gpu[i] - h_y_cpu[i]));

    float mean_ms = std::accumulate(run_ms.begin(), run_ms.end(), 0.f) / N_MEASURE;
    float min_ms  = *std::min_element(run_ms.begin(), run_ms.end());

    long long nnz = 0;
    for (int d = 0; d < A.num_diags; ++d) nnz += A.diag_lengths[d];

    double gf_mean = 2.0 * nnz / (mean_ms * 1e-3) / 1e9;
    double gf_peak = 2.0 * nnz / (min_ms  * 1e-3) / 1e9;

    printf("  NNZ       : %lld\n", nnz);
    printf("  CPU       : %.3f ms\n", cpu_ms);
    printf("  GPU mean  : %.3f ms  (%.2f GFlop/s)\n", mean_ms, gf_mean);
    printf("  GPU min   : %.3f ms  (%.2f GFlop/s)\n", min_ms,  gf_peak);
    printf("  Max |err| : %.2e  %s\n", max_err,
           max_err < 1e-3f ? "[PASS]" : "[FAIL]");

    dA.free_all();
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_tasks); cudaFree(d_dlist);
}

/* ============================================================
 * main
 * ============================================================ */
int main()
{
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 8) {
        fprintf(stderr, "WARNING: wmma::precision::tf32 requires sm_80+.\n");
    }

    /* 1. Tridiagonal */
    run_test("Tridiagonal (3 diag)", 65536, 65536, {-1, 0, 1});

    /* 2. Pentadiagonal */
    run_test("Pentadiagonal (5 diag)", 65536, 65536, {-2,-1,0,1,2});

    /* 3. Banded bw=16 (33 diag) */
    { std::vector<int> v; for (int d=-16;d<=16;++d) v.push_back(d);
      run_test("Banded bw=16 (33 diag)", 65536, 65536, v); }

    /* 4. Banded bw=64 (129 diag) */
    { std::vector<int> v; for (int d=-64;d<=64;++d) v.push_back(d);
      run_test("Banded bw=64 (129 diag)", 65536, 65536, v); }

    /* 5. Scattered */
    run_test("Scattered 7 diag", 131072, 131072, {-512,-32,-1,0,1,32,512});

    /* 6. Non-square banded */
    { std::vector<int> v; for (int d=-8;d<=8;++d) v.push_back(d);
      run_test("Rect 256K×128K bw=8", 262144, 131072, v); }

    /* 7. Many diagonals (stress) */
    { std::vector<int> v; for (int d=-250;d<=250;++d) v.push_back(d);
      run_test("Banded bw=250 (501 diag)", 32768, 32768, v); }

    printf("\nDone.\n");
    return 0;
}
