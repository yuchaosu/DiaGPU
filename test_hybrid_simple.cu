/*
 * test_hybrid_simple.cu — Correctness test for hybrid_kernel_simple.
 *
 * Builds the same HybridPlan and KernelArgs as the production kernel,
 * but launches hybrid_kernel_simple instead. Compares against a CPU
 * reference to verify correctness.
 *
 * Compile:
 *   nvcc test_hybrid_simple.cu diag_hybrid_kernel.cu -o test_hybrid_simple -std=c++17
 *
 * (Links diag_hybrid_kernel.cu for build_hybrid_plan and launch helpers.
 *  The simple kernel is defined in diag_hybrid_simple.cu, included below.)
 */

#include "diag_hybrid_kernel.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

/* ── Include the simple kernel directly (it's not in the .cuh) ── */
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

/* ── Upload helper ── */
template <typename T>
static T* upload(const std::vector<T>& h)
{
    if (h.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

/* ── Download helper ── */
template <typename T>
static std::vector<T> download(const T* d_ptr, size_t count)
{
    std::vector<T> h(count);
    CUDA_CHECK(cudaMemcpy(h.data(), d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

/* ── Build a DiagMatrix with deterministic values ── */
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
        for (int p = 0; p < len; ++p)
            M.values.push_back(
                1.0f + seed * static_cast<float>((abs(offsets[i]) * 13 + p) % 97) * 0.01f);
        base += len;
    }
    return M;
}

/* ── CPU reference: naive triple-loop ── */
static void cpu_reference(const DiagMatrix& A, const DiagMatrix& B,
                           int M, int N,
                           std::vector<float>& C_dense)
{
    C_dense.assign(static_cast<size_t>(M) * N, 0.0f);

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
}

/* ── Scatter GPU diagonal output to dense and compare ── */
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

/* ── Run one test ── */
static bool run_test(const char* name,
                     int M, int K, int N,
                     const std::vector<int>& a_offsets,
                     const std::vector<int>& b_offsets,
                     float tol = 1e-3f)
{
    printf("[%s] M=%d K=%d N=%d  A_diags=%zu  B_diags=%zu\n",
           name, M, K, N, a_offsets.size(), b_offsets.size());

    DiagMatrix A = make_diag_matrix(M, K, a_offsets, 1.0f);
    DiagMatrix B = make_diag_matrix(K, N, b_offsets, 2.0f);
    sort_diag_matrix_by_offset(A);

    /* CPU reference */
    std::vector<float> C_ref;
    cpu_reference(A, B, M, N, C_ref);

    /* Build plan (same as production kernel) */
    HybridPlan plan = build_hybrid_plan(A, B, M, K, N);
    printf("  plan: tasks=%zu  c_diags=%zu  a_contrib=%zu  b_contrib=%zu  smem=%d bytes\n",
           plan.tasks.size(), plan.c_diags.size(),
           plan.a_contrib.size(), plan.b_contrib.size(), plan.max_smem);

    if (plan.tasks.empty()) {
        printf("  SKIP (no tasks)\n\n");
        return true;
    }

    /* Upload */
    float* d_A_vals    = upload(A.values);
    int*   d_A_offsets = upload(A.offsets);
    int*   d_A_starts  = upload(A.diag_starts);
    int*   d_A_lengths = upload(A.diag_lengths);
    float* d_B_vals    = upload(B.values);
    int*   d_B_offsets = upload(B.offsets);
    int*   d_B_starts  = upload(B.diag_starts);
    int*   d_B_lengths = upload(B.diag_lengths);

    HybridTask*  d_tasks     = upload(plan.tasks);
    HybridCDiag* d_cdiags    = upload(plan.c_diags);
    int*         d_acontrib  = upload(plan.a_contrib);
    int*         d_bcontrib  = upload(plan.b_contrib.empty()
                                     ? std::vector<int>{0} : plan.b_contrib);
    PartBMeta*   d_part_b    = upload(plan.part_b_meta);

    float* d_C_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C_vals,
               static_cast<size_t>(plan.total_c_values) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C_vals, 0,
               static_cast<size_t>(plan.total_c_values) * sizeof(float)));

    /* Assemble args */
    HybridKernelArgs kargs = {};
    kargs.tasks       = d_tasks;
    kargs.n_tasks     = static_cast<int>(plan.tasks.size());
    kargs.max_smem    = plan.max_smem;
    kargs.c_diags     = d_cdiags;
    kargs.n_c_diags   = static_cast<int>(plan.c_diags.size());
    kargs.a_contrib   = d_acontrib;
    kargs.b_contrib   = d_bcontrib;
    kargs.part_b_meta = d_part_b;
    kargs.A_vals      = d_A_vals;
    kargs.A_offsets   = d_A_offsets;
    kargs.A_starts    = d_A_starts;
    kargs.A_lengths   = d_A_lengths;
    kargs.A_num_diags = A.num_diags;
    kargs.B_vals      = d_B_vals;
    kargs.B_offsets   = d_B_offsets;
    kargs.B_starts    = d_B_starts;
    kargs.B_lengths   = d_B_lengths;
    kargs.C_vals      = d_C_vals;

    /* Launch simple kernel (one block per task) */
    int grid = kargs.n_tasks;
    int smem = kargs.max_smem;

    /* Need extra smem for the null slot: add (chunk_b) floats.
     * Conservative: add max possible chunk_b ≈ (HYBRID_TILE + 8) * 4 bytes */
    smem += (HYBRID_TILE + 16) * sizeof(float);

    int dev_max_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&dev_max_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    smem = std::min(smem, dev_max_smem);

    CUDA_CHECK(cudaFuncSetAttribute(hybrid_kernel_simple,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    hybrid_kernel_simple<<<grid, HYBRID_BLOCK, smem>>>(kargs);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Download and compare */
    auto C_gpu = download(d_C_vals, static_cast<size_t>(plan.total_c_values));
    float err = compare_result(C_gpu, plan, C_ref, M, N);

    bool pass = err <= tol;
    printf("  max error: %.6e  %s\n\n", err, pass ? "PASS" : "FAIL");

    /* Cleanup */
    cudaFree(d_A_vals);    cudaFree(d_A_offsets);
    cudaFree(d_A_starts);  cudaFree(d_A_lengths);
    cudaFree(d_B_vals);    cudaFree(d_B_offsets);
    cudaFree(d_B_starts);  cudaFree(d_B_lengths);
    cudaFree(d_tasks);     cudaFree(d_cdiags);
    cudaFree(d_acontrib);  cudaFree(d_bcontrib);
    cudaFree(d_part_b);    cudaFree(d_C_vals);

    return pass;
}

/* ── Generate consecutive offsets: [-half, ..., +half] ── */
static std::vector<int> range_offsets(int half)
{
    std::vector<int> v;
    for (int d = -half; d <= half; ++d) v.push_back(d);
    return v;
}

/* ── Main ── */
int main()
{
    int all_pass = 0;

    /* Small: easy to debug */
    all_pass |= !run_test("tiny 8x8, 3 diags",
                           8, 8, 8, {-1, 0, 1}, {-1, 0, 1});

    all_pass |= !run_test("small 16x16, 5 diags",
                           16, 16, 16, {-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2});

    /* Medium: exercises tiling */
    all_pass |= !run_test("medium 256x256, 11 diags",
                           256, 256, 256, range_offsets(5), range_offsets(5));

    /* Larger: exercises partitioning (>53 A diags) */
    all_pass |= !run_test("large 512x512, 41 diags",
                           512, 512, 512, range_offsets(20), range_offsets(20));

    /* Non-square */
    all_pass |= !run_test("rect 256x128x256, 7 diags",
                           256, 128, 256, {-3, -1, 0, 1, 2, 4, 5}, {-2, -1, 0, 1, 3, 4, 6});

    /* Stress: many diags, forces multiple A-partitions */
    all_pass |= !run_test("stress 1024x1024, 101 diags",
                           1024, 1024, 1024, range_offsets(50), range_offsets(50));

    printf("========================================\n");
    printf("Result: %s\n", all_pass == 0 ? "ALL PASS" : "SOME FAILED");
    return all_pass;
}
