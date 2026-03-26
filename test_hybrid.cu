/* ============================================================
 * test_hybrid.cu
 *
 * Correctness verification for diag_hybrid_kernel.
 *
 * Strategy: for each test case, compute C = A × B on the CPU
 * using a naive triple-loop reference, then run the GPU kernel
 * (both sequential and pipelined launch paths), download the
 * result, and compare element-by-element with tolerance 1e-3.
 *
 * Test cases:
 *   1. Tiny 8×8, 1 diagonal each  — trivial, easy to trace
 *   2. Small 32×32, 4 diagonals  — corner path only (≤16 pairs)
 *   3. Medium 256×256, 20 diagonals  — mix of corner and heavy
 *   4. Heavy-only 128×128, 20 A diags + 20 B diags (dC=0 gets
 *      20 contributors → heavy path)
 *   5. Edge diagonals: extreme offsets near ±(N-1)
 *
 * Compile:
 *   nvcc test_hybrid.cu diag_hybrid_kernel.cu -o test_hybrid -std=c++17
 * ============================================================ */

#include "diag_hybrid_kernel.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
 * CPU reference: naive O(nnz_A × nnz_B) multiply
 *
 * Fills C_dense (row-major M×N) directly from the diagonal
 * format without any tiling.  Used as ground truth.
 * ============================================================ */
static void cpu_reference(const DiagMatrix& A, const DiagMatrix& B,
                           int M, int /*K*/, int N,
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
                int k   = a_sc + pa;      // column of A = inner index
                int pb  = k - b_sr;
                if (pb < 0 || pb >= b_len) continue;
                int col = b_sc + pb;
                if (row >= M || col >= N) continue;
                float av = A.values[A.diag_starts[ai] + pa];
                float bv = B.values[B.diag_starts[bi] + pb];
                C_dense[static_cast<size_t>(row) * N + col] += av * bv;
            }
        }
    }
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
 *
 * Build a DiagMatrix from a list of offsets.  Values are filled
 * with a deterministic formula so results are reproducible.
 *
 *   val(diag d, pos p) = seed_a * (|d| * 13 + p) % 97 * 0.01 + 1.0
 * ============================================================ */
static DiagMatrix make_diag_matrix(int rows, int cols,
                                   const std::vector<int>& offsets,
                                   float seed)
{
    DiagMatrix M;
    M.rows     = rows;
    M.cols     = cols;
    M.num_diags = static_cast<int>(offsets.size());
    M.offsets  = offsets;
    M.diag_starts.resize(M.num_diags);
    M.diag_lengths.resize(M.num_diags);

    int base = 0;
    for (int i = 0; i < M.num_diags; ++i) {
        int len = DiagMatrix::diag_length(rows, cols, offsets[i]);
        M.diag_starts[i]  = base;
        M.diag_lengths[i] = len;
        int d = offsets[i];
        for (int p = 0; p < len; ++p) {
            float v = 1.0f + seed * static_cast<float>((abs(d) * 13 + p) % 97) * 0.01f;
            M.values.push_back(v);
        }
        base += len;
    }
    return M;
}

/* ============================================================
 * compare_result
 *
 * Scatter the diagonal-format GPU result back into a dense
 * M×N matrix and compare with the CPU reference.
 *
 * Returns true if max absolute error < tol.
 * ============================================================ */
static bool compare_result(const std::vector<float>& C_gpu_diag,
                            const HybridPlan& plan,
                            const std::vector<float>& C_cpu_dense,
                            int M, int N,
                            float tol,
                            const char* label)
{
    /* Scatter diagonal format → dense. */
    std::vector<float> C_gpu_dense(static_cast<size_t>(M) * N, 0.0f);
    for (const auto& cd : plan.c_diags) {
        int c_sr = cd.c_sr;
        int c_sc = (cd.c_offset >= 0) ? cd.c_offset : 0;
        for (int k = 0; k < cd.length; ++k) {
            int row = c_sr + k;
            int col = c_sc + k;
            if (row < M && col < N) {
                C_gpu_dense[static_cast<size_t>(row) * N + col] =
                    C_gpu_diag[cd.values_start + k];
            }
        }
    }

    float max_err = 0.0f;
    int   err_row = -1, err_col = -1;
    float err_gpu = 0.0f, err_cpu = 0.0f;

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float g = C_gpu_dense[static_cast<size_t>(r) * N + c];
            float ref = C_cpu_dense[static_cast<size_t>(r) * N + c];
            float e = fabsf(g - ref);
            if (e > max_err) {
                max_err = e;
                err_row = r; err_col = c;
                err_gpu = g; err_cpu = ref;
            }
        }
    }

    bool ok = (max_err < tol);
    printf("  %-30s  max_err=%.2e  %s\n",
           label, static_cast<double>(max_err),
           ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("    worst: [%d,%d]  gpu=%.6f  ref=%.6f\n",
               err_row, err_col,
               static_cast<double>(err_gpu),
               static_cast<double>(err_cpu));
    }
    return ok;
}

/* ============================================================
 * run_test
 *
 * Full end-to-end test for one (A, B, M, K, N) configuration.
 * Runs:
 *   1. CPU reference
 *   2. GPU sequential  (launch_hybrid)
 *   3. GPU pipelined   (launch_hybrid_pipelined)
 * and compares all against the CPU reference.
 *
 * Returns true if all comparisons pass.
 * ============================================================ */
static bool run_test(const char* name,
                     DiagMatrix A, DiagMatrix B,
                     int M, int K, int N,
                     float tol = 1e-3f)
{
    printf("\n[%s]  M=%d K=%d N=%d  A_diags=%d  B_diags=%d\n",
           name, M, K, N, A.num_diags, B.num_diags);

    /* ---- 0. Sort A by offset (required by the kernel). ---- */
    sort_diag_matrix_by_offset(A);

    /* ---- 1. CPU reference. ---- */
    std::vector<float> C_ref;
    cpu_reference(A, B, M, K, N, C_ref);

    /* ---- 2. Build hybrid plan. ---- */
    HybridPlan plan = build_hybrid_plan(A, B, M, K, N);
    printf("  corner_tasks=%zu  s1_tasks=%zu  s2_tasks=%zu  "
           "pairs=%zu  partial_buf=%d floats\n",
           plan.corner_tasks.size(), plan.s1_tasks.size(),
           plan.s2_tasks.size(),     plan.pairs.size(),
           plan.partial_buf_size);

    /* ---- 3. Upload all read-only arrays. ---- */
    float* d_A_vals    = upload(A.values);
    int*   d_A_offsets = upload(A.offsets);
    int*   d_A_starts  = upload(A.diag_starts);
    int*   d_A_lengths = upload(A.diag_lengths);

    float* d_B_vals    = upload(B.values);
    int*   d_B_offsets = upload(B.offsets);
    int*   d_B_starts  = upload(B.diag_starts);
    int*   d_B_lengths = upload(B.diag_lengths);

    /* Build and upload B_lookup (size 2n-1, indexed by d_b + n-1). */
    auto b_lookup = build_b_diag_lookup(B, N);
    int* d_B_lookup = upload(b_lookup);

    int B_off_min = *std::min_element(B.offsets.begin(), B.offsets.end());
    int B_off_max = *std::max_element(B.offsets.begin(), B.offsets.end());

    /* Plan tables. */
    HybridCornerTask* d_corner = upload(plan.corner_tasks);
    HybridS1Task*     d_s1     = upload(plan.s1_tasks);
    HybridS2Task*     d_s2     = upload(plan.s2_tasks);
    HybridCDiag*      d_cdiags = upload(plan.c_diags);
    HybridPair*       d_pairs  = upload(plan.pairs);

    /* Output buffer. */
    float* d_C_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C_vals,
               static_cast<size_t>(plan.total_c_values) * sizeof(float)));

    /* Partial buffer (may be zero-size for corner-only cases). */
    float* d_partial = nullptr;
    if (plan.partial_buf_size > 0)
        CUDA_CHECK(cudaMalloc(&d_partial,
                   static_cast<size_t>(plan.partial_buf_size) * sizeof(float)));

    /* ---- 4. Assemble HybridKernelArgs. ---- */
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

    bool all_ok = true;

    /* ---- 5a. Sequential launch (launch_hybrid). ---- */
    CUDA_CHECK(cudaMemset(d_C_vals, 0,
               static_cast<size_t>(plan.total_c_values) * sizeof(float)));
    if (d_partial)
        CUDA_CHECK(cudaMemset(d_partial, 0,
                   static_cast<size_t>(plan.partial_buf_size) * sizeof(float)));

    launch_hybrid(kargs);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        auto C_gpu = download(d_C_vals,
                              static_cast<size_t>(plan.total_c_values));
        all_ok &= compare_result(C_gpu, plan, C_ref, M, N, tol,
                                 "sequential (launch_hybrid)");
    }

    /* ---- 5b. Pipelined launch (launch_hybrid_pipelined). ---- */
    if (kargs.n_s1 > 0 || kargs.n_s2 > 0) {
        /*
         * ctrl layout: (2 + 2*n_s2) ints
         *   [s1_next | s2_claim | pending[0..n_s2) | num_partitions[0..n_s2)]
         *
         * We fill num_partitions[] on the host, upload once, then
         * init_fused_ctrl() copies it into pending[] before each launch.
         */
        int n_s2 = kargs.n_s2;
        std::vector<int> h_ctrl(2 + 2 * n_s2, 0);
        for (int i = 0; i < n_s2; ++i)
            h_ctrl[2 + n_s2 + i] = plan.s2_tasks[i].num_partials;
        int* d_ctrl = upload(h_ctrl);

        CUDA_CHECK(cudaMemset(d_C_vals, 0,
                   static_cast<size_t>(plan.total_c_values) * sizeof(float)));
        if (d_partial)
            CUDA_CHECK(cudaMemset(d_partial, 0,
                       static_cast<size_t>(plan.partial_buf_size) * sizeof(float)));

        launch_hybrid_pipelined(kargs, d_ctrl);
        CUDA_CHECK(cudaDeviceSynchronize());

        {
            auto C_gpu = download(d_C_vals,
                                  static_cast<size_t>(plan.total_c_values));
            all_ok &= compare_result(C_gpu, plan, C_ref, M, N, tol,
                                     "pipelined (launch_hybrid_pipelined)");
        }

        CUDA_CHECK(cudaFree(d_ctrl));
    } else {
        printf("  %-30s  skipped (no heavy tasks)\n",
               "pipelined (launch_hybrid_pipelined)");
    }

    /* ---- 6. Cleanup. ---- */
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
    printf("Device: %s  (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    bool all_pass = true;

    /* ----------------------------------------------------------
     * Test 1: Tiny 8×8, single diagonal each.
     * A has only main diagonal (offset 0), same for B.
     * C[0][k] = A[0][k] * B[0][k].
     * Contributor count = 1 → corner path.
     * ---------------------------------------------------------- */
    {
        auto A = make_diag_matrix(8, 8, {0}, 1.0f);
        auto B = make_diag_matrix(8, 8, {0}, 2.0f);
        all_pass &= run_test("tiny_8x8_single_diag", A, B, 8, 8, 8);
    }

    /* ----------------------------------------------------------
     * Test 2: 32×32, 4 A diagonals × 4 B diagonals.
     * Max contributors per dC = 4 → all corner path.
     * Exercises boundary elements and B_lookup for non-zero offsets.
     * ---------------------------------------------------------- */
    {
        auto A = make_diag_matrix(32, 32, {-1, 0, 1, 2}, 1.0f);
        auto B = make_diag_matrix(32, 32, {-2, 0, 1, 3}, 2.0f);
        all_pass &= run_test("small_32x32_corner_only", A, B, 32, 32, 32);
    }

    /* ----------------------------------------------------------
     * Test 3: 256×256, 10 A diagonals × 10 B diagonals.
     * dC = 0 gets up to 10 contributors → still corner (≤16).
     * Exercises longer tiles and multiple tile segments.
     * ---------------------------------------------------------- */
    {
        std::vector<int> offs_a, offs_b;
        for (int d = -4; d <= 5; ++d) offs_a.push_back(d);
        for (int d = -5; d <= 4; ++d) offs_b.push_back(d);
        auto A = make_diag_matrix(256, 256, offs_a, 1.0f);
        auto B = make_diag_matrix(256, 256, offs_b, 2.0f);
        all_pass &= run_test("medium_256x256_10diags", A, B, 256, 256, 256);
    }

    /* ----------------------------------------------------------
     * Test 4: 256×256, 20 A diagonals × 20 B diagonals.
     * dC = 0 gets 20 contributors → heavy path (>16).
     * Tests the full heavy s1 + s2 pipeline.
     * ---------------------------------------------------------- */
    {
        std::vector<int> offs_a, offs_b;
        for (int d = -9; d <= 10; ++d) offs_a.push_back(d);
        for (int d = -10; d <= 9; ++d) offs_b.push_back(d);
        auto A = make_diag_matrix(256, 256, offs_a, 1.0f);
        auto B = make_diag_matrix(256, 256, offs_b, 2.0f);
        all_pass &= run_test("heavy_256x256_20diags", A, B, 256, 256, 256);
    }

    /* ----------------------------------------------------------
     * Test 5: 512×512, 30 A diagonals × 30 B diagonals.
     * dC = 0 gets 30 contributors → definitely heavy.
     * Multiple tile segments per diagonal (512 / 256 = 2).
     * Also exercises the fused pipelined kernel under load.
     * ---------------------------------------------------------- */
    {
        std::vector<int> offs_a, offs_b;
        for (int d = -14; d <= 15; ++d) offs_a.push_back(d);
        for (int d = -15; d <= 14; ++d) offs_b.push_back(d);
        auto A = make_diag_matrix(512, 512, offs_a, 1.5f);
        auto B = make_diag_matrix(512, 512, offs_b, 0.7f);
        all_pass &= run_test("heavy_512x512_30diags", A, B, 512, 512, 512);
    }

    /* ----------------------------------------------------------
     * Test 6: Extreme offsets — diagonals near the matrix corners.
     * A has offset +(N-2) and B has offset -(N-2), so the only
     * output diagonal is dC = 0 with length 2.
     * Verifies that boundary index arithmetic is correct.
     * ---------------------------------------------------------- */
    {
        int sz = 64;
        auto A = make_diag_matrix(sz, sz, {sz - 2, 0, -(sz - 2)}, 1.0f);
        auto B = make_diag_matrix(sz, sz, {-(sz - 2), 0, sz - 2}, 2.0f);
        all_pass &= run_test("extreme_offsets_64x64", A, B, sz, sz, sz);
    }

    /* ----------------------------------------------------------
     * Test 7: Non-square matrices — A is M×K, B is K×N.
     * Verifies that c_sr, a_sr, b_sr are all computed correctly
     * when rows ≠ cols.
     * ---------------------------------------------------------- */
    {
        auto A = make_diag_matrix(128, 64, {-3, 0, 3, 6}, 1.0f);
        auto B = make_diag_matrix(64, 256, {-2, 0, 2, 4}, 2.0f);
        all_pass &= run_test("nonsquare_128x64x256", A, B, 128, 64, 256);
    }

    /* ----------------------------------------------------------
     * Test 8: Single-element diagonals (length 1) at corners.
     * Stress-tests bounds checking in compute_overlap.
     * ---------------------------------------------------------- */
    {
        int sz = 16;
        std::vector<int> offs;
        for (int d = -(sz - 1); d <= (sz - 1); ++d) offs.push_back(d);
        auto A = make_diag_matrix(sz, sz, offs, 1.0f);
        auto B = make_diag_matrix(sz, sz, offs, 1.0f);
        all_pass &= run_test("all_diags_16x16", A, B, sz, sz, sz);
    }

    /* ---------------------------------------------------------- */
    printf("\n============================================\n");
    printf("Result: %s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("============================================\n");

    return all_pass ? 0 : 1;
}
