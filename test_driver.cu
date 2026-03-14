/* ============================================================
 * test_driver.cu
 *
 * End-to-end test for the DiagSpMM framework.
 *
 * Uses a small 4×4 example to verify correctness:
 *
 *   A (4x4) diagonals:
 *     d=-1 : [1, 2, 3]
 *     d= 0 : [4, 5, 6, 7]
 *     d= 1 : [8, 9, 10]
 *
 *   B (4x4) diagonals:
 *     d= 0 : [1, 1, 1, 1]
 *     d= 1 : [2, 2, 2]
 *
 *   A as dense:       B as dense:
 *     4  8  0  0        1  2  0  0
 *     1  5  9  0        0  1  2  0
 *     0  2  6 10        0  0  1  2
 *     0  0  3  7        0  0  0  1
 *
 *   C = A × B (dense):
 *     4  16  16   0
 *     1   7  19  18
 *     0   2  10  22
 *     0   0   3  13
 *
 *   C diagonals (expected):
 *     d=-1 : [1, 2, 3]
 *     d= 0 : [4, 7, 10, 13]
 *     d= 1 : [16, 19, 22]
 *     d= 2 : [16, 18]
 *
 * Contributor structure for this example:
 *
 *   d_c = -1:  { (d_a=-1, d_b=0) }
 *     1 group (a=-1), 1 pair
 *
 *   d_c =  0:  { (d_a=-1, d_b=1), (d_a=0, d_b=0) }
 *     2 groups: group(a=-1) with 1 pair, group(a=0) with 1 pair
 *     smemA loaded twice (once per group), each reused by 1 pair
 *
 *   d_c =  1:  { (d_a=0, d_b=1), (d_a=1, d_b=0) }
 *     2 groups: group(a=0) with 1 pair, group(a=1) with 1 pair
 *
 *   d_c =  2:  { (d_a=1, d_b=1) }
 *     1 group (a=1), 1 pair
 *
 * packedB example for d_c=0, group(a=-1), pair(d_b=1):
 *   tile covers positions [0..3] of output diagonal 0
 *   b_map_offset = c_sr + d_a - b_sr = 0 + (-1) - 0 = -1
 *   q=0: p_b = -1+0 = -1 → invalid → 0.0
 *   q=1: p_b = -1+1 =  0 → B.values[4+0] = 2.0
 *   q=2: p_b = -1+2 =  1 → B.values[4+1] = 2.0
 *   q=3: p_b = -1+3 =  2 → B.values[4+2] = 2.0
 *   → packedB for this pair = [0.0, 2.0, 2.0, 2.0] (padded to 32)
 *
 * Compile:
 *   nvcc test_driver.cu diag_kernel.cu -o test_diag -std=c++17
 * ============================================================ */

#include "diag_types.cuh"
#include "diag_host_preprocess.cuh"
#include "diag_kernel.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <nvtx3/nvToolsExt.h>

/* ---- NVTX colour palette for easier timeline reading ---- */
static const uint32_t NVTX_PREPROCESS = 0xFF4287f5; /* blue   */
static const uint32_t NVTX_UPLOAD     = 0xFFf5a142; /* orange */
static const uint32_t NVTX_KERNEL     = 0xFF42f554; /* green  */
static const uint32_t NVTX_DOWNLOAD   = 0xFFf54242; /* red    */

static inline void nvtx_push(const char* name, uint32_t colour) {
    nvtxEventAttributes_t attr = {};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = colour;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}
#define NVTX_POP() nvtxRangePop()

/* ============================================================
 * CUDA error checking macro
 * ============================================================ */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/* ============================================================
 * CPU reference: dense diagonal multiplication
 *
 * Directly computes C = A × B using the diagonal structure
 * without any tiling or packing—purely for correctness check.
 * ============================================================ */
static void
cpu_diag_multiply(const DiagMatrix& A, const DiagMatrix& B,
                  int M, int K, int N,
                  std::vector<float>& C_dense)
{
    C_dense.assign(M * N, 0.0f);

    for (int ai = 0; ai < A.num_diags; ++ai) {
        int d_a  = A.offsets[ai];
        int a_sr = DiagMatrix::diag_start_row(d_a);
        int a_sc = DiagMatrix::diag_start_col(d_a);
        int a_len = A.diag_lengths[ai];

        for (int bi = 0; bi < B.num_diags; ++bi) {
            int d_b  = B.offsets[bi];
            int b_sr = DiagMatrix::diag_start_row(d_b);
            int b_sc = DiagMatrix::diag_start_col(d_b);
            int b_len = B.diag_lengths[bi];

            for (int pa = 0; pa < a_len; ++pa) {
                int row = a_sr + pa;
                int k   = a_sc + pa;         // column of A = row of B
                int pb  = k - b_sr;
                if (pb < 0 || pb >= b_len) continue;
                int col = b_sc + pb;          // column of B = column of C
                float a_val = A.values[A.diag_starts[ai] + pa];
                float b_val = B.values[B.diag_starts[bi] + pb];
                C_dense[row * N + col] += a_val * b_val;
            }
        }
    }
}

/* ============================================================
 * Helper: allocate on device and copy from host vector
 * ============================================================ */
template <typename T>
static T* upload(const std::vector<T>& h_data)
{
    if (h_data.empty()) return nullptr;
    T* d_ptr = nullptr;
    size_t bytes = h_data.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, h_data.data(), bytes,
                          cudaMemcpyHostToDevice));
    return d_ptr;
}

/* ============================================================
 * Helper: print preprocessing summary
 * ============================================================ */
static void
print_preprocess_summary(const PreprocessResult& pr)
{
    printf("=== Preprocessing Summary ===\n");
    printf("  Output diagonals : %zu\n", pr.output_diags.size());
    printf("  Total C values   : %d\n",  pr.total_c_values);
    printf("  Tasks            : %zu\n", pr.tasks.size());
    printf("    LIGHT          : %zu\n", pr.light_task_ids.size());
    printf("    MEDIUM         : %zu\n", pr.medium_task_ids.size());
    printf("    HEAVY          : %zu\n", pr.heavy_task_ids.size());
    printf("    WIDE           : %zu\n", pr.wide_task_ids.size());
    printf("  Groups           : %zu\n", pr.groups.size());
    printf("  Pairs            : %zu\n", pr.pairs.size());
    printf("  PackedB floats   : %zu\n", pr.packedB.size());
    printf("\n");

    for (size_t i = 0; i < pr.tasks.size(); ++i) {
        const Task& t = pr.tasks[i];
        const char* bkt = (t.bucket == 0) ? "LIGHT" :
                          (t.bucket == 1) ? "MEDIUM" :
                          (t.bucket == 2) ? "HEAVY" : "WIDE";
        printf("  Task %zu: c_diag_offset=%d  p=[%d..%d)  "
               "groups=%d  work=%d  bucket=%s\n",
               i, t.c_offset, t.p_begin, t.p_begin + t.p_len,
               t.group_count, t.work_est, bkt);

        for (int gi = 0; gi < t.group_count; ++gi) {
            const Group& g = pr.groups[t.group_begin + gi];
            printf("    Group %d: a_offset=%d  a_map_offset=%d  "
                   "a_diag_len=%d  pairs=%d\n",
                   gi, g.a_offset, g.a_map_offset,
                   g.a_diag_len, g.pair_count);

            for (int pi = 0; pi < g.pair_count; ++pi) {
                const PairMeta& p = pr.pairs[g.pair_begin + pi];
                printf("      Pair %d: b_offset=%d  valid=[%d..%d)  "
                       "packedB_offset=%d\n",
                       pi, p.b_offset,
                       p.out_valid_begin, p.out_valid_end,
                       p.packedB_offset);
            }
        }
    }
    printf("\n");
}

/* ============================================================
 * MAIN
 * ============================================================ */
int main()
{
    printf("DiagSpMM Framework Test (with per-bucket dispatch)\n");
    printf("===================================================\n\n");

    /* ---- Build input matrices ---- */
    const int M = 4, K = 4, N = 4;

    DiagMatrix A;
    A.rows = M;  A.cols = K;  A.num_diags = 3;
    A.offsets     = { -1,  0,  1 };
    A.values      = {  1,  2,  3,          // d=-1, length 3
                       4,  5,  6,  7,      // d= 0, length 4
                       8,  9, 10 };        // d= 1, length 3
    A.diag_starts  = { 0, 3, 7 };
    A.diag_lengths = { 3, 4, 3 };

    DiagMatrix B;
    B.rows = K;  B.cols = N;  B.num_diags = 2;
    B.offsets     = {  0,  1 };
    B.values      = {  1,  1,  1,  1,      // d= 0, length 4
                       2,  2,  2 };        // d= 1, length 3
    B.diag_starts  = { 0, 4 };
    B.diag_lengths = { 4, 3 };

    /* ---- CPU reference ---- */
    std::vector<float> C_dense;
    cpu_diag_multiply(A, B, M, K, N, C_dense);

    printf("CPU reference C (dense):\n");
    for (int i = 0; i < M; ++i) {
        printf("  ");
        for (int j = 0; j < N; ++j)
            printf("%6.1f", C_dense[i * N + j]);
        printf("\n");
    }
    printf("\n");

    /* ---- Host preprocessing ---- */
    nvtx_push("host_preprocess", NVTX_PREPROCESS);
    PreprocessResult pr = build_all(A, B, M, K, N);
    NVTX_POP();
    print_preprocess_summary(pr);

    /* ---- Upload to device ---- */
    nvtx_push("device_upload", NVTX_UPLOAD);

    /* A values */
    float* d_A_values = upload(A.values);

    /* Preprocessed tables */
    Task*       d_tasks   = upload(pr.tasks);
    Group*      d_groups  = upload(pr.groups);
    PairMeta*   d_pairs   = upload(pr.pairs);
    float*      d_packedB = upload(pr.packedB);
    OutputDiag* d_c_diags = upload(pr.output_diags);

    /* Upload per-bucket task id lists for separate kernel dispatch. */
    int* d_light_task_ids  = upload(pr.light_task_ids);
    int* d_medium_task_ids = upload(pr.medium_task_ids);
    int* d_heavy_task_ids  = upload(pr.heavy_task_ids);
    int* d_wide_task_ids   = upload(pr.wide_task_ids);

    const int num_light  = static_cast<int>(pr.light_task_ids.size());
    const int num_medium = static_cast<int>(pr.medium_task_ids.size());
    const int num_heavy  = static_cast<int>(pr.heavy_task_ids.size());
    const int num_wide   = static_cast<int>(pr.wide_task_ids.size());

    /* Output C (zero-initialized) */
    float* d_C_values = nullptr;
    size_t c_bytes = pr.total_c_values * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_C_values, c_bytes));
    CUDA_CHECK(cudaMemset(d_C_values, 0, c_bytes));
    NVTX_POP(); /* device_upload */

    /* ---- Warm-up launch (excluded from profiling) ----
     * Forces CUDA context + JIT init before the timed region.
     * Without this, the first kernel launch includes driver
     * overhead that is irrelevant to kernel performance.      */
    {
        nvtx_push("warmup", 0xFF888888);
        launch_light_kernel(d_tasks, d_light_task_ids, d_groups, d_pairs,
                            d_A_values, d_packedB, d_c_diags,
                            d_C_values, num_light);
        launch_medium_kernel(d_tasks, d_medium_task_ids, d_groups, d_pairs,
                             d_A_values, d_packedB, d_c_diags,
                             d_C_values, num_medium);
        launch_heavy_kernel(d_tasks, d_heavy_task_ids, d_groups, d_pairs,
                            d_A_values, d_packedB, d_c_diags,
                            d_C_values, num_heavy);
        launch_wide_kernel(d_tasks, d_wide_task_ids, d_groups, d_pairs,
                           d_A_values, d_packedB, d_c_diags,
                           d_C_values, num_wide);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_C_values, 0, c_bytes));
        NVTX_POP();
    }

    /* ---- Timed kernel launch (per-bucket dispatch) ---- */
    nvtx_push("kernel_all_buckets", NVTX_KERNEL);

    /* Light bucket: multi-task-per-CTA, one warp per task. */
    if (num_light > 0) {
        nvtx_push("kernel_light", 0xFF88CC88);
        launch_light_kernel(d_tasks, d_light_task_ids, d_groups, d_pairs,
                            d_A_values, d_packedB, d_c_diags,
                            d_C_values, num_light);
        NVTX_POP();
    }

    /* Medium bucket: one CTA per task, 128 threads. */
    if (num_medium > 0) {
        nvtx_push("kernel_medium", 0xFF42f554);
        launch_medium_kernel(d_tasks, d_medium_task_ids, d_groups, d_pairs,
                             d_A_values, d_packedB, d_c_diags,
                             d_C_values, num_medium);
        NVTX_POP();
    }

    /* Heavy bucket: one CTA per task, 256 threads, double-buffered smemA. */
    if (num_heavy > 0) {
        nvtx_push("kernel_heavy", 0xFFCC4444);
        launch_heavy_kernel(d_tasks, d_heavy_task_ids, d_groups, d_pairs,
                            d_A_values, d_packedB, d_c_diags,
                            d_C_values, num_heavy);
        NVTX_POP();
    }

    /* Wide bucket: one CTA per task, 128 threads, 4 outputs per thread. */
    if (num_wide > 0) {
        nvtx_push("kernel_wide", 0xFF8844CC);
        launch_wide_kernel(d_tasks, d_wide_task_ids, d_groups, d_pairs,
                           d_A_values, d_packedB, d_c_diags,
                           d_C_values, num_wide);
        NVTX_POP();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP(); /* kernel_all_buckets */

    /* ---- Download results ---- */
    nvtx_push("device_download", NVTX_DOWNLOAD);
    std::vector<float> h_C_values(pr.total_c_values, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_C_values.data(), d_C_values, c_bytes,
                          cudaMemcpyDeviceToHost));
    NVTX_POP(); /* device_download */

    /* ---- Verify against CPU reference ---- */
    printf("=== GPU Output (diagonal format) ===\n");

    bool pass = true;
    for (size_t di = 0; di < pr.output_diags.size(); ++di) {
        const OutputDiag& od = pr.output_diags[di];
        printf("  d_c = %2d:  [", od.offset);
        for (int p = 0; p < od.length; ++p) {
            float gpu_val = h_C_values[od.values_start + p];
            if (p > 0) printf(", ");
            printf("%.1f", gpu_val);

            /* Compare with CPU dense reference */
            int row = od.start_row + p;
            int col = od.start_col + p;
            float cpu_val = C_dense[row * N + col];
            if (std::fabs(gpu_val - cpu_val) > 1e-5f) {
                printf("(!= %.1f)", cpu_val);
                pass = false;
            }
        }
        printf("]\n");
    }
    printf("\n");

    if (pass) {
        printf("PASS: GPU output matches CPU reference.\n");
    } else {
        printf("FAIL: Mismatch detected!\n");
    }

    /* ---- Cleanup ---- */
    cudaFree(d_A_values);
    cudaFree(d_tasks);
    cudaFree(d_groups);
    cudaFree(d_pairs);
    cudaFree(d_packedB);
    cudaFree(d_c_diags);
    cudaFree(d_light_task_ids);
    cudaFree(d_medium_task_ids);
    cudaFree(d_heavy_task_ids);
    cudaFree(d_wide_task_ids);
    cudaFree(d_C_values);

    return pass ? 0 : 1;
}
