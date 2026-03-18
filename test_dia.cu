/* ============================================================
 * test_driver.cu
 *
 * End-to-end test for the DiagSpMM framework.
 *
 * Uses a 1024×1024 example with 21 diagonals per matrix to
 * exercise all four kernel buckets (light, medium, heavy, wide).
 *
 *   A (1024×1024): 21 diagonals, offsets d ∈ {-10, -9, ..., 0, ..., 9, 10}
 *   B (1024×1024): 21 diagonals, same offsets
 *
 * Values are generated procedurally:
 *   A[diag d, position p] = 1.0 + 0.01 * ((d + 10) * 13 + p) % 97
 *   B[diag d, position p] = 1.0 + 0.01 * ((d + 10) * 7  + p) % 89
 *
 * Output C has up to 41 diagonals (d_c ∈ {-20, ..., 0, ..., 20}).
 * Near the center (d_c ≈ 0) up to 21 contributor pairs exist per
 * output diagonal, generating heavy tiles; near the edges (|d_c| ≈ 20)
 * only 1 pair exists, generating light tiles.
 *
 * Compile:
 *   nvcc test_driver.cu diag_kernel.cu -o test_diag -std=c++17
 * ============================================================ */

#include "diag_types.cuh"
#include "diag_host_preprocess.cuh"
#include "diag_kernel.cuh"

#include <cstdio>
#include <cstdlib>
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

    /* Print first few tasks per bucket for debugging */
    int shown[4] = {};   // count per bucket
    const int MAX_PER_BUCKET = 3;
    for (size_t i = 0; i < pr.tasks.size(); ++i) {
        const Task& t = pr.tasks[i];
        int b = t.bucket;
        if (b < 0 || b > 3) b = 1;
        if (shown[b] >= MAX_PER_BUCKET) continue;
        shown[b]++;
        const char* bkt = (t.bucket == 0) ? "LIGHT" :
                          (t.bucket == 1) ? "MEDIUM" :
                          (t.bucket == 2) ? "HEAVY" : "WIDE";
        printf("  Task %zu: c_diag_offset=%d  p=[%d..%d)  p_len=%d  "
               "groups=%d  work=%d  bucket=%s\n",
               i, t.c_offset, t.p_begin, t.p_begin + t.p_len,
               t.p_len, t.group_count, t.work_est, bkt);
    }
    printf("  ... (%zu total tasks)\n\n", pr.tasks.size());

    /* DEBUG: Check for tile size vs kernel expectation mismatches */
    printf("=== DEBUG: Bucket tile-size audit ===\n");
    printf("  LIGHT kernel expects tile_size = %d (WARP_SIZE)\n", TILE_SIZE_LIGHT);
    printf("  MEDIUM kernel expects tile_size = %d\n", TILE_SIZE);
    printf("  HEAVY kernel expects tile_size = %d\n", TILE_SIZE_HEAVY);
    printf("  WIDE kernel expects tile_size = %d\n", WIDE_TILE_SIZE);
    int oversize_light = 0, oversize_heavy = 0;
    for (size_t i = 0; i < pr.tasks.size(); ++i) {
        const Task& t = pr.tasks[i];
        if (t.bucket == 0 && t.p_len > TILE_SIZE_LIGHT) {
            if (oversize_light < 3)
                printf("  WARNING: LIGHT task %zu has p_len=%d > %d!\n",
                       i, t.p_len, TILE_SIZE_LIGHT);
            oversize_light++;
        }
        if (t.bucket == 2 && t.p_len > TILE_SIZE_HEAVY) {
            if (oversize_heavy < 3)
                printf("  WARNING: HEAVY task %zu has p_len=%d > %d!\n",
                       i, t.p_len, TILE_SIZE_HEAVY);
            oversize_heavy++;
        }
    }
    if (oversize_light > 0)
        printf("  TOTAL: %d LIGHT tasks with p_len > %d (BUG: kernel will only compute first %d elements!)\n",
               oversize_light, TILE_SIZE_LIGHT, TILE_SIZE_LIGHT);
    if (oversize_heavy > 0)
        printf("  TOTAL: %d HEAVY tasks with p_len > %d\n",
               oversize_heavy, TILE_SIZE_HEAVY);
    if (oversize_light == 0 && oversize_heavy == 0)
        printf("  OK: All tasks have tile sizes matching their kernel.\n");
    printf("\n");
}

/* ============================================================
 * Helper: build a 1024×1024 DiagMatrix with 21 diagonals
 *         (offsets -10 to +10) and procedural values.
 * ============================================================ */
static DiagMatrix
build_1024_matrix(int rows, int cols,
                  int seed_a, int seed_b)
{
    DiagMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.num_diags = 21;

    int val_offset = 0;
    for (int i = 0; i < 21; ++i) {
        int d = i - 10;   // offsets: -10, -9, ..., 0, ..., 9, 10
        int len = DiagMatrix::diag_length(rows, cols, d);
        mat.offsets.push_back(d);
        mat.diag_starts.push_back(val_offset);
        mat.diag_lengths.push_back(len);
        for (int p = 0; p < len; ++p) {
            float v = 1.0f + 0.01f * (float)(((d + 10) * seed_a + p) % seed_b);
            mat.values.push_back(v);
        }
        val_offset += len;
    }
    return mat;
}

/* ============================================================
 * MAIN
 * ============================================================ */
int main()
{
    printf("DiagSpMM Framework Test — 1024×1024, 21 diagonals\n");
    printf("==================================================\n\n");

    /* ---- Build input matrices ---- */
    const int M = 1024, K = 1024, N = 1024;

    DiagMatrix A = build_1024_matrix(M, K, 13, 97);
    DiagMatrix B = A;

    printf("A: %dx%d, %d diagonals, %zu values\n",
           A.rows, A.cols, A.num_diags, A.values.size());
    printf("B: %dx%d, %d diagonals, %zu values\n",
           B.rows, B.cols, B.num_diags, B.values.size());
    printf("\n");

    /* ---- CPU reference ---- */
    std::vector<float> C_dense;
    cpu_diag_multiply(A, B, M, K, N, C_dense);

    printf("CPU reference C — sample values (top-left 4×4):\n");
    for (int i = 0; i < 4; ++i) {
        printf("  ");
        for (int j = 0; j < 4; ++j)
            printf("%10.3f", C_dense[i * N + j]);
        printf("\n");
    }
    printf("\n");

    /* ---- Host preprocessing ---- */
    nvtx_push("host_preprocess", NVTX_PREPROCESS);
    PreprocessResult pr = build_all_adaptive(A, B, M, K, N);
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

    /* ---- Kernel launch (per-bucket dispatch) ---- */
    nvtx_push("kernel_all_buckets", NVTX_KERNEL);

    if (num_light > 0) {
        nvtx_push("kernel_light", 0xFF88CC88);
        launch_light_kernel(d_tasks, d_light_task_ids, d_groups, d_pairs,
                            d_A_values, d_packedB, d_c_diags,
                            d_C_values, num_light);
        NVTX_POP();
    }

    if (num_medium > 0) {
        nvtx_push("kernel_medium", 0xFF42f554);
        launch_medium_kernel(d_tasks, d_medium_task_ids, d_groups, d_pairs,
                             d_A_values, d_packedB, d_c_diags,
                             d_C_values, num_medium);
        NVTX_POP();
    }

    if (num_heavy > 0) {
        nvtx_push("kernel_heavy", 0xFFCC4444);
        launch_heavy_kernel(d_tasks, d_heavy_task_ids, d_groups, d_pairs,
                            d_A_values, d_packedB, d_c_diags,
                            d_C_values, num_heavy);
        NVTX_POP();
    }

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
    printf("=== Verification ===\n");

    bool pass = true;
    int mismatches = 0;
    const float tol = 1e-2f;   /* allow small fp accumulation error */

    for (size_t di = 0; di < pr.output_diags.size(); ++di) {
        const OutputDiag& od = pr.output_diags[di];
        for (int p = 0; p < od.length; ++p) {
            float gpu_val = h_C_values[od.values_start + p];
            int row = od.start_row + p;
            int col = od.start_col + p;
            float cpu_val = C_dense[row * N + col];
            float diff = std::fabs(gpu_val - cpu_val);
            if (diff > tol) {
                if (mismatches < 10) {
                    printf("  MISMATCH d_c=%d p=%d (%d,%d): gpu=%.4f cpu=%.4f diff=%.4f\n",
                           od.offset, p, row, col, gpu_val, cpu_val, diff);
                }
                ++mismatches;
                pass = false;
            }
        }
    }

    if (pass) {
        printf("  PASS: All %d output values match CPU reference (tol=%.0e).\n",
               pr.total_c_values, tol);
    } else {
        printf("  FAIL: %d / %d mismatches (tol=%.0e).\n",
               mismatches, pr.total_c_values, tol);
    }

    /* Print a few sample output diagonals for sanity */
    printf("\n=== Sample GPU output diagonals ===\n");
    for (size_t di = 0; di < pr.output_diags.size(); ++di) {
        const OutputDiag& od = pr.output_diags[di];
        printf("  d_c = %3d  (len=%4d):  [", od.offset, od.length);
        int show = std::min(od.length, 4);
        for (int p = 0; p < show; ++p) {
            if (p > 0) printf(", ");
            printf("%.3f", h_C_values[od.values_start + p]);
        }
        if (od.length > 4) printf(", ...");
        printf("]\n");
    }
    printf("\n");

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
