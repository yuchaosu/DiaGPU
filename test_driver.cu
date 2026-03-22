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
#include <cstring>
#include <cmath>
#include <vector>
#include <nvtx3/nvToolsExt.h>
#include <cusparse.h>

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
    printf("  (zero-metadata: no Groups or PairMeta)\n");
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
               "work=%d  bucket=%s\n",
               i, t.c_offset, t.p_begin, t.p_begin + t.p_len,
               t.p_len, t.work_est, bkt);
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
 * cuSPARSE error checking macro
 * ============================================================ */
#define CUSPARSE_CHECK(call)                                           \
    do {                                                               \
        cusparseStatus_t st = (call);                                  \
        if (st != CUSPARSE_STATUS_SUCCESS) {                           \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",          \
                    __FILE__, __LINE__, (int)st);                       \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* ============================================================
 * cuSPARSE CSR SpGEMM benchmark
 *
 * Converts DIA matrices A and B to CSR, then runs
 * cusparseSpGEMM to compute C = A × B.
 * Returns elapsed GPU time in milliseconds.
 * Optionally returns the dense result for verification.
 * ============================================================ */
static float
cusparse_spgemm_benchmark(const DiagMatrix& A, const DiagMatrix& B,
                          int M, int K, int N,
                          std::vector<float>* C_dense_out = nullptr)
{
    /* ---- Convert DIA → COO → CSR on host ---- */
    /* Build COO triplets for A */
    std::vector<int>   A_row, A_col;
    std::vector<float> A_val;
    for (int di = 0; di < A.num_diags; ++di) {
        int d   = A.offsets[di];
        int sr  = DiagMatrix::diag_start_row(d);
        int sc  = DiagMatrix::diag_start_col(d);
        int len = A.diag_lengths[di];
        for (int p = 0; p < len; ++p) {
            A_row.push_back(sr + p);
            A_col.push_back(sc + p);
            A_val.push_back(A.values[A.diag_starts[di] + p]);
        }
    }
    int A_nnz = (int)A_val.size();

    /* Build COO triplets for B */
    std::vector<int>   B_row, B_col;
    std::vector<float> B_val;
    for (int di = 0; di < B.num_diags; ++di) {
        int d   = B.offsets[di];
        int sr  = DiagMatrix::diag_start_row(d);
        int sc  = DiagMatrix::diag_start_col(d);
        int len = B.diag_lengths[di];
        for (int p = 0; p < len; ++p) {
            B_row.push_back(sr + p);
            B_col.push_back(sc + p);
            B_val.push_back(B.values[B.diag_starts[di] + p]);
        }
    }
    int B_nnz = (int)B_val.size();

    /* COO → CSR: build row pointers */
    auto coo_to_csr = [](const std::vector<int>& rows,
                         const std::vector<int>& cols,
                         const std::vector<float>& vals,
                         int num_rows, int nnz,
                         std::vector<int>& csr_offsets,
                         std::vector<int>& csr_cols,
                         std::vector<float>& csr_vals) {
        /* Sort by (row, col) */
        std::vector<int> idx(nnz);
        for (int i = 0; i < nnz; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return rows[a] < rows[b] ||
                   (rows[a] == rows[b] && cols[a] < cols[b]);
        });
        csr_cols.resize(nnz);
        csr_vals.resize(nnz);
        for (int i = 0; i < nnz; ++i) {
            csr_cols[i] = cols[idx[i]];
            csr_vals[i] = vals[idx[i]];
        }
        csr_offsets.assign(num_rows + 1, 0);
        for (int i = 0; i < nnz; ++i)
            csr_offsets[rows[idx[i]] + 1]++;
        for (int i = 0; i < num_rows; ++i)
            csr_offsets[i + 1] += csr_offsets[i];
    };

    std::vector<int> hA_offsets, hA_cols, hB_offsets, hB_cols;
    std::vector<float> hA_vals, hB_vals;
    coo_to_csr(A_row, A_col, A_val, M, A_nnz, hA_offsets, hA_cols, hA_vals);
    coo_to_csr(B_row, B_col, B_val, K, B_nnz, hB_offsets, hB_cols, hB_vals);

    /* ---- Upload CSR to device ---- */
    int *dA_offsets, *dA_cols, *dB_offsets, *dB_cols;
    float *dA_vals, *dB_vals;

    CUDA_CHECK(cudaMalloc(&dA_offsets, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dA_cols,    A_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dA_vals,    A_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB_offsets, (K + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dB_cols,    B_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dB_vals,    B_nnz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA_offsets, hA_offsets.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_cols,    hA_cols.data(),    A_nnz * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_vals,    hA_vals.data(),    A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_offsets, hB_offsets.data(), (K + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_cols,    hB_cols.data(),    B_nnz * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_vals,    hB_vals.data(),    B_nnz * sizeof(float), cudaMemcpyHostToDevice));

    /* ---- cuSPARSE setup ---- */
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA, matB, matC;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, M, K, A_nnz,
        dA_offsets, dA_cols, dA_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matB, K, N, B_nnz,
        dB_offsets, dB_cols, dB_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    /* C is created with 0 nnz; SpGEMM will fill in the structure. */
    CUSPARSE_CHECK(cusparseCreateCsr(&matC, M, N, 0,
        NULL, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    float alpha = 1.0f, beta = 0.0f;
    cusparseSpGEMMDescr_t spgemm_desc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemm_desc));

    /* ---- SpGEMM work estimation (phase 1) ---- */
    size_t bufSize1 = 0, bufSize2 = 0;
    void *dBuf1 = NULL, *dBuf2 = NULL;

    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc,
        &bufSize1, NULL));
    CUDA_CHECK(cudaMalloc(&dBuf1, bufSize1));
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc,
        &bufSize1, dBuf1));

    /* ---- SpGEMM compute (phase 2) ---- */
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc,
        &bufSize2, NULL));
    CUDA_CHECK(cudaMalloc(&dBuf2, bufSize2));

    /* Warmup run */
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc,
        &bufSize2, dBuf2));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Reset descriptor and matC for the timed run (SpGEMM is stateful) */
    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemm_desc));
    CUSPARSE_CHECK(cusparseDestroySpMat(matC));
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemm_desc));
    CUSPARSE_CHECK(cusparseCreateCsr(&matC, M, N, 0,
        NULL, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cudaFree(dBuf1); cudaFree(dBuf2);
    dBuf1 = NULL; dBuf2 = NULL;
    bufSize1 = 0; bufSize2 = 0;

    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc,
        &bufSize1, NULL));
    CUDA_CHECK(cudaMalloc(&dBuf1, bufSize1));
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc,
        &bufSize1, dBuf1));
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc,
        &bufSize2, NULL));
    CUDA_CHECK(cudaMalloc(&dBuf2, bufSize2));

    /* ---- Timed run ---- */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc,
        &bufSize2, dBuf2));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float cusparse_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cusparse_ms, ev_start, ev_stop));

    /* ---- Copy C back ---- */
    CUSPARSE_CHECK(cusparseSpGEMM_copy(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc));

    /* Read C's CSR structure */
    int64_t C_rows, C_cols, C_nnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_rows, &C_cols, &C_nnz));

    int *dC_offsets, *dC_cols;
    float *dC_vals;
    CUSPARSE_CHECK(cusparseCsrSetPointers(matC,
        /* the pointers are already allocated by SpGEMM_copy,
           we just need to retrieve them */
        NULL, NULL, NULL));
    /* Actually retrieve the device pointers */
    {
        void *p_off, *p_col, *p_val;
        cusparseIndexType_t off_t, col_t;
        cusparseIndexBase_t base;
        cudaDataType val_t;
        CUSPARSE_CHECK(cusparseCsrGet(matC, &C_rows, &C_cols, &C_nnz,
            &p_off, &p_col, &p_val,
            &off_t, &col_t, &base, &val_t));
        dC_offsets = (int*)p_off;
        dC_cols    = (int*)p_col;
        dC_vals    = (float*)p_val;
    }

    /* Optionally reconstruct dense output for verification */
    if (C_dense_out) {
        std::vector<int>   hC_offsets(C_rows + 1);
        std::vector<int>   hC_cols(C_nnz);
        std::vector<float> hC_vals(C_nnz);
        CUDA_CHECK(cudaMemcpy(hC_offsets.data(), dC_offsets, (C_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC_cols.data(),    dC_cols,    C_nnz * sizeof(int),        cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC_vals.data(),    dC_vals,    C_nnz * sizeof(float),      cudaMemcpyDeviceToHost));

        C_dense_out->assign(M * N, 0.0f);
        for (int i = 0; i < (int)C_rows; ++i)
            for (int j = hC_offsets[i]; j < hC_offsets[i + 1]; ++j)
                (*C_dense_out)[i * N + hC_cols[j]] = hC_vals[j];
    }

    printf("  cuSPARSE SpGEMM: C_nnz = %lld\n", (long long)C_nnz);

    /* ---- Cleanup ---- */
    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemm_desc));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matB));
    CUSPARSE_CHECK(cusparseDestroySpMat(matC));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    cudaFree(dA_offsets); cudaFree(dA_cols); cudaFree(dA_vals);
    cudaFree(dB_offsets); cudaFree(dB_cols); cudaFree(dB_vals);
    cudaFree(dBuf1);      cudaFree(dBuf2);
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return cusparse_ms;
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

    sort_diag_matrix_by_offset(A);
    sort_diag_matrix_by_offset(B);
    std::vector<int> b_lookup = build_b_diag_lookup(B, N);

    Task*       d_tasks     = upload(pr.tasks);
    OutputDiag* d_c_diags   = upload(pr.output_diags);
    float*      d_A_values  = upload(A.values);
    int*        d_A_offsets = upload(A.offsets);
    int*        d_A_starts  = upload(A.diag_starts);
    int*        d_A_lengths = upload(A.diag_lengths);
    float*      d_B_values  = upload(B.values);
    int*        d_B_starts  = upload(B.diag_starts);
    int*        d_B_lengths = upload(B.diag_lengths);
    int*        d_B_lookup  = upload(b_lookup);

    int* d_light_task_ids  = upload(pr.light_task_ids);
    int* d_medium_task_ids = upload(pr.medium_task_ids);
    int* d_heavy_task_ids  = upload(pr.heavy_task_ids);
    int* d_wide_task_ids   = upload(pr.wide_task_ids);

    const int num_light  = static_cast<int>(pr.light_task_ids.size());
    const int num_medium = static_cast<int>(pr.medium_task_ids.size());
    const int num_heavy  = static_cast<int>(pr.heavy_task_ids.size());
    const int num_wide   = static_cast<int>(pr.wide_task_ids.size());

    float* d_C_values = nullptr;
    size_t c_bytes = pr.total_c_values * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_C_values, c_bytes));
    CUDA_CHECK(cudaMemset(d_C_values, 0, c_bytes));
    NVTX_POP();

    /* Build base KernelArgs */
    KernelArgs ka = {};
    ka.tasks       = d_tasks;
    ka.c_diags     = d_c_diags;
    ka.C_values    = d_C_values;
    ka.A_values    = d_A_values;
    ka.A_offsets   = d_A_offsets;
    ka.A_starts    = d_A_starts;
    ka.A_lengths   = d_A_lengths;
    ka.A_num_diags = A.num_diags;
    ka.B_values    = d_B_values;
    ka.B_starts    = d_B_starts;
    ka.B_lengths   = d_B_lengths;
    ka.B_num_diags = B.num_diags;
    ka.B_diag_lookup = d_B_lookup;
    ka.n           = N;
    ka.B_offset_min = *std::min_element(B.offsets.begin(), B.offsets.end());
    ka.B_offset_max = *std::max_element(B.offsets.begin(), B.offsets.end());

    /* Helper lambda for per-bucket launch */
    auto launch_bucket = [&](int* ids, int count, auto launcher) {
        if (count == 0) return;
        ka.task_ids = ids; ka.num_tasks = count;
        launcher(ka);
    };

    /* ---- Warm-up launch ---- */
    {
        nvtx_push("warmup", 0xFF888888);
        launch_bucket(d_light_task_ids,  num_light,  launch_light_kernel);
        launch_bucket(d_medium_task_ids, num_medium, launch_medium_kernel);
        launch_bucket(d_heavy_task_ids,  num_heavy,  launch_heavy_kernel);
        launch_bucket(d_wide_task_ids,   num_wide,   launch_wide_kernel);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_C_values, 0, c_bytes));
        NVTX_POP();
    }

    /* ---- CUDA events for per-bucket and total timing ---- */
    cudaEvent_t ev_total_start, ev_total_stop;
    cudaEvent_t ev_light_start, ev_light_stop;
    cudaEvent_t ev_medium_start, ev_medium_stop;
    cudaEvent_t ev_heavy_start, ev_heavy_stop;
    cudaEvent_t ev_wide_start, ev_wide_stop;
    CUDA_CHECK(cudaEventCreate(&ev_total_start));
    CUDA_CHECK(cudaEventCreate(&ev_total_stop));
    CUDA_CHECK(cudaEventCreate(&ev_light_start));
    CUDA_CHECK(cudaEventCreate(&ev_light_stop));
    CUDA_CHECK(cudaEventCreate(&ev_medium_start));
    CUDA_CHECK(cudaEventCreate(&ev_medium_stop));
    CUDA_CHECK(cudaEventCreate(&ev_heavy_start));
    CUDA_CHECK(cudaEventCreate(&ev_heavy_stop));
    CUDA_CHECK(cudaEventCreate(&ev_wide_start));
    CUDA_CHECK(cudaEventCreate(&ev_wide_stop));

    /* ---- Timed kernel launch (per-bucket dispatch) ---- */
    nvtx_push("kernel_all_buckets", NVTX_KERNEL);
    CUDA_CHECK(cudaEventRecord(ev_total_start));

    if (num_light > 0) {
        nvtx_push("kernel_light", 0xFF88CC88);
        CUDA_CHECK(cudaEventRecord(ev_light_start));
        launch_bucket(d_light_task_ids, num_light, launch_light_kernel);
        CUDA_CHECK(cudaEventRecord(ev_light_stop));
        NVTX_POP();
    }
    if (num_medium > 0) {
        nvtx_push("kernel_medium", 0xFF42f554);
        CUDA_CHECK(cudaEventRecord(ev_medium_start));
        launch_bucket(d_medium_task_ids, num_medium, launch_medium_kernel);
        CUDA_CHECK(cudaEventRecord(ev_medium_stop));
        NVTX_POP();
    }
    if (num_heavy > 0) {
        nvtx_push("kernel_heavy", 0xFFCC4444);
        CUDA_CHECK(cudaEventRecord(ev_heavy_start));
        launch_bucket(d_heavy_task_ids, num_heavy, launch_heavy_kernel);
        CUDA_CHECK(cudaEventRecord(ev_heavy_stop));
        NVTX_POP();
    }
    if (num_wide > 0) {
        nvtx_push("kernel_wide", 0xFF8844CC);
        CUDA_CHECK(cudaEventRecord(ev_wide_start));
        launch_bucket(d_wide_task_ids, num_wide, launch_wide_kernel);
        CUDA_CHECK(cudaEventRecord(ev_wide_stop));
        NVTX_POP();
    }

    CUDA_CHECK(cudaEventRecord(ev_total_stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();

    /* ---- Collect DiaGPU timing ---- */
    float ms_total = 0, ms_light = 0, ms_medium = 0, ms_heavy = 0, ms_wide = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev_total_start, ev_total_stop));
    if (num_light  > 0) CUDA_CHECK(cudaEventElapsedTime(&ms_light,  ev_light_start,  ev_light_stop));
    if (num_medium > 0) CUDA_CHECK(cudaEventElapsedTime(&ms_medium, ev_medium_start, ev_medium_stop));
    if (num_heavy  > 0) CUDA_CHECK(cudaEventElapsedTime(&ms_heavy,  ev_heavy_start,  ev_heavy_stop));
    if (num_wide   > 0) CUDA_CHECK(cudaEventElapsedTime(&ms_wide,   ev_wide_start,   ev_wide_stop));

    printf("=== DiaGPU Kernel Timing ===\n");
    printf("  Light  : %8.4f ms  (%d tasks)\n", ms_light,  num_light);
    printf("  Medium : %8.4f ms  (%d tasks)\n", ms_medium, num_medium);
    printf("  Heavy  : %8.4f ms  (%d tasks)\n", ms_heavy,  num_heavy);
    printf("  Wide   : %8.4f ms  (%d tasks)\n", ms_wide,   num_wide);
    printf("  TOTAL  : %8.4f ms\n\n", ms_total);

    CUDA_CHECK(cudaEventDestroy(ev_total_start));
    CUDA_CHECK(cudaEventDestroy(ev_total_stop));
    CUDA_CHECK(cudaEventDestroy(ev_light_start));
    CUDA_CHECK(cudaEventDestroy(ev_light_stop));
    CUDA_CHECK(cudaEventDestroy(ev_medium_start));
    CUDA_CHECK(cudaEventDestroy(ev_medium_stop));
    CUDA_CHECK(cudaEventDestroy(ev_heavy_start));
    CUDA_CHECK(cudaEventDestroy(ev_heavy_stop));
    CUDA_CHECK(cudaEventDestroy(ev_wide_start));
    CUDA_CHECK(cudaEventDestroy(ev_wide_stop));

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

    /* ============================================================
     * cuSPARSE SpGEMM benchmark for comparison
     * ============================================================ */
    printf("=== cuSPARSE SpGEMM Benchmark ===\n");
    float cusparse_ms = cusparse_spgemm_benchmark(A, B, M, K, N);
    printf("  cuSPARSE compute time: %8.4f ms\n\n", cusparse_ms);

    printf("=== Summary ===\n");
    printf("  DiaGPU   : %8.4f ms\n", ms_total);
    printf("  cuSPARSE : %8.4f ms\n", cusparse_ms);
    if (cusparse_ms > 0.0f)
        printf("  Speedup  : %.2fx\n", cusparse_ms / ms_total);
    printf("\n");

    /* ---- Cleanup ---- */
    cudaFree(d_tasks); cudaFree(d_c_diags);
    cudaFree(d_A_values); cudaFree(d_A_offsets);
    cudaFree(d_A_starts); cudaFree(d_A_lengths);
    cudaFree(d_B_values); cudaFree(d_B_starts);
    cudaFree(d_B_lengths); cudaFree(d_B_lookup);
    cudaFree(d_light_task_ids); cudaFree(d_medium_task_ids);
    cudaFree(d_heavy_task_ids); cudaFree(d_wide_task_ids);
    cudaFree(d_C_values);

    return pass ? 0 : 1;
}
