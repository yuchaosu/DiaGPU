/* ============================================================
 * bench_compare.cu
 *
 * Benchmark comparison of three SpMM algorithms for
 * structured sparse (diagonal) matrices:
 *
 *   1. DiagSpMM   — Our algorithm (A-stationary, packed-B,
 *                   zero-atomic, per-bucket kernels)
 *   2. Paper HM   — Algorithm 2 from Haque et al. 2024
 *                   (one thread per nnz of A, atomicAdd)
 *   3. cuSPARSE   — NVIDIA cusparseSpGEMM (CSR-based)
 *
 * All three are timed using cudaEvent (kernel-only, excluding
 * preprocessing and data transfer). Results are verified
 * against a CPU reference.
 *
 * Compile:
 *   nvcc -std=c++17 -O2 -arch=sm_86 \
 *        bench_compare.cu paper_hm_kernel.cu diag_kernel.cu \
 *        -lcusparse -o bench_compare
 * ============================================================ */

#include "diag_types.cuh"
#include "diag_host_preprocess.cuh"
#include "diag_kernel.cuh"
#include "paper_hm.cuh"

#include <cusparse.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

/* ============================================================
 * Error checking macros
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
 * Common types
 * ============================================================ */
/* Default iteration count for averaging. */
static int NUM_ITERS = 100;

struct BenchResult {
    float gpu_ms;                /* average kernel-only GPU time */
    std::vector<float> C_dense;  /* n*n dense output (from last iter) */
    bool  pass;                  /* correctness vs CPU ref */
};

/* ============================================================
 * Helpers
 * ============================================================ */
template <typename T>
static T* upload_vec(const std::vector<T>& h_data)
{
    if (h_data.empty()) return nullptr;
    T* d_ptr = nullptr;
    size_t bytes = h_data.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, h_data.data(), bytes,
                          cudaMemcpyHostToDevice));
    return d_ptr;
}

/* Build a DiagMatrix with given offsets and procedural values. */
static DiagMatrix
build_diag_matrix(int n, const std::vector<int>& offsets, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);

    DiagMatrix mat;
    mat.rows = n;
    mat.cols = n;
    mat.num_diags = static_cast<int>(offsets.size());

    int val_offset = 0;
    for (int d : offsets) {
        int len = DiagMatrix::diag_length(n, n, d);
        mat.offsets.push_back(d);
        mat.diag_starts.push_back(val_offset);
        mat.diag_lengths.push_back(len);
        for (int p = 0; p < len; ++p)
            mat.values.push_back(dist(rng));
        val_offset += len;
    }
    return mat;
}

/* CPU reference: diagonal-aware SpMM. */
static void
cpu_diag_multiply(const DiagMatrix& A, const DiagMatrix& B,
                  int n, std::vector<float>& C_dense)
{
    C_dense.assign(n * n, 0.0f);
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
                int k   = a_sc + pa;
                int pb  = k - b_sr;
                if (pb < 0 || pb >= b_len) continue;
                int col = b_sc + pb;
                C_dense[row * n + col] +=
                    A.values[A.diag_starts[ai] + pa]
                  * B.values[B.diag_starts[bi] + pb];
            }
        }
    }
}

/* Verify GPU dense output against CPU reference.
 * Only checks positions on the expected output diagonals. */
static bool
verify_result(const float* gpu, const float* cpu, int n, float tol)
{
    int mismatches = 0;
    for (int i = 0; i < n * n; ++i) {
        float diff = std::fabs(gpu[i] - cpu[i]);
        if (diff > tol) {
            if (mismatches < 3) {
                int r = i / n, c = i % n;
                fprintf(stderr, "    MISMATCH (%d,%d): gpu=%.6f cpu=%.6f diff=%.6f\n",
                        r, c, gpu[i], cpu[i], diff);
            }
            ++mismatches;
        }
    }
    return mismatches == 0;
}

/* Build dense matrix from DiagMatrix. */
static void
diag_to_dense(const DiagMatrix& M, float* dense)
{
    int n = M.rows;
    memset(dense, 0, n * n * sizeof(float));
    for (int di = 0; di < M.num_diags; ++di) {
        int d   = M.offsets[di];
        int sr  = DiagMatrix::diag_start_row(d);
        int sc  = DiagMatrix::diag_start_col(d);
        int len = M.diag_lengths[di];
        for (int p = 0; p < len; ++p)
            dense[(sr + p) * n + (sc + p)] = M.values[M.diag_starts[di] + p];
    }
}

/* ============================================================
 * Helper: build KernelArgs and upload all device data.
 * Returns a KernelArgs struct and a list of device pointers
 * to free later.
 * ============================================================ */
struct DeviceState {
    KernelArgs args;
    std::vector<void*> allocs;  /* device pointers to free */
    size_t c_bytes;
};

static DeviceState
upload_diagspmm(const DiagMatrix& A, const DiagMatrix& B, int n,
                const PreprocessResult& pr,
                const std::vector<int>& b_lookup)
{
    DeviceState ds;
    ds.c_bytes = pr.total_c_values * sizeof(float);

    auto up_f = [&](const std::vector<float>& v) -> float* {
        float* d = upload_vec(v);
        ds.allocs.push_back(d);
        return d;
    };
    auto up_i = [&](const std::vector<int>& v) -> int* {
        int* d = upload_vec(v);
        ds.allocs.push_back(d);
        return d;
    };
    auto up_t = [&](const std::vector<Task>& v) -> Task* {
        Task* d = upload_vec(v);
        ds.allocs.push_back(d);
        return d;
    };
    auto up_od = [&](const std::vector<OutputDiag>& v) -> OutputDiag* {
        OutputDiag* d = upload_vec(v);
        ds.allocs.push_back(d);
        return d;
    };

    ds.args.tasks       = up_t(pr.tasks);
    ds.args.c_diags     = up_od(pr.output_diags);
    ds.args.A_values    = up_f(A.values);
    ds.args.A_offsets   = up_i(A.offsets);
    ds.args.A_starts    = up_i(A.diag_starts);
    ds.args.A_lengths   = up_i(A.diag_lengths);
    ds.args.A_num_diags = A.num_diags;
    ds.args.B_values    = up_f(B.values);
    ds.args.B_starts    = up_i(B.diag_starts);
    ds.args.B_lengths   = up_i(B.diag_lengths);
    ds.args.B_num_diags = B.num_diags;
    ds.args.B_diag_lookup = up_i(b_lookup);
    ds.args.n           = n;

    /* C output */
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C, ds.c_bytes));
    ds.allocs.push_back(d_C);
    ds.args.C_values = d_C;

    return ds;
}

static void free_device_state(DeviceState& ds)
{
    for (void* p : ds.allocs) cudaFree(p);
    ds.allocs.clear();
}

/* ============================================================
 *            BENCHMARK 1: DiagSpMM (our algorithm)
 * ============================================================ */
static BenchResult
bench_diagspmm(const DiagMatrix& A, const DiagMatrix& B, int n)
{
    BenchResult res;

    PreprocessResult pr = build_all_adaptive(A, B, n, n, n);
    std::vector<int> b_lookup = build_b_diag_lookup(B, n);

    DeviceState ds = upload_diagspmm(A, B, n, pr, b_lookup);

    /* Per-bucket task id uploads */
    int* d_light_ids  = upload_vec(pr.light_task_ids);
    int* d_medium_ids = upload_vec(pr.medium_task_ids);
    int* d_heavy_ids  = upload_vec(pr.heavy_task_ids);
    int* d_wide_ids   = upload_vec(pr.wide_task_ids);

    int nl = (int)pr.light_task_ids.size();
    int nm = (int)pr.medium_task_ids.size();
    int nh = (int)pr.heavy_task_ids.size();
    int nw = (int)pr.wide_task_ids.size();

    /* Helper to launch all buckets. */
    auto launch_all = [&]() {
        KernelArgs ka = ds.args;
        if (nl > 0) { ka.task_ids = d_light_ids;  ka.num_tasks = nl; launch_light_kernel(ka); }
        if (nm > 0) { ka.task_ids = d_medium_ids; ka.num_tasks = nm; launch_medium_kernel(ka); }
        if (nh > 0) { ka.task_ids = d_heavy_ids;  ka.num_tasks = nh; launch_heavy_kernel(ka); }
        if (nw > 0) { ka.task_ids = d_wide_ids;   ka.num_tasks = nw; launch_wide_kernel(ka); }
    };

    /* Warmup */
    CUDA_CHECK(cudaMemset(ds.args.C_values, 0, ds.c_bytes));
    launch_all();
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed runs */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        CUDA_CHECK(cudaMemset(ds.args.C_values, 0, ds.c_bytes));
        launch_all();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    res.gpu_ms = total_ms / NUM_ITERS;

    /* Download and convert to dense */
    std::vector<float> h_C(pr.total_c_values);
    CUDA_CHECK(cudaMemcpy(h_C.data(), ds.args.C_values, ds.c_bytes,
                          cudaMemcpyDeviceToHost));

    res.C_dense.assign(n * n, 0.0f);
    for (size_t di = 0; di < pr.output_diags.size(); ++di) {
        const OutputDiag& od = pr.output_diags[di];
        for (int p = 0; p < od.length; ++p) {
            int row = od.start_row + p;
            int col = od.start_col + p;
            res.C_dense[row * n + col] = h_C[od.values_start + p];
        }
    }

    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free_device_state(ds);
    cudaFree(d_light_ids); cudaFree(d_medium_ids);
    cudaFree(d_heavy_ids); cudaFree(d_wide_ids);

    return res;
}

/* ============================================================
 *        BENCHMARK 2: Paper Algorithm 2 (HM + atomics)
 * ============================================================ */
static BenchResult
bench_paper_hm(const DiagMatrix& A_dia, const DiagMatrix& B_dia, int n)
{
    BenchResult res;

    /* Convert DiagMatrix to dense, then to HM. */
    std::vector<float> denseA(n * n), denseB(n * n);
    diag_to_dense(A_dia, denseA.data());
    diag_to_dense(B_dia, denseB.data());

    /* Collect offsets */
    std::vector<int> offA(A_dia.offsets.begin(), A_dia.offsets.end());
    std::vector<int> offB(B_dia.offsets.begin(), B_dia.offsets.end());

    HMMatrix A_hm = dense_to_hm(denseA.data(), n, offA);
    HMMatrix B_hm = dense_to_hm(denseB.data(), n, offB);
    HMMatrix C_hm = compute_c_hm_structure(A_hm, B_hm, n);
    std::vector<int> c_lookup = build_c_diag_lookup(C_hm, n);

    /* Upload */
    float* d_A_vals    = upload_vec(A_hm.values);
    int*   d_A_offsets = upload_vec(A_hm.diag_offsets);
    int*   d_A_starts  = upload_vec(A_hm.diag_starts);
    int*   d_A_lengths = upload_vec(A_hm.diag_lengths);

    float* d_B_vals    = upload_vec(B_hm.values);
    int*   d_B_offsets = upload_vec(B_hm.diag_offsets);
    int*   d_B_starts  = upload_vec(B_hm.diag_starts);
    int*   d_B_lengths = upload_vec(B_hm.diag_lengths);

    float* d_C_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C_vals, C_hm.total_nz * sizeof(float)));
    int*   d_C_offsets = upload_vec(C_hm.diag_offsets);
    int*   d_C_starts  = upload_vec(C_hm.diag_starts);
    int*   d_C_lengths = upload_vec(C_hm.diag_lengths);
    int*   d_C_lookup  = upload_vec(c_lookup);

    const int block_size = 256;
    int grid_size = (A_hm.total_nz + block_size - 1) / block_size;

    size_t c_hm_bytes = C_hm.total_nz * sizeof(float);

    /* Warmup */
    CUDA_CHECK(cudaMemset(d_C_vals, 0, c_hm_bytes));
    hm_structured_sparse_matmul_kernel<<<grid_size, block_size>>>(
        d_A_vals, d_A_offsets, d_A_starts, d_A_lengths, A_hm.num_diags,
        d_B_vals, d_B_offsets, d_B_starts, d_B_lengths, B_hm.num_diags,
        d_C_vals, d_C_offsets, d_C_starts, d_C_lengths, C_hm.num_diags,
        d_C_lookup, A_hm.total_nz, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed runs: NUM_ITERS iterations. */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        CUDA_CHECK(cudaMemset(d_C_vals, 0, c_hm_bytes));
        hm_structured_sparse_matmul_kernel<<<grid_size, block_size>>>(
            d_A_vals, d_A_offsets, d_A_starts, d_A_lengths, A_hm.num_diags,
            d_B_vals, d_B_offsets, d_B_starts, d_B_lengths, B_hm.num_diags,
            d_C_vals, d_C_offsets, d_C_starts, d_C_lengths, C_hm.num_diags,
            d_C_lookup, A_hm.total_nz, n);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    res.gpu_ms = total_ms / NUM_ITERS;

    /* Download and convert to dense */
    std::vector<float> h_C(C_hm.total_nz);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C_vals,
               C_hm.total_nz * sizeof(float), cudaMemcpyDeviceToHost));

    HMMatrix C_result = C_hm;
    C_result.values = h_C;
    res.C_dense.assign(n * n, 0.0f);
    hm_to_dense(C_result, res.C_dense.data());

    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A_vals); cudaFree(d_A_offsets);
    cudaFree(d_A_starts); cudaFree(d_A_lengths);
    cudaFree(d_B_vals); cudaFree(d_B_offsets);
    cudaFree(d_B_starts); cudaFree(d_B_lengths);
    cudaFree(d_C_vals); cudaFree(d_C_offsets);
    cudaFree(d_C_starts); cudaFree(d_C_lengths);
    cudaFree(d_C_lookup);

    return res;
}

/* ============================================================
 *         BENCHMARK 3: cuSPARSE SpGEMM (CSR-based)
 * ============================================================ */

/* Convert DiagMatrix to CSR on host. */
static void
dia_to_csr(const DiagMatrix& M, int n,
           std::vector<int>& offsets,
           std::vector<int>& cols,
           std::vector<float>& vals)
{
    /* Build COO */
    struct Triplet { int r, c; float v; };
    std::vector<Triplet> coo;
    for (int di = 0; di < M.num_diags; ++di) {
        int d   = M.offsets[di];
        int sr  = DiagMatrix::diag_start_row(d);
        int sc  = DiagMatrix::diag_start_col(d);
        int len = M.diag_lengths[di];
        for (int p = 0; p < len; ++p)
            coo.push_back({sr + p, sc + p, M.values[M.diag_starts[di] + p]});
    }
    std::sort(coo.begin(), coo.end(), [](const Triplet& a, const Triplet& b) {
        return a.r < b.r || (a.r == b.r && a.c < b.c);
    });

    int nnz = (int)coo.size();
    offsets.assign(n + 1, 0);
    cols.resize(nnz);
    vals.resize(nnz);
    for (int i = 0; i < nnz; ++i) {
        offsets[coo[i].r + 1]++;
        cols[i] = coo[i].c;
        vals[i] = coo[i].v;
    }
    for (int i = 0; i < n; ++i)
        offsets[i + 1] += offsets[i];
}

/* Run one complete cuSPARSE SpGEMM cycle on pre-uploaded CSR data.
 * Follows the official NVIDIA sample exactly:
 *   1. Create matC with pre-allocated dC_csrOffsets
 *   2. workEstimation (query + execute)
 *   3. compute (query + execute)  ← timed
 *   4. SpMatGetSize → allocate dC_columns, dC_values
 *   5. cusparseCsrSetPointers
 *   6. SpGEMM_copy
 * Returns kernel time (step 3 only) via cudaEvent. */
static float
cusparse_single_run(cusparseHandle_t handle,
                    cusparseSpMatDescr_t matA,
                    cusparseSpMatDescr_t matB,
                    int n,
                    std::vector<float>* C_dense_out)
{
    float alpha = 1.0f, beta = 0.0f;

    /* Pre-allocate C row offsets (required by the API). */
    int* dC_csrOffsets = NULL;
    CUDA_CHECK(cudaMalloc(&dC_csrOffsets, (n + 1) * sizeof(int)));

    cusparseSpMatDescr_t matC;
    CUSPARSE_CHECK(cusparseCreateCsr(&matC, n, n, 0,
        dC_csrOffsets, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpGEMMDescr_t desc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&desc));

    /* Phase 1: work estimation */
    size_t buf1 = 0, buf2 = 0;
    void *dBuf1 = NULL, *dBuf2 = NULL;

    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, desc, &buf1, NULL));
    CUDA_CHECK(cudaMalloc(&dBuf1, buf1));
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, desc, &buf1, dBuf1));

    /* Phase 2: compute — first call queries buffer size */
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, desc, &buf2, NULL));
    CUDA_CHECK(cudaMalloc(&dBuf2, buf2));

    /* Timed compute (actual SpGEMM) */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, desc, &buf2, dBuf2));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

    /* Phase 3: get C structure size, allocate, set pointers, copy. */
    int64_t C_rows, C_cols, C_nnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_rows, &C_cols, &C_nnz));

    int*   dC_columns = NULL;
    float* dC_values  = NULL;
    CUDA_CHECK(cudaMalloc(&dC_columns, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dC_values,  C_nnz * sizeof(float)));

    /* Update matC with the newly allocated column/value arrays. */
    CUSPARSE_CHECK(cusparseCsrSetPointers(matC,
        dC_csrOffsets, dC_columns, dC_values));

    /* Copy the final product into matC. */
    CUSPARSE_CHECK(cusparseSpGEMM_copy(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPGEMM_DEFAULT, desc));

    /* Optionally extract dense result */
    if (C_dense_out) {
        std::vector<int>   hC_off(C_rows + 1);
        std::vector<int>   hC_col(C_nnz);
        std::vector<float> hC_val(C_nnz);
        CUDA_CHECK(cudaMemcpy(hC_off.data(), dC_csrOffsets, (C_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC_col.data(), dC_columns,    C_nnz * sizeof(int),        cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC_val.data(), dC_values,     C_nnz * sizeof(float),      cudaMemcpyDeviceToHost));

        C_dense_out->assign(n * n, 0.0f);
        for (int i = 0; i < (int)C_rows; ++i)
            for (int j = hC_off[i]; j < hC_off[i + 1]; ++j)
                (*C_dense_out)[i * n + hC_col[j]] = hC_val[j];
    }

    /* Cleanup this run */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(desc));
    CUSPARSE_CHECK(cusparseDestroySpMat(matC));
    cudaFree(dBuf1);
    cudaFree(dBuf2);
    cudaFree(dC_csrOffsets);
    cudaFree(dC_columns);
    cudaFree(dC_values);

    return ms;
}

static BenchResult
bench_cusparse(const DiagMatrix& A_dia, const DiagMatrix& B_dia, int n)
{
    BenchResult res;

    /* Convert to CSR */
    std::vector<int> hA_off, hA_col, hB_off, hB_col;
    std::vector<float> hA_val, hB_val;
    dia_to_csr(A_dia, n, hA_off, hA_col, hA_val);
    dia_to_csr(B_dia, n, hB_off, hB_col, hB_val);
    int A_nnz = (int)hA_val.size();
    int B_nnz = (int)hB_val.size();

    /* Upload CSR (persistent across iterations) */
    int *dA_off, *dA_col, *dB_off, *dB_col;
    float *dA_val, *dB_val;
    CUDA_CHECK(cudaMalloc(&dA_off, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dA_col, A_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dA_val, A_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB_off, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dB_col, B_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dB_val, B_nnz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA_off, hA_off.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_col, hA_col.data(), A_nnz * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_val, hA_val.data(), A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_off, hB_off.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_col, hB_col.data(), B_nnz * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_val, hB_val.data(), B_nnz * sizeof(float), cudaMemcpyHostToDevice));

    /* Create persistent handle and input matrix descriptors */
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA, matB;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, n, A_nnz,
        dA_off, dA_col, dA_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matB, n, n, B_nnz,
        dB_off, dB_col, dB_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    /* Warmup (discard result) */
    cusparse_single_run(handle, matA, matB, n, nullptr);

    /* Timed runs: each iteration is a clean SpGEMM cycle.
     * Accumulate per-iteration cudaEvent times. */
    float total_ms = 0.0f;

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        /* Last iteration: also extract dense output for verification */
        bool last = (iter == NUM_ITERS - 1);
        float ms = cusparse_single_run(handle, matA, matB, n,
                                       last ? &res.C_dense : nullptr);
        total_ms += ms;
    }

    res.gpu_ms = total_ms / NUM_ITERS;

    /* Cleanup */
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matB));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    cudaFree(dA_off); cudaFree(dA_col); cudaFree(dA_val);
    cudaFree(dB_off); cudaFree(dB_col); cudaFree(dB_val);

    return res;
}

/* ============================================================
 *                        MAIN
 *
 * Usage:
 *   ./bench_compare [options]
 *
 * Options:
 *   -n <size>      Matrix dimension (default: runs preset list)
 *   -d <diags>     Number of diagonals (must be odd, default: 21)
 *   -i <iters>     Number of iterations for averaging (default: 100)
 *
 * Examples:
 *   ./bench_compare                        # run preset test suite
 *   ./bench_compare -n 2048 -d 21         # single test: 2048x2048, 21 diags
 *   ./bench_compare -n 4096 -d 41 -i 50  # 4096x4096, 41 diags, 50 iters
 * ============================================================ */

static void print_usage(const char* prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("  -n <size>    Matrix dimension (default: preset list)\n");
    printf("  -d <diags>   Number of diagonals, must be odd (default: 21)\n");
    printf("  -i <iters>   Iterations for averaging (default: 100)\n");
    printf("  -h           Show this help\n");
}

struct TestCase {
    int n;
    int num_diags;  /* total diagonals (odd: centered around 0) */
};

static void
run_test(const TestCase& tc, float tol)
{
    int n  = tc.n;
    int nd = tc.num_diags;
    int hb = (nd - 1) / 2;   /* half-bandwidth */

    /* Build diagonal offsets: {-hb, ..., 0, ..., +hb} */
    std::vector<int> offsets;
    for (int d = -hb; d <= hb; ++d)
        offsets.push_back(d);

    /* Build input matrices */
    DiagMatrix A = build_diag_matrix(n, offsets, 42);
    DiagMatrix B = build_diag_matrix(n, offsets, 137);

    int nnzA = (int)A.values.size();

    /* CPU reference */
    std::vector<float> C_ref;
    cpu_diag_multiply(A, B, n, C_ref);

    /* Run benchmarks */
    BenchResult r_diag = bench_diagspmm(A, B, n);
    r_diag.pass = verify_result(r_diag.C_dense.data(), C_ref.data(), n, tol);

    BenchResult r_hm = bench_paper_hm(A, B, n);
    r_hm.pass = verify_result(r_hm.C_dense.data(), C_ref.data(), n, tol);

    BenchResult r_csp = bench_cusparse(A, B, n);
    r_csp.pass = verify_result(r_csp.C_dense.data(), C_ref.data(), n, tol);

    /* Print row */
    printf("  %-6d  %-6d  %-7d  %12.4f  %12.4f  %12.4f  %-6s  %-6s  %-6s\n",
           n, nd, nnzA,
           r_diag.gpu_ms, r_hm.gpu_ms, r_csp.gpu_ms,
           r_diag.pass ? "PASS" : "FAIL",
           r_hm.pass   ? "PASS" : "FAIL",
           r_csp.pass  ? "PASS" : "FAIL");
}

static void
print_table_header()
{
    printf("  %-6s  %-6s  %-7s  %12s  %12s  %12s  %-6s  %-6s  %-6s\n",
           "n", "diags", "nnzA", "DiagSpMM", "Paper-HM", "cuSPARSE",
           "DSP", "PHM", "cuSP");
    printf("  %-6s  %-6s  %-7s  %12s  %12s  %12s  %-6s  %-6s  %-6s\n",
           "", "", "", "(ms)", "(ms)", "(ms)",
           "pass?", "pass?", "pass?");
    printf("  ");
    for (int i = 0; i < 86; ++i) printf("-");
    printf("\n");
}

int main(int argc, char** argv)
{
    int cli_n = 0;         /* 0 = use preset list */
    int cli_diags = 21;
    NUM_ITERS = 100;

    /* Parse CLI arguments */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            cli_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            cli_diags = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            NUM_ITERS = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Ensure diags is odd */
    if (cli_diags % 2 == 0) {
        cli_diags++;
        fprintf(stderr, "  Warning: diag count rounded up to odd: %d\n", cli_diags);
    }
    if (NUM_ITERS < 1) NUM_ITERS = 1;

    printf("================================================================\n");
    printf("  SpMM Benchmark: DiagSpMM vs Paper-HM vs cuSPARSE\n");
    printf("  Iterations per test: %d (average reported)\n", NUM_ITERS);
    printf("================================================================\n\n");

    /* Print GPU info */
    {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        printf("  GPU: %s (%d SMs, %.0f MHz)\n\n",
               prop.name, prop.multiProcessorCount,
               prop.clockRate / 1000.0);
    }

    const float tol = 1e-2f;

    print_table_header();

    if (cli_n > 0) {
        /* Single user-specified test case */
        run_test({cli_n, cli_diags}, tol);
    } else {
        /* Preset test suite */
        std::vector<TestCase> cases = {
            { 512,  11},
            { 512,  21},
            {1024,  11},
            {1024,  21},
            {1024,  41},
            {2048,  11},
            {2048,  21},
            {2048,  41},
            {4096,  11},
            {4096,  21},
            {4096,  41},
        };
        for (const auto& tc : cases)
            run_test(tc, tol);
    }

    printf("\n  Notes:\n");
    printf("  - Times are kernel-only (cudaEvent), excluding preprocessing/transfer.\n");
    printf("  - Each benchmark: 1 warmup + %d timed iterations (average reported).\n", NUM_ITERS);
    printf("  - Verification tolerance: %.0e\n", tol);

    printf("\n================================================================\n");
    printf("  Done.\n");
    printf("================================================================\n");

    return 0;
}
