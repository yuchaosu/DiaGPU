/* ============================================================
 * bench_cusparse.cu
 *
 * Throughput comparison: the TF32 tensor-core diagonal SpMV
 * kernel (launch_tc_spmv_dense) vs. cuSPARSE generic SpMV on
 * the SAME banded matrix.
 *
 * cuSPARSE's generic API has no DIA format, so the band is fed
 * to cuSPARSE as CSR (fp32).  Both consume identical numeric
 * values; the TC kernel rounds to TF32 internally, so its
 * result is checked against a CPU reference with a TF32-level
 * relative tolerance.
 *
 * The matrix is generated directly from its diagonal offsets
 * (no dense N x N buffer) so N can be large.
 *
 * Build:
 *   nvcc -std=c++17 -arch=sm_90 bench_cusparse.cu \
 *        tc_spmv_dense_kernel.cu -o bench_cusparse -lcusparse
 * ============================================================ */

#include "dia_reconstruct.cuh"

#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e__ = (call);                                          \
        if (e__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                         cudaGetErrorString(e__), __FILE__, __LINE__);     \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#define CUSPARSE_CHECK(call)                                               \
    do {                                                                   \
        cusparseStatus_t s__ = (call);                                     \
        if (s__ != CUSPARSE_STATUS_SUCCESS) {                              \
            std::fprintf(stderr, "cuSPARSE error %d (%s) at %s:%d\n",      \
                         (int)s__, cusparseGetErrorString(s__),            \
                         __FILE__, __LINE__);                              \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

/* Deterministic value for band entry H[r, r+d] and for x[i]. */
static inline float hval(int r, int d)
{
    unsigned h = static_cast<unsigned>(r) * 2654435761u
               ^ static_cast<unsigned>(d + 1024) * 40503u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return static_cast<float>(h & 0xFFFF) / 65536.0f * 2.0f - 1.0f;
}
static inline float xval(int i)
{
    unsigned h = static_cast<unsigned>(i) * 2246822519u;
    h ^= h >> 13; h *= 0x85ebca6bu; h ^= h >> 16;
    return static_cast<float>(h & 0xFFFF) / 65536.0f * 2.0f - 1.0f;
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    std::vector<int> offsets = {-9, -5, -2, -1, 0, 1, 2, 5, 9};
    std::sort(offsets.begin(), offsets.end());          /* ascending for CSR */
    const int num_diags = static_cast<int>(offsets.size());

    std::printf("Banded SpMV throughput  —  N = %d, diagonals = %d\n",
                N, num_diags);
    std::printf("offsets = {");
    for (int d : offsets) std::printf(" %d", d);
    std::printf(" }\n\n");

    /* ---------------- x ---------------- */
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = xval(i);

    /* ---------------- CSR (ascending columns per row) ---------------- */
    std::vector<int>   h_rowptr(N + 1, 0);
    std::vector<int>   h_colidx;
    std::vector<float> h_csrval;
    h_colidx.reserve(static_cast<size_t>(N) * num_diags);
    h_csrval.reserve(static_cast<size_t>(N) * num_diags);
    for (int r = 0; r < N; ++r) {
        for (int d : offsets) {
            const int c = r + d;
            if (c >= 0 && c < N) {
                h_colidx.push_back(c);
                h_csrval.push_back(hval(r, d));
            }
        }
        h_rowptr[r + 1] = static_cast<int>(h_colidx.size());
    }
    const int nnz = static_cast<int>(h_colidx.size());

    /* ---------------- CPU reference y = H * x ---------------- */
    std::vector<float> y_ref(N, 0.0f);
    for (int r = 0; r < N; ++r) {
        float s = 0.0f;
        for (int p = h_rowptr[r]; p < h_rowptr[r + 1]; ++p)
            s += h_csrval[p] * x[h_colidx[p]];
        y_ref[r] = s;
    }
    float ymax = 1e-30f;
    for (float v : y_ref) ymax = std::max(ymax, std::fabs(v));

    /* ---------------- Recon (descending offsets) ---------------- */
    ReconMatrix R;
    R.rows = N; R.cols = N; R.num_diags = num_diags;
    R.diag_offsets = offsets;
    std::sort(R.diag_offsets.begin(), R.diag_offsets.end(),
              std::greater<int>());
    R.values.assign(static_cast<size_t>(num_diags) * N, 0.0f);
    for (int k = 0; k < num_diags; ++k) {
        const int d = R.diag_offsets[k];
        for (int c = 0; c < N; ++c) {
            const int row = c - d;
            if (row >= 0 && row < N)
                R.values[static_cast<size_t>(k) * N + c] = hval(row, d);
        }
    }

    /* ---------------- device buffers ---------------- */
    int   *d_rowptr, *d_colidx, *d_off;
    float *d_csrval, *d_val, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_rowptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colidx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrval, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_off,    num_diags * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val,    R.values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,      N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rowptr, h_rowptr.data(), (N + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colidx, h_colidx.data(), nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrval, h_csrval.data(), nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off, R.diag_offsets.data(), num_diags * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, R.values.data(),
                          R.values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* ---------------- cuSPARSE setup ---------------- */
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, N, N, nnz,
                                     d_rowptr, d_colidx, d_csrval,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_32F));
    const float alpha = 1.0f, beta = 0.0f;
    size_t buf_sz = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &buf_sz));
    void* d_buf = nullptr;
    if (buf_sz) CUDA_CHECK(cudaMalloc(&d_buf, buf_sz));

    ReconView RV{ N, N, num_diags, d_off, d_val };

    auto verify = [&](const char* tag, float tol) {
        std::vector<float> y(N);
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float err = 0.0f;
        for (int i = 0; i < N; ++i)
            err = std::max(err, std::fabs(y[i] - y_ref[i]));
        const float rel = err / ymax;
        std::printf("    %-10s rel error = %.2e  %s\n", tag, rel,
                    rel < tol ? "[OK]" : "[FAIL]");
        return rel < tol;
    };

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    const int WARMUP = 20, ITERS = 200;

    auto time_ms = [&](std::function<void()> fn) {
        for (int i = 0; i < WARMUP; ++i) fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < ITERS; ++i) fn();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        return ms / ITERS;
    };

    /* ---------------- cuSPARSE ---------------- */
    CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(float)));
    CUSPARSE_CHECK(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, d_buf));
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok_cs = verify("cuSPARSE", 1e-4f);
    float ms_cs = time_ms([&]{
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                     vecX, &beta, vecY, CUDA_R_32F,
                     CUSPARSE_SPMV_CSR_ALG2, d_buf);
    });

    /* ---------------- TC kernel (dense / WMMA + smem) ---------------- */
    CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(float)));
    launch_tc_spmv_dense(RV, d_x, N, d_y, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok_tc = verify("TC dense", 1e-2f);
    float ms_tc = time_ms([&]{ launch_tc_spmv_dense(RV, d_x, N, d_y, 0); });

    /* ---------------- TC kernel (register-direct PTX) ---------------- */
    CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(float)));
    launch_tc_spmv_regdirect(RV, d_x, N, d_y, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok_rd = verify("TC regdirect", 1e-2f);
    float ms_rd = time_ms([&]{ launch_tc_spmv_regdirect(RV, d_x, N, d_y, 0); });

    /* ---------------- report ---------------- */
    const double useful_flop = 2.0 * nnz;                  /* per SpMV */
    const long   n_tiles  = (N + MMA_M - 1) / MMA_M;
    const long   n_batch  = (num_diags + MMA_K - 1) / MMA_K;
    const double issued_flop = 2.0 * n_tiles * n_batch
                             * (double)MMA_M * MMA_N * MMA_K;
    auto gflops = [&](double ms){ return useful_flop / (ms * 1e6); };

    std::printf("\n  nnz = %d  (%.2f%% dense)\n", nnz,
                100.0 * nnz / ((double)N * N));
    std::printf("  %-14s %9.4f ms   %8.2f GFLOP/s\n",
                "cuSPARSE",      ms_cs, gflops(ms_cs));
    std::printf("  %-14s %9.4f ms   %8.2f GFLOP/s\n",
                "TC dense",      ms_tc, gflops(ms_tc));
    std::printf("  %-14s %9.4f ms   %8.2f GFLOP/s\n",
                "TC regdirect",  ms_rd, gflops(ms_rd));
    std::printf("  speedup  regdirect / dense    = %.2fx\n", ms_tc / ms_rd);
    std::printf("  speedup  regdirect / cuSPARSE = %.2fx\n", ms_cs / ms_rd);
    std::printf("  TC MMA utilization = %.1f%%  (useful %.2e / issued %.2e flop)\n",
                100.0 * useful_flop / issued_flop, useful_flop, issued_flop);

    std::printf("\n%s\n",
                (ok_cs && ok_tc && ok_rd) ? "RESULTS VERIFIED"
                                          : "VERIFICATION FAILED");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(d_buf); cudaFree(d_rowptr); cudaFree(d_colidx); cudaFree(d_csrval);
    cudaFree(d_off); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    return (ok_cs && ok_tc && ok_rd) ? 0 : 1;
}
