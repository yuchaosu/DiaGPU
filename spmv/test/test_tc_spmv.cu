/* ============================================================
 * test_tc_spmv.cu
 *
 * GPU correctness harness for tc_spmv_regdirect_kernel.
 *
 *   1. Build a banded dense H + random x.
 *   2. dense_to_dia(H) -> build_recon(DIA) -> Recon  (real code).
 *   3. Upload Recon + x, run launch_tc_spmv_regdirect -> y_gpu.
 *   4. Compare y_gpu against:
 *        - y_recon : CPU spmv_from_recon (same identity as kernel)
 *        - y_ref   : dense H * x reference
 *
 * The kernel multiplies in TF32, so the pass threshold is a
 * relative tolerance (~1e-2), not exact equality.
 *
 * Build:
 *   nvcc -std=c++17 -arch=sm_90 test_tc_spmv.cu \
 *        ../src/tc_spmv_regdirect_kernel.cu -o test_tc_spmv
 * ============================================================ */

#include "../src/dia_reconstruct.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                         cudaGetErrorString(err__), __FILE__, __LINE__);   \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

/* ---- DIA build + CPU reference (mirror of test_reconstruct.cpp) ---- */
static DiaMatrix dense_to_dia(const float* H, int rows, int cols)
{
    DiaMatrix D;
    D.rows = rows;
    D.cols = cols;
    for (int d = -(rows - 1); d <= cols - 1; ++d) {
        const int sr  = (d >= 0) ? 0 : -d;
        const int sc  = (d >= 0) ? d : 0;
        const int len = std::min(rows - sr, cols - sc);
        if (len <= 0) continue;

        bool nz = false;
        for (int j = 0; j < len; ++j)
            if (H[(sr + j) * cols + (sc + j)] != 0.0f) { nz = true; break; }
        if (!nz) continue;

        D.offsets.push_back(d);
        D.diag_starts.push_back(static_cast<int>(D.values.size()));
        D.diag_lengths.push_back(len);
        for (int j = 0; j < len; ++j)
            D.values.push_back(H[(sr + j) * cols + (sc + j)]);
    }
    return D;
}

static std::vector<float> spmv_from_recon(const ReconMatrix& R,
                                          const std::vector<float>& x)
{
    std::vector<float> y(R.rows, 0.0f);
    for (int r = 0; r < R.rows; ++r) {
        float s = 0.0f;
        for (int k = 0; k < R.num_diags; ++k) {
            const int d = R.diag_offsets[k];
            const int c = r + d;
            if (c >= 0 && c < R.cols)
                s += R.values[static_cast<size_t>(k) * R.cols + c] * x[c];
        }
        y[r] = s;
    }
    return y;
}

/* ---- one test case ---- */
static bool run_case(const char* name, int N,
                     const std::vector<int>& offsets)
{
    /* deterministic band matrix + x */
    std::vector<float> H(static_cast<size_t>(N) * N, 0.0f);
    unsigned seed = 0x1234567u + static_cast<unsigned>(N);
    auto rng = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>((seed >> 8) & 0xFFFF) / 65536.0f * 2.0f - 1.0f;
    };
    for (int r = 0; r < N; ++r)
        for (int d : offsets) {
            const int c = r + d;
            if (c >= 0 && c < N) H[static_cast<size_t>(r) * N + c] = rng();
        }
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = rng();

    /* CPU references */
    std::vector<float> y_ref(N, 0.0f);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            y_ref[r] += H[static_cast<size_t>(r) * N + c] * x[c];

    DiaMatrix   dia = dense_to_dia(H.data(), N, N);
    ReconMatrix R   = build_recon(dia);
    std::vector<float> y_recon = spmv_from_recon(R, x);

    /* upload */
    int   *d_off = nullptr;
    float *d_val = nullptr, *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_off, R.diag_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, R.values.size()       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,   x.size()              * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,   N                     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_off, R.diag_offsets.data(),
                          R.diag_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, R.values.data(),
                          R.values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(float)));

    ReconView RV{ R.rows, R.cols, R.num_diags, d_off, d_val };

    float ymax = 1e-30f;
    for (float v : y_ref) ymax = std::max(ymax, std::fabs(v));

    auto check = [&](const char* tag,
                     void (*launch)(ReconView, const float*, int, float*, cudaStream_t)) {
        CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(float)));
        launch(RV, d_x, static_cast<int>(x.size()), d_y, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> y_gpu(N);
        CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float err_recon = 0.0f, err_ref = 0.0f;
        for (int i = 0; i < N; ++i) {
            err_recon = std::max(err_recon, std::fabs(y_gpu[i] - y_recon[i]));
            err_ref   = std::max(err_ref,   std::fabs(y_gpu[i] - y_ref[i]));
        }
        const float rel = err_ref / ymax;
        const bool  ok  = rel < 1e-2f;   /* TF32 tolerance */
        std::printf("  %-16s %-10s N=%-4d diags=%2d  |gpu-recon|=%.2e  "
                    "rel|gpu-ref|=%.2e  %s\n",
                    name, tag, N, R.num_diags, err_recon, rel,
                    ok ? "[OK]" : "[FAIL]");
        return ok;
    };

    bool ok = true;
    ok &= check("regdirect", launch_tc_spmv_regdirect);

    cudaFree(d_off); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    return ok;
}

int main()
{
    std::printf("TF32 tensor-core diagonal SpMV — GPU correctness\n");
    bool all = true;

    /* tridiagonal, exact tile multiple */
    all &= run_case("tridiagonal",      48, {-1, 0, 1});
    /* tridiagonal, ragged last tile (not a multiple of 16) */
    all &= run_case("tridiag ragged",   37, {-1, 0, 1});
    /* pentadiagonal */
    all &= run_case("pentadiagonal",    64, {-2, -1, 0, 1, 2});
    /* 9 diagonals > MMA_K=8 -> exercises the k-batch loop */
    all &= run_case("9-diag wide band",  80, {-9, -5, -2, -1, 0, 1, 2, 5, 9});
    /* asymmetric / off-center band */
    all &= run_case("asymmetric band",   71, {-3, 0, 2, 4, 7});
    /* single diagonal (main only) */
    all &= run_case("main diagonal",     16, {0});

    std::printf("\n%s\n", all ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all ? 0 : 1;
}
