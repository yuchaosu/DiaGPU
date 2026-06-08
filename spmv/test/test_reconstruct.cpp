/* ============================================================
 * test_reconstruct.cpp
 *
 * Pure-C++ test for the diagonal reconstruction declared in
 * dia_reconstruct.cuh.  The DiaMatrix / ReconMatrix types and
 * the build_recon() function are copied verbatim (CUDA bits
 * stripped) so this test exercises the SAME code the kernel
 * relies on.
 *
 * The test pipeline:
 *   1. ORIGINAL  H_dense   -> DIA storage
 *   2. CONSTRUCT  build_recon(DIA) -> Recon
 *   3. RESULT    spmv_from_recon(Recon, x) -> y, vs. H_dense * x
 *   4. DECONSTRUCT  Recon -> H_dense  (inverse rearrangement)
 *
 * Build:
 *   g++ -std=c++17 -O2 test_reconstruct.cpp -o test_reconstruct
 * ============================================================ */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

/* =====================================================================
 *      ---- BEGIN  code copied from dia_reconstruct.cuh ----
 * Identical to the .cuh contents (sans CUDA headers / device qualifiers).
 * If the .cuh changes, mirror the change here.
 * ===================================================================== */

constexpr int MMA_M  = 16;
constexpr int MMA_N  =  8;
constexpr int MMA_K  =  8;
constexpr int TILE_M = MMA_M;

struct DiaMatrix {
    int                rows;
    int                cols;
    std::vector<int>   offsets;
    std::vector<int>   diag_starts;
    std::vector<int>   diag_lengths;
    std::vector<float> values;
};

struct ReconMatrix {
    int                rows;          /* = H.rows */
    int                cols;          /* = H.cols */
    int                num_diags;
    std::vector<int>   diag_offsets;  /* DESCENDING */
    std::vector<float> values;        /* num_diags * cols, row-major */
};

inline ReconMatrix build_recon(const DiaMatrix& H)
{
    ReconMatrix R;
    R.rows         = H.rows;
    R.cols         = H.cols;
    R.num_diags    = static_cast<int>(H.offsets.size());
    R.diag_offsets = H.offsets;
    std::sort(R.diag_offsets.begin(), R.diag_offsets.end(),
              std::greater<int>());
    R.values.assign(static_cast<size_t>(R.num_diags) * R.cols, 0.0f);

    for (int k = 0; k < R.num_diags; ++k) {
        const int d = R.diag_offsets[k];

        int di = -1;
        for (int i = 0; i < static_cast<int>(H.offsets.size()); ++i)
            if (H.offsets[i] == d) { di = i; break; }
        if (di < 0) continue;

        const int sc   = (d >= 0) ? d : 0;
        const int len  = H.diag_lengths[di];
        const int base = H.diag_starts[di];
        for (int j = 0; j < len; ++j) {
            const int col = sc + j;
            R.values[static_cast<size_t>(k) * R.cols + col] =
                H.values[base + j];
        }
    }
    return R;
}

/* =====================================================================
 *      ---- END  code copied from dia_reconstruct.cuh ----
 * ===================================================================== */

/* ============================================================
 * dense_to_dia  — bring a dense H into the same DIA storage
 * format that build_recon expects.
 * ============================================================ */
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

/* ============================================================
 * spmv_from_recon  — SpMV that mirrors what the TC kernel
 * computes:  y[r] = sum_k Recon[k, r + d_k] * x[r + d_k].
 * ============================================================ */
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
                s += R.values[static_cast<size_t>(k) * R.cols + c]
                     * x[c];
        }
        y[r] = s;
    }
    return y;
}

/* ============================================================
 * recon_to_dense  — inverse of build_recon.  Walks each
 * Recon[k, c] back to H[c - d_k, c] to recover the dense H.
 * Demonstrates the rearrangement is invertible (no info loss).
 * ============================================================ */
static std::vector<float> recon_to_dense(const ReconMatrix& R)
{
    std::vector<float> H(static_cast<size_t>(R.rows) * R.cols, 0.0f);
    for (int k = 0; k < R.num_diags; ++k) {
        const int d = R.diag_offsets[k];
        for (int c = 0; c < R.cols; ++c) {
            const int r_orig = c - d;
            if (r_orig >= 0 && r_orig < R.rows)
                H[static_cast<size_t>(r_orig) * R.cols + c] =
                    R.values[static_cast<size_t>(k) * R.cols + c];
        }
    }
    return H;
}

/* ---------------- pretty-print helpers ---------------- */
static void print_matrix(const char* label, const float* M,
                         int rows, int cols)
{
    std::printf("%s  [%d x %d]\n", label, rows, cols);
    for (int r = 0; r < rows; ++r) {
        std::printf("  ");
        for (int c = 0; c < cols; ++c)
            std::printf("%6.2f ", M[r * cols + c]);
        std::printf("\n");
    }
    std::printf("\n");
}

static void print_vector(const char* label, const float* v, int n)
{
    std::printf("%s  [%d]\n  ", label, n);
    for (int i = 0; i < n; ++i) std::printf("%6.2f ", v[i]);
    std::printf("\n\n");
}

/* ============================================================ */
int main()
{
    /* ---------- 1. ORIGINAL ---------- */
    const int N = 5;
    float H_dense[N * N] = {
        1, 2, 3, 0, 0,
        4, 5, 6, 7, 0,
        8, 9, 8, 7, 6,
        0, 5, 4, 3, 2,
        0, 0, 1, 2, 3,
    };
    float x_arr[N] = {1, 2, 3, 4, 5};
    std::vector<float> x(x_arr, x_arr + N);

    print_matrix("1. ORIGINAL  H_dense", H_dense, N, N);
    print_vector("   x", x.data(), N);

    /* reference y = H * x */
    std::vector<float> y_ref(N, 0.0f);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            y_ref[r] += H_dense[r * N + c] * x[c];

    /* ---------- 2. CONSTRUCT  via the real build_recon ---------- */
    DiaMatrix dia = dense_to_dia(H_dense, N, N);
    ReconMatrix R = build_recon(dia);

    std::printf("   DIA storage: %zu diagonals, offsets = {",
                dia.offsets.size());
    for (int o : dia.offsets) std::printf(" %d", o);
    std::printf(" }\n");
    std::printf("   Recon offsets (descending): {");
    for (int o : R.diag_offsets) std::printf(" %d", o);
    std::printf(" }\n\n");

    print_matrix("2. CONSTRUCT  Recon  =  build_recon(DIA)",
                 R.values.data(), R.num_diags, R.cols);

    /* ---------- 3. RESULT  via spmv_from_recon ---------- */
    std::vector<float> y_recon = spmv_from_recon(R, x);

    print_vector("3. RESULT  y from Recon", y_recon.data(), N);
    print_vector("   y_ref = H * x", y_ref.data(), N);

    float max_err_y = 0.0f;
    for (int i = 0; i < N; ++i)
        max_err_y = std::max(max_err_y,
                             std::fabs(y_recon[i] - y_ref[i]));
    std::printf("   max |y_recon - y_ref| = %.6f  %s\n\n",
                max_err_y, max_err_y < 1e-3f ? "[OK]" : "[FAIL]");

    /* ---------- 4. DECONSTRUCT  Recon -> dense H ---------- */
    std::vector<float> H_back = recon_to_dense(R);

    print_matrix("4. DECONSTRUCT  H recovered from Recon",
                 H_back.data(), N, N);

    float max_err_H = 0.0f;
    for (int i = 0; i < N * N; ++i)
        max_err_H = std::max(max_err_H,
                             std::fabs(H_back[i] - H_dense[i]));
    std::printf("   max |H_back - H_dense| = %.6f  %s\n",
                max_err_H, max_err_H < 1e-3f ? "[OK]" : "[FAIL]");

    return 0;
}
