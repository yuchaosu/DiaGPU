/* ============================================================
 * dia_reconstruct.cuh
 *
 * Single-operand reconstruction of a band Hamiltonian for
 * tensor-core SpMV.  The compact DIA storage is lifted into a
 * dense (K x N) matrix Recon, where K is the number of active
 * diagonals and N is the matrix column count:
 *
 *   Recon[k, c]  =  H[c - d_k, c]    if 0 <= c - d_k < rows,
 *                =  0                otherwise.
 *
 *   diag_offsets are sorted DESCENDING (most positive d_k first,
 *   most negative last).  Each row of Recon is the d_k diagonal
 *   padded with zeros:
 *     - upper diagonals (d_k > 0): d_k zeros padded in FRONT
 *     - main          (d_k == 0): no padding
 *     - lower diagonals (d_k < 0): |d_k| zeros padded BEHIND
 *
 * Recon is the only matrix that has to be materialised — the
 * input vector x is consumed in place.  SpMV recovers y by
 *
 *   y[r] = sum_k Recon[k, r + d_k] * x[r + d_k]      (out-of-
 *                                                     range
 *                                                     terms drop)
 *
 * which the kernel evaluates as a tensor-core MMA whose
 * diagonal entries are the y values (see tc_spmv_dense_kernel.cu).
 * ============================================================ */
#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <functional>
#include <vector>

/* ============================================================
 * Tile + WMMA geometry (TF32, m16n16k8 — the only TF32 shape
 * the nvcuda::wmma API supports).
 *   One CTA = one warp = one output tile of MMA_M (= 16) rows.
 *   The K dimension is batched MMA_K (= 8) diagonals at a time.
 *   A single MMA per batch produces a 16 x 16 accumulator whose
 *   diagonal entries C[r, r] are the 16 output rows y[tile+r].
 * ============================================================ */
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K =  8;
constexpr int TILE_M = MMA_M;

/* ============================================================
 * Host-side compact DIA matrix (owning).
 * ============================================================ */
struct DiaMatrix {
    int                rows;
    int                cols;
    std::vector<int>   offsets;
    std::vector<int>   diag_starts;
    std::vector<int>   diag_lengths;
    std::vector<float> values;
};

/* ============================================================
 * Host-side reconstructed (K x cols) matrix.
 * ============================================================ */
struct ReconMatrix {
    int                rows;          /* = H.rows */
    int                cols;          /* = H.cols  (and ncols of values) */
    int                num_diags;
    std::vector<int>   diag_offsets;  /* [num_diags], DESCENDING */
    std::vector<float> values;        /* num_diags * cols, row-major */
};

/* ------------------------------------------------------------
 * build_recon
 *   Rearrange the DIA values into a (K x cols) dense matrix
 *   with rows sorted by diagonal offset descending.
 * ------------------------------------------------------------ */
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

        /* Find this diagonal in the input DIA storage. */
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

/* ============================================================
 * Device-side view of Recon (non-owning).
 * ============================================================ */
struct ReconView {
    int          rows;
    int          cols;
    int          num_diags;
    const int*   diag_offsets;   /* [num_diags] */
    const float* values;         /* num_diags * cols */
};

/* ============================================================
 * Kernel launcher  (implementation in tc_spmv_dense_kernel.cu).
 * ============================================================ */
void launch_tc_spmv_dense(
    ReconView    R,
    const float* d_x,
    int          x_size,
    float*       d_y,
    cudaStream_t stream = 0);

/* ============================================================
 * Register-direct variant (implementation in
 * tc_spmv_regdirect_kernel.cu).  Same diagonal mapping and
 * math as launch_tc_spmv_dense, but rewritten with the two
 * Drawloom-style mechanics that a band actually benefits from:
 *
 *   1. raw mma.sync.m16n8k8.f32.tf32 PTX (the native TF32 shape)
 *      with operands streamed global -> registers — NO shared-
 *      memory staging, no __syncwarp.
 *   2. the 16 diagonal results are read straight out of the
 *      accumulator registers (known lane/register map) — NO
 *      store_matrix_sync, no smem round-trip.
 *
 * Deliberately omitted (irrelevant to a uniform band): row
 * reordering / LSH densification, the deep cp.async pipeline
 * (only ceil(num_diags/8) batches to overlap), and 2:4 sparse
 * tensor cores.
 * ============================================================ */
void launch_tc_spmv_regdirect(
    ReconView    R,
    const float* d_x,
    int          x_size,
    float*       d_y,
    cudaStream_t stream = 0);
