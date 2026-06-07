/* ============================================================
 * tc_spmv_dense_kernel.cu
 *
 * Tensor-core SpMV kernel driven by the single Recon (K x N)
 * matrix built in dia_reconstruct.cuh.  One CTA = one warp =
 * one output tile of MMA_M (= 16) rows.  No second operand is
 * pre-built; the kernel gathers the corresponding x values
 * inline.
 *
 * For a tile starting at row tile_row, the SpMV identity
 *
 *   y[tile_row + r] = sum_k Recon[k, tile_row + r + d_k]
 *                            * x[tile_row + r + d_k]
 *
 * is realised by a matrix multiply C = A * B with
 *
 *   A[r, k] = Recon[k, tile_row + r + d_k]      (M x K)
 *   B[k, c] = x[tile_row + c + d_k]             (K x N)
 *
 * Then  C[r, c] = sum_k Recon[k, ...] * x[tile_row + c + d_k]
 * and the diagonal C[r, r] equals y[tile_row + r] — the off-
 * diagonal entries are cross-row products we discard.
 *
 * The nvcuda::wmma TF32 path only supports the m16n16k8 shape,
 * so the accumulator is a full 16 x 16 tile.  N = 16 >= M = 16
 * means a SINGLE MMA per k-batch covers all 16 diagonal entries
 * y[tile_row + 0 .. + 15] — no top/bot split is needed.
 *
 * The diagonals spanning > MMA_K diagonals are handled by
 * batching the k axis in steps of MMA_K, accumulating into c.
 *
 * Per-CTA shared memory:
 *   A_smem : 16 *  8 = 128 floats
 *   B_smem :  8 * 16 = 128 floats
 *   C_smem : 16 * 16 = 256 floats     (output extraction)
 * Total = 512 floats = 2048 bytes per CTA.
 *
 * Requires sm_80+ (Ampere / Hopper) for wmma::precision::tf32.
 * ============================================================ */

#include "dia_reconstruct.cuh"

#include <mma.h>
using namespace nvcuda;

constexpr int A_SZ     = MMA_M * MMA_K;   /* 16 x  8 */
constexpr int B_SZ     = MMA_K * MMA_N;   /*  8 x 16 */
constexpr int C_SZ     = MMA_M * MMA_N;   /* 16 x 16 */
constexpr int CTA_SMEM = A_SZ + B_SZ + C_SZ;

/* ============================================================
 * Kernel
 * ============================================================ */
__global__ void tc_spmv_dense_kernel(
    ReconView    R,
    const float* x,
    int          x_size,
    float*       y)
{
    __shared__ float smem[CTA_SMEM];
    float* A_smem = smem;
    float* B_smem = A_smem + A_SZ;
    float* C_smem = B_smem + B_SZ;

    const int tile_row = static_cast<int>(blockIdx.x) * MMA_M;
    if (tile_row >= R.rows) return;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    /* ====================================================
     * Walk the diagonal axis in MMA_K-wide batches.
     * ==================================================== */
    for (int dk = 0; dk < R.num_diags; dk += MMA_K) {

        /* ----------------------------------------------------
         * Gather A_smem  (16 x 8, row-major)
         *   A_smem[r * MMA_K + kk] = Recon[dk+kk, tile_row+r+d_{dk+kk}]
         * 128 elements / 32 lanes = 4 per lane.
         * ---------------------------------------------------- */
        #pragma unroll
        for (int e = 0; e < A_SZ / 32; ++e) {
            const int f  = e * 32 + lane;
            const int r  = f / MMA_K;
            const int kk = f % MMA_K;
            const int di = dk + kk;

            float val = 0.0f;
            if (di < R.num_diags) {
                const int d   = R.diag_offsets[di];
                const int col = tile_row + r + d;
                if (col >= 0 && col < R.cols)
                    val = R.values[di * R.cols + col];
            }
            A_smem[r * MMA_K + kk] = val;
        }

        /* ----------------------------------------------------
         * Gather B_smem  (8 x 16, row-major)
         *   B_smem[kk * MMA_N + c] = x[tile_row + c + d_{dk+kk}]
         * 128 elements / 32 lanes = 4 per lane.
         * ---------------------------------------------------- */
        #pragma unroll
        for (int e = 0; e < B_SZ / 32; ++e) {
            const int f  = e * 32 + lane;
            const int kk = f / MMA_N;
            const int c  = f % MMA_N;
            const int di = dk + kk;

            float val = 0.0f;
            if (di < R.num_diags) {
                const int d  = R.diag_offsets[di];
                const int xi = tile_row + c + d;
                if (xi >= 0 && xi < x_size) val = x[xi];
            }
            B_smem[kk * MMA_N + c] = val;
        }
        __syncwarp();

        /* ----------------------------------------------------
         * Load fragments and issue one MMA.
         *   a_frag (row-major, ldm = MMA_K)
         *   b_frag (row-major, ldm = MMA_N)
         * TF32 fragments require explicit float -> tf32 rounding
         * of each element before the multiply.
         * ---------------------------------------------------- */
        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K,
                       wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K,
                       wmma::precision::tf32, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, A_smem, MMA_K);
        wmma::load_matrix_sync(b_frag, B_smem, MMA_N);

        #pragma unroll
        for (int t = 0; t < a_frag.num_elements; ++t)
            a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
        #pragma unroll
        for (int t = 0; t < b_frag.num_elements; ++t)
            b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    /* ====================================================
     * Diagonal extraction.
     *   y[tile_row + r] = C[r, r]   for r in [0, 16)
     * store_matrix_sync stores row-major with ldm = MMA_N.
     * ==================================================== */
    wmma::store_matrix_sync(C_smem, c_frag, MMA_N, wmma::mem_row_major);
    if (lane < MMA_M) {
        const int r = tile_row + lane;
        if (r < R.rows) y[r] = C_smem[lane * MMA_N + lane];
    }
}

/* ============================================================
 * Host launcher.
 * ============================================================ */
void launch_tc_spmv_dense(
    ReconView    R,
    const float* d_x,
    int          x_size,
    float*       d_y,
    cudaStream_t stream)
{
    const int n_tiles = (R.rows + MMA_M - 1) / MMA_M;
    if (n_tiles == 0) return;
    tc_spmv_dense_kernel<<<n_tiles, 32, 0, stream>>>(R, d_x, x_size, d_y);
}
