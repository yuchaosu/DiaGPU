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
 *   A[r, k] = Recon[k, tile_row + r + d_k]      (M x K_TC)
 *   B[k, c] = x[tile_row + c + d_k]             (K_TC x N)
 *
 * Then  C[r, c] = sum_k Recon[k, ...] * x[tile_row + c + d_k]
 * and the diagonal C[r, r] equals y[tile_row + r] — the off-
 * diagonal entries are cross-row products we discard.
 *
 * Because WMMA m16n8k8 only produces an (M x N) = (16 x 8)
 * accumulator per MMA, two MMAs cover the 16 diagonal entries
 * of the conceptual (16 x 16) C:
 *
 *   MMA #1  c_top += A * B_top   -> y[tile_row +  0 .. +  7]
 *   MMA #2  c_bot += A * B_bot   -> y[tile_row +  8 .. + 15]
 *
 * The diagonals span > MMA_K diagonals are handled by batching
 * the k axis in steps of MMA_K, accumulating into c_top / c_bot.
 *
 * Per-CTA shared memory:
 *   A_smem  : 16 * 8 = 128 floats
 *   Bt_smem :  8 * 8 =  64 floats
 *   Bb_smem :  8 * 8 =  64 floats
 *   C_smem  : 16 * 8 = 128 floats     (output extraction)
 * Total = 384 floats = 1536 bytes per CTA.
 *
 * Requires sm_80+ (Ampere / Hopper) for wmma::precision::tf32.
 * ============================================================ */

#include "dia_reconstruct.cuh"

#include <mma.h>
using namespace nvcuda;

constexpr int A_SZ     = MMA_M * MMA_K;
constexpr int B_SZ     = MMA_K * MMA_N;
constexpr int C_SZ     = MMA_M * MMA_N;
constexpr int CTA_SMEM = A_SZ + 2 * B_SZ + C_SZ;

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
    float* A_smem  = smem;
    float* Bt_smem = A_smem  + A_SZ;
    float* Bb_smem = Bt_smem + B_SZ;
    float* C_smem  = Bb_smem + B_SZ;

    const int tile_row = static_cast<int>(blockIdx.x) * MMA_M;
    if (tile_row >= R.rows) return;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_top, c_bot;
    wmma::fill_fragment(c_top, 0.0f);
    wmma::fill_fragment(c_bot, 0.0f);

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
         * Gather B_top and B_bot  (8 x 8 each, row-major)
         *   Bt_smem[kk * MMA_N + c] = x[tile_row + c       + d_{dk+kk}]
         *   Bb_smem[kk * MMA_N + c] = x[tile_row + c + MMA_N + d_{dk+kk}]
         * 2 * 64 = 128 elements / 32 = 4 per lane.
         * ---------------------------------------------------- */
        #pragma unroll
        for (int e = 0; e < (2 * B_SZ) / 32; ++e) {
            const int f    = e * 32 + lane;
            const bool top = (f < B_SZ);
            const int  g   = top ? f : f - B_SZ;
            const int  kk  = g / MMA_N;
            const int  c   = g % MMA_N;
            const int  di  = dk + kk;

            float val = 0.0f;
            if (di < R.num_diags) {
                const int d  = R.diag_offsets[di];
                const int xi = tile_row + c + (top ? 0 : MMA_N) + d;
                if (xi >= 0 && xi < x_size) val = x[xi];
            }
            (top ? Bt_smem : Bb_smem)[g] = val;
        }
        __syncwarp();

        /* ----------------------------------------------------
         * Load fragments and issue two MMAs.
         *   a_frag  (row-major, ldm = MMA_K)
         *   bt/bb_frag (row-major, ldm = MMA_N)
         * ---------------------------------------------------- */
        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K,
                       wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K,
                       wmma::precision::tf32, wmma::row_major> bt_frag, bb_frag;

        wmma::load_matrix_sync(a_frag,  A_smem,  MMA_K);
        wmma::load_matrix_sync(bt_frag, Bt_smem, MMA_N);
        wmma::load_matrix_sync(bb_frag, Bb_smem, MMA_N);

        wmma::mma_sync(c_top, a_frag, bt_frag, c_top);
        wmma::mma_sync(c_bot, a_frag, bb_frag, c_bot);
    }

    /* ====================================================
     * Diagonal extraction.
     *   For r in [0, 8):    y[tile_row + r] = C_top[r, r]
     *   For i in [0, 8):    y[tile_row + 8 + i] = C_bot[8+i, i]
     * store_matrix_sync stores row-major with ldm = MMA_N.
     * ==================================================== */
    wmma::store_matrix_sync(C_smem, c_top, MMA_N, wmma::mem_row_major);
    if (lane < MMA_N) {
        const int r = tile_row + lane;
        if (r < R.rows) y[r] = C_smem[lane * MMA_N + lane];
    }
    wmma::store_matrix_sync(C_smem, c_bot, MMA_N, wmma::mem_row_major);
    if (lane < MMA_N) {
        const int r = tile_row + MMA_N + lane;
        if (r < R.rows) y[r] = C_smem[(MMA_N + lane) * MMA_N + lane];
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
