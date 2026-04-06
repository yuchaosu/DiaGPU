/* ============================================================
 * dia_spmv.cuh
 *
 * Standalone GPU SpMV for DIA (diagonal) format: y = A * x
 *
 * Algorithmic design inspired by Drawloom (Zhang et al., PPoPP'26):
 *   "Exploiting Efficient Mapping and Pipelined Execution for
 *    Accelerating SpMV on Tensor Cores"
 *
 * Key techniques:
 *
 *   1. Per-tile diagonal classification (host preprocessing)
 *      For each output tile [tile_row, tile_row+TILE), only the
 *      diagonals whose value range overlaps the tile are included.
 *      Corner tiles (near matrix boundaries) have fewer contributing
 *      diagonals; tiles with zero contributing diagonals are skipped
 *      entirely.  Analogous to the paper's row classification that
 *      avoids mapping zero-padded rows onto TC blocks.
 *
 *   2. Tensor Core MMA accumulation (WMMA m16n8k8, TF32)
 *      For each output tile, one warp handles 32 rows as two
 *      16-row WMMA groups.  For each batch of MMA_K=8 contributing
 *      diagonals, the warp:
 *        (a) Cooperatively loads A values into smem (A[diag][row],
 *            col-major — adjacent threads read the same diagonal,
 *            giving coalesced A_vals[] access).
 *        (b) Cooperatively loads B_top[diag][j] = x[r+j+off] and
 *            B_bot[diag][j] = x[r+8+j+off] into smem.
 *        (c) Issues two wmma::mma_sync calls (one per top/bottom
 *            half) to accumulate into TC register fragments.
 *      After all batches, results are extracted from the diagonal
 *      of the 16×8 accumulator fragments:
 *        C_top[j][j] = y[r_base+j]    for j = 0..7
 *        C_bot[j+8][j] = y[r_base+8+j] for j = 0..7
 *      This matches the paper's ArbitWeave "diagonal result"
 *      extraction pattern.
 *
 * Shared memory per warp (static, 6 KB total for 4 warps):
 *   A_smem  : MMA_K × MMA_M floats = 8×16  (col-major)
 *   B_top   : MMA_K × MMA_N floats = 8×8   (row-major)
 *   B_bot   : MMA_K × MMA_N floats = 8×8   (row-major)
 *   C_smem  : MMA_M × MMA_N floats = 16×8  (output extraction)
 *
 * TC result efficiency:  E_res = 16 / (16×8) = 12.5% per MMA.
 * In DIA format each row has unique x values so the diagonal
 * extraction pattern (not stripe sharing) is unavoidable.
 * E_comp = 100% after classification (only non-zero positions
 * are staged in A_smem).
 *
 * Requires sm_80+ (Ampere / Hopper) for wmma::precision::tf32.
 * ============================================================ */
#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

/* ============================================================
 * Compile-time constants
 * ============================================================ */
constexpr int SPMV_TILE          = 128;   // rows per CTA
constexpr int SPMV_BLOCK         = 128;   // threads per block (= 4 warps)
constexpr int SPMV_BLOCKS_PER_SM =   4;  // target occupancy

/* WMMA shape (TF32, sm_80+) */
constexpr int MMA_M = 16;
constexpr int MMA_N =  8;
constexpr int MMA_K =  8;   /* diagonals processed per MMA call */

/* Derived warp layout */
constexpr int WARPS_PER_BLOCK   = SPMV_BLOCK / 32;          /* 4 */
constexpr int ROWS_PER_WARP     = SPMV_TILE / WARPS_PER_BLOCK; /* 32 */
constexpr int GROUPS_PER_WARP   = ROWS_PER_WARP / MMA_M;    /* 2 */

/* Smem layout per warp (floats):
 *   WS_A : A matrix  (col-major, MMA_K cols × MMA_M rows)
 *   WS_B : B_top / B_bot (row-major, MMA_K rows × MMA_N cols, each)
 *   WS_C : C output  (row-major, MMA_M rows × MMA_N cols)       */
constexpr int WS_A   = MMA_K * MMA_M;          /* 128 */
constexpr int WS_B   = MMA_K * MMA_N;          /*  64 */
constexpr int WS_C   = MMA_M * MMA_N;          /* 128 */
constexpr int WS_TOT = WS_A + 2 * WS_B + WS_C; /* 384 floats = 1536 bytes/warp */

constexpr int SPMV_SMEM_BYTES = WARPS_PER_BLOCK * WS_TOT * sizeof(float); /* 6144 */

/* ============================================================
 * Host-side DIA matrix (owning, variable-length diagonals)
 * ============================================================ */
struct DiaSpmvMatrix {
    int rows, cols, num_diags;
    std::vector<int>   offsets;
    std::vector<float> values;
    std::vector<int>   diag_starts;
    std::vector<int>   diag_lengths;

    static int diag_start_row(int d) { return d >= 0 ?  0 : -d; }
    static int diag_start_col(int d) { return d >= 0 ?  d :  0; }
    static int diag_length(int rows, int cols, int d) {
        int sr = diag_start_row(d), sc = diag_start_col(d);
        int len = std::min(rows - sr, cols - sc);
        return len > 0 ? len : 0;
    }
};

/* ============================================================
 * Task — one output tile with its contributing diagonal list
 * ============================================================ */
struct SpMVTask {
    int tile_row;    /* first row of tile                          */
    int diag_begin;  /* offset into diag_list[] on device          */
    int diag_count;  /* number of contributing diagonals           */
};

/* ============================================================
 * Kernel argument bundle (passed by value → constant memory)
 * ============================================================ */
struct SpMVArgs {
    int             rows, cols;
    const SpMVTask* tasks;
    int             n_tasks;
    const int*      diag_list;   /* flat contributing diagonal indices */
    const float*    A_vals;
    const int*      A_offsets;
    const int*      A_starts;
    const int*      A_lengths;
    const float*    x;
    float*          y;
};

/* ============================================================
 * Host-side plan (result of preprocessing)
 * ============================================================ */
struct SpMVPlan {
    std::vector<SpMVTask> tasks;
    std::vector<int>      diag_list;
};

/* ============================================================
 * Preprocessing: build per-tile contributing diagonal lists
 * ============================================================ */
SpMVPlan build_spmv_plan(const DiaSpmvMatrix& A);

/* ============================================================
 * Kernel declaration (uses static smem of SPMV_SMEM_BYTES)
 * ============================================================ */
__global__ void __launch_bounds__(SPMV_BLOCK, SPMV_BLOCKS_PER_SM)
dia_spmv_kernel(SpMVArgs args);

/* ============================================================
 * Host launch wrapper
 * ============================================================ */
void launch_dia_spmv(const SpMVPlan& plan,
                     const SpMVTask* d_tasks,
                     const int*      d_diag_list,
                     int             rows,
                     int             cols,
                     const float*    d_A_vals,
                     const int*      d_A_offsets,
                     const int*      d_A_starts,
                     const int*      d_A_lengths,
                     const float*    d_x,
                     float*          d_y,
                     cudaStream_t    stream = 0);
