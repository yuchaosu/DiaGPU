/* ============================================================
 * dia_spmv.cu
 *
 * DIA SpMV using Tensor Core MMA (WMMA m16n8k8, TF32).
 *
 * Diagonal classification (preprocessing):
 *   For each output tile [tile_row, tile_row+TILE), only the
 *   diagonals that actually overlap the tile are included in
 *   the task's diagonal list.  Corner tiles at the top/bottom
 *   of the matrix have fewer contributing diagonals.  Tiles
 *   with zero contributing diagonals produce no task.
 *
 * MMA layout — "diagonals as rows" (fits DIA format):
 *
 *   In the TC B matrix, each ROW corresponds to one contributing
 *   diagonal of A.  The row contains K_mma=8 consecutive values
 *   from that diagonal.  This is the natural DIA access pattern:
 *   A_vals[d][p_base .. p_base+7] is a coalesced 8-element read.
 *
 *   In the TC A matrix (matrix_a), each ROW corresponds to one
 *   output position.  The row holds the K_mma=8 x values needed
 *   by each of the K_mma contributing diagonals:
 *     A_TC[pos][d] = x[tile_row + pos + off[d]]
 *   (stored col-major: adjacent threads read the same diagonal's
 *    x-slice → 16 coalesced x reads per diagonal).
 *
 * MMA computation:
 *   C[pos][pos'] = Σ_d  A_TC[pos][d] * B[d][pos']
 *               = Σ_d  x[tile_row+pos+off_d] * A_vals[d][p_d+pos']
 *
 *   Diagonal extraction:
 *     C_top[j][j]   = y[tile_row + j]     for j = 0..7
 *     C_bot[j+8][j] = y[tile_row + 8 + j] for j = 0..7
 *
 *   Two passes per batch: B_top holds A values for positions 0..7,
 *   B_bot holds positions 8..15.  A_TC (x values) is loaded once
 *   and shared across both mma_sync calls.
 *
 * Shared memory per warp (static, 6 KB total for 4 warps):
 *   A_TC_smem : MMA_K × MMA_M floats = 8×16  (col-major, x values)
 *   B_top     : MMA_K × MMA_N floats = 8×8   (row-major, A values pos 0..7)
 *   B_bot     : MMA_K × MMA_N floats = 8×8   (row-major, A values pos 8..15)
 *   C_smem    : MMA_M × MMA_N floats = 16×8  (output extraction)
 *
 * Requires sm_80+ (Ampere / Hopper) for wmma::precision::tf32.
 * ============================================================ */

#include "dia_spmv.cuh"

#include <mma.h>
using namespace nvcuda;

/* ============================================================
 * Kernel
 * ============================================================ */
__global__ void __launch_bounds__(SPMV_BLOCK, SPMV_BLOCKS_PER_SM)
dia_spmv_kernel(SpMVArgs args)
{
    /* Static shared memory: 4 warps × 384 floats = 6 KB */
    __shared__ float smem[WARPS_PER_BLOCK * WS_TOT];

    const int tid  = static_cast<int>(threadIdx.x);
    const int wid  = tid >> 5;    /* warp index 0..3   */
    const int lane = tid & 31;    /* lane 0..31        */

    const int task_id = static_cast<int>(blockIdx.x);
    if (task_id >= args.n_tasks) return;

    const SpMVTask task     = args.tasks[task_id];
    const int      tile_row = task.tile_row;
    const int      n_d      = task.diag_count;
    const int*     d_list   = args.diag_list + task.diag_begin;

    /* Per-warp smem:
     *   A_s  : x values  (col-major, cols = diagonals, rows = output positions)
     *   Bt_s : A values  (row-major, rows = diagonals, cols = positions 0..7)
     *   Bb_s : A values  (row-major, rows = diagonals, cols = positions 8..15)
     *   C_s  : MMA output (16×8 float, row-major)                             */
    float* A_s  = smem + wid * WS_TOT;
    float* Bt_s = A_s  + WS_A;
    float* Bb_s = Bt_s + WS_B;
    float* C_s  = Bb_s + WS_B;

    for (int grp = 0; grp < GROUPS_PER_WARP; ++grp) {

        const int r_base = tile_row + wid * ROWS_PER_WARP + grp * MMA_M;

        wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_top, c_bot;
        wmma::fill_fragment(c_top, 0.0f);
        wmma::fill_fragment(c_bot, 0.0f);

        for (int dk = 0; dk < n_d; dk += MMA_K) {

            /* ====================================================
             * 1. Load A_TC (x values) into smem — col-major
             *
             *    A_TC_smem[col * MMA_M + row]
             *      col  = diagonal slot  0..MMA_K-1
             *      row  = output position 0..MMA_M-1
             *      value = x[r_base + row + off[d_list[dk + col]]]
             *
             *    WS_A = MMA_K * MMA_M = 128 elements, 4 per thread.
             *    Thread 0..15 (e=0, col=0): read x[r_base+0..15+off_d0]
             *    — 16 consecutive x elements → coalesced.
             * ==================================================== */
            #pragma unroll
            for (int e = 0; e < WS_A / 32; ++e) {
                const int f   = e * 32 + lane;
                const int col = f / MMA_M;   /* diagonal slot   0..7  */
                const int row = f % MMA_M;   /* output position 0..15 */

                float val = 0.0f;
                const int di = dk + col;
                if (di < n_d) {
                    const int d     = d_list[di];
                    const int off   = args.A_offsets[d];
                    const int xrow  = r_base + row;
                    const int x_idx = xrow + off;
                    if (xrow < args.rows && x_idx >= 0 && x_idx < args.cols)
                        val = args.x[x_idx];
                }
                A_s[f] = val;   /* A_TC_smem[col*MMA_M + row] */
            }

            /* ====================================================
             * 2. Load B_top and B_bot (A values) into smem — row-major
             *
             *    Each ROW of B = one contributing diagonal of A.
             *    This fits DIA format: A_vals[d][p..p+7] is a
             *    contiguous 8-element read per diagonal.
             *
             *    B_top_smem[diag * MMA_N + j] = A_vals[d][p_d + j]
             *      j = 0..7  (positions within top half, 0..7)
             *    B_bot_smem[diag * MMA_N + j] = A_vals[d][p_d + 8 + j]
             *      j = 0..7  (positions within bottom half, 8..15)
             *
             *    2 * WS_B = 128 elements total, 4 per thread.
             *    Thread 0..7 (e=0, top, diag=0): read A_vals[d0][p..p+7]
             *    — 8 consecutive diagonal values → coalesced.
             * ==================================================== */
            #pragma unroll
            for (int e = 0; e < 2 * WS_B / 32; ++e) {
                const int f    = e * 32 + lane;
                const bool top = (f < WS_B);
                const int  g   = top ? f : f - WS_B;     /* 0..63 in section */
                const int  diag = g / MMA_N;              /* diagonal slot 0..7 */
                const int  j    = g % MMA_N;              /* col 0..7           */

                float val = 0.0f;
                const int di = dk + diag;
                if (di < n_d) {
                    const int d   = d_list[di];
                    const int off = args.A_offsets[d];
                    const int sr  = (off >= 0) ? 0 : -off;

                    /* Position in the diagonal for the j-th column of B:
                     *   B_top col j → output position (r_base + j)
                     *   B_bot col j → output position (r_base + MMA_N + j)  */
                    const int out_pos = top ? (r_base + j) : (r_base + MMA_N + j);
                    const int p       = out_pos - sr;   /* position in diagonal */
                    const int a_len   = args.A_lengths[d];

                    if (p >= 0 && p < a_len && out_pos < args.rows)
                        val = args.A_vals[args.A_starts[d] + p];
                }
                (top ? Bt_s : Bb_s)[g] = val;
            }

            /* Smem writes visible to all lanes */
            __syncwarp();

            /* ====================================================
             * 3. Load WMMA fragments
             *
             *    matrix_a (col-major, ldm = MMA_M = 16):
             *      A_TC[pos][diag] = A_s[diag * MMA_M + pos] = x value
             *
             *    matrix_b (row-major, ldm = MMA_N = 8):
             *      B[diag][j] = B_s[diag * MMA_N + j] = A value
             *      Each row = one diagonal → "diagonal in rows".
             * ==================================================== */
            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K,
                           wmma::precision::tf32, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K,
                           wmma::precision::tf32, wmma::row_major> bt_frag, bb_frag;

            wmma::load_matrix_sync(a_frag,   A_s,  MMA_M);  /* col-major, ldm=16 */
            wmma::load_matrix_sync(bt_frag,  Bt_s, MMA_N);  /* row-major, ldm=8  */
            wmma::load_matrix_sync(bb_frag,  Bb_s, MMA_N);  /* row-major, ldm=8  */

            /* ====================================================
             * 4. MMA accumulate
             *
             *    c_top[j][j] = Σ_d x[r+j+off_d] * A_vals[d][p_d+j]
             *                = y[r_base + j]            j = 0..7
             *
             *    c_bot[j+8][j] = Σ_d x[r+j+8+off_d] * A_vals[d][p_d+8+j]
             *                  = y[r_base + 8 + j]      j = 0..7
             * ==================================================== */
            wmma::mma_sync(c_top, a_frag, bt_frag, c_top);
            wmma::mma_sync(c_bot, a_frag, bb_frag, c_bot);
        }

        /* ====================================================
         * 5. Extract results from TC accumulator
         *
         *    store_matrix_sync (row-major, ldm = MMA_N = 8):
         *      C_s[row * 8 + col] = C[row][col]
         *
         *    Top half:    C_s[j*8 + j]        = y[r_base + j]
         *    Bottom half: C_s[(j+8)*8 + j]    = y[r_base + 8 + j]
         * ==================================================== */
        wmma::store_matrix_sync(C_s, c_top, MMA_N, wmma::mem_row_major);
        if (lane < MMA_N) {
            const int my_row = r_base + lane;
            if (my_row < args.rows)
                args.y[my_row] = C_s[lane * MMA_N + lane];          /* C[j][j]   */
        }

        /* store_matrix_sync is a warp barrier; all lanes complete
         * the top-half read before the second store overwrites C_s. */
        wmma::store_matrix_sync(C_s, c_bot, MMA_N, wmma::mem_row_major);
        if (lane < MMA_N) {
            const int my_row = r_base + MMA_N + lane;
            if (my_row < args.rows)
                args.y[my_row] = C_s[(MMA_N + lane) * MMA_N + lane]; /* C[j+8][j] */
        }
    }
}

/* ============================================================
 * HOST: build_spmv_plan
 *
 * Diagonal classification: for each output tile, enumerate
 * the diagonals that have at least one element overlapping it.
 *
 * Diagonal d overlaps tile [r_begin, r_end) iff:
 *   r_begin < sr + len   AND   r_end > sr
 * where sr = max(0, -off) and len = A_lengths[d].
 * ============================================================ */
SpMVPlan build_spmv_plan(const DiaSpmvMatrix& A)
{
    SpMVPlan plan;
    const int n_tiles = (A.rows + SPMV_TILE - 1) / SPMV_TILE;
    int diag_ptr = 0;

    for (int t = 0; t < n_tiles; ++t) {
        const int r_begin = t * SPMV_TILE;
        const int r_end   = r_begin + SPMV_TILE;

        std::vector<int> contrib;
        for (int d = 0; d < A.num_diags; ++d) {
            const int off = A.offsets[d];
            const int sr  = (off >= 0) ? 0 : -off;
            const int len = A.diag_lengths[d];
            if (r_begin < sr + len && r_end > sr)
                contrib.push_back(d);
        }

        if (contrib.empty()) continue;

        SpMVTask task;
        task.tile_row   = r_begin;
        task.diag_begin = diag_ptr;
        task.diag_count = static_cast<int>(contrib.size());
        plan.tasks.push_back(task);
        for (int idx : contrib) plan.diag_list.push_back(idx);
        diag_ptr += task.diag_count;
    }
    return plan;
}

/* ============================================================
 * HOST: launch_dia_spmv
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
                     cudaStream_t    stream)
{
    if (plan.tasks.empty()) return;

    SpMVArgs args;
    args.rows      = rows;
    args.cols      = cols;
    args.tasks     = d_tasks;
    args.n_tasks   = static_cast<int>(plan.tasks.size());
    args.diag_list = d_diag_list;
    args.A_vals    = d_A_vals;
    args.A_offsets = d_A_offsets;
    args.A_starts  = d_A_starts;
    args.A_lengths = d_A_lengths;
    args.x         = d_x;
    args.y         = d_y;

    dia_spmv_kernel<<<args.n_tasks, SPMV_BLOCK, 0, stream>>>(args);
}
