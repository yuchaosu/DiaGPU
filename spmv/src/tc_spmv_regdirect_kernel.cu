/* ============================================================
 * tc_spmv_regdirect_kernel.cu
 *
 * Register-direct TF32 tensor-core diagonal SpMV.
 *
 * The diagonal of a dense A*B is y, with the two mechanics
 * borrowed from Drawloom that a uniform band benefits from:
 *
 *   1. raw `mma.sync.m16n8k8.row.col.f32.tf32.tf32.f32` PTX, with
 *      A (Recon) and B (x) streamed global -> registers.  No
 *      shared-memory staging, no load_matrix_sync, no __syncwarp.
 *   2. the 16 diagonal outputs are read directly from the
 *      accumulator registers using the fixed m16n8k8 C-fragment
 *      lane map — no store_matrix_sync, no smem round-trip.
 *
 * Mapping (identical to the dense kernel):
 *   A[m,k] = Recon[dk+k, tile_row+m+d_{dk+k}]      (16 x 8)
 *   Bt[k,n]= x[tile_row+n     + d_{dk+k}]          ( 8 x 8)
 *   Bb[k,n]= x[tile_row+n + 8 + d_{dk+k}]          ( 8 x 8)
 *   C_top  = A*Bt   -> y[tile_row + 0..7]  = C_top[r , r]
 *   C_bot  = A*Bb   -> y[tile_row + 8..15] = C_bot[8+i, i]
 *
 * m16n8k8 fragment layout (PTX ISA), groupID g = lane>>2,
 * thread-in-group t = lane&3:
 *   A : a0=A[g,t]   a1=A[g+8,t]   a2=A[g,t+4]   a3=A[g+8,t+4]
 *   B : b0=B[t,g]   b1=B[t+4,g]
 *   C : c0=C[g,2t]  c1=C[g,2t+1]  c2=C[g+8,2t]  c3=C[g+8,2t+1]
 *
 * So the wanted diagonal entries live in exactly the lanes with
 * g == 2t  (-> c0 / c2) or g == 2t+1 (-> c1 / c3); those 8 lanes
 * write the 16 results straight to global.
 *
 * Requires sm_80+ for the tf32 MMA.
 * ============================================================ */

#include "dia_reconstruct.cuh"

#include <cstdint>

constexpr int WARPS_PER_BLOCK = 4;   /* 128 threads: better occupancy than 1 warp */

/* float -> tf32 (round to nearest, packed in the low bits of a .b32). */
__device__ __forceinline__ uint32_t to_tf32(float f)
{
    uint32_t u;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(f));
    return u;
}

/* mma.sync.m16n8k8 tf32, accumulating in place. */
__device__ __forceinline__ void mma_m16n8k8_tf32(
    float* acc, const uint32_t* A, const uint32_t* B)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]));
}

__global__ void tc_spmv_regdirect_kernel(
    ReconView    R,
    const float* x,
    int          x_size,
    float*       y)
{
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp =
        static_cast<int>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5);
    const int tile_row = global_warp * MMA_M;          /* 16 rows / warp */
    if (tile_row >= R.rows) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g    = lane >> 2;                         /* groupID    0..7 */
    const int t    = lane & 3;                          /* in-group   0..3 */

    float acc_top[4] = {0.f, 0.f, 0.f, 0.f};
    float acc_bot[4] = {0.f, 0.f, 0.f, 0.f};

    /* Walk the diagonal axis in MMA_K (=8) wide batches. */
    for (int dk = 0; dk < R.num_diags; dk += MMA_K) {
        const int di0 = dk + t;        /* diagonals this lane touches */
        const int di4 = dk + t + 4;
        const int off0 = (di0 < R.num_diags) ? R.diag_offsets[di0] : 0;
        const int off4 = (di4 < R.num_diags) ? R.diag_offsets[di4] : 0;
        const bool v0 = (di0 < R.num_diags);
        const bool v4 = (di4 < R.num_diags);

        /* --- A fragment: A[m,k] = Recon[di, tile_row+m+d_di] --- */
        uint32_t A[4] = {0u, 0u, 0u, 0u};
        if (v0) {
            int c = tile_row + g + off0;
            if (c >= 0 && c < R.cols) A[0] = to_tf32(R.values[(size_t)di0 * R.cols + c]);
            c = tile_row + g + 8 + off0;
            if (c >= 0 && c < R.cols) A[1] = to_tf32(R.values[(size_t)di0 * R.cols + c]);
        }
        if (v4) {
            int c = tile_row + g + off4;
            if (c >= 0 && c < R.cols) A[2] = to_tf32(R.values[(size_t)di4 * R.cols + c]);
            c = tile_row + g + 8 + off4;
            if (c >= 0 && c < R.cols) A[3] = to_tf32(R.values[(size_t)di4 * R.cols + c]);
        }

        /* --- B_top: Bt[k,n] = x[tile_row+n+d_di]  (n = g) --- */
        uint32_t Bt[2] = {0u, 0u};
        if (v0) { int xi = tile_row + g + off0;     if (xi >= 0 && xi < x_size) Bt[0] = to_tf32(x[xi]); }
        if (v4) { int xi = tile_row + g + off4;     if (xi >= 0 && xi < x_size) Bt[1] = to_tf32(x[xi]); }

        /* --- B_bot: Bb[k,n] = x[tile_row+n+8+d_di] --- */
        uint32_t Bb[2] = {0u, 0u};
        if (v0) { int xi = tile_row + g + 8 + off0; if (xi >= 0 && xi < x_size) Bb[0] = to_tf32(x[xi]); }
        if (v4) { int xi = tile_row + g + 8 + off4; if (xi >= 0 && xi < x_size) Bb[1] = to_tf32(x[xi]); }

        mma_m16n8k8_tf32(acc_top, A, Bt);
        mma_m16n8k8_tf32(acc_bot, A, Bb);
    }

    /* --- Register-resident diagonal extraction (no smem). ---
     * Lanes with g==2t hold C[g,g] in c0 and C[g+8,g] in c2;
     * lanes with g==2t+1 hold them in c1 and c3. */
    if (g == 2 * t) {
        const int r_top = tile_row + g;
        const int r_bot = tile_row + 8 + g;
        if (r_top < R.rows) y[r_top] = acc_top[0];   /* C_top[g , g] */
        if (r_bot < R.rows) y[r_bot] = acc_bot[2];   /* C_bot[g+8, g] */
    } else if (g == 2 * t + 1) {
        const int r_top = tile_row + g;
        const int r_bot = tile_row + 8 + g;
        if (r_top < R.rows) y[r_top] = acc_top[1];
        if (r_bot < R.rows) y[r_bot] = acc_bot[3];
    }
}

void launch_tc_spmv_regdirect(
    ReconView    R,
    const float* d_x,
    int          x_size,
    float*       d_y,
    cudaStream_t stream)
{
    const int n_tiles = (R.rows + MMA_M - 1) / MMA_M;
    if (n_tiles == 0) return;
    const int n_blocks = (n_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    tc_spmv_regdirect_kernel<<<n_blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
        R, d_x, x_size, d_y);
}

/* ------------------------------------------------------------------ *
 * Fused real+imag SpMV:  yr = H*xr ,  yi = H*xi  (same real H).
 *
 * The recon (A) fragment is loaded ONCE per (warp, diagonal-batch) and
 * fed to FOUR MMAs — top/bottom x real/imag — so the dominant recon
 * traffic is read once instead of twice across two separate launches.
 * At large q the matrix (D*N floats) dwarfs the vectors, so this ~halves
 * the DRAM traffic of a complex apply. Single-state NV=2 batch (see
 * dailyNote/2026-06-21.md §5); a wider TC-SpMM needs block/multi-state.
 * Costs more registers (4 accumulators) -> watch occupancy.
 * ------------------------------------------------------------------ */
__global__ void tc_spmv_regdirect_fused_kernel(
    ReconView    R,
    const float* xr,
    const float* xi,
    int          x_size,
    float*       yr,
    float*       yi)
{
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp =
        static_cast<int>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5);
    const int tile_row = global_warp * MMA_M;
    if (tile_row >= R.rows) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    float acc_top_r[4] = {0.f, 0.f, 0.f, 0.f};
    float acc_bot_r[4] = {0.f, 0.f, 0.f, 0.f};
    float acc_top_i[4] = {0.f, 0.f, 0.f, 0.f};
    float acc_bot_i[4] = {0.f, 0.f, 0.f, 0.f};

    for (int dk = 0; dk < R.num_diags; dk += MMA_K) {
        const int di0 = dk + t;
        const int di4 = dk + t + 4;
        const int off0 = (di0 < R.num_diags) ? R.diag_offsets[di0] : 0;
        const int off4 = (di4 < R.num_diags) ? R.diag_offsets[di4] : 0;
        const bool v0 = (di0 < R.num_diags);
        const bool v4 = (di4 < R.num_diags);

        /* --- A fragment: loaded once, reused for real AND imag. --- */
        uint32_t A[4] = {0u, 0u, 0u, 0u};
        if (v0) {
            int c = tile_row + g + off0;
            if (c >= 0 && c < R.cols) A[0] = to_tf32(R.values[(size_t)di0 * R.cols + c]);
            c = tile_row + g + 8 + off0;
            if (c >= 0 && c < R.cols) A[1] = to_tf32(R.values[(size_t)di0 * R.cols + c]);
        }
        if (v4) {
            int c = tile_row + g + off4;
            if (c >= 0 && c < R.cols) A[2] = to_tf32(R.values[(size_t)di4 * R.cols + c]);
            c = tile_row + g + 8 + off4;
            if (c >= 0 && c < R.cols) A[3] = to_tf32(R.values[(size_t)di4 * R.cols + c]);
        }

        /* --- B fragments for both components (n = g, and n = g+8). --- */
        uint32_t Bt_r[2] = {0u, 0u}, Bb_r[2] = {0u, 0u};
        uint32_t Bt_i[2] = {0u, 0u}, Bb_i[2] = {0u, 0u};
        if (v0) {
            int xt = tile_row + g + off0;
            if (xt >= 0 && xt < x_size) { Bt_r[0] = to_tf32(xr[xt]); Bt_i[0] = to_tf32(xi[xt]); }
            int xb = tile_row + g + 8 + off0;
            if (xb >= 0 && xb < x_size) { Bb_r[0] = to_tf32(xr[xb]); Bb_i[0] = to_tf32(xi[xb]); }
        }
        if (v4) {
            int xt = tile_row + g + off4;
            if (xt >= 0 && xt < x_size) { Bt_r[1] = to_tf32(xr[xt]); Bt_i[1] = to_tf32(xi[xt]); }
            int xb = tile_row + g + 8 + off4;
            if (xb >= 0 && xb < x_size) { Bb_r[1] = to_tf32(xr[xb]); Bb_i[1] = to_tf32(xi[xb]); }
        }

        mma_m16n8k8_tf32(acc_top_r, A, Bt_r);
        mma_m16n8k8_tf32(acc_bot_r, A, Bb_r);
        mma_m16n8k8_tf32(acc_top_i, A, Bt_i);
        mma_m16n8k8_tf32(acc_bot_i, A, Bb_i);
    }

    if (g == 2 * t) {
        const int r_top = tile_row + g;
        const int r_bot = tile_row + 8 + g;
        if (r_top < R.rows) { yr[r_top] = acc_top_r[0]; yi[r_top] = acc_top_i[0]; }
        if (r_bot < R.rows) { yr[r_bot] = acc_bot_r[2]; yi[r_bot] = acc_bot_i[2]; }
    } else if (g == 2 * t + 1) {
        const int r_top = tile_row + g;
        const int r_bot = tile_row + 8 + g;
        if (r_top < R.rows) { yr[r_top] = acc_top_r[1]; yi[r_top] = acc_top_i[1]; }
        if (r_bot < R.rows) { yr[r_bot] = acc_bot_r[3]; yi[r_bot] = acc_bot_i[3]; }
    }
}

void launch_tc_spmv_regdirect_fused(
    ReconView    R,
    const float* d_xr,
    const float* d_xi,
    int          x_size,
    float*       d_yr,
    float*       d_yi,
    cudaStream_t stream)
{
    const int n_tiles = (R.rows + MMA_M - 1) / MMA_M;
    if (n_tiles == 0) return;
    const int n_blocks = (n_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    tc_spmv_regdirect_fused_kernel<<<n_blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
        R, d_xr, d_xi, x_size, d_yr, d_yi);
}
