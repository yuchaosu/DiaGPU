/* ============================================================
 * hm_optimized_kernel.cu
 *
 * Optimized HM: A-stationary, atomic writes, minimal inner loop.
 *
 * Grid structure: one kernel launch per A diagonal.
 *   → Zero binary search (thread knows its diagonal).
 *   → All threads in a warp process adjacent A positions
 *     on the same diagonal → perfectly coalesced.
 *
 * Shared memory: preloaded B/C metadata for the current A diagonal.
 *   b_sr[bi], b_start[bi], b_len[bi], c_base[bi]
 *   → Zero L1 reads in inner loop (all from smem, ~5 cycles).
 *
 * Inner loop per B diagonal (~6 ops):
 *   p_b = k - smem_b_sr[bi]
 *   if valid: b_val = B_vals[smem_b_start[bi] + p_b]
 *   atomicAdd(&C[smem_c_base[bi] + row], a_val * b_val)
 * ============================================================ */

#include "hm_optimized_kernel.cuh"

/* ============================================================
 * Kernel: processes all elements of ONE A diagonal.
 *
 * Block: 256 threads. Grid: ceil(A_diag_length / 256).
 * Each thread handles one A element on this diagonal.
 *
 * Shared memory holds B/C metadata for this A diagonal:
 *   smem_b_sr[bi], smem_b_start[bi], smem_b_len[bi], smem_c_base[bi]
 * Size: 4 * B_num_diags * sizeof(int) = ~3.2 KB for 201 diags.
 * ============================================================ */
__global__ void
__launch_bounds__(256, 4)
hm_optimized_kernel(HMOptArgs args, int ai)
{
    /* Shared memory: B metadata for this A diagonal. */
    extern __shared__ int smem[];
    int* smem_b_sr    = smem;
    int* smem_b_start = smem + args.B_num_diags;
    int* smem_b_len   = smem + args.B_num_diags * 2;
    int* smem_c_base  = smem + args.B_num_diags * 3;

    const int tid = threadIdx.x;
    const int B_nd = args.B_num_diags;

    /* Collaborative load of B/C metadata into shared memory. */
    for (int i = tid; i < B_nd; i += blockDim.x) {
        smem_b_sr[i]    = args.b_sr[i];
        smem_b_start[i] = args.b_start[i];
        smem_b_len[i]   = args.b_len[i];
        smem_c_base[i]  = args.c_base[ai * B_nd + i];
    }
    __syncthreads();

    /* This thread's A element. */
    const int a_len = args.A_lengths[ai];
    const int p_a   = blockIdx.x * blockDim.x + tid;
    if (p_a >= a_len) return;

    const int d_a  = args.A_offsets[ai];
    const int a_sr = (d_a >= 0) ? 0 : -d_a;
    const int a_sc = (d_a >= 0) ? d_a : 0;
    const int row  = a_sr + p_a;
    const int k    = a_sc + p_a;   /* column of A = row of B */

    const float a_val = args.A_values[args.A_starts[ai] + p_a];
    if (a_val == 0.0f) return;

    /* Inner loop: iterate B diagonals.
     * All metadata from shared memory (~5 cycle reads).
     * No binary search, no C_diag_lookup, no branches
     * except the p_b bounds check (predicated). */
    for (int bi = 0; bi < B_nd; ++bi) {
        const int c_wr = smem_c_base[bi];
        if (c_wr < 0) continue;  /* output diagonal doesn't exist */

        const int p_b = k - smem_b_sr[bi];
        if (p_b < 0 || p_b >= smem_b_len[bi]) continue;

        const float b_val = args.B_values[smem_b_start[bi] + p_b];

        /* c_wr = C_val_starts[d_c+n-1] - c_sr(d_c)
         * Final address = c_wr + row. */
        atomicAdd(&args.C_values[c_wr + row], a_val * b_val);
    }
}

/* ============================================================
 * Launch: one kernel per A diagonal.
 * ============================================================ */
void launch_hm_optimized(HMOptArgs args, cudaStream_t stream)
{
    const int block = 256;
    const size_t smem_bytes = 4 * args.B_num_diags * sizeof(int);

    for (int ai = 0; ai < args.A_num_diags; ++ai) {
        int a_len = args.A_lengths[ai];
        if (a_len == 0) continue;
        int grid = (a_len + block - 1) / block;
        hm_optimized_kernel<<<grid, block, smem_bytes, stream>>>(args, ai);
    }
}
