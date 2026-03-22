/* ============================================================
 * diag_batched_kernel.cu
 *
 * Output-diagonal-batched SpMM.
 *
 * Each warp: 32 threads × BATCH_K output diags.
 * A loaded ONCE per A diagonal, reused K times.
 * Register accumulation.  Zero atomics.  Zero shared memory.
 *
 * Per-FMA cost:
 *   ~0 A read (amortized over K, hits L1 from pre-read)
 *   1 B global read
 *   1 register FMA
 *   0 atomics
 * ============================================================ */

#include "diag_batched_kernel.cuh"

/* ============================================================
 * Device helpers
 * ============================================================ */
__device__ __forceinline__ int
bat_lower_bound(const int* __restrict__ arr, int n, int val)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < val) lo = mid + 1;
        else                hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int
bat_upper_bound(const int* __restrict__ arr, int n, int val)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] <= val) lo = mid + 1;
        else                 hi = mid;
    }
    return lo;
}

/* ============================================================
 * BATCHED KERNEL
 *
 * Work item = (d_c_batch, pos_tile).
 * Flattened to 1D index for grid-stride scheduling.
 *
 * Each warp handles one work item:
 *   - 32 threads → 32 consecutive row positions
 *   - BATCH_K consecutive output diagonals
 *   - BATCH_K register accumulators
 *
 * Inner loop: for each A diagonal in the UNION range,
 *   load a_val ONCE, then unrolled loop over K output diags.
 *   #pragma unroll ensures acc[k] stays in registers.
 * ============================================================ */
__global__ void
__launch_bounds__(128, 8)
diag_spmm_batched_kernel(BatchedArgs args)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_cta = blockDim.x / WARP_SIZE;  /* 4 */

    const int global_warp = static_cast<int>(blockIdx.x) * warps_per_cta
                          + warp_id;
    const int total_warps = static_cast<int>(gridDim.x) * warps_per_cta;

    const int n     = args.n;
    const int n_m_1 = n - 1;

    for (int item = global_warp; item < args.total_items; item += total_warps) {
        /* Decode work item → (d_c_batch, pos_tile). */
        const int batch_idx = item / args.num_pos_tiles;
        const int tile_idx  = item % args.num_pos_tiles;

        const int d_c_base = args.d_c_min + batch_idx * BATCH_K;
        const int p_begin  = tile_idx * WARP_SIZE;
        const int row      = p_begin + lane_id;

        if (row >= n) continue;

        /* K register accumulators — kept in registers via #pragma unroll. */
        float acc[BATCH_K];
        #pragma unroll
        for (int k = 0; k < BATCH_K; ++k) acc[k] = 0.0f;

        /* How many valid output diags in this batch? */
        const int d_c_end = min(d_c_base + BATCH_K, args.d_c_max + 1);
        const int batch_size = d_c_end - d_c_base;

        /* UNION A range for this batch:
         * Valid d_a satisfies d_a + d_b = d_c for some d_c in [d_c_base, d_c_end)
         * and d_b in [B_min, B_max].
         * → d_a in [d_c_base - B_max, d_c_end - 1 - B_min]. */
        const int d_a_lo = d_c_base - args.B_offset_max;
        const int d_a_hi = d_c_end - 1 - args.B_offset_min;

        const int ai_begin = bat_lower_bound(
            args.A_offsets, args.A_num_diags, d_a_lo);
        const int ai_end = bat_upper_bound(
            args.A_offsets, args.A_num_diags, d_a_hi);

        /* Iterate A diagonals in the union range. */
        for (int ai = ai_begin; ai < ai_end; ++ai) {
            const int d_a  = args.A_offsets[ai];
            const int a_sr = (d_a >= 0) ? 0 : -d_a;
            const int p_a  = row - a_sr;

            /* Load A value ONCE — reused across K output diags. */
            const float a_val = (p_a >= 0 && p_a < args.A_lengths[ai])
                              ? args.A_values[args.A_starts[ai] + p_a]
                              : 0.0f;

            if (a_val == 0.0f) continue;

            /* Which of the K output diags does this A diagonal contribute to?
             * d_c = d_a + d_b, d_b in [B_min, B_max]
             * → d_c in [d_a + B_min, d_a + B_max]
             * Intersect with [d_c_base, d_c_base + batch_size): */
            const int k_lo = max(0, d_a + args.B_offset_min - d_c_base);
            const int k_hi = min(batch_size, d_a + args.B_offset_max - d_c_base + 1);

            const int k_row_plus_da = row + d_a;  /* = k in A*B = column of A */

            /* Unrolled inner loop over K output diags.
             * #pragma unroll ensures acc[k] is register-allocated.
             * The branch (k >= k_lo && k < k_hi) compiles to predication. */
            #pragma unroll
            for (int k = 0; k < BATCH_K; ++k) {
                if (k >= k_lo && k < k_hi) {
                    const int d_c = d_c_base + k;
                    const int d_b = d_c - d_a;

                    const int bi = args.B_diag_lookup[d_b + n_m_1];
                    /* For symmetric contiguous offsets, bi is always valid
                     * within [k_lo, k_hi). Check kept for generality. */
                    if (bi >= 0) {
                        const int b_sr = (d_b >= 0) ? 0 : -d_b;
                        const int p_b  = k_row_plus_da - b_sr;

                        const float b_val =
                            (p_b >= 0 && p_b < args.B_lengths[bi])
                            ? args.B_values[args.B_starts[bi] + p_b]
                            : 0.0f;

                        acc[k] += a_val * b_val;
                    }
                }
            }
        }

        /* Write back K accumulators to C diagonal storage. */
        #pragma unroll
        for (int k = 0; k < BATCH_K; ++k) {
            if (k < batch_size && acc[k] != 0.0f) {
                const int d_c = d_c_base + k;
                const int c_sr = (d_c >= 0) ? 0 : -d_c;
                const int p_c  = row - c_sr;
                const int idx  = d_c + n_m_1;

                if (idx >= 0 && idx < 2 * n - 1) {
                    const int c_start = args.C_val_starts[idx];
                    const int c_len   = args.C_diag_lens[idx];
                    if (c_start >= 0 && p_c >= 0 && p_c < c_len)
                        args.C_values[c_start + p_c] = acc[k];
                }
            }
        }
    }
}

/* ============================================================
 * Launch wrapper
 * ============================================================ */
static int bat_get_sm_count() {
    static int cached = 0;
    if (cached == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached,
                               cudaDevAttrMultiProcessorCount, dev);
    }
    return cached;
}

void launch_batched_kernel(BatchedArgs args, cudaStream_t stream)
{
    if (args.total_items == 0) return;

    cudaFuncSetAttribute(diag_spmm_batched_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0);

    const int block_size = 128;          /* 4 warps per CTA */
    const int warps_per_cta = block_size / WARP_SIZE;
    int sm = bat_get_sm_count();
    int grid_size = sm * 8;              /* target 8 CTAs per SM */

    int min_ctas = (args.total_items + warps_per_cta - 1) / warps_per_cta;
    if (grid_size > min_ctas) grid_size = min_ctas;

    diag_spmm_batched_kernel<<<grid_size, block_size, 0, stream>>>(args);
}
