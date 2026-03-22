/* ============================================================
 * diag_rowtiled_kernel.cu
 *
 * Row-tiled, A-stationary, shared-memory accumulation SpMM.
 *
 * Zero atomics.  Near-optimal memory traffic.
 * See diag_rowtiled_kernel.cuh for design overview.
 * ============================================================ */

#include "diag_rowtiled_kernel.cuh"

/* ============================================================
 * Device helpers: binary search on sorted offset array
 * ============================================================ */
__device__ __forceinline__ int
rt_lower_bound(const int* __restrict__ arr, int n, int val)
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
rt_upper_bound(const int* __restrict__ arr, int n, int val)
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
 * ROW-TILED KERNEL
 *
 * Each CTA owns blockDim.x consecutive rows.
 * Thread tid handles row = r_begin + tid.
 *
 * Outer loop: chunks of output diagonals (size chunk_d).
 * Middle loop: A diagonals (a_val loaded once, reused).
 * Inner loop: B diagonals producing d_c in this chunk.
 *
 * Shared memory layout: float acc[chunk_d][blockDim.x]
 *   acc[d_c_local][tid] accumulates C[row][col] where
 *   d_c = chunk_lo + d_c_local, col = row + d_c.
 *
 * No bank conflicts: consecutive tid → consecutive addresses.
 * ============================================================ */
__global__ void
diag_spmm_rowtiled_kernel(RowTiledArgs args)
{
    extern __shared__ float smem[];  /* acc[chunk_d][blockDim.x] */

    const int R   = blockDim.x;
    const int tid = threadIdx.x;
    const int r_begin = blockIdx.x * R;
    const int row = r_begin + tid;
    const int n   = args.n;
    const int n_m_1 = n - 1;

    /* Out-of-bounds threads still participate in syncthreads
     * but skip all loads and accumulation. */
    const bool active = (row < n);

    /* Process output diagonals in chunks. */
    for (int chunk_lo = args.d_c_min;
         chunk_lo <= args.d_c_max;
         chunk_lo += args.chunk_d)
    {
        const int chunk_hi = min(chunk_lo + args.chunk_d - 1,
                                 args.d_c_max);
        const int chunk_size = chunk_hi - chunk_lo + 1;

        /* Zero accumulators for this chunk. */
        for (int d = 0; d < chunk_size; ++d)
            smem[d * R + tid] = 0.0f;
        __syncthreads();

        if (active) {
            /* A-stationary: iterate A diagonals. */
            for (int ai = 0; ai < args.A_num_diags; ++ai) {
                const int d_a  = args.A_offsets[ai];
                const int a_sr = (d_a >= 0) ? 0 : -d_a;
                const int p_a  = row - a_sr;

                /* Load A value ONCE — reused across all B iters. */
                const float a_val =
                    (p_a >= 0 && p_a < args.A_lengths[ai])
                    ? args.A_values[args.A_starts[ai] + p_a]
                    : 0.0f;

                if (a_val == 0.0f) continue;  /* skip zero A */

                /* Which B diagonals produce d_c in [chunk_lo, chunk_hi]?
                 * d_c = d_a + d_b  →  d_b in [chunk_lo - d_a, chunk_hi - d_a]
                 * Binary search B_offsets (sorted) for this range. */
                const int d_b_lo = chunk_lo - d_a;
                const int d_b_hi = chunk_hi - d_a;
                const int bi_begin = rt_lower_bound(
                    args.B_offsets, args.B_num_diags, d_b_lo);
                const int bi_end = rt_upper_bound(
                    args.B_offsets, args.B_num_diags, d_b_hi);

                /* Inner loop: iterate matching B diagonals. */
                const int k = row + d_a;  /* column of A = row of B */

                for (int bi = bi_begin; bi < bi_end; ++bi) {
                    const int d_b  = args.B_offsets[bi];
                    const int b_sr = (d_b >= 0) ? 0 : -d_b;
                    const int p_b  = k - b_sr;

                    float b_val =
                        (p_b >= 0 && p_b < args.B_lengths[bi])
                        ? args.B_values[args.B_starts[bi] + p_b]
                        : 0.0f;

                    const int d_c_local = (d_a + d_b) - chunk_lo;
                    smem[d_c_local * R + tid] += a_val * b_val;
                }
            }
        }

        __syncthreads();

        /* Write back this chunk to C diagonal storage. */
        if (active) {
            for (int d = 0; d < chunk_size; ++d) {
                float val = smem[d * R + tid];
                if (val == 0.0f) continue;  /* skip zero output */

                int d_c = chunk_lo + d;
                int c_sr = (d_c >= 0) ? 0 : -d_c;
                int p_c  = row - c_sr;
                int idx  = d_c + n_m_1;

                if (idx >= 0 && idx < 2 * n - 1) {
                    int c_start = args.C_val_starts[idx];
                    int c_len   = args.C_diag_lens[idx];
                    if (c_start >= 0 && p_c >= 0 && p_c < c_len) {
                        args.C_values[c_start + p_c] = val;
                    }
                }
            }
        }

        __syncthreads();
    }
}

/* ============================================================
 * Launch wrapper
 * ============================================================ */
static int rt_get_sm_count() {
    static int cached = 0;
    if (cached == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached,
                               cudaDevAttrMultiProcessorCount, dev);
    }
    return cached;
}

void launch_rowtiled_kernel(RowTiledArgs args,
                            int block_size,
                            cudaStream_t stream)
{
    int grid = (args.n + block_size - 1) / block_size;

    /* Dynamic shared memory: chunk_d * block_size * sizeof(float) */
    size_t smem_bytes = args.chunk_d * block_size * sizeof(float);

    /* Opt-in for > 48 KB shared memory if needed. */
    cudaFuncSetAttribute(diag_spmm_rowtiled_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes);

    diag_spmm_rowtiled_kernel<<<grid, block_size, smem_bytes, stream>>>(
        args);
}
