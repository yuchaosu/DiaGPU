/* ============================================================
 * diag_rowtiled_kernel.cu
 *
 * Row-tiled, A-stationary SpMM with shared-memory A cache.
 *
 * KEY INSIGHT:
 *   Shared memory holds A values (READ-ONLY, loaded once).
 *   Accumulation stays in REGISTERS (fast, 1 cycle FMA).
 *
 *   Old row-tiled: smem for read-write accumulators = 20-30 cyc/FMA
 *   This kernel:   smem for read-only A cache       =  5 cyc/read
 *                  register for accumulation         =  1 cyc/FMA
 *
 * DESIGN:
 *   Each CTA owns R consecutive rows (R = blockDim.x).
 *   Pre-load ALL A diagonal values into shared memory (once).
 *   Iterate output diagonals with sliding window (zero binary search).
 *   For each output diagonal, register-accumulate:
 *     acc += smemA[ai][tid] * B_values[...]
 *   Write acc directly to C (zero atomics).
 *
 * PER-FMA COST:
 *   1 smem read (A):     ~5 cycles (pipelined, zero bank conflicts)
 *   1 global read (B):   ~200 cycles (pipelined)
 *   1 register FMA:      ~1 cycle
 *   0 atomics
 *
 * vs Paper HM:
 *   1 global read (B):   ~200 cycles
 *   1 atomic write (C):  ~4-100 cycles
 * ============================================================ */

#include "diag_rowtiled_kernel.cuh"

/* ============================================================
 * Device helpers
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
 * KERNEL: Shared-memory A cache + register accumulation
 *
 * Shared memory layout: smemA[ai * R + tid]
 *   - ai indexes A diagonals (0 .. A_num_diags-1)
 *   - tid indexes thread position (0 .. R-1)
 *   - Consecutive tid → consecutive addresses → zero bank conflicts
 *
 * Sliding window for A range:
 *   As d_c increments by 1, ai_begin and ai_end each advance
 *   by at most 1 (monotonic). Total advance across all d_c:
 *   O(A_num_diags). Amortized O(1) per d_c.
 *   → Zero binary search overhead.
 * ============================================================ */
__global__ void
diag_spmm_rowtiled_kernel(RowTiledArgs args)
{
    extern __shared__ float smemA[];  /* [A_num_diags][R] */

    const int R   = blockDim.x;
    const int tid = threadIdx.x;
    const int row = blockIdx.x * R + tid;
    const int n   = args.n;
    const int n_m_1 = n - 1;

    /* ---- Phase 1: Pre-load ALL A values into shared memory ---- */
    for (int ai = 0; ai < args.A_num_diags; ++ai) {
        float val = 0.0f;
        if (row < n) {
            int d_a = args.A_offsets[ai];
            int a_sr = (d_a >= 0) ? 0 : -d_a;
            int p_a  = row - a_sr;
            if (p_a >= 0 && p_a < args.A_lengths[ai])
                val = args.A_values[args.A_starts[ai] + p_a];
        }
        smemA[ai * R + tid] = val;
    }
    __syncthreads();

    if (row >= n) return;

    /* ---- Phase 2: Iterate output diagonals with sliding window ---- */

    /* Sliding window: valid A diagonal indices for current d_c.
     * A_offsets[ai] must be in [d_c - B_offset_max, d_c - B_offset_min].
     * As d_c increases by 1, the window shifts right by 1.
     * ai_begin and ai_end advance monotonically → O(1) amortized. */
    int ai_begin = 0;
    int ai_end   = 0;

    const int b_off_min = args.B_offsets[0];                      /* sorted ascending */
    const int b_off_max = args.B_offsets[args.B_num_diags - 1];

    for (int d_c = args.d_c_min; d_c <= args.d_c_max; ++d_c) {
        /* Advance upper bound: include A diags where d_a <= d_c - b_off_min */
        const int d_a_hi = d_c - b_off_min;
        while (ai_end < args.A_num_diags && args.A_offsets[ai_end] <= d_a_hi)
            ++ai_end;

        /* Advance lower bound: exclude A diags where d_a < d_c - b_off_max */
        const int d_a_lo = d_c - b_off_max;
        while (ai_begin < ai_end && args.A_offsets[ai_begin] < d_a_lo)
            ++ai_begin;

        if (ai_begin >= ai_end) continue;

        /* ---- Accumulate in register ---- */
        float acc = 0.0f;

        for (int ai = ai_begin; ai < ai_end; ++ai) {
            float a_val = smemA[ai * R + tid];  /* ~5 cycles, zero bank conflict */
            if (a_val == 0.0f) continue;

            int d_a = args.A_offsets[ai];
            int d_b = d_c - d_a;

            /* B diagonal lookup.  For symmetric contiguous offsets,
             * bi is guaranteed valid here (sliding window ensures it).
             * The lookup is kept for generality. */
            int bi = args.B_diag_lookup[d_b + n_m_1];
            if (bi < 0) continue;

            int b_sr = (d_b >= 0) ? 0 : -d_b;
            int p_b  = row + d_a - b_sr;

            float b_val = (p_b >= 0 && p_b < args.B_lengths[bi])
                        ? args.B_values[args.B_starts[bi] + p_b]
                        : 0.0f;

            acc += a_val * b_val;
        }

        /* ---- Write back to C diagonal storage ---- */
        if (acc != 0.0f) {
            int c_sr = (d_c >= 0) ? 0 : -d_c;
            int p_c  = row - c_sr;
            int idx  = d_c + n_m_1;
            if (idx >= 0 && idx < 2 * n - 1) {
                int c_start = args.C_val_starts[idx];
                int c_len   = args.C_diag_lens[idx];
                if (c_start >= 0 && p_c >= 0 && p_c < c_len)
                    args.C_values[c_start + p_c] = acc;
            }
        }
    }
}

/* ============================================================
 * Launch wrapper
 * ============================================================ */
void launch_rowtiled_kernel(RowTiledArgs args,
                            int block_size,
                            cudaStream_t stream)
{
    int grid = (args.n + block_size - 1) / block_size;

    /* Shared memory: A_num_diags * block_size * sizeof(float) */
    size_t smem_bytes = (size_t)args.A_num_diags * block_size * sizeof(float);

    /* Opt-in for > 48 KB shared memory if needed. */
    cudaFuncSetAttribute(diag_spmm_rowtiled_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes);

    /* Prefer larger shared memory carveout for this kernel. */
    cudaFuncSetAttribute(diag_spmm_rowtiled_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    diag_spmm_rowtiled_kernel<<<grid, block_size, smem_bytes, stream>>>(
        args);
}
