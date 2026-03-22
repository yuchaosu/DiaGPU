/* ============================================================
 * diag_rowtiled_kernel.cuh
 *
 * Row-tiled, A-stationary SpMM kernel for diagonal storage.
 *
 * KEY PROPERTIES:
 *   - ZERO atomics (each CTA exclusively owns a block of rows)
 *   - A-load amortization: each A value loaded once per chunk,
 *     reused across all B diagonal iterations
 *   - Shared-memory accumulation across output diagonals
 *   - Coalesced A/B reads and C writes
 *
 * DESIGN:
 *   Each CTA owns R consecutive rows (R = blockDim.x).
 *   Phase 1: Pre-load ALL A values into shared memory (read-only).
 *   Phase 2: Iterate output diags with sliding window:
 *     For each d_c:
 *       acc (register) += smemA[ai][tid] * B_values[...]
 *       Write acc to C (direct, zero atomics)
 *
 *   smem is READ-ONLY (A cache, ~5 cyc/read).
 *   Accumulation is in REGISTERS (~1 cyc/FMA).
 *   A loaded once into smem, reused across all output diagonals.
 *   Sliding window: zero binary search (O(1) amortized per d_c).
 *
 * Requires: A_offsets and B_offsets sorted ascending.
 * ============================================================ */
#pragma once

#include "diag_types.cuh"
#include <cuda_runtime.h>

/* ============================================================
 * RowTiledArgs — everything the rowtiled kernel needs.
 * Passed by value → constant memory.
 * ============================================================ */
struct RowTiledArgs {
    /* Matrix dimension */
    int n;

    /* A diagonal metadata (sorted ascending by offset) */
    const float* A_values;
    const int*   A_offsets;
    const int*   A_starts;
    const int*   A_lengths;
    int          A_num_diags;

    /* B diagonal metadata (sorted ascending by offset) */
    const float* B_values;
    const int*   B_offsets;
    const int*   B_starts;
    const int*   B_lengths;
    int          B_num_diags;

    /* B diagonal lookup: B_diag_lookup[d_b + (n-1)] = bi or -1 */
    const int*   B_diag_lookup;

    /* Output C — diagonal format.
     * C_val_starts[d_c + (n-1)] = starting offset in C_values
     *   for output diagonal d_c, or -1 if not present.
     * C_diag_lens[d_c + (n-1)]  = length of diagonal d_c.     */
    float*       C_values;
    const int*   C_val_starts;   /* size 2n-1 */
    const int*   C_diag_lens;    /* size 2n-1 */

    /* Output diagonal range */
    int          d_c_min;        /* smallest output diagonal offset */
    int          d_c_max;        /* largest output diagonal offset */
    int          num_out_diags;  /* = d_c_max - d_c_min + 1 */
};

/* ============================================================
 * Kernel declaration
 *
 * Dynamic shared memory: A_num_diags * blockDim.x * sizeof(float)
 * ============================================================ */
__global__ void
diag_spmm_rowtiled_kernel(RowTiledArgs args);

/* ============================================================
 * Host launch wrapper
 * ============================================================ */
void launch_rowtiled_kernel(RowTiledArgs args,
                            int block_size = 128,
                            cudaStream_t stream = 0);
