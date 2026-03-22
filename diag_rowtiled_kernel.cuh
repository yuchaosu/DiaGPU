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
 *   Output diagonals are processed in chunks of D.
 *   For each chunk:
 *     1. Zero smem accumulators: float acc[D][R]
 *     2. For each A diagonal d_a:
 *          load a_val (register, coalesced, REUSED across B iters)
 *          For each B diagonal d_b producing d_c in this chunk:
 *            load b_val (coalesced)
 *            smem[d_c - chunk_lo][tid] += a_val * b_val
 *     3. Write back: direct store to C diagonal format
 *
 * Requires: A_offsets and B_offsets sorted ascending.
 *           Symmetric offsets (A.offsets == B.offsets) for best
 *           performance, but works for any sorted offsets.
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

    /* Chunk size for shared memory tiling (= B_num_diags typically) */
    int          chunk_d;
};

/* ============================================================
 * Kernel declaration
 *
 * Dynamic shared memory: chunk_d * blockDim.x * sizeof(float)
 * ============================================================ */
__global__ void
diag_spmm_rowtiled_kernel(RowTiledArgs args);

/* ============================================================
 * Host launch wrapper
 * ============================================================ */
void launch_rowtiled_kernel(RowTiledArgs args,
                            int block_size = 128,
                            cudaStream_t stream = 0);
