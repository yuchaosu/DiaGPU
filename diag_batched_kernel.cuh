/* ============================================================
 * diag_batched_kernel.cuh
 *
 * Output-diagonal-batched SpMM kernel.
 *
 * Each warp handles 32 positions across K consecutive output
 * diagonals simultaneously.  K register accumulators.
 * A value loaded ONCE per A diagonal, reused across K FMAs.
 *
 * Zero atomics.  Near-optimal A amortization.
 * Register-only accumulation (no shared memory).
 *
 * Requires: A_offsets sorted ascending.
 * ============================================================ */
#pragma once

#include "diag_types.cuh"
#include <cuda_runtime.h>

/* Batch size — number of output diagonals per warp.
 * K=32: 32 register accumulators + ~30 other = ~62 regs/thread.
 * High occupancy (8 CTAs/SM). L1 reuse via tile-major ordering. */
constexpr int BATCH_K = 32;

/* ============================================================
 * BatchedArgs — everything the batched kernel needs.
 * ============================================================ */
struct BatchedArgs {
    int n;

    /* A (sorted ascending by offset) */
    const float* A_values;
    const int*   A_offsets;
    const int*   A_starts;
    const int*   A_lengths;
    int          A_num_diags;

    /* B (sorted ascending by offset) */
    const float* B_values;
    const int*   B_starts;
    const int*   B_lengths;
    int          B_num_diags;

    /* B diagonal lookup and range */
    const int*   B_diag_lookup;  /* size 2n-1 */
    int          B_offset_min;
    int          B_offset_max;

    /* Output C diagonal format */
    float*       C_values;
    const int*   C_val_starts;   /* size 2n-1, -1 if not present */
    const int*   C_diag_lens;    /* size 2n-1 */

    /* Output diagonal range */
    int          d_c_min;
    int          d_c_max;

    /* Total work items: num_d_c_batches × num_pos_tiles */
    int          num_d_c_batches;  /* ceil((d_c_max - d_c_min + 1) / BATCH_K) */
    int          num_pos_tiles;    /* ceil(n / 32) */
    int          total_items;      /* num_d_c_batches * num_pos_tiles */
};

__global__ void diag_spmm_batched_kernel(BatchedArgs args);

void launch_batched_kernel(BatchedArgs args, cudaStream_t stream = 0);
