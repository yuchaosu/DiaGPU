/* ============================================================
 * hm_optimized_kernel.cuh
 *
 * Optimized HM-style SpMM for symmetric diagonal storage.
 *
 * Improvements over baseline HM:
 *   1. Zero binary search (grid = 1 block per A diagonal segment)
 *   2. B/C metadata preloaded into shared memory (zero L1 pressure)
 *   3. Precomputed C write base addresses (eliminate C_diag_lookup)
 *   4. B range narrowing (skip out-of-bounds B diags per thread)
 *   5. Inner loop: ~6 instructions per FMA (vs ~16 in baseline)
 * ============================================================ */
#pragma once

#include "diag_types.cuh"
#include <cuda_runtime.h>

struct HMOptArgs {
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

    /* Precomputed per-B-diagonal metadata (size B_num_diags each).
     * Loaded into smem once per CTA for fast inner loop access.
     *   b_sr[bi]         = max(0, -B_offsets[bi])
     *   b_start[bi]      = B.diag_starts[bi]
     *   b_len[bi]        = B.diag_lengths[bi]
     *
     * Per-(A_diag, B_diag) pair, the C write base address:
     *   For output diagonal d_c = d_a + d_b:
     *     c_write_base = C_val_starts[d_c + n-1] - c_sr(d_c)
     *   So the final write is: C[c_write_base + row]
     *   This is precomputed per (ai, bi) pair.
     *
     * c_base[ai * B_num_diags + bi] = C_val_starts[d_c+n-1] - c_sr(d_c)
     *   or -1 if the output diagonal doesn't exist.             */
    const int*   b_sr;          /* [B_num_diags] */
    const int*   b_start;       /* [B_num_diags] */
    const int*   b_len;         /* [B_num_diags] */
    const int*   c_base;        /* [A_num_diags * B_num_diags] */

    /* Output C */
    float*       C_values;
};

__global__ void hm_optimized_kernel(HMOptArgs args, int ai);

void launch_hm_optimized(HMOptArgs args, cudaStream_t stream = 0);
