/* ============================================================
 * paper_hm_hash.cuh
 *
 * Hash-table-accumulator variant of the HM kernel.
 *
 * Same A-centric dispatch as hm_structured_sparse_matmul_kernel,
 * but accumulates into a per-block shared-memory hash table
 * instead of doing global atomicAdd per (A, B-diag) update.
 * The table is flushed to global C once at end of block.
 *
 *   keys[] : flat global C index (C_starts[c_diag] + p_c) or EMPTY
 *   vals[] : partial sum
 *   probe  : linear, bounded; overflow falls back to global atomicAdd
 * ============================================================ */
#pragma once

#include "paper_hm.cuh"

/* Tunables (power-of-two hash size required). */
constexpr int HM_HASH_BLOCK     = 256;
constexpr int HM_HASH_TABLE     = 2048;
constexpr int HM_HASH_TABLE_LOG = 11;   /* log2(HM_HASH_TABLE) */
constexpr int HM_HASH_PROBE_MAX = 16;
constexpr int HM_HASH_EMPTY     = -1;

__global__ void
hm_hash_structured_sparse_matmul_kernel(
    const float* __restrict__ A_vals,
    const int*   __restrict__ A_offsets,
    const int*   __restrict__ A_starts,
    const int*   __restrict__ A_lengths,
    int                       A_num_diags,
    const float* __restrict__ B_vals,
    const int*   __restrict__ B_offsets,
    const int*   __restrict__ B_starts,
    const int*   __restrict__ B_lengths,
    int                       B_num_diags,
    float*       __restrict__ C_vals,
    const int*   __restrict__ C_offsets,
    const int*   __restrict__ C_starts,
    const int*   __restrict__ C_lengths,
    int                       C_num_diags,
    const int*   __restrict__ C_diag_lookup,
    int                       nzA,
    int                       n);

void launch_hm_hash(
    const float* A_vals, const int* A_offsets, const int* A_starts,
    const int* A_lengths, int A_num_diags,
    const float* B_vals, const int* B_offsets, const int* B_starts,
    const int* B_lengths, int B_num_diags,
    float* C_vals, const int* C_offsets, const int* C_starts,
    const int* C_lengths, int C_num_diags,
    const int* C_diag_lookup,
    int nzA, int n,
    cudaStream_t stream = 0);
