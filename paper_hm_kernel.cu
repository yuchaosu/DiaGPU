/* ============================================================
 * paper_hm_kernel.cu
 *
 * GPU kernel for Algorithm 2 (Structured Sparse Matrix
 * Multiplication using HM storage) from:
 *   Haque, Parvez, Hossain — Algorithms 2024, 17, 31.
 *
 * One thread per nonzero of A. For each A element, iterates
 * over all B diagonals, computes the matching B element index
 * and the target C element, and updates via atomicAdd.
 * ============================================================ */

#include "paper_hm.cuh"

/* Device helper: binary search to find which diagonal an element
 * index belongs to. Returns the diagonal index (0-based). */
__device__ static int
find_diagonal_bs(const int* __restrict__ start_dg, int num_diags, int idx)
{
    int lo = 0, hi = num_diags - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (start_dg[mid] <= idx)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

__global__ void
__launch_bounds__(256)
hm_structured_sparse_matmul_kernel(
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
    int                       n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nzA) return;

    /* Find which diagonal of A this element belongs to. */
    int a_diag_local = find_diagonal_bs(A_starts, A_num_diags, i);
    int d_a     = A_offsets[a_diag_local];
    int a_start = A_starts[a_diag_local];
    int a_ind   = i - a_start;

    float a_val = A_vals[i];

    /* Matrix coordinates of this A element. */
    int a_sr = (d_a >= 0) ? 0 : -d_a;
    int a_sc = (d_a >= 0) ? d_a : 0;
    int row = a_sr + a_ind;
    int k   = a_sc + a_ind;

    /* For each diagonal of B, find the B element at row k. */
    for (int bi = 0; bi < B_num_diags; ++bi) {
        int d_b  = B_offsets[bi];
        int b_sr = (d_b >= 0) ? 0 : -d_b;
        int b_sc = (d_b >= 0) ? d_b : 0;

        int p_b = k - b_sr;
        if (p_b < 0 || p_b >= B_lengths[bi]) continue;

        int col = b_sc + p_b;
        int d_c = col - row;

        int c_lookup_idx = d_c + (n - 1);
        int c_diag_local = C_diag_lookup[c_lookup_idx];
        if (c_diag_local < 0) continue;

        int c_sr = (d_c >= 0) ? 0 : -d_c;
        int p_c  = row - c_sr;
        if (p_c < 0 || p_c >= C_lengths[c_diag_local]) continue;

        float b_val = B_vals[B_starts[bi] + p_b];
        atomicAdd(&C_vals[C_starts[c_diag_local] + p_c], a_val * b_val);
    }
}
