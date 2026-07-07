/* ============================================================
 * spmv_zeroskip_kernels.cuh — zero-skipped, load-balanced CUDA-core SpMV.
 *
 * The dense-DIA kernel (cuda_spmv_dia) reads the full reconstructed band —
 * num_diags * cols floats — INCLUDING the interior structural zeros of each
 * diagonal (measured ~72% of a real heis band). SpMV is memory-bound, so those
 * zero reads are pure wasted bandwidth. These kernels store ONLY the nonzeros
 * (CSR, built by dropping zeros) so the matrix traffic drops to nnz*(val+col).
 *
 * Two balance strategies over the same CSR:
 *   - scalar : one thread per row. Minimal-overhead zero-skip; imbalanced when
 *              per-row nnz varies (heis rows vary by index residue mod pattern).
 *   - warp   : one warp per row, lanes stride the row's nonzeros + warp-reduce.
 *              Balances rows of uneven nnz; atomic-free (lane 0 writes y[row]).
 *
 * MEASURED (A100, sim/spmv_zeroskip_bench.cu, bit-exact vs dense-DIA):
 *   matrix          fill    csr_scalar vs dense-DIA cuda_spmv_dia
 *   heis_18 (27.9%)         2.02x    heis_20 (27.6%)  1.46x
 *   bh_18   (20.8%)         1.56x    sweepheis(100%)  0.94x
 * => 'scalar' wins 1.46-2.0x whenever the band has interior zeros; the gain
 * tracks the fill (fewer zeros -> less win) and it only loses on a fully-dense
 * band, where the col index doubles matrix bytes with no zeros to skip. The
 * 'warp' kernel LOSES on these thin rows (avg ~9.5 nnz << 32 lanes) — for a
 * fixed-band Hamiltonian, per-row nnz is bounded and scalar is already balanced;
 * the lever is the zero-skip, not the balance. Since the SpMV operator H is FIXED
 * across all evolution steps (unlike the SpMSpM fill-in chain), the one-time CSR
 * build amortizes over every step — this is the right place to skip zeros.
 * ============================================================ */
#pragma once
#include <cuda_runtime.h>

/* Compressed (CSR) operand: row_ptr[rows+1], col_idx[nnz], val[nnz]. */
struct CsrView { int rows; const int* row_ptr; const int* col_idx; const float* val; };

/* ---- scalar: one thread per row, register accumulate, only nonzeros read. ---- */
__global__ void spmv_csr_scalar(CsrView A, const float* __restrict__ x,
                                 float* __restrict__ y)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= A.rows) return;
    const int b = A.row_ptr[r], e = A.row_ptr[r + 1];
    float acc = 0.f;
    for (int j = b; j < e; ++j) acc += A.val[j] * x[A.col_idx[j]];
    y[r] = acc;
}

/* ---- warp: one warp per row, lanes stride the row, warp-reduce (balanced). ---- */
__global__ void spmv_csr_warp(CsrView A, const float* __restrict__ x,
                              float* __restrict__ y)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= A.rows) return;
    const int b = A.row_ptr[warp], e = A.row_ptr[warp + 1];
    float acc = 0.f;
    for (int j = b + lane; j < e; j += 32) acc += A.val[j] * x[A.col_idx[j]];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[warp] = acc;
}
