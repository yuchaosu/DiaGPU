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

/* ============================================================
 * DIA-NATIVE zero-skip (keeps the diagonal structure of the SAME recon the TC
 * kernel consumes — does NOT go to arbitrary-column CSR).
 *
 * Per row r, store only the NONZERO diagonals as (didx, val), where didx is a
 * 1-byte index into the shared diagonal-offset array (D <= 256). The column is
 * recovered as col = r + offset[didx], so x is read per-diagonal-regular (not a
 * scattered CSR gather), and the per-nonzero metadata is 1 byte, not the 4-byte
 * CSR col index -> LESS traffic than CSR (val+1 = 5 B/nz vs val+col = 8 B/nz)
 * and much less than the dense recon (num_diags*n floats, ~72% of them zero).
 * Atomic-free: one thread per row accumulates y[r] in a register.
 *
 * offsets[] must be the SAME descending diagonal-offset order build_recon uses,
 * so this and the TC kernel are fed identical H.
 *
 * MEASURED (A100, sim/spmv_dia_vs_tc.cu, bit-exact vs dense fp32; same recon):
 *   matrix          fill   scalar vs tensor-core   scalar vs dense-CUDA
 *   heis_18 (27.9%)        5.13x                   2.48x
 *   heis_20 (27.6%)        4.18x                   2.05x
 *   bh_18   (20.8%)        5.31x                   2.52x
 *   tfim_18 (100%)         1.93x (dense-CUDA 2.3x wins — no zeros to skip)
 * => on interior-sparse bands the DIA-native zero-skip beats the tensor-core
 * kernel 4-5x AND is exact (TC carries TF32 error ~1e-4 rel). The 1-byte didx
 * beats a 4-byte CSR col (2.7-4.0x less matrix traffic vs 1.75-2.5x). On a fully
 * dense band, fall back to dense cuda_spmv_dia. The warp variant loses on thin
 * bands (per-row nnz << 32) — scalar is the right choice; balance is not the
 * lever, the zero-skip is.
 * ============================================================ */
__global__ void spmv_dia_zeroskip(int rows, int D,
    const int*  __restrict__ row_ptr,
    const unsigned char* __restrict__ didx,
    const float* __restrict__ val,
    const int*  __restrict__ offsets,
    const float* __restrict__ x, float* __restrict__ y)
{
    extern __shared__ int soff[];                     // D diagonal offsets
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offsets[k];
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int b = row_ptr[r], e = row_ptr[r + 1];
    float acc = 0.f;
    for (int j = b; j < e; ++j) acc += val[j] * x[r + soff[didx[j]]];
    y[r] = acc;
}

/* ---- 2-byte diagonal-index variant for WIDE bands (D > 256, up to 65535). ----
 * Identical to spmv_dia_zeroskip but didx is uint16 so it covers the fill-in
 * families (O2 D~5329 etc.) while STAYING DIA-based (col = row + offset[didx]);
 * we never fall back to a 4-byte-col CSR gather. Traffic 6 B/nz (val+2) vs the
 * 1-byte kernel's 5 B/nz — still below CSR's 8 B/nz. Same shared-offset scheme. */
__global__ void spmv_dia_zeroskip_u16(int rows, int D,
    const int*  __restrict__ row_ptr,
    const unsigned short* __restrict__ didx,
    const float* __restrict__ val,
    const int*  __restrict__ offsets,
    const float* __restrict__ x, float* __restrict__ y)
{
    extern __shared__ int soff[];                     // D diagonal offsets
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offsets[k];
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int b = row_ptr[r], e = row_ptr[r + 1];
    float acc = 0.f;
    for (int j = b; j < e; ++j) acc += val[j] * x[r + soff[didx[j]]];
    y[r] = acc;
}

/* ---- int64 row-pointer variants for LARGE matrices (nnz > INT32_MAX). ----
 * heis_28 / qmaxcut_28 / fermi_28 have nnz ~4-7e9, overflowing 32-bit row_ptr
 * and the j index. These take a 64-bit row_ptr and index the nonzero arrays
 * with a 64-bit j; col indices stay 32-bit (col < n <= 2^28). One thread/row. */
__global__ void spmv_dia_zeroskip_i64(long long rows, int D,
    const long long* __restrict__ row_ptr,
    const unsigned char* __restrict__ didx,
    const float* __restrict__ val,
    const int*  __restrict__ offsets,
    const float* __restrict__ x, float* __restrict__ y)
{
    extern __shared__ int soff[];
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offsets[k];
    __syncthreads();
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const long long b = row_ptr[r], e = row_ptr[r + 1];
    float acc = 0.f;
    for (long long j = b; j < e; ++j) acc += val[j] * x[r + soff[didx[j]]];
    y[r] = acc;
}

__global__ void spmv_dia_zeroskip_u16_i64(long long rows, int D,
    const long long* __restrict__ row_ptr,
    const unsigned short* __restrict__ didx,
    const float* __restrict__ val,
    const int*  __restrict__ offsets,
    const float* __restrict__ x, float* __restrict__ y)
{
    extern __shared__ int soff[];
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offsets[k];
    __syncthreads();
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const long long b = row_ptr[r], e = row_ptr[r + 1];
    float acc = 0.f;
    for (long long j = b; j < e; ++j) acc += val[j] * x[r + soff[didx[j]]];
    y[r] = acc;
}

/* Balanced (warp-per-row) DIA-native variant — for matrices whose per-row nnz
 * is skewed. On a fixed narrow band per-row nnz is bounded, so scalar usually
 * wins; kept for the wide/fill-in case. */
__global__ void spmv_dia_zeroskip_warp(int rows, int D,
    const int*  __restrict__ row_ptr,
    const unsigned char* __restrict__ didx,
    const float* __restrict__ val,
    const int*  __restrict__ offsets,
    const float* __restrict__ x, float* __restrict__ y)
{
    extern __shared__ int soff[];
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offsets[k];
    __syncthreads();
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const int b = row_ptr[warp], e = row_ptr[warp + 1];
    float acc = 0.f;
    for (int j = b + lane; j < e; j += 32) acc += val[j] * x[warp + soff[didx[j]]];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[warp] = acc;
}
