#pragma once
#include "dia_reconstruct.cuh"

// ============================================================================
// Flexible CUDA-core diagonal kernels (no tensor cores, no wasted output).
//
// The TC kernel forces a row-coupled diagonal gather onto an m16n8 MMA and
// keeps only the tile diagonal (~16 of 128 outputs) -> ~6% of the MMA's FLOPs
// are useful. SpMV is memory-bound, so that waste is hidden behind the loads;
// these kernels compute ONLY the needed elements, in full fp32.
//
// DIA layout (ReconView, from build_recon):
//   diag_offsets[k]                          : signed offset of diagonal k
//   values[k*cols + c]  with c = i + off_k   : matrix entry A[i][i+off_k]
//   => y[i] = sum_k values[k*cols + (i+off_k)] * x[i+off_k]   (valid cols only)
// ============================================================================

// ---- SpMV: one thread per row, fp32 accumulate, zero wasted compute. -------
__global__ void cuda_spmv_dia(ReconView R, const float* __restrict__ x,
                              int x_size, float* __restrict__ y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R.rows) return;
    const int nd = R.num_diags;
    float acc = 0.f;
    for (int k = 0; k < nd; ++k) {
        const int c = i + R.diag_offsets[k];
        if (c >= 0 && c < R.cols)
            acc += R.values[(size_t)k * R.cols + c] * x[c];
    }
    y[i] = acc;
}

// ---- SpMM: one thread per row, NV vectors, matrix value read once per (row,
// diag) and reused across all NV vectors -> matrix traffic / NV (the SpMM
// arithmetic-intensity win). X is column-major: X[v*cols + c]; same for Y.
template <int NV>
__global__ void cuda_spmm_dia(ReconView R, const float* __restrict__ X,
                              int x_size, float* __restrict__ Y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R.rows) return;
    const int nd = R.num_diags, cols = R.cols;
    float acc[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) acc[v] = 0.f;
    for (int k = 0; k < nd; ++k) {
        const int c = i + R.diag_offsets[k];
        if (c >= 0 && c < cols) {
            const float a = R.values[(size_t)k * cols + c];   // reused across NV
#pragma unroll
            for (int v = 0; v < NV; ++v)
                acc[v] += a * X[(size_t)v * cols + c];
        }
    }
#pragma unroll
    for (int v = 0; v < NV; ++v) Y[(size_t)v * R.rows + i] = acc[v];
}
