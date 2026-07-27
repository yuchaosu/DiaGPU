/* ============================================================
 * common.cuh — shared pieces for the paper benchmark drivers.
 *
 * Holds: timing/upload/error helpers, the locked didx SpMV kernel
 * (per-nonzero diagonal-index plan) + its host builder, and the
 * HamSim/libdiaq kernels copied verbatim (templated on precision)
 * from github.com/srikarchundury/diaq for research benchmarking
 * with attribution (CC BY-NC-ND; not redistributed).
 * ============================================================ */
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

template<class F> static float tms(F f, int wu, int it){
    for (int i = 0; i < wu; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < it; ++i) f();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / it;
}

template<typename T> static T* dupload(const std::vector<T>& h){
    T* d; CUDA_CHECK(cudaMalloc(&d, h.size()*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static double rel_l2(const std::vector<double>& ref, const float* got, int n){
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i){ double d = ref[i]-(double)got[i]; num += d*d; den += ref[i]*ref[i]; }
    return den > 0 ? std::sqrt(num/den) : std::sqrt(num);
}

/* ---- locked SpMV: per-nonzero diagonal-index plan (didx) -----------------
 * Host emits each row's exact work list (didx,val) per TRUE nonzero;
 * column COMPUTED as r + offset[didx] (offset table in smem), never
 * stored. 1-byte index for D<=256, else 2-byte. NV vectors share each
 * matrix read (fused real/imag apply).                                  */
template <typename IT, int NV>
__global__ void spmv_didx_kernel(int n, int D,
    const int* __restrict__ rp, const IT* __restrict__ didx,
    const float* __restrict__ val, const int* __restrict__ offs,
    const float* __restrict__ X, float* __restrict__ Y)
{
    extern __shared__ int soff[];
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offs[k];
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float acc[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) acc[v] = 0.f;
    const int e = rp[r + 1];
    for (int j = rp[r]; j < e; ++j) {
        const int c = r + soff[didx[j]];
        const float a = val[j];
#pragma unroll
        for (int v = 0; v < NV; ++v) acc[v] += a * X[(size_t)v * n + c];
    }
#pragma unroll
    for (int v = 0; v < NV; ++v) Y[(size_t)v * n + r] = acc[v];
}

struct DidxPlan {
    std::vector<int>            rp;
    std::vector<unsigned char>  d8;    /* D <= 256 */
    std::vector<unsigned short> d16;   /* D  > 256 */
    std::vector<float>          val;
    bool wide = false;
};

inline DidxPlan build_didx_plan(
    int n, const std::vector<int>& offsets, const std::vector<size_t>& starts,
    const std::vector<int>& lengths, const std::vector<float>& values)
{
    DidxPlan P; P.wide = offsets.size() > 256;
    std::vector<std::vector<std::pair<int,float>>> rows(n);
    for (size_t k = 0; k < offsets.size(); ++k) {
        int off = offsets[k]; size_t s = starts[k]; int len = lengths[k];
        for (int p = 0; p < len; ++p) {
            float v = values[s + p]; if (v == 0.f) continue;
            int row = off >= 0 ? p : p - off;
            rows[row].push_back({(int)k, v});
        }
    }
    P.rp.assign(n + 1, 0);
    for (int r = 0; r < n; ++r) {
        P.rp[r+1] = P.rp[r] + (int)rows[r].size();
        for (auto& e : rows[r]) {
            if (P.wide) P.d16.push_back((unsigned short)e.first);
            else        P.d8.push_back((unsigned char)e.first);
            P.val.push_back(e.second);
        }
    }
    return P;
}

/* ================= HamSim/libdiaq kernels, verbatim (templated) =========== */

// diaq/src/spHamSim_gpu_all_fused.cu : spmv_row_diag_sparse_kernel
template <typename VT>
__global__ void diaq_spmv_row_kernel(
    unsigned int numRows, unsigned int numCols, unsigned int numDiags,
    const int * __restrict__ dIndices,
    const unsigned int * __restrict__ diagOffsets,
    const unsigned int * __restrict__ diagLens,
    const VT * __restrict__ A_real, const VT * __restrict__ A_imag,
    const VT * __restrict__ x_real, const VT * __restrict__ x_imag,
    VT * __restrict__ y_real, VT * __restrict__ y_imag)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRows) return;
    VT sum_r = 0.0, sum_i = 0.0;
    for (unsigned int k = 0; k < numDiags; ++k) {
        int d = dIndices[k];
        int j = (int)i + d;
        if (j < 0 || (unsigned int)j >= numCols) continue;
        int p = (d >= 0) ? (int)i : ((int)i + d);
        if (p < 0) continue;
        unsigned int len = diagLens[k];
        if ((unsigned int)p >= len) continue;
        unsigned int idxA = diagOffsets[k] + (unsigned int)p;
        VT ar = A_real[idxA], ai = A_imag[idxA];
        VT xr = x_real[(unsigned int)j], xi = x_imag[(unsigned int)j];
        sum_r += ar * xr - ai * xi;
        sum_i += ar * xi + ai * xr;
    }
    y_real[i] = sum_r;
    y_imag[i] = sum_i;
}

// diaq/src/spGEMM_gpu.cu : product_kernel  (SPGEMM_TILE_T = 256)
#define DIAQ_SPGEMM_TILE_T 256
template <typename VT>
__global__ void diaq_product_kernel(
    int numRes,
    const int * __restrict__ resDiagIndices,
    const unsigned int * __restrict__ resDiagOffsets,
    const unsigned int * __restrict__ resDiagLens,
    const unsigned int * __restrict__ resPairOffset,
    const int * __restrict__ pair_dA,
    const int * __restrict__ pair_dB,
    const unsigned int * __restrict__ pairSlotA,
    const unsigned int * __restrict__ pairSlotB,
    const unsigned int * __restrict__ A_diagOffsets,
    const unsigned int * __restrict__ A_diagLens,
    const unsigned int * __restrict__ B_diagOffsets,
    const unsigned int * __restrict__ B_diagLens,
    const VT * __restrict__ A_real, const VT * __restrict__ A_imag,
    const VT * __restrict__ B_real, const VT * __restrict__ B_imag,
    VT * __restrict__ C_real, VT * __restrict__ C_imag)
{
    int resSlot = blockIdx.x;
    if (resSlot >= numRes) return;
    int dC = resDiagIndices[resSlot];
    unsigned int offC = resDiagOffsets[resSlot];
    unsigned int lenC = resDiagLens[resSlot];
    unsigned int pairStart = resPairOffset[resSlot];
    unsigned int pairEnd   = resPairOffset[resSlot + 1];

    for (unsigned int p = pairStart; p < pairEnd; ++p) {
        int dA = pair_dA[p];
        int dB = pair_dB[p];
        unsigned int slotA = pairSlotA[p], slotB = pairSlotB[p];
        unsigned int offA = A_diagOffsets[slotA], lenA = A_diagLens[slotA];
        unsigned int offB = B_diagOffsets[slotB], lenB = B_diagLens[slotB];
        bool first = (p == pairStart);

        for (unsigned int t0 = 0; t0 < lenC; t0 += DIAQ_SPGEMM_TILE_T) {
            unsigned int tEnd = t0 + DIAQ_SPGEMM_TILE_T;
            if (tEnd > lenC) tEnd = lenC;
            for (unsigned int t = t0 + threadIdx.x; t < tEnd; t += blockDim.x) {
                int i, j;
                if (dC >= 0) { i = (int)t;      j = (int)t + dC; }
                else         { i = (int)t - dC; j = (int)t;      }
                int pA = (dA >= 0) ? i : (i + dA);
                int pB = (dB >= 0) ? (i + dA) : j;
                if (pA < 0 || pB < 0) continue;
                if ((unsigned int)pA >= lenA || (unsigned int)pB >= lenB) continue;
                unsigned int idxA = offA + (unsigned int)pA;
                unsigned int idxB = offB + (unsigned int)pB;
                unsigned int idxC = offC + t;
                VT ar = A_real[idxA], ai = A_imag[idxA];
                VT br = B_real[idxB], bi = B_imag[idxB];
                VT cr = ar * br - ai * bi;
                VT ci = ar * bi + ai * br;
                if (first) { C_real[idxC]  = cr; C_imag[idxC]  = ci; }
                else       { C_real[idxC] += cr; C_imag[idxC] += ci; }
            }
            __syncthreads();
        }
    }
}
