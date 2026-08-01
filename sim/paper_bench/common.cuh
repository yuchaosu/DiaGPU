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
#include <cstdlib>
#include <atomic>
#include <thread>
#include <chrono>
#include <nvml.h>

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

/* ---- ENERGY=1 hook: measure J/apply for whatever tms() times -------------
 * Loops the kernel for >=ENERGY_SECS (default 6s, >> the ~100ms update
 * period of the power counter), samples NVML power at 25ms, integrates,
 * and subtracts an idle baseline sampled once per process.  Emits
 *   ENERGYRAW,<J_per_call_net>,<avg_W>,<idle_W>,<iters>
 * on stdout immediately BEFORE the caller's own CSV row, so post-processing
 * pairs each ENERGYRAW with the row that follows it.                       */
static double g_idle_W = -1.0;
static nvmlDevice_t g_nvml_dev;
static bool g_nvml_ok = false;
static inline double energy_sample_W(){
    unsigned mw = 0;
    return nvmlDeviceGetPowerUsage(g_nvml_dev, &mw) == NVML_SUCCESS ? mw / 1000.0 : -1.0;
}
template<class F> static void energy_probe(F f, float ms_per_call){
    const char* env = getenv("ENERGY");
    if (!env || !*env || ms_per_call <= 0) return;
    if (!g_nvml_ok) {
        if (nvmlInit() != NVML_SUCCESS) return;
        if (nvmlDeviceGetHandleByIndex(0, &g_nvml_dev) != NVML_SUCCESS) return;
        g_nvml_ok = true;
        /* idle baseline: 2s of samples with the GPU quiet */
        CUDA_CHECK(cudaDeviceSynchronize());
        double acc = 0; int cnt = 0;
        for (int i = 0; i < 80; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            double w = energy_sample_W(); if (w > 0){ acc += w; ++cnt; }
        }
        g_idle_W = cnt ? acc / cnt : -1.0;
    }
    const double secs = atof(env) > 1.0 ? atof(env) : 6.0;
    long it = std::max(32L, (long)(secs * 1000.0 / ms_per_call));
    std::atomic<bool> stop{false};
    std::atomic<long> nsmp{0};
    double joules = 0;
    std::thread th([&]{
        auto t0 = std::chrono::steady_clock::now(); auto tp = t0;
        while (!stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            auto tn = std::chrono::steady_clock::now();
            double w = energy_sample_W();
            if (w > 0){ joules += w * std::chrono::duration<double>(tn - tp).count(); ++nsmp; }
            tp = tn;
        }
    });
    auto w0 = std::chrono::steady_clock::now();
    for (long i = 0; i < it; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    stop = true; th.join();
    const double avg_W = wall > 0 ? joules / wall : -1;
    const double net_J = (avg_W > 0 && g_idle_W > 0) ? (avg_W - g_idle_W) * wall / it : -1;
    printf("ENERGYRAW,%.6e,%.1f,%.1f,%ld\n", net_J, avg_W, g_idle_W, it);
}

template<class F> static float tms(F f, int wu, int it){
    for (int i = 0; i < wu; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < it; ++i) f();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    energy_probe(f, ms / it);       /* inert unless ENERGY=1 */
    return ms / it;
}


/* symmetric didx: upper diagonals stored once, each row reads v_d at two
 * positions (+d term and mirror); 1B meta = 7-bit upper slot | side bit.
 * Promoted to the MAINLINE SpMV symmetric mode (2026-08-01). */
__global__ static void spmv_didx_sym_kernel(int n, int Dup,
    const int* __restrict__ rp, const unsigned char* __restrict__ meta,
    const int* __restrict__ offs_up, const long long* __restrict__ starts_up,
    const float* __restrict__ val,
    const float* __restrict__ X, float* __restrict__ Y)
{
    extern __shared__ char smraw[];
    long long* sst = (long long*)smraw;
    int*       soff = (int*)(smraw + Dup * 8);
    for (int k = threadIdx.x; k < Dup; k += blockDim.x){ sst[k] = starts_up[k]; soff[k] = offs_up[k]; }
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float acc = 0.f;
    const int e = rp[r + 1];
    for (int j = rp[r]; j < e; ++j) {
        const unsigned char m = meta[j];
        const int s = m & 127, side = m >> 7;
        const int d = soff[s];
        acc += val[sst[s] + (side ? r - d : r)] * X[side ? r - d : r + d];
    }
    Y[r] = acc;
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

/* two-pass build: count per-row nonzeros, prefix-sum, then scatter into the
 * final arrays — no per-row vectors, O(stored) with tight constants (the
 * plan-build time is part of the paper's amortization table).  Within a row,
 * entries land in ascending diagonal order because the outer loop is over
 * diagonals and the per-row cursor advances monotonically. */
inline DidxPlan build_didx_plan(
    int n, const std::vector<int>& offsets, const std::vector<size_t>& starts,
    const std::vector<int>& lengths, const std::vector<float>& values)
{
    DidxPlan P; P.wide = offsets.size() > 256;
    P.rp.assign(n + 1, 0);
    for (size_t k = 0; k < offsets.size(); ++k) {
        int off = offsets[k]; size_t s = starts[k]; int len = lengths[k];
        for (int p = 0; p < len; ++p)
            if (values[s + p] != 0.f) ++P.rp[(off >= 0 ? p : p - off) + 1];
    }
    for (int r = 0; r < n; ++r) P.rp[r+1] += P.rp[r];
    size_t nnz = P.rp[n];
    P.val.resize(nnz);
    if (P.wide) P.d16.resize(nnz); else P.d8.resize(nnz);
    std::vector<int> cur(P.rp.begin(), P.rp.end() - 1);
    for (size_t k = 0; k < offsets.size(); ++k) {
        int off = offsets[k]; size_t s = starts[k]; int len = lengths[k];
        for (int p = 0; p < len; ++p) {
            float v = values[s + p]; if (v == 0.f) continue;
            int r = off >= 0 ? p : p - off;
            int j = cur[r]++;
            P.val[j] = v;
            if (P.wide) P.d16[j] = (unsigned short)k;
            else        P.d8[j]  = (unsigned char)k;
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
