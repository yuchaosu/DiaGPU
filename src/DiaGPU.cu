#include "DiaGPU.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/scatter.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef CUDA_CALL
#define CUDA_CALL(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)
#endif

// ---------- device helpers ----------
__device__ __forceinline__ int start_row(int offset){ return (offset >= 0 ? 0 : -offset); }
// NOTE: end_row is EXCLUSIVE (matches your original kernel)
__device__ __forceinline__ int end_row(int N,int offset){ return (offset >= 0 ? N - offset : N); }

// verify if current row is inside the current diagonal (local index or -1)
__device__ __forceinline__
int local_index_or_neg1(int N, int offset, int i) {
    int s = (offset >= 0 ? 0 : -offset);
    int e = (offset >= 0 ? N - offset : N);
    return (i >= s && i < e) ? (i - s) : -1;
}

// ---------- compact plan to keep B on device ----------
struct DiaBPlan {
    int N = 0;
    int numDiagB = 0;
    thrust::device_vector<int>   offsetsB;
    thrust::device_vector<int>   ptrB;
    thrust::device_vector<float> valsB;
};

DiaBPlan* create_B_plan(const DiagListF32& B) {
    auto* p = new DiaBPlan();
    p->N = B.N;
    p->numDiagB = (int)B.offsets.size();
    p->offsetsB.assign(B.offsets.begin(), B.offsets.end());
    p->ptrB.assign(B.ptr.begin(), B.ptr.end());
    p->valsB.assign(B.vals.begin(), B.vals.end());
    return p;
}
void destroy_B_plan(DiaBPlan* plan) { delete plan; }

// ---------- KERNEL 1: Non-blocking (your original per-output-diagonal kernel) ----------
template <typename T>
__global__ void spmspm_perc_kernel(
    int N,
    // A
    int numDiagA, const int* __restrict__ offsetsA, const int* __restrict__ ptrA, const T* __restrict__ valsA,
    // B (device-resident via plan)
    int numDiagB, const int* __restrict__ offsetsB, const int* __restrict__ ptrB, const T* __restrict__ valsB,
    // C meta (temporary, pre-compaction)
    int numDiagC, const int* __restrict__ offsetsC, const int* __restrict__ lenC, const int* __restrict__ ptrC,
    // pairs contributing to each C diagonal (cPairPtr[c]..cPairPtr[c+1])
    const int* __restrict__ cPairPtr, const int* __restrict__ cPairA, const int* __restrict__ cPairB,
    // outputs
    T* __restrict__ outValsC,       // concatenated values for all C diagonals
    T* __restrict__ absSumC         // per-diagonal sum of |values|
){
    const int cDiagIdx = blockIdx.y;                      // which C diagonal
    if (cDiagIdx >= numDiagC) return;

    const int Lc  = lenC[cDiagIdx];                       // length of this C diagonal
    const int t   = blockIdx.x * blockDim.x + threadIdx.x;// local position along this diagonal
    T acc         = T(0);
    const int offsetC = offsetsC[cDiagIdx];

    // Traverse all (A,B) pairs that form this C diagonal
    for (int p = cPairPtr[cDiagIdx]; p < cPairPtr[cDiagIdx+1]; ++p) {
        const int dA = cPairA[p], dB = cPairB[p];
        const int offsetA = offsetsA[dA], offsetB = offsetsB[dB];

        // i-range intersection in EXCLUSIVE form [i_low, i_high_ex)
        int i_low = start_row(offsetA);
        int i_high_ex = end_row(N, offsetA);

        int tmp = start_row(offsetC);            if (tmp > i_low)      i_low = tmp;
        tmp     = end_row(N, offsetC);           if (tmp < i_high_ex)  i_high_ex = tmp;

        tmp     = start_row(offsetB) - offsetA;  if (tmp > i_low)      i_low = tmp;
        tmp     = end_row(N, offsetB) - offsetA; if (tmp < i_high_ex)  i_high_ex = tmp;

        if (i_low >= i_high_ex) continue;

        // Convert to local index window along C diagonal
        const int t0 = i_low - start_row(offsetC);
        const int t1 = i_high_ex - start_row(offsetC);   // exclusive

        if (t >= t0 && t < t1) {
            const int i  = t + start_row(offsetC);
            const int tA = local_index_or_neg1(N, offsetA, i);           if (tA < 0) continue;
            const int tB = local_index_or_neg1(N, offsetB, i + offsetA); if (tB < 0) continue;

            const T a = valsA[ptrA[dA] + tA];
            const T b = valsB[ptrB[dB] + tB];
            acc += a * b;
        }
    }

    if (t < Lc) {
        outValsC[ptrC[cDiagIdx] + t] = acc;
        atomicAdd(&absSumC[cDiagIdx], acc >= 0 ? acc : -acc);
    }
}

// ---------- HOST: Build C offsets & contributing pairs (shared by all paths) ----------
static void build_C_offsets_and_pairs(
    int N,
    const std::vector<int>& offsetsA,
    const std::vector<int>& offsetsB,
    std::vector<int>& offsetsC,
    std::vector<int>& lenC,
    std::vector<int>& cPairPtr,
    std::vector<int>& cPairA,
    std::vector<int>& cPairB)
{
    if (offsetsA.empty() || offsetsB.empty()) {
        offsetsC.clear(); lenC.clear(); cPairPtr.assign(1,0);
        cPairA.clear(); cPairB.clear();
        return;
    }

    const int bmin = offsetsB.front(), bmax = offsetsB.back();
    std::vector<int> Bindex(bmax - bmin + 1, -1);
    for (int i = 0; i < (int)offsetsB.size(); ++i) Bindex[offsetsB[i] - bmin] = i;

    auto diag_len_host = [](int N, int offset) {
        int a = (offset>=0?offset:-offset); int L=N-a; return L>0?L:0;
    };

    std::unordered_map<int, std::vector<std::pair<int,int>>> buckets; // c_off -> list of (dA,dB)
    buckets.reserve(offsetsA.size() * 2);

    for (int dA = 0; dA < (int)offsetsA.size(); ++dA) {
        const int offsetA = offsetsA[dA];
        const int clamp_min = std::max(offsetA + offsetsB.front(), -(N - 1));
        const int clamp_max = std::min(offsetA + offsetsB.back(),   (N - 1));
        for (int offsetC = clamp_min; offsetC <= clamp_max; ++offsetC) {
            const int offsetB = offsetC - offsetA;
            if (offsetB < bmin || offsetB > bmax) continue;
            const int dB = Bindex[offsetB - bmin];
            if (dB < 0) continue;
            if (diag_len_host(N, offsetC) <= 0) continue;
            buckets[offsetC].emplace_back(dA, dB);
        }
    }

    offsetsC.clear(); offsetsC.reserve(buckets.size());
    for (auto& kv : buckets) offsetsC.push_back(kv.first);
    std::sort(offsetsC.begin(), offsetsC.end());

    lenC.resize(offsetsC.size());
    cPairPtr.clear(); cPairA.clear(); cPairB.clear();
    cPairPtr.push_back(0);
    for (int idx = 0; idx < (int)offsetsC.size(); ++idx) {
        const int offsetC = offsetsC[idx];
        auto& vec = buckets[offsetC];
        // length of C diagonal
        int a = (offsetC>=0?offsetC:-offsetC); int L=N-a; lenC[idx] = (L>0?L:0);
        for (auto& pr : vec) { cPairA.push_back(pr.first); cPairB.push_back(pr.second); }
        cPairPtr.push_back((int)cPairA.size());
    }
}

// ---------- PUBLIC API 1: Non-blocking multiply (your original path) ----------
void multiply_sparse_noPad_with_timing_copy_plan(const DiagListF32& A,
                                                 const DiaBPlan* planB,
                                                 float /*eps*/,
                                                 DiagListF32& C_out,
                                                 float* kernel_ms,
                                                 float* kernel_plus_copy_ms)
{
    const int N = A.N;
    // 1) Build output structure + contributing pairs
    std::vector<int> offsetsC, lenC, cPairPtr, cPairA, cPairB;
    {
        std::vector<int> hBoff(planB->offsetsB.size());
        thrust::copy(planB->offsetsB.begin(), planB->offsetsB.end(), hBoff.begin());
        build_C_offsets_and_pairs(N, A.offsets, hBoff, offsetsC, lenC, cPairPtr, cPairA, cPairB);
    }

    const int numDiagC = (int)offsetsC.size();
    if (numDiagC == 0) {
        C_out = {N, {}, {0}, {}};
        if (kernel_ms) *kernel_ms = 0.f;
        if (kernel_plus_copy_ms) *kernel_plus_copy_ms = 0.f;
        return;
    }

    // ptrC & sizes
    std::vector<int> ptrC(numDiagC + 1, 0);
    for (int i=0;i<numDiagC;++i) ptrC[i+1] = ptrC[i] + lenC[i];
    const int nnzC = ptrC.back();

    // 2) Upload A + C-meta + pairs (B stays on device via plan)
    thrust::device_vector<int>   d_offsetsA(A.offsets.begin(), A.offsets.end());
    thrust::device_vector<int>   d_ptrA(A.ptr.begin(),         A.ptr.end());
    thrust::device_vector<float> d_valsA(A.vals.begin(),       A.vals.end());

    thrust::device_vector<int>   d_offsetsC(offsetsC.begin(),  offsetsC.end());
    thrust::device_vector<int>   d_lenC(lenC.begin(),          lenC.end());
    thrust::device_vector<int>   d_ptrC(ptrC.begin(),          ptrC.end());

    thrust::device_vector<int>   d_cPairPtr(cPairPtr.begin(),  cPairPtr.end());
    thrust::device_vector<int>   d_cPairA(cPairA.begin(),      cPairA.end());
    thrust::device_vector<int>   d_cPairB(cPairB.begin(),      cPairB.end());

    thrust::device_vector<float> d_outValsC(nnzC, 0.0f);
    thrust::device_vector<float> d_valsC_final(nnzC);
    thrust::device_vector<float> d_absSumC(numDiagC, 0.0f);

    // 3) Launch + time
    int maxLenC = 0; for (int L : lenC) maxLenC = std::max(maxLenC, L);
    const dim3 block(256,1,1);
    const dim3 grid((maxLenC + block.x - 1)/block.x, numDiagC, 1);

    cudaEvent_t k0,k1,k2;
    CUDA_CALL(cudaEventCreate(&k0));
    CUDA_CALL(cudaEventCreate(&k1));
    CUDA_CALL(cudaEventCreate(&k2));

    CUDA_CALL(cudaEventRecord(k0, 0));
    spmspm_perc_kernel<float><<<grid, block>>>(
        N,
        (int)A.offsets.size(),
        thrust::raw_pointer_cast(d_offsetsA.data()),
        thrust::raw_pointer_cast(d_ptrA.data()),
        thrust::raw_pointer_cast(d_valsA.data()),
        planB->numDiagB,
        thrust::raw_pointer_cast(planB->offsetsB.data()),
        thrust::raw_pointer_cast(planB->ptrB.data()),
        thrust::raw_pointer_cast(planB->valsB.data()),
        numDiagC,
        thrust::raw_pointer_cast(d_offsetsC.data()),
        thrust::raw_pointer_cast(d_lenC.data()),
        thrust::raw_pointer_cast(d_ptrC.data()),
        thrust::raw_pointer_cast(d_cPairPtr.data()),
        thrust::raw_pointer_cast(d_cPairA.data()),
        thrust::raw_pointer_cast(d_cPairB.data()),
        thrust::raw_pointer_cast(d_outValsC.data()),
        thrust::raw_pointer_cast(d_absSumC.data()));
    CUDA_CALL(cudaEventRecord(k1, 0));

    CUDA_CALL(cudaMemcpyAsync(
        thrust::raw_pointer_cast(d_valsC_final.data()),
        thrust::raw_pointer_cast(d_outValsC.data()),
        nnzC * sizeof(float),
        cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaEventRecord(k2, 0));
    CUDA_CALL(cudaEventSynchronize(k2));

    float ms_kernel = 0.f, ms_total = 0.f;
    CUDA_CALL(cudaEventElapsedTime(&ms_kernel, k0, k1));
    CUDA_CALL(cudaEventElapsedTime(&ms_total,  k0, k2));
    CUDA_CALL(cudaEventDestroy(k0));
    CUDA_CALL(cudaEventDestroy(k1));
    CUDA_CALL(cudaEventDestroy(k2));
    if (kernel_ms)           *kernel_ms = ms_kernel;
    if (kernel_plus_copy_ms) *kernel_plus_copy_ms = ms_total;

    // 4) Download raw result
    std::vector<int>   h_offsetsC(offsetsC.size());
    std::vector<int>   h_ptrC(ptrC.size());
    std::vector<float> h_valsC(nnzC);
    thrust::copy(d_offsetsC.begin(), d_offsetsC.end(), h_offsetsC.begin());
    thrust::copy(d_ptrC.begin(),     d_ptrC.end(),     h_ptrC.begin());
    thrust::copy(d_valsC_final.begin(), d_valsC_final.end(), h_valsC.begin());

    C_out.N = N;
    C_out.offsets = std::move(h_offsetsC);
    C_out.ptr     = std::move(h_ptrC);
    C_out.vals    = std::move(h_valsC);
}

// =======================================================================================
//                        GROUPED (K-SEGMENTED) VARIANTS
// =======================================================================================

// ---------- KERNEL 2: C-centric grouped (more parallelism; atomics on C) ----------
template <typename T>
__global__ void spmspm_perc_grouped_kernel(
    int N,
    int seg_size,                             // K-segment size (K==N here)
    // A
    int numDiagA, const int* __restrict__ offsetsA, const int* __restrict__ ptrA, const T* __restrict__ valsA,
    // B
    int numDiagB, const int* __restrict__ offsetsB, const int* __restrict__ ptrB, const T* __restrict__ valsB,
    // C meta
    int numDiagC, const int* __restrict__ offsetsC, const int* __restrict__ lenC, const int* __restrict__ ptrC,
    // contributing pairs for each C diagonal
    const int* __restrict__ cPairPtr, const int* __restrict__ cPairA, const int* __restrict__ cPairB,
    // outputs
    T* __restrict__ outValsC,
    T* __restrict__ absSumC
){
    const int cDiagIdx = blockIdx.y;                       // which C diagonal
    if (cDiagIdx >= numDiagC) return;

    const int seg  = blockIdx.z;                           // which K-segment
    const int k_begin = seg * seg_size;
    const int k_end   = min(N, k_begin + seg_size);        // exclusive

    const int t  = blockIdx.x * blockDim.x + threadIdx.x;  // index along C diagonal
    const int Lc = lenC[cDiagIdx];
    const int offsetC = offsetsC[cDiagIdx];

    T acc = T(0);

    // Iterate contributing (A,B) diagonal pairs for this C diagonal
    for (int p = cPairPtr[cDiagIdx]; p < cPairPtr[cDiagIdx+1]; ++p) {
        const int dA = cPairA[p];
        const int dB = cPairB[p];
        const int offsetA = offsetsA[dA];
        const int offsetB = offsetsB[dB];

        // Base i-range [i_low, i_high_ex) from A, C, B constraints (exclusive hi)
        int i_low = start_row(offsetA);
        int i_high_ex = end_row(N, offsetA);

        int tmp = start_row(offsetC);            if (tmp > i_low)      i_low = tmp;
        tmp     = end_row(N, offsetC);           if (tmp < i_high_ex)  i_high_ex = tmp;

        tmp     = start_row(offsetB) - offsetA;  if (tmp > i_low)      i_low = tmp;
        tmp     = end_row(N, offsetB) - offsetA; if (tmp < i_high_ex)  i_high_ex = tmp;

        // Segment constraint: k=i+offsetA in [k_begin, k_end) → i in [k_begin - offsetA, k_end - offsetA)
        tmp = k_begin - offsetA;  if (tmp > i_low)      i_low = tmp;
        tmp = k_end   - offsetA;  if (tmp < i_high_ex)  i_high_ex = tmp;

        if (i_low >= i_high_ex) continue;

        // Local t-window
        const int t0 = i_low - start_row(offsetC);
        const int t1 = i_high_ex - start_row(offsetC);

        if (t >= t0 && t < t1) {
            const int i  = t + start_row(offsetC);
            const int k  = i + offsetA;  // guaranteed within segment
            const int tA = local_index_or_neg1(N, offsetA, i); if (tA < 0) continue;
            const int tB = local_index_or_neg1(N, offsetB, k); if (tB < 0) continue;

            const T a = valsA[ptrA[dA] + tA];
            const T b = valsB[ptrB[dB] + tB];
            acc += a * b;
        }
    }

    if (t < Lc) {
        // Multiple segments contribute to the same C element → atomic
        atomicAdd(&outValsC[ptrC[cDiagIdx] + t], acc);
        const T v = (acc >= 0 ? acc : -acc);
        if (v != T(0)) atomicAdd(&absSumC[cDiagIdx], v);
    }
}

// ---------- HOST 2: C-centric grouped (segmented) launcher ----------
void multiply_sparse_grouped_with_timing_copy_plan(const DiagListF32& A,
                                                   const DiaBPlan* planB,
                                                   int seg_size,                 // e.g., 1024/2048/4096
                                                   float /*eps*/,
                                                   DiagListF32& C_out,
                                                   float* kernel_ms,
                                                   float* kernel_plus_copy_ms)
{
    const int N = A.N;
    // 1) Build C structure & pairs
    std::vector<int> offsetsC, lenC, cPairPtr, cPairA, cPairB;
    {
        std::vector<int> hBoff(planB->offsetsB.size());
        thrust::copy(planB->offsetsB.begin(), planB->offsetsB.end(), hBoff.begin());
        build_C_offsets_and_pairs(N, A.offsets, hBoff, offsetsC, lenC, cPairPtr, cPairA, cPairB);
    }

    const int numDiagC = (int)offsetsC.size();
    if (numDiagC == 0) {
        C_out = {N, {}, {0}, {}};
        if (kernel_ms) *kernel_ms = 0.f;
        if (kernel_plus_copy_ms) *kernel_plus_copy_ms = 0.f;
        return;
    }

    // ptrC
    std::vector<int> ptrC(numDiagC + 1, 0);
    for (int i = 0; i < numDiagC; ++i) ptrC[i+1] = ptrC[i] + lenC[i];
    const int nnzC = ptrC.back();

    // 2) Upload buffers
    thrust::device_vector<int>   d_offsetsA(A.offsets.begin(), A.offsets.end());
    thrust::device_vector<int>   d_ptrA(A.ptr.begin(),         A.ptr.end());
    thrust::device_vector<float> d_valsA(A.vals.begin(),       A.vals.end());

    thrust::device_vector<int>   d_offsetsC(offsetsC.begin(),  offsetsC.end());
    thrust::device_vector<int>   d_lenC(lenC.begin(),          lenC.end());
    thrust::device_vector<int>   d_ptrC(ptrC.begin(),          ptrC.end());

    thrust::device_vector<int>   d_cPairPtr(cPairPtr.begin(),  cPairPtr.end());
    thrust::device_vector<int>   d_cPairA(cPairA.begin(),      cPairA.end());
    thrust::device_vector<int>   d_cPairB(cPairB.begin(),      cPairB.end());

    thrust::device_vector<float> d_outValsC(nnzC, 0.0f);
    thrust::device_vector<float> d_valsC_final(nnzC);
    thrust::device_vector<float> d_absSumC(numDiagC, 0.0f);

    // 3) Grid/block (3D): tiles along C, C-diagonals, K-segments
    int maxLenC = 0; for (int L : lenC) maxLenC = std::max(maxLenC, L);
    const int threads = 256;
    const int tiles_x = (maxLenC + threads - 1) / threads;
    const int numSeg  = (N + seg_size - 1) / seg_size;

    dim3 block(threads, 1, 1);
    dim3 grid(tiles_x, numDiagC, numSeg);

    // 4) Timing + launch
    cudaEvent_t k0,k1,k2;
    CUDA_CALL(cudaEventCreate(&k0));
    CUDA_CALL(cudaEventCreate(&k1));
    CUDA_CALL(cudaEventCreate(&k2));

    CUDA_CALL(cudaEventRecord(k0, 0));
    spmspm_perc_grouped_kernel<float><<<grid, block>>>(
        N, seg_size,
        (int)A.offsets.size(),
        thrust::raw_pointer_cast(d_offsetsA.data()),
        thrust::raw_pointer_cast(d_ptrA.data()),
        thrust::raw_pointer_cast(d_valsA.data()),
        planB->numDiagB,
        thrust::raw_pointer_cast(planB->offsetsB.data()),
        thrust::raw_pointer_cast(planB->ptrB.data()),
        thrust::raw_pointer_cast(planB->valsB.data()),
        numDiagC,
        thrust::raw_pointer_cast(d_offsetsC.data()),
        thrust::raw_pointer_cast(d_lenC.data()),
        thrust::raw_pointer_cast(d_ptrC.data()),
        thrust::raw_pointer_cast(d_cPairPtr.data()),
        thrust::raw_pointer_cast(d_cPairA.data()),
        thrust::raw_pointer_cast(d_cPairB.data()),
        thrust::raw_pointer_cast(d_outValsC.data()),
        thrust::raw_pointer_cast(d_absSumC.data()));
    CUDA_CALL(cudaEventRecord(k1, 0));

    CUDA_CALL(cudaMemcpyAsync(
        thrust::raw_pointer_cast(d_valsC_final.data()),
        thrust::raw_pointer_cast(d_outValsC.data()),
        nnzC * sizeof(float),
        cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaEventRecord(k2, 0));
    CUDA_CALL(cudaEventSynchronize(k2));

    float ms_kernel = 0.f, ms_total = 0.f;
    CUDA_CALL(cudaEventElapsedTime(&ms_kernel, k0, k1));
    CUDA_CALL(cudaEventElapsedTime(&ms_total,  k0, k2));
    CUDA_CALL(cudaEventDestroy(k0));
    CUDA_CALL(cudaEventDestroy(k1));
    CUDA_CALL(cudaEventDestroy(k2));
    if (kernel_ms)           *kernel_ms = ms_kernel;
    if (kernel_plus_copy_ms) *kernel_plus_copy_ms = ms_total;

    // 5) Download
    std::vector<int>   h_offsetsC(offsetsC.size());
    std::vector<int>   h_ptrC(ptrC.size());
    std::vector<float> h_valsC(nnzC);
    thrust::copy(d_offsetsC.begin(), d_offsetsC.end(), h_offsetsC.begin());
    thrust::copy(d_ptrC.begin(),     d_ptrC.end(),     h_ptrC.begin());
    thrust::copy(d_valsC_final.begin(), d_valsC_final.end(), h_valsC.begin());

    C_out.N = N;
    C_out.offsets = std::move(h_offsetsC);
    C_out.ptr     = std::move(h_ptrC);
    C_out.vals    = std::move(h_valsC);
}

// ---------- A-centric adjacency (for reuse kernel) ----------
static void build_Acentric_pairs(
    int N,
    const std::vector<int>& offsetsA,
    const std::vector<int>& offsetsB,
    const std::vector<int>& offsetsC,   // unique & sorted
    std::vector<int>& aPairPtr,         // size = numDiagA + 1
    std::vector<int>& aPairBIdx,        // flattened B indices
    std::vector<int>& aPairCIdx)        // flattened C indices
{
    aPairPtr.assign(offsetsA.size()+1, 0);
    aPairBIdx.clear();
    aPairCIdx.clear();
    if (offsetsA.empty() || offsetsB.empty() || offsetsC.empty()) return;

    const int bmin = offsetsB.front(), bmax = offsetsB.back();
    std::vector<int> Bindex(bmax - bmin + 1, -1);
    for (int i = 0; i < (int)offsetsB.size(); ++i) Bindex[offsetsB[i] - bmin] = i;

    const int cmin = offsetsC.front(), cmax = offsetsC.back();
    std::vector<int> Cindex(cmax - cmin + 1, -1);
    for (int i = 0; i < (int)offsetsC.size(); ++i) Cindex[offsetsC[i] - cmin] = i;

    auto diag_len = [](int N, int off){ int a = (off>=0?off:-off); int L=N-a; return L>0?L:0; };

    int acc = 0;
    for (int aIdx = 0; aIdx < (int)offsetsA.size(); ++aIdx) {
        const int a_off = offsetsA[aIdx];
        const int clamp_min = std::max(a_off + offsetsB.front(), -(N - 1));
        const int clamp_max = std::min(a_off + offsetsB.back(),   (N - 1));
        for (int c_off = clamp_min; c_off <= clamp_max; ++c_off) {
            if (c_off < cmin || c_off > cmax) continue;
            const int b_off = c_off - a_off;
            if (b_off < bmin || b_off > bmax) continue;
            const int bIdx = Bindex[b_off - bmin];
            const int cIdx = Cindex[c_off - cmin];
            if (bIdx < 0 || cIdx < 0) continue;
            if (diag_len(N, c_off) <= 0) continue;

            aPairBIdx.push_back(bIdx);
            aPairCIdx.push_back(cIdx);
            ++acc;
        }
        aPairPtr[aIdx+1] = acc;
    }
}

// ---------- KERNEL 3: A-centric grouped (shared-memory A reuse) ----------
template <typename T, int TILE_I=256>
__global__ void spmspm_Acentric_grouped_kernel(
    int N, int seg_size,
    // A
    int numDiagA, const int* __restrict__ offsetsA,
    const int* __restrict__ ptrA, const T* __restrict__ valsA,
    // B
    int numDiagB, const int* __restrict__ offsetsB,
    const int* __restrict__ ptrB, const T* __restrict__ valsB,
    // C (linearized by diagonals)
    int numDiagC, const int* __restrict__ offsetsC,
    const int* __restrict__ lenC, const int* __restrict__ ptrC,
    // A-centric adjacency
    const int* __restrict__ aPairPtr,   // len = numDiagA+1
    const int* __restrict__ aPairBIdx,  // flattened
    const int* __restrict__ aPairCIdx,  // flattened
    // Output
    T* __restrict__ outValsC
){
    extern __shared__ T sA[];  // TILE_I

    const int aIdx = blockIdx.y;
    const int seg  = blockIdx.z;
    if (aIdx >= numDiagA) return;

    const int a_off = offsetsA[aIdx];
    const int k_begin = seg * seg_size;
    const int k_end   = min(N, k_begin + seg_size); // exclusive

    // Base i-range for A (exclusive hi)
    int i_low = start_row(a_off);
    int i_high_ex = end_row(N, a_off);

    // Segment window: k = i + a_off in [k_begin, k_end) => i in [k_begin - a_off, k_end - a_off)
    int tmp = k_begin - a_off; if (tmp > i_low)      i_low = tmp;
    tmp     = k_end   - a_off; if (tmp < i_high_ex)  i_high_ex = tmp;
    if (i_low >= i_high_ex) return;

    // Tile i-range
    const int tile_id    = blockIdx.x;
    const int tile_start = i_low + tile_id * TILE_I;
    if (tile_start >= i_high_ex) return;
    const int tile_end_ex = min(tile_start + TILE_I, i_high_ex);
    const int tile_len    = tile_end_ex - tile_start;

    // Stage A stripe into shared memory
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
        const int i  = tile_start + t;
        const int tA = local_index_or_neg1(N, a_off, i); // guaranteed valid
        sA[t] = valsA[ptrA[aIdx] + tA];
    }
    __syncthreads();

    // Loop over partners for this A-diagonal
    const int p0 = aPairPtr[aIdx];
    const int p1 = aPairPtr[aIdx+1];
    for (int p = p0; p < p1; ++p) {
        const int bIdx = aPairBIdx[p];
        const int cIdx = aPairCIdx[p];

        const int b_off = offsetsB[bIdx];
        const int c_off = offsetsC[cIdx];

        // Intersect this tile with B window (k in [start_row(b_off), end_row(b_off))) and C window (i in [start_row(c_off), end_row(c_off)))
        int lo = tile_start;
        int hi_ex = tile_end_ex;

        tmp = start_row(b_off) - a_off; if (tmp > lo)     lo    = tmp;
        tmp = end_row(N, b_off) - a_off; if (tmp < hi_ex) hi_ex = tmp;

        tmp = start_row(c_off); if (tmp > lo)     lo    = tmp;
        tmp = end_row(N, c_off); if (tmp < hi_ex) hi_ex = tmp;

        if (lo >= hi_ex) continue;

        const int t0 = lo - tile_start;
        const int t1 = hi_ex - tile_start; // exclusive

        for (int t = threadIdx.x + t0; t < t1; t += blockDim.x) {
            const int i   = tile_start + t;
            const int k   = i + a_off;
            const int tB  = local_index_or_neg1(N, b_off, k); // valid by construction
            const T prod  = sA[t] * valsB[ptrB[bIdx] + tB];

            const int tC  = i - start_row(c_off);
            atomicAdd(&outValsC[ptrC[cIdx] + tC], prod);
        }
        __syncthreads(); // reuse sA for next partner
    }
}

// ---------- HOST 3: A-centric grouped (reuse) launcher ----------
void multiply_sparse_grouped_Areuse_with_timing_copy_plan(const DiagListF32& A,
                                                          const DiaBPlan* planB,
                                                          int seg_size,   // e.g., 1024–4096
                                                          DiagListF32& C_out,
                                                          float* kernel_ms,
                                                          float* kernel_plus_copy_ms)
{
    const int N = A.N;

    // Build C (needed to size output and map partners to cIdx)
    std::vector<int> offsetsC, lenC, cPairPtr_dummy, cPairA_dummy, cPairB_dummy;
    {
        std::vector<int> hBoff(planB->offsetsB.size());
        thrust::copy(planB->offsetsB.begin(), planB->offsetsB.end(), hBoff.begin());
        build_C_offsets_and_pairs(N, A.offsets, hBoff, offsetsC, lenC, cPairPtr_dummy, cPairA_dummy, cPairB_dummy);
    }
    if (offsetsC.empty()) {
        C_out = {N, {}, {0}, {}};
        if (kernel_ms) *kernel_ms = 0.f;
        if (kernel_plus_copy_ms) *kernel_plus_copy_ms = 0.f;
        return;
    }

    // ptrC
    std::vector<int> ptrC(offsetsC.size()+1, 0);
    for (int i=0;i<(int)offsetsC.size();++i) ptrC[i+1] = ptrC[i] + lenC[i];
    const int nnzC = ptrC.back();

    // Build A-centric adjacency
    std::vector<int> aPairPtr, aPairBIdx, aPairCIdx;
    {
        std::vector<int> hBoff(planB->offsetsB.size());
        thrust::copy(planB->offsetsB.begin(), planB->offsetsB.end(), hBoff.begin());
        build_Acentric_pairs(N, A.offsets, hBoff, offsetsC, aPairPtr, aPairBIdx, aPairCIdx);
    }

    // Upload device buffers
    thrust::device_vector<int>   d_offsetsA(A.offsets.begin(), A.offsets.end());
    thrust::device_vector<int>   d_ptrA(A.ptr.begin(),         A.ptr.end());
    thrust::device_vector<float> d_valsA(A.vals.begin(),       A.vals.end());

    // use B directly from planB
    const int* d_offsetsB = thrust::raw_pointer_cast(planB->offsetsB.data());
    const int* d_ptrB     = thrust::raw_pointer_cast(planB->ptrB.data());
    const float* d_valsB  = thrust::raw_pointer_cast(planB->valsB.data());

    thrust::device_vector<int>   d_offsetsC(offsetsC.begin(),  offsetsC.end());
    thrust::device_vector<int>   d_lenC(lenC.begin(),          lenC.end());
    thrust::device_vector<int>   d_ptrC(ptrC.begin(),          ptrC.end());

    thrust::device_vector<int>   d_aPairPtr(aPairPtr.begin(),  aPairPtr.end());
    thrust::device_vector<int>   d_aPairBIdx(aPairBIdx.begin(),aPairBIdx.end());
    thrust::device_vector<int>   d_aPairCIdx(aPairCIdx.begin(),aPairCIdx.end());

    thrust::device_vector<float> d_outValsC(nnzC, 0.0f);
    thrust::device_vector<float> d_valsC_final(nnzC);

    // Grid/block for A-centric kernel
    const int TILE_I = 256;
    int maxLenA = 0;
    for (int a_off : A.offsets) {
        const int s = (a_off >= 0 ? 0 : -a_off);
        const int e = (a_off >= 0 ? N - a_off : N);
        maxLenA = std::max(maxLenA, e - s);
    }
    const int threads = 256;
    const int tiles_x = (maxLenA + TILE_I - 1) / TILE_I;
    const int numSeg  = (N + seg_size - 1) / seg_size;

    dim3 block(threads, 1, 1);
    dim3 grid(tiles_x, (int)A.offsets.size(), numSeg);
    size_t shmem = TILE_I * sizeof(float);

    // Timing/launch
    cudaEvent_t k0,k1,k2;
    CUDA_CALL(cudaEventCreate(&k0));
    CUDA_CALL(cudaEventCreate(&k1));
    CUDA_CALL(cudaEventCreate(&k2));

    CUDA_CALL(cudaEventRecord(k0, 0));
    spmspm_Acentric_grouped_kernel<float, 256><<<grid, block, shmem>>>(
        N, seg_size,
        (int)A.offsets.size(),
        thrust::raw_pointer_cast(d_offsetsA.data()),
        thrust::raw_pointer_cast(d_ptrA.data()),
        thrust::raw_pointer_cast(d_valsA.data()),
        planB->numDiagB,
        d_offsetsB,
        d_ptrB,
        d_valsB,
        (int)offsetsC.size(),
        thrust::raw_pointer_cast(d_offsetsC.data()),
        thrust::raw_pointer_cast(d_lenC.data()),
        thrust::raw_pointer_cast(d_ptrC.data()),
        thrust::raw_pointer_cast(d_aPairPtr.data()),
        thrust::raw_pointer_cast(d_aPairBIdx.data()),
        thrust::raw_pointer_cast(d_aPairCIdx.data()),
        thrust::raw_pointer_cast(d_outValsC.data()));
    CUDA_CALL(cudaEventRecord(k1, 0));

    CUDA_CALL(cudaMemcpyAsync(
        thrust::raw_pointer_cast(d_valsC_final.data()),
        thrust::raw_pointer_cast(d_outValsC.data()),
        nnzC * sizeof(float),
        cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaEventRecord(k2, 0));
    CUDA_CALL(cudaEventSynchronize(k2));

    float ms_kernel = 0.f, ms_total = 0.f;
    CUDA_CALL(cudaEventElapsedTime(&ms_kernel, k0, k1));
    CUDA_CALL(cudaEventElapsedTime(&ms_total,  k0, k2));
    CUDA_CALL(cudaEventDestroy(k0));
    CUDA_CALL(cudaEventDestroy(k1));
    CUDA_CALL(cudaEventDestroy(k2));
    if (kernel_ms)           *kernel_ms = ms_kernel;
    if (kernel_plus_copy_ms) *kernel_plus_copy_ms = ms_total;

    // Download C
    std::vector<int>   h_offsetsC(offsetsC.size());
    std::vector<int>   h_ptrC(ptrC.size());
    std::vector<float> h_valsC(nnzC);
    thrust::copy(d_offsetsC.begin(), d_offsetsC.end(), h_offsetsC.begin());
    thrust::copy(d_ptrC.begin(),     d_ptrC.end(),     h_ptrC.begin());
    thrust::copy(d_valsC_final.begin(), d_valsC_final.end(), h_valsC.begin());
    C_out.N      = N;
    C_out.offsets= std::move(h_offsetsC);
    C_out.ptr    = std::move(h_ptrC);
    C_out.vals   = std::move(h_valsC);
}

// ---------- (Optional) tiny dispatcher ----------
enum class GroupMode { None, Ccentric, AcentricReuse };
void multiply_sparse_auto_with_timing_copy_plan(const DiagListF32& A,
                                                const DiaBPlan* planB,
                                                float eps,
                                                int seg_size,
                                                GroupMode mode,
                                                DiagListF32& C_out,
                                                float* k_ms,
                                                float* kpc_ms)
{
    if (mode == GroupMode::None || seg_size <= 0) {
        multiply_sparse_noPad_with_timing_copy_plan(A, planB, eps, C_out, k_ms, kpc_ms);
    } else if (mode == GroupMode::AcentricReuse) {
        multiply_sparse_grouped_Areuse_with_timing_copy_plan(A, planB, seg_size, C_out, k_ms, kpc_ms);
    } else {
        multiply_sparse_grouped_with_timing_copy_plan(A, planB, seg_size, eps, C_out, k_ms, kpc_ms);
    }
}
