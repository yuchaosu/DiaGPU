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
__device__ __forceinline__ int end_row(int N,int offset){ return (offset >= 0 ? N - offset : N); }

//verify if current row is inside the current diagonal
__device__ __forceinline__
int local_index_or_neg1(int N, int offset, int i) {
    int s = (offset >= 0 ? 0 : -offset);
    int e = (offset >= 0 ? N - offset : N);
    return (i >= s && i < e) ? (i - s) : -1;
}

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

void destroy_B_plan(DiaBPlan* plan) {
    delete plan;
}

// ---------- kernel: per-output-diagonal (C) numeric multiply ----------
template <typename T>
__global__ void spmspm_perc_kernel(
    int N,
    // A
    int numDiagA, const int* __restrict__ offsetsA, const int* __restrict__ ptrA, const T* __restrict__ valsA,
    // B
    int numDiagB, const int* __restrict__ offsetsB, const int* __restrict__ ptrB, const T* __restrict__ valsB,
    // C meta (temporary, pre-compaction)
    int numDiagC, const int* __restrict__ offsetsC, const int* __restrict__ lenC, const int* __restrict__ ptrC,
    // pairs that contribute to each C diagonal
    const int* __restrict__ cPairPtr, const int* __restrict__ cPairA, const int* __restrict__ cPairB,
    // outputs
    T* __restrict__ outValsC,       // concatenated values for all C diagonals
    T* __restrict__ absSumC         // per-diagonal sum of |values| to decide pruning
){
    int cDiagIdx = blockIdx.y;                      // which diagonal of C this block computes
    if (cDiagIdx >= numDiagC) return;

    int Lc  = lenC[cDiagIdx];                       // length of this C diagonal
    int t   = blockIdx.x * blockDim.x + threadIdx.x;// local position along this diagonal
    T acc   = T(0);
    int offsetC = offsetsC[cDiagIdx];               // offset of this C diagonal

    // Traverse all contributing (offsetA, offsetB) pairs for this C diagonal
    for (int p = cPairPtr[cDiagIdx]; p < cPairPtr[cDiagIdx+1]; ++p) {
        int dA = cPairA[p], dB = cPairB[p];
        int offsetA = offsetsA[dA], offsetB = offsetsB[dB];

        // Intersection of valid row ranges for A(offsetA), C(offsetC), and B(offsetB) shifted by offsetA
        int i_min = start_row(offsetA);
        int tmp   = start_row(offsetC);              if (tmp > i_min) i_min = tmp;
        tmp       = start_row(offsetB) - offsetA;    if (tmp > i_min) i_min = tmp;

        int i_max = end_row(N, offsetA);
        tmp       = end_row(N, offsetC);             if (tmp < i_max) i_max = tmp;
        tmp       = end_row(N, offsetB) - offsetA;   if (tmp < i_max) i_max = tmp;

        if (i_min >= i_max) continue;

        // Convert to local index window along C diagonal
        int t0 = i_min - start_row(offsetC);
        int t1 = i_max - start_row(offsetC);

        if (t >= t0 && t < t1) {
            int i  = t + start_row(offsetC);
            int tA = local_index_or_neg1(N, offsetA, i);           if (tA < 0) continue;
            int tB = local_index_or_neg1(N, offsetB, i + offsetA); if (tB < 0) continue;

            T a = valsA[ptrA[dA] + tA];
            T b = valsB[ptrB[dB] + tB];
            acc += a * b;
            //print offset A value A, offset B value B and partial sum C offset C
            //printf("A[%d] = %g, B[%d] = %g, C[%d] += %g, C[%d] = %g\n", offsetA, a, offsetB, b, offsetC, a * b, offsetC, acc);
        }
    }

    if (t < Lc) {
        outValsC[ptrC[cDiagIdx] + t] = acc;
        atomicAdd(&absSumC[cDiagIdx], acc >= 0 ? acc : -acc);
    }
}



// ---------- host: build output-diagonals (C) and contributing pairs ----------
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

    int bmin = offsetsB.front(), bmax = offsetsB.back();
    std::vector<int> Bindex(bmax - bmin + 1, -1);
    for (int i = 0; i < (int)offsetsB.size(); ++i) Bindex[offsetsB[i] - bmin] = i;

    auto diag_len_host = [](int N, int offset) { int a=(offset>=0?offset:-offset); int L=N-a; return L>0?L:0; };

    std::unordered_map<int, std::vector<std::pair<int,int>>> buckets; // key: offsetC
    buckets.reserve(offsetsA.size() * 2);

    for (int dA = 0; dA < (int)offsetsA.size(); ++dA) {
        int offsetA = offsetsA[dA];
        int clamp_min = std::max(offsetA + offsetsB.front(), -(N - 1));
        int clamp_max = std::min(offsetA + offsetsB.back(),   (N - 1));
        for (int offsetC = clamp_min; offsetC <= clamp_max; ++offsetC) {
            int offsetB = offsetC - offsetA;
            if (offsetB < bmin || offsetB > bmax) continue;
            int dB = Bindex[offsetB - bmin];
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
        int offsetC = offsetsC[idx];
        auto& vec = buckets[offsetC];
        lenC[idx] = diag_len_host(N, offsetC);
        for (auto& pr : vec) { cPairA.push_back(pr.first); cPairB.push_back(pr.second); }
        cPairPtr.push_back((int)cPairA.size());
    }
}

// ---------- thrust functors ----------
struct IsNonZero { __host__ __device__ int operator()(float s) const { return s != 0.0f; } };
struct GreaterThanEps { float eps; __host__ __device__ int operator()(float s) const { return fabsf(s) > eps ? 1 : 0; } };
struct MaskLen { __host__ __device__ int operator()(int k, int L) const { return k ? L : 0; } };

// ---------- public API ----------
void multiply_sparse_noPad_with_timing_copy_plan(const DiagListF32& A,
                                                 const DiaBPlan* planB,
                                                 float /*eps*/,
                                                 DiagListF32& C_out,
                                                 float* kernel_ms,
                                                 float* kernel_plus_copy_ms)
{
    const int N = A.N;
    // 1) Build output structure and contributing pairs (host)
    std::vector<int> offsetsC, lenC, cPairPtr, cPairA, cPairB;
    build_C_offsets_and_pairs(N, A.offsets, 
                              /*B.offsets*/ std::vector<int>(planB->offsetsB.begin(), planB->offsetsB.end()),
                              offsetsC, lenC, cPairPtr, cPairA, cPairB);

    const int numDiagC = (int)offsetsC.size();
    if (numDiagC == 0) {
        C_out = {N, {}, {0}, {}};
        if (kernel_ms) *kernel_ms = 0.f;
        if (kernel_plus_copy_ms) *kernel_plus_copy_ms = 0.f;
        return;
    }

    // ptrC
    std::vector<int> ptrC(numDiagC + 1, 0);
    for (int i=0;i<numDiagC;++i) ptrC[i+1] = ptrC[i] + lenC[i];
    const int nnzC = ptrC.back();

    // 2) Upload A + C-meta + pairs (B is already on device via plan)
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
    thrust::device_vector<float> d_absSumC(numDiagC, 0.0f); // kept if kernel signature needs it

    // 3) Launch + time (kernel-only and kernel+copy)
    int maxLenC = 0; for (int L : lenC) maxLenC = std::max(maxLenC, L);
    const dim3 block(256,1);
    const dim3 grid((maxLenC + block.x - 1)/block.x, numDiagC);

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

    // Device→device copy to simulate cuSPARSE "copy"
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

    // 4) Download raw result (no compaction)
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



// DiagListF32 power_repeat_right_timed(const DiagListF32& A, int power, float eps,
//                                      float* total_kernel_ms, float* total_kernel_copy_ms) {
//     if (total_kernel_ms) *total_kernel_ms = 0.f;
//     if (total_kernel_copy_ms) *total_kernel_copy_ms = 0.f;
//     if (power <= 1) return A;

//     DiagListF32 C = A;
//     for (int k = 2; k <= power; ++k) {
//         DiagListF32 next;
//         float ms_k = 0.f, ms_kc = 0.f;
//         multiply_sparse_noPad_with_timing_copy(C, A, eps, next, &ms_k, &ms_kc);
//         if (total_kernel_ms)        *total_kernel_ms        += ms_k;
//         if (total_kernel_copy_ms)   *total_kernel_copy_ms   += ms_kc;
//         C = std::move(next);
//         // avoid dump_list here if you care about fair timing
//     }
//     return C;
// }

