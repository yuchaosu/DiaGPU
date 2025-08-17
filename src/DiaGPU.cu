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
            printf("A[%d] = %g, B[%d] = %g, C[%d] += %g, C[%d] = %g\n", offsetA, a, offsetB, b, offsetC, a * b, offsetC, acc);
        }
    }

    if (t < Lc) {
        outValsC[ptrC[cDiagIdx] + t] = acc;
        atomicAdd(&absSumC[cDiagIdx], acc >= 0 ? acc : -acc);
    }
}

// ---------- kernel: compact kept C diagonals (drop all-zero ones) ----------
template <typename T>
__global__ void compact_segments_kernel(
    int numDiagC,
    const int* __restrict__ keepC,
    const int* __restrict__ offsetsC,
    const int* __restrict__ lenC,
    const int* __restrict__ ptrC,
    const T*   __restrict__ valsC,
    const int* __restrict__ compactIndexC,   // new compact index for each kept diagonal
    const int* __restrict__ ptrCompact,      // ptr array for compacted layout
    int*       __restrict__ offsetsCompact,
    T*         __restrict__ valsCompact)
{
    int c = blockIdx.y;
    if (c >= numDiagC || !keepC[c]) return;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int L = lenC[c]; if (t >= L) return;

    int pos  = compactIndexC[c];
    valsCompact[ptrCompact[pos] + t] = valsC[ptrC[c] + t];
    if (t == 0) offsetsCompact[pos] = offsetsC[c];
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
void multiply_sparse_noPad(const DiagListF32& A,
                           const DiagListF32& B,
                           float eps,
                           DiagListF32& C_out)
{
    const int N = A.N;

    // 1) Build C’s diagonal set and contributing pairs (host)
    std::vector<int> offsetsC, lenC, cPairPtr, cPairA, cPairB;
    build_C_offsets_and_pairs(N, A.offsets, B.offsets, offsetsC, lenC, cPairPtr, cPairA, cPairB);
    //print cPairPtr, cPairA, cPairB
    for (int i = 0; i < (int)cPairPtr.size(); ++i) {
        printf("cPairPtr[%d] = %d\n", i, cPairPtr[i]);
    }
    for (int i = 0; i < (int)cPairA.size(); ++i) {
        printf("cPairA[%d] = %d\n", i, cPairA[i]);
    }
    for (int i = 0; i < (int)cPairB.size(); ++i) {
        printf("cPairB[%d] = %d\n", i, cPairB[i]);
    }

    int numDiagC = (int)offsetsC.size();
    if (numDiagC == 0) { C_out = {N, {}, {0}, {}}; return; }

    // 2) Temporary (pre-compaction) ptr for C
    std::vector<int> ptrC(numDiagC + 1, 0);
    for (int i = 0; i < numDiagC; ++i) ptrC[i+1] = ptrC[i] + lenC[i];
    int nnzC = ptrC.back();

    // 3) Upload inputs/meta
    thrust::device_vector<int>   d_offsetsA(A.offsets.begin(), A.offsets.end());
    thrust::device_vector<int>   d_ptrA(A.ptr.begin(),         A.ptr.end());
    thrust::device_vector<float> d_valsA(A.vals.begin(),       A.vals.end());

    thrust::device_vector<int>   d_offsetsB(B.offsets.begin(), B.offsets.end());
    thrust::device_vector<int>   d_ptrB(B.ptr.begin(),         B.ptr.end());
    thrust::device_vector<float> d_valsB(B.vals.begin(),       B.vals.end());

    thrust::device_vector<int>   d_offsetsC(offsetsC.begin(), offsetsC.end());
    thrust::device_vector<int>   d_lenC(lenC.begin(), lenC.end());
    thrust::device_vector<int>   d_ptrC(ptrC.begin(), ptrC.end());

    thrust::device_vector<int>   d_cPairPtr(cPairPtr.begin(), cPairPtr.end());
    thrust::device_vector<int>   d_cPairA(cPairA.begin(),     cPairA.end());
    thrust::device_vector<int>   d_cPairB(cPairB.begin(),     cPairB.end());

    thrust::device_vector<float> d_outValsC(nnzC, 0.0f);
    thrust::device_vector<float> d_absSumC(numDiagC, 0.0f);

    // 4) Launch numeric
    int maxLenC = 0; for (int L : lenC) maxLenC = std::max(maxLenC, L);
    dim3 block(256, 1);
    dim3 grid((maxLenC + block.x - 1) / block.x, numDiagC);

    spmspm_perc_kernel<float><<<grid, block>>>(
        N,
        (int)A.offsets.size(),
        thrust::raw_pointer_cast(d_offsetsA.data()),
        thrust::raw_pointer_cast(d_ptrA.data()),
        thrust::raw_pointer_cast(d_valsA.data()),
        (int)B.offsets.size(),
        thrust::raw_pointer_cast(d_offsetsB.data()),
        thrust::raw_pointer_cast(d_ptrB.data()),
        thrust::raw_pointer_cast(d_valsB.data()),
        numDiagC,
        thrust::raw_pointer_cast(d_offsetsC.data()),
        thrust::raw_pointer_cast(d_lenC.data()),
        thrust::raw_pointer_cast(d_ptrC.data()),
        thrust::raw_pointer_cast(d_cPairPtr.data()),
        thrust::raw_pointer_cast(d_cPairA.data()),
        thrust::raw_pointer_cast(d_cPairB.data()),
        thrust::raw_pointer_cast(d_outValsC.data()),
        thrust::raw_pointer_cast(d_absSumC.data()));
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Download temp C (pre-compaction)
    std::vector<int>    h_offsetsC(offsetsC.size());
    std::vector<int>    h_lenC(lenC.size());
    std::vector<int>    h_ptrC(ptrC.size());
    std::vector<float>  h_outValsC(nnzC);
    std::vector<float>  h_absSumC(numDiagC);

    thrust::copy(d_offsetsC.begin(), d_offsetsC.end(), h_offsetsC.begin());
    thrust::copy(d_lenC.begin(),     d_lenC.end(),     h_lenC.begin());
    thrust::copy(d_ptrC.begin(),     d_ptrC.end(),     h_ptrC.begin());
    thrust::copy(d_outValsC.begin(), d_outValsC.end(), h_outValsC.begin());
    thrust::copy(d_absSumC.begin(),  d_absSumC.end(),  h_absSumC.begin());

    // Host-side compaction: keep diagonals with |sum| > eps (or != 0 if eps==0)
    C_out.N = N;
    C_out.offsets.clear();
    C_out.ptr.clear();
    C_out.vals.clear();

    C_out.ptr.push_back(0);  // ptr[0] = 0

    for (int c = 0; c < numDiagC; ++c) {
        float s = h_absSumC[c];
        bool keep = (eps == 0.0f) ? (s != 0.0f) : (std::fabs(s) > eps);
        if (!keep) continue;

        // append this diagonal in order
        C_out.offsets.push_back(h_offsetsC[c]);

        int L  = h_lenC[c];                     // expected length = N - |offset|
        int p0 = h_ptrC[c], p1 = h_ptrC[c+1];   // slice in the temp buffer
        // safety guard: if anything is off, clamp by the slice
        if (p1 - p0 < L) L = p1 - p0;

        // copy values
        C_out.vals.insert(C_out.vals.end(), h_outValsC.begin() + p0, h_outValsC.begin() + p0 + L);

        // advance ptr
        C_out.ptr.push_back(C_out.ptr.back() + L);
    }

    // (Optional but recommended) sanity: ptr/offset lengths must match
    if ((int)C_out.ptr.size() != (int)C_out.offsets.size() + 1 ||
        (int)C_out.vals.size() != C_out.ptr.back()) {
        fprintf(stderr, "[compact-host] layout mismatch: ptr.size=%zu offs.size=%zu vals=%zu ptr.back()=%d\n",
                C_out.ptr.size(), C_out.offsets.size(), C_out.vals.size(), C_out.ptr.back());
        std::exit(1);
    }
    //print C_out
    dump_list("C_out", C_out);

}

DiagListF32 power_repeat_right(const DiagListF32& A, int power, float eps) {
    if (power <= 1) return A;
    DiagListF32 C = A;
    for (int k = 2; k <= power; ++k) {
        DiagListF32 next;
        multiply_sparse_noPad(C, A, eps, next);   // C = C * A
        C = std::move(next);
        dump_list("power_repeat_right", C);
        std::cout << std::endl;
    }
    return C;
}
