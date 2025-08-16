#include "DiaGPU.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <map>
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

// ===== internal packed form (device-friendly) =====
struct PackedDiagF32 {
    int N = 0;
    std::vector<int>   offsets; // sorted, size K
    std::vector<int>   ptr;     // size K+1
    std::vector<float> vals;    // sum(N - |o|)
};

static inline int diag_len(int N, int o) { int a = (o >= 0 ? o : -o); int L = N - a; return L > 0 ? L : 0; }

// pack/unpack between public map and internal packed
static PackedDiagF32 pack_from_map(int N, const DiagMapF32& M) {
    PackedDiagF32 P; P.N = N;
    P.offsets.reserve(M.size());
    for (auto &kv : M) P.offsets.push_back(kv.first);
    std::sort(P.offsets.begin(), P.offsets.end());
    P.ptr.assign(P.offsets.size() + 1, 0);
    for (size_t d = 0; d < P.offsets.size(); ++d) {
        int L = diag_len(N, P.offsets[d]);
        P.ptr[d+1] = P.ptr[d] + L;
    }
    P.vals.assign(P.ptr.back(), 0.0f);
    for (size_t d = 0; d < P.offsets.size(); ++d) {
        int o = P.offsets[d];
        auto it = M.find(o);
        if (it == M.end()) continue; // shouldn’t happen
        const auto& v = it->second;
        int L = diag_len(N, o);
        if ((int)v.size() != L) {
            fprintf(stderr, "[pack] length mismatch at offset %d: expected %d, got %zu\n", o, L, v.size());
            std::exit(1);
        }
        std::copy(v.begin(), v.end(), P.vals.begin() + P.ptr[d]);
    }
    return P;
}

static DiagMapF32 unpack_to_map(int N,
                                const std::vector<int>& offsets,
                                const std::vector<int>& ptr,
                                const std::vector<float>& vals)
{
    DiagMapF32 M;
    for (size_t d = 0; d + 1 < ptr.size(); ++d) {
        int o = offsets[d];
        int L = ptr[d+1] - ptr[d];
        if (L <= 0) continue;
        std::vector<float> v(L);
        std::copy(vals.begin() + ptr[d], vals.begin() + ptr[d+1], v.begin());
        // Optionally skip if all zeros (but our compaction already prunes)
        M.emplace(o, std::move(v));
    }
    return M;
}

// ===== device helpers =====
__device__ __forceinline__ int s_of(int o){ return (o >= 0 ? 0 : -o); }
__device__ __forceinline__ int e_of(int N,int o){ return (o >= 0 ? N - o : N); }

__device__ __forceinline__
int local_t_or_neg1_safe(int N, int o, int i) {
    int s = (o >= 0 ? 0 : -o);
    int e = (o >= 0 ? N - o : N);
    return (i >= s && i < e) ? (i - s) : -1;
}

// ===== kernels (per-q; no atomics for outputs) =====
template <typename T>
__global__ void spmspm_perq_kernel(
    int N,
    // A
    int KA, const int* __restrict__ offA, const int* __restrict__ ptrA, const T* __restrict__ valA,
    // B
    int KB, const int* __restrict__ offB, const int* __restrict__ ptrB, const T* __restrict__ valB,
    // output meta
    int KQ, const int* __restrict__ offQ, const int* __restrict__ lenQ, const int* __restrict__ ptrQ,
    // pair lists
    const int* __restrict__ qPairPtr, const int* __restrict__ qPairA, const int* __restrict__ qPairB,
    // outputs
    T* __restrict__ outVals,
    T* __restrict__ diagAbsSum // one per q
){
    int qid = blockIdx.y; if (qid >= KQ) return;
    int Lq = lenQ[qid];
    int t  = blockIdx.x * blockDim.x + threadIdx.x;
    T acc = T(0);
    int q = offQ[qid];

    for (int p = qPairPtr[qid]; p < qPairPtr[qid+1]; ++p) {
        int dA = qPairA[p], dB = qPairB[p];
        int oA = offA[dA],   oB = offB[dB];

        int i_min = s_of(oA);
        int tmp   = s_of(q);        if (tmp > i_min) i_min = tmp;
        tmp       = s_of(oB) - oA;  if (tmp > i_min) i_min = tmp;

        int i_max = e_of(N, oA);
        tmp       = e_of(N, q);     if (tmp < i_max) i_max = tmp;
        tmp       = e_of(N, oB)-oA; if (tmp < i_max) i_max = tmp;

        if (i_min >= i_max) continue;

        int t0 = i_min - s_of(q);
        int t1 = i_max - s_of(q);

        if (t >= t0 && t < t1) {
            int i  = t + s_of(q);
            int tA = local_t_or_neg1_safe(N, oA, i);       if (tA < 0) continue;
            int tB = local_t_or_neg1_safe(N, oB, i + oA);  if (tB < 0) continue;

            T a = valA[ptrA[dA] + tA];
            T b = valB[ptrB[dB] + tB];
            acc += a * b;
        }
    }

    // write + accumulate abs sum (simple, safe version)
    if (t < Lq) {
        outVals[ptrQ[qid] + t] = acc;
        atomicAdd(&diagAbsSum[qid], acc >= 0 ? acc : -acc);
    }
}

template <typename T>
__global__ void compact_segments_kernel(
    int KQ,
    const int* __restrict__ keep,
    const int* __restrict__ offQ,
    const int* __restrict__ lenQ,
    const int* __restrict__ ptrQ,
    const T*   __restrict__ valQ,
    const int* __restrict__ newIdxQ,
    const int* __restrict__ newPtr,
    int*       __restrict__ newOff,
    T*         __restrict__ newVals)
{
    int q = blockIdx.y; if (q >= KQ || !keep[q]) return;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int L = lenQ[q]; if (t >= L) return;
    int pos  = newIdxQ[q];
    newVals[newPtr[pos] + t] = valQ[ptrQ[q] + t];
    if (t == 0) newOff[pos] = offQ[q];
}

// ===== host helpers (pairs per q) =====
static void build_q_and_pairs(
    int N,
    const std::vector<int>& offA,
    const std::vector<int>& offB,
    std::vector<int>& offQ,
    std::vector<int>& lenQ,
    std::vector<int>& qPairPtr,
    std::vector<int>& qPairA,
    std::vector<int>& qPairB)
{
    int bmin = offB.front(), bmax = offB.back();
    std::vector<int> Bindex(bmax - bmin + 1, -1);
    for (int i = 0; i < (int)offB.size(); ++i) Bindex[offB[i] - bmin] = i;

    std::unordered_map<int, std::vector<std::pair<int,int>>> buckets;
    buckets.reserve(offA.size() * 2);

    for (int dA = 0; dA < (int)offA.size(); ++dA) {
        int oA = offA[dA];
        int clamp_min = std::max(oA + offB.front(), -(N - 1));
        int clamp_max = std::min(oA + offB.back(),   (N - 1));
        for (int q = clamp_min; q <= clamp_max; ++q) {
            int oB = q - oA;
            if (oB < bmin || oB > bmax) continue;
            int dB = Bindex[oB - bmin];
            if (dB < 0) continue;
            if (diag_len(N, q) <= 0) continue;
            buckets[q].emplace_back(dA, dB);
        }
    }

    offQ.clear(); offQ.reserve(buckets.size());
    for (auto& kv : buckets) offQ.push_back(kv.first);
    std::sort(offQ.begin(), offQ.end());

    lenQ.resize(offQ.size());
    qPairPtr.clear(); qPairA.clear(); qPairB.clear();
    qPairPtr.push_back(0);
    for (int idx = 0; idx < (int)offQ.size(); ++idx) {
        int q = offQ[idx];
        auto& vec = buckets[q];
        lenQ[idx] = diag_len(N, q);
        for (auto& pr : vec) { qPairA.push_back(pr.first); qPairB.push_back(pr.second); }
        qPairPtr.push_back((int)qPairA.size());
    }
}

// small thrust functors (no extended-lambda required)
struct IsNonZero { __host__ __device__ int operator()(float s) const { return s != 0.0f; } };
struct GreaterThanEps { float eps; __host__ __device__ int operator()(float s) const { return fabsf(s) > eps ? 1 : 0; } };
struct MaskLen { __host__ __device__ int operator()(int k, int L) const { return k ? L : 0; } };

// ===== public API (maps) =====
void multiply_sparse_noPad(int N,
                           const DiagMapF32& Amap,
                           const DiagMapF32& Bmap,
                           float eps,
                           DiagMapF32& C_out)
{
    // pack both sides
    PackedDiagF32 A = pack_from_map(N, Amap);
    PackedDiagF32 B = pack_from_map(N, Bmap);

    // build output-diagonal set and pair lists
    std::vector<int> offQ, lenQ, qPairPtr, qPairA, qPairB;
    build_q_and_pairs(N, A.offsets, B.offsets, offQ, lenQ, qPairPtr, qPairA, qPairB);
    int KQ = (int)offQ.size();
    if (KQ == 0) { C_out.clear(); return; }

    // nominal ptr for temporary (pre-compaction)
    std::vector<int> ptrQ(KQ + 1, 0);
    for (int i = 0; i < KQ; ++i) ptrQ[i+1] = ptrQ[i] + lenQ[i];
    int nnzQ = ptrQ.back();

    // upload inputs/meta
    thrust::device_vector<int>   d_offA(A.offsets.begin(), A.offsets.end());
    thrust::device_vector<int>   d_ptrA(A.ptr.begin(),     A.ptr.end());
    thrust::device_vector<float> d_valA(A.vals.begin(),    A.vals.end());

    thrust::device_vector<int>   d_offB(B.offsets.begin(), B.offsets.end());
    thrust::device_vector<int>   d_ptrB(B.ptr.begin(),     B.ptr.end());
    thrust::device_vector<float> d_valB(B.vals.begin(),    B.vals.end());

    thrust::device_vector<int>   d_offQ(offQ.begin(), offQ.end());
    thrust::device_vector<int>   d_lenQ(lenQ.begin(), lenQ.end());
    thrust::device_vector<int>   d_ptrQ(ptrQ.begin(), ptrQ.end());

    thrust::device_vector<int>   d_qPairPtr(qPairPtr.begin(), qPairPtr.end());
    thrust::device_vector<int>   d_qPairA(qPairA.begin(),     qPairA.end());
    thrust::device_vector<int>   d_qPairB(qPairB.begin(),     qPairB.end());

    thrust::device_vector<float> d_outVals(nnzQ, 0.0f);
    thrust::device_vector<float> d_diagSum(KQ, 0.0f);

    // launch numeric
    int maxL = 0; for (int L : lenQ) maxL = std::max(maxL, L);
    dim3 block(256, 1);
    dim3 grid((maxL + block.x - 1) / block.x, KQ);

    spmspm_perq_kernel<float><<<grid, block>>>(
        N,
        (int)A.offsets.size(),
        thrust::raw_pointer_cast(d_offA.data()),
        thrust::raw_pointer_cast(d_ptrA.data()),
        thrust::raw_pointer_cast(d_valA.data()),
        (int)B.offsets.size(),
        thrust::raw_pointer_cast(d_offB.data()),
        thrust::raw_pointer_cast(d_ptrB.data()),
        thrust::raw_pointer_cast(d_valB.data()),
        KQ,
        thrust::raw_pointer_cast(d_offQ.data()),
        thrust::raw_pointer_cast(d_lenQ.data()),
        thrust::raw_pointer_cast(d_ptrQ.data()),
        thrust::raw_pointer_cast(d_qPairPtr.data()),
        thrust::raw_pointer_cast(d_qPairA.data()),
        thrust::raw_pointer_cast(d_qPairB.data()),
        thrust::raw_pointer_cast(d_outVals.data()),
        thrust::raw_pointer_cast(d_diagSum.data()));
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // keep flags (drop all-zero diagonals)
    thrust::device_vector<int> d_keep(KQ);
    if (eps == 0.0f) {
        thrust::transform(d_diagSum.begin(), d_diagSum.end(), d_keep.begin(), IsNonZero{});
    } else {
        thrust::transform(d_diagSum.begin(), d_diagSum.end(), d_keep.begin(), GreaterThanEps{eps});
    }

    // scans for compaction
    thrust::device_vector<int> d_newIdxQ(KQ);
    thrust::exclusive_scan(d_keep.begin(), d_keep.end(), d_newIdxQ.begin(), 0);

    thrust::device_vector<int> d_keepLen(KQ);
    thrust::transform(d_keep.begin(), d_keep.end(), d_lenQ.begin(), d_keepLen.begin(), MaskLen{});

    thrust::device_vector<int> d_newPtr(KQ + 1);
    d_newPtr[0] = 0;
    thrust::exclusive_scan(d_keepLen.begin(), d_keepLen.end(), d_newPtr.begin() + 1, 0);

    int newK = 0;
    if (KQ > 0) {
        int lastKeep = d_keep[KQ - 1];
        int lastIdx  = d_newIdxQ[KQ - 1];
        newK = lastIdx + lastKeep;
    }
    int newNNZ = d_newPtr[KQ];

    thrust::device_vector<int>   d_newOff(newK);
    thrust::device_vector<float> d_newVals(newNNZ);

    // copy kept segments
    dim3 cblock(256, 1);
    dim3 cgrid((maxL + cblock.x - 1)/cblock.x, KQ);
    compact_segments_kernel<float><<<cgrid, cblock>>>(
        KQ,
        thrust::raw_pointer_cast(d_keep.data()),
        thrust::raw_pointer_cast(d_offQ.data()),
        thrust::raw_pointer_cast(d_lenQ.data()),
        thrust::raw_pointer_cast(d_ptrQ.data()),
        thrust::raw_pointer_cast(d_outVals.data()),
        thrust::raw_pointer_cast(d_newIdxQ.data()),
        thrust::raw_pointer_cast(d_newPtr.data()),
        thrust::raw_pointer_cast(d_newOff.data()),
        thrust::raw_pointer_cast(d_newVals.data()));
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // download & unpack to public map
    std::vector<int>   h_off(newK);
    std::vector<int>   h_ptr(newK + 1);
    std::vector<float> h_val(newNNZ);
    thrust::copy(d_newOff.begin(), d_newOff.end(), h_off.begin());
    thrust::copy(d_newPtr.begin(), d_newPtr.begin() + (newK + 1), h_ptr.begin());
    thrust::copy(d_newVals.begin(), d_newVals.end(), h_val.begin());
    C_out = unpack_to_map(N, h_off, h_ptr, h_val);
}

DiagMapF32 power_repeat_right(int N, const DiagMapF32& A, int power, float eps) {
    if (power <= 1) return A;
    DiagMapF32 C = A;
    for (int k = 2; k <= power; ++k) {
        DiagMapF32 next;
        multiply_sparse_noPad(N, C, A, eps, next);
        C.swap(next);
    }
    return C;
}
