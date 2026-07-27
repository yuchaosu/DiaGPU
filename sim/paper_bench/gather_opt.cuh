/* ============================================================
 * gather_opt.cuh — optimized pure-diagonal (vector-vector) SpMV.
 *
 * Shared by opt bench and the paper driver.  Host preprocesses the
 * interior zeros once per operator: each diagonal is cut ONLY at
 * zero gaps >= MINGAP (coarse affine segments, O(diags) of them),
 * packed 16B-aligned; per-y-tile FULL/PARTIAL lists let the kernel
 * skip non-covering segments entirely and run the FULL loop with no
 * bounds test; float4 loads on the segment stream.
 * ============================================================ */
#pragma once
#include "../../spmv/src/spmv_gather.cuh"   /* DSeg */
#include <vector>
#include <algorithm>

/* scalar tile kernel: ILP positions/thread, strided by blockDim */
template <int ILP, int NV>
__global__ void spmv_tile_kernel(
    const float* __restrict__ Av, const DSeg* __restrict__ segs,
    const int* __restrict__ fullPtr, const int* __restrict__ fullIdx,
    const int* __restrict__ partPtr, const int* __restrict__ partIdx,
    const float* __restrict__ X, int n, float* __restrict__ Y)
{
    const int tile = blockIdx.x;
    const int p0 = tile * blockDim.x * ILP;
    float acc[ILP][NV];
#pragma unroll
    for (int r=0;r<ILP;++r)
#pragma unroll
        for (int v=0;v<NV;++v) acc[r][v]=0.f;

    for (int q = fullPtr[tile]; q < fullPtr[tile+1]; ++q) {
        const DSeg s = segs[fullIdx[q]];
#pragma unroll
        for (int r=0;r<ILP;++r) {
            const int p = p0 + r*blockDim.x + threadIdx.x;
            const float a = Av[s.base + (p - s.lo)];
#pragma unroll
            for (int v=0;v<NV;++v) acc[r][v] += a * X[(size_t)v*n + (p + s.off)];
        }
    }
    for (int q = partPtr[tile]; q < partPtr[tile+1]; ++q) {
        const DSeg s = segs[partIdx[q]];
#pragma unroll
        for (int r=0;r<ILP;++r) {
            const int p = p0 + r*blockDim.x + threadIdx.x;
            const int t = p - s.lo;
            if ((unsigned)t >= (unsigned)s.len) continue;
            const float a = Av[s.base + t];
#pragma unroll
            for (int v=0;v<NV;++v) acc[r][v] += a * X[(size_t)v*n + (p + s.off)];
        }
    }
#pragma unroll
    for (int r=0;r<ILP;++r) {
        const int p = p0 + r*blockDim.x + threadIdx.x;
        if (p < n)
#pragma unroll
            for (int v=0;v<NV;++v) Y[(size_t)v*n + p] = acc[r][v];
    }
}

/* vec4 tile kernel: thread owns 4 consecutive lanes; segment loads LDG.128 */
template <int NV>
__global__ void spmv_tile_vec4_kernel(
    const float* __restrict__ Av, const DSeg* __restrict__ segs,
    const int* __restrict__ fullPtr, const int* __restrict__ fullIdx,
    const int* __restrict__ partPtr, const int* __restrict__ partIdx,
    const float* __restrict__ X, int n, float* __restrict__ Y)
{
    const int tile = blockIdx.x;
    const int p0 = (tile * blockDim.x + threadIdx.x) * 4;
    float acc[4][NV];
#pragma unroll
    for (int r=0;r<4;++r)
#pragma unroll
        for (int v=0;v<NV;++v) acc[r][v]=0.f;

    for (int q = fullPtr[tile]; q < fullPtr[tile+1]; ++q) {
        const DSeg s = segs[fullIdx[q]];
        const float4 a = *(const float4*)(Av + s.base + (p0 - s.lo));
        const int c = p0 + s.off;
        if ((s.off & 3) == 0) {
#pragma unroll
            for (int v=0;v<NV;++v) {
                const float4 xv = *(const float4*)(X + (size_t)v*n + c);
                acc[0][v] += a.x*xv.x; acc[1][v] += a.y*xv.y;
                acc[2][v] += a.z*xv.z; acc[3][v] += a.w*xv.w;
            }
        } else {
#pragma unroll
            for (int v=0;v<NV;++v) {
                acc[0][v] += a.x*X[(size_t)v*n + c+0];
                acc[1][v] += a.y*X[(size_t)v*n + c+1];
                acc[2][v] += a.z*X[(size_t)v*n + c+2];
                acc[3][v] += a.w*X[(size_t)v*n + c+3];
            }
        }
    }
    for (int q = partPtr[tile]; q < partPtr[tile+1]; ++q) {
        const DSeg s = segs[partIdx[q]];
#pragma unroll
        for (int r=0;r<4;++r) {
            const int p = p0 + r;
            const int t = p - s.lo;
            if ((unsigned)t >= (unsigned)s.len) continue;
            const float a = Av[s.base + t];
#pragma unroll
            for (int v=0;v<NV;++v) acc[r][v] += a * X[(size_t)v*n + (p + s.off)];
        }
    }
    if (p0 + 3 < n) {
#pragma unroll
        for (int v=0;v<NV;++v)
            *(float4*)(Y + (size_t)v*n + p0) = make_float4(acc[0][v],acc[1][v],acc[2][v],acc[3][v]);
    } else {
#pragma unroll
        for (int r=0;r<4;++r) if (p0+r < n)
#pragma unroll
            for (int v=0;v<NV;++v) Y[(size_t)v*n + p0+r] = acc[r][v];
    }
}

struct TilePlan {
    std::vector<DSeg> segs;
    std::vector<float> packed;
    std::vector<int> fullPtr, fullIdx, partPtr, partIdx;
    int ntiles = 0;
};

/* FULL must cover the NOMINAL tile end a+POS: the last, truncated tile then
 * classifies everything PARTIAL, so the test-free FULL loop never reads past
 * a segment. */
inline void fill_tile_lists(TilePlan& T, int n, int POS)
{
    T.ntiles = (n + POS - 1) / POS;
    T.fullPtr.assign(T.ntiles + 1, 0); T.partPtr.assign(T.ntiles + 1, 0);
    for (int t = 0; t < T.ntiles; ++t) {
        int a = t * POS, b = std::min(a + POS, n);
        T.fullPtr[t] = (int)T.fullIdx.size(); T.partPtr[t] = (int)T.partIdx.size();
        for (size_t k = 0; k < T.segs.size(); ++k) {
            const DSeg& s = T.segs[k];
            if (s.lo >= b || s.lo + s.len <= a) continue;
            if (s.lo <= a && s.lo + s.len >= a + POS) T.fullIdx.push_back((int)k);
            else                                      T.partIdx.push_back((int)k);
        }
    }
    T.fullPtr[T.ntiles] = (int)T.fullIdx.size(); T.partPtr[T.ntiles] = (int)T.partIdx.size();
}

/* coarse zero-skip pack: split diagonals only at zero gaps >= MINGAP; every
 * kept span stays affine; bases aligned so base ≡ lo (mod 4) for float4.  */
inline TilePlan build_coarse_plan(const DiaHost& H, int POS, int MINGAP)
{
    TilePlan T;
    const int n = H.n;
    for (size_t k = 0; k < H.offsets.size(); ++k) {
        int off = H.offsets[k], len = H.lengths[k];
        int lo = off < 0 ? -off : 0;
        size_t s = H.starts[k];
        int t = 0;
        while (t < len) {
            while (t < len && H.values[s + t] == 0.f) ++t;
            if (t >= len) break;
            int start = t, lastnz = t;
            while (t < len) {
                if (H.values[s + t] != 0.f) { lastnz = t; ++t; continue; }
                int g = t; while (g < len && H.values[s + g] == 0.f) ++g;
                if (g - t >= MINGAP) break;
                t = g;
            }
            int slo = lo + start, seglen = lastnz - start + 1;
            size_t base = T.packed.size();
            while (((base - (size_t)(slo & 3)) & 3) != 0) { T.packed.push_back(0.f); ++base; }
            for (int u = start; u <= lastnz; ++u) T.packed.push_back(H.values[s + u]);
            T.segs.push_back(DSeg{ base, off, slo, seglen });
        }
    }
    fill_tile_lists(T, n, POS);
    return T;
}
