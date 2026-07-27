/* ============================================================
 * opt.cu — PURE-DIAGONAL SpMV, optimized, measured.
 *
 * Constraint: values stay diagonal-major (packed per diagonal); the
 * kernel is the vector-multiply-accumulate over diagonals
 *     y[p] += H_d[p-lo_d] * x[p+d]
 * with register accumulation (output-stationary).  No row/col-major
 * value ordering anywhere.
 *
 * Variants:
 *   V0  flat plan          : every tile scans all diagonals, bounds
 *                            test per (position, diagonal)  [baseline]
 *   V1  tile lists         : host lists, per y-tile, only the
 *                            INTERSECTING diagonals, split FULL
 *                            (covers whole tile -> NO bounds test,
 *                            branch-free) / PARTIAL (edges only)
 *   V2  V1 + float4        : thread owns 4 consecutive lanes; packer
 *                            aligns each diagonal so the matrix load
 *                            is one LDG.128; x vectorized when
 *                            off%4==0; y stored as float4
 * References printed for context: padded row-dense, cuSPARSE-class
 * didx numbers come from the paper drivers, not here.
 *
 * usage: opt <dia_file> [iters=200]
 * ============================================================ */
#include "../dia_io.hpp"
#include "../../spmv/src/cuda_dia_kernels.cuh"
#include "../../spmv/src/spmv_gather.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

template<class F> static float tms(F f,int wu,int it){
    for(int i=0;i<wu;++i)f(); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
    for(int i=0;i<it;++i)f(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); cudaEventDestroy(s); cudaEventDestroy(e); return ms/it; }

template<typename T> static T* dupload(const std::vector<T>& h){
    T* d; CUDA_CHECK(cudaMalloc(&d,h.size()*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); return d; }

static double rel_l2(const std::vector<double>& ref,const float* got,int n){
    double num=0,den=0; for(int i=0;i<n;++i){double d=ref[i]-(double)got[i];num+=d*d;den+=ref[i]*ref[i];}
    return den>0?std::sqrt(num/den):std::sqrt(num); }

/* ---------------- V1: per-tile diagonal lists, FULL/PARTIAL ---------------- */
/* tile t covers positions [t*POS, min((t+1)*POS, n)).  fullPtr/partPtr are
 * CSR-style lists of DSeg indices.  FULL inner loop has NO bounds test.     */
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
        const DSeg s = segs[fullIdx[q]];              /* warp-uniform, L1 */
#pragma unroll
        for (int r=0;r<ILP;++r) {
            const int p = p0 + r*blockDim.x + threadIdx.x;
            const float a = Av[s.base + (p - s.lo)];  /* NO bounds test */
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

/* ---------------- V2: V1 + float4 (4 consecutive lanes / thread) ----------
 * Requires: seg.base aligned so base-lo ≡ 0 (mod 4) (packer guarantees),
 * p0 and thread lanes multiples of 4.  x vectorized when off%4==0.        */
template <int NV>
__global__ void spmv_tile_vec4_kernel(
    const float* __restrict__ Av, const DSeg* __restrict__ segs,
    const int* __restrict__ fullPtr, const int* __restrict__ fullIdx,
    const int* __restrict__ partPtr, const int* __restrict__ partIdx,
    const float* __restrict__ X, int n, float* __restrict__ Y)
{
    const int tile = blockIdx.x;
    const int p0 = (tile * blockDim.x + threadIdx.x) * 4;   /* 4 consecutive */
    float acc[4][NV];
#pragma unroll
    for (int r=0;r<4;++r)
#pragma unroll
        for (int v=0;v<NV;++v) acc[r][v]=0.f;

    for (int q = fullPtr[tile]; q < fullPtr[tile+1]; ++q) {
        const DSeg s = segs[fullIdx[q]];
        const float4 a = *(const float4*)(Av + s.base + (p0 - s.lo));  /* LDG.128 */
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

/* ---------------- host: aligned pack + tile lists ---------------- */
struct TilePlan {
    std::vector<DSeg> segs;
    std::vector<float> packed;                 /* aligned copy of values   */
    std::vector<int> fullPtr, fullIdx, partPtr, partIdx;
    int ntiles = 0;
};

static void fill_tile_lists(TilePlan& T, int n, int POS);

static TilePlan build_tile_plan(const DiaHost& H, int POS)
{
    TilePlan T;
    const int n = H.n;
    /* pack each diagonal at a base with base ≡ lo (mod 4) -> float4-able */
    for (size_t k = 0; k < H.offsets.size(); ++k) {
        int off = H.offsets[k], len = H.lengths[k];
        int lo = off < 0 ? -off : 0;
        size_t base = T.packed.size();
        while (((base - (size_t)(lo & 3)) & 3) != 0) { T.packed.push_back(0.f); ++base; }
        for (int t = 0; t < len; ++t) T.packed.push_back(H.values[H.starts[k] + t]);
        T.segs.push_back(DSeg{ base, off, lo, len });
    }
    fill_tile_lists(T, n, POS);
    return T;
}

/* Coarse zero-skip: split a diagonal ONLY at zero gaps >= MINGAP.  Each piece
 * keeps affine (self-indexing) addressing; small gaps stay stored so the
 * segment count stays ~O(diags), unlike the every-gap split.  Big periodic
 * holes (heis: gap 3*2^i) are neither stored, read, nor iterated (tile lists
 * skip them entirely).  Pure diagonal order throughout.                     */
static TilePlan build_coarse_plan(const DiaHost& H, int POS, int MINGAP)
{
    TilePlan T;
    const int n = H.n;
    for (size_t k = 0; k < H.offsets.size(); ++k) {
        int off = H.offsets[k], len = H.lengths[k];
        int lo = off < 0 ? -off : 0;
        size_t s = H.starts[k];
        int t = 0;
        while (t < len) {
            while (t < len && H.values[s + t] == 0.f) ++t;   /* skip zeros   */
            if (t >= len) break;
            int start = t, lastnz = t;
            while (t < len) {
                if (H.values[s + t] != 0.f) { lastnz = t; ++t; continue; }
                int g = t; while (g < len && H.values[s + g] == 0.f) ++g;
                if (g - t >= MINGAP) break;                  /* big gap: cut */
                t = g;                                       /* absorb small */
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

static void fill_tile_lists(TilePlan& T, int n, int POS)
{
    T.ntiles = (n + POS - 1) / POS;
    T.fullPtr.assign(T.ntiles + 1, 0); T.partPtr.assign(T.ntiles + 1, 0);
    for (int t = 0; t < T.ntiles; ++t) {
        int a = t * POS, b = std::min(a + POS, n);
        T.fullPtr[t] = (int)T.fullIdx.size(); T.partPtr[t] = (int)T.partIdx.size();
        for (size_t k = 0; k < T.segs.size(); ++k) {
            const DSeg& s = T.segs[k];
            if (s.lo >= b || s.lo + s.len <= a) continue;        /* no overlap */
            /* FULL must cover the NOMINAL tile end a+POS (not b): the last,
             * truncated tile then classifies everything PARTIAL, so the
             * test-free FULL loop can never read past a segment. */
            if (s.lo <= a && s.lo + s.len >= a + POS) T.fullIdx.push_back((int)k);
            else                                      T.partIdx.push_back((int)k);
        }
    }
    T.fullPtr[T.ntiles] = (int)T.fullIdx.size(); T.partPtr[T.ntiles] = (int)T.partIdx.size();
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr,"usage: %s <dia_file> [iters=200]\n",argv[0]); return 1; }
    int iters = argc > 2 ? atoi(argv[2]) : 200;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();
    printf("=== %s  n=%d  diags=%d ===\n", argv[1], n, nd);

    std::vector<float> X(2*(size_t)n);
    srand(42);
    for (size_t i = 0; i < X.size(); ++i) X[i] = (float)rand()/RAND_MAX - 0.5f;
    std::vector<double> yr(n,0.0), yi(n,0.0);
    for (int k = 0; k < nd; ++k){
        int off=H.offsets[k]; size_t s=H.starts[k]; int len=H.lengths[k];
        for (int p=0;p<len;++p){ int r=off>=0?p:p-off, c=off>=0?p+off:p;
            double v=(double)H.values[s+p]; yr[r]+=v*(double)X[c]; yi[r]+=v*(double)X[n+c]; }
    }
    float *dX=dupload(X), *dY; CUDA_CHECK(cudaMalloc(&dY,2*(size_t)n*4));
    std::vector<float> out(2*(size_t)n);
    const int TPB=256;
    auto chk1=[&](){ CUDA_CHECK(cudaMemcpy(out.data(),dY,(size_t)n*4,cudaMemcpyDeviceToHost));
                     return rel_l2(yr,out.data(),n); };
    auto chk2=[&](){ CUDA_CHECK(cudaMemcpy(out.data(),dY,2*(size_t)n*4,cudaMemcpyDeviceToHost));
                     return std::max(rel_l2(yr,out.data(),n),rel_l2(yi,out.data()+n,n)); };

    /* V0: flat plan baseline */
    {
        float* dAv = dupload(H.values);
        auto plan = build_dense_plan(H.offsets,H.starts,H.lengths);
        DSeg* dS = dupload(plan); int nseg=(int)plan.size();
        const int ILP = n>=65536?4:1;
        int b1=(n+TPB-1)/TPB, b4=(n+TPB*4-1)/(TPB*4);
        auto l1=[&](){ if(ILP==4) spmv_gather_kernel<128,4,1><<<b4,TPB>>>(dAv,dS,nseg,dX,n,dY);
                       else       spmv_gather_kernel<128,1,1><<<b1,TPB>>>(dAv,dS,nseg,dX,n,dY); };
        auto l2=[&](){ if(ILP==4) spmv_gather_kernel<128,4,2><<<b4,TPB>>>(dAv,dS,nseg,dX,n,dY);
                       else       spmv_gather_kernel<128,1,2><<<b1,TPB>>>(dAv,dS,nseg,dX,n,dY); };
        l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V0 flat     NV1 : %9.5f ms  relerr %.2e\n", tms(l1,10,iters), chk1());
        l2(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V0 flat     NV2 : %9.5f ms  relerr %.2e\n", tms(l2,10,iters), chk2());
        cudaFree(dAv); cudaFree(dS);
    }

    /* V1: tile lists (scalar) */
    for (int ILP : {1,4}) {
        const int POS = TPB*ILP;
        TilePlan T = build_tile_plan(H, POS);
        float* dAv = dupload(T.packed); DSeg* dS = dupload(T.segs);
        int *dfp=dupload(T.fullPtr), *dfi=dupload(T.fullIdx),
            *dpp=dupload(T.partPtr), *dpi=dupload(T.partIdx);
        double avgFull = T.fullIdx.size()/(double)T.ntiles, avgPart = T.partIdx.size()/(double)T.ntiles;
        auto l1=[&](){ if(ILP==4) spmv_tile_kernel<4,1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY);
                       else       spmv_tile_kernel<1,1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        auto l2=[&](){ if(ILP==4) spmv_tile_kernel<4,2><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY);
                       else       spmv_tile_kernel<1,2><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V1 tile i%d  NV1 : %9.5f ms  relerr %.2e   (avg full %.1f, part %.1f per tile)\n",
               ILP, tms(l1,10,iters), chk1(), avgFull, avgPart);
        l2(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V1 tile i%d  NV2 : %9.5f ms  relerr %.2e\n", ILP, tms(l2,10,iters), chk2());
        cudaFree(dAv);cudaFree(dS);cudaFree(dfp);cudaFree(dfi);cudaFree(dpp);cudaFree(dpi);
    }

    /* V2: tile lists + float4 */
    {
        const int POS = TPB*4;
        TilePlan T = build_tile_plan(H, POS);
        float* dAv = dupload(T.packed); DSeg* dS = dupload(T.segs);
        int *dfp=dupload(T.fullPtr), *dfi=dupload(T.fullIdx),
            *dpp=dupload(T.partPtr), *dpi=dupload(T.partIdx);
        auto l1=[&](){ spmv_tile_vec4_kernel<1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        auto l2=[&](){ spmv_tile_vec4_kernel<2><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V2 vec4     NV1 : %9.5f ms  relerr %.2e\n", tms(l1,10,iters), chk1());
        l2(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V2 vec4     NV2 : %9.5f ms  relerr %.2e\n", tms(l2,10,iters), chk2());
        cudaFree(dAv);cudaFree(dS);cudaFree(dfp);cudaFree(dfi);cudaFree(dpp);cudaFree(dpi);
    }

    /* V3: coarse zero-skip (split only at gaps >= MINGAP) + tile lists/vec4 */
    for (int MINGAP : {256, 1024, 4096}) {
        const int POS = TPB*4;
        TilePlan T = build_coarse_plan(H, POS, MINGAP);
        double stored = 0; for (auto& s : T.segs) stored += s.len;
        float* dAv = dupload(T.packed); DSeg* dS = dupload(T.segs);
        int *dfp=dupload(T.fullPtr), *dfi=dupload(T.fullIdx),
            *dpp=dupload(T.partPtr), *dpi=dupload(T.partIdx);
        double avgFull = T.fullIdx.size()/(double)T.ntiles, avgPart = T.partIdx.size()/(double)T.ntiles;
        auto ls=[&](){ spmv_tile_kernel<4,1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        auto lv=[&](){ spmv_tile_vec4_kernel<1><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        auto lv2=[&](){ spmv_tile_vec4_kernel<2><<<T.ntiles,TPB>>>(dAv,dS,dfp,dfi,dpp,dpi,dX,n,dY); };
        ls(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V3 g%-5d  NV1s : %9.5f ms  relerr %.2e   (%zu segs, reads %.1f%%, full %.1f part %.1f/tile)\n",
               MINGAP, tms(ls,10,iters), chk1(), T.segs.size(), 100.0*stored/H.nnz, avgFull, avgPart);
        lv(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V3 g%-5d  NV1v : %9.5f ms  relerr %.2e\n", MINGAP, tms(lv,10,iters), chk1());
        lv2(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  V3 g%-5d  NV2v : %9.5f ms  relerr %.2e\n", MINGAP, tms(lv2,10,iters), chk2());
        cudaFree(dAv);cudaFree(dS);cudaFree(dfp);cudaFree(dfi);cudaFree(dpp);cudaFree(dpi);
    }

    /* reference: padded row-dense */
    {
        std::vector<float> pad((size_t)nd*n,0.f);
        for (int k=0;k<nd;++k){ int off=H.offsets[k]; size_t s=H.starts[k]; int len=H.lengths[k];
            int c0=off>=0?off:0; for(int p=0;p<len;++p) pad[(size_t)k*n+c0+p]=H.values[s+p]; }
        float* dV=dupload(pad); int* dOff=dupload(std::vector<int>(H.offsets));
        ReconView R{n,n,nd,dOff,dV}; int blocks=(n+TPB-1)/TPB;
        auto l=[&](){ cuda_spmv_dia<<<blocks,TPB>>>(R,dX,n,dY); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  ref row-dense   : %9.5f ms  relerr %.2e\n", tms(l,10,iters), chk1());
        cudaFree(dV); cudaFree(dOff);
    }
    return 0;
}
