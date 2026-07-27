/* ============================================================
 * spmv_gather.cuh — diagonal-direct (position-tiled) SpMV.
 *
 * Same schedule machinery as the SpMSpM gather kernels: the operator
 * is a list of SEGMENTS (contiguous nonzero runs of a diagonal),
 * y is covered by a 1-D grid of uniform position tiles, each block
 * stages the segment metadata in shared memory in MAXD-sized chunks
 * and accumulates its ILP positions in registers. SpMV is the
 * degenerate offset convolution: the per-output "pair list" of the
 * SpMSpM kernel collapses to the segment list, with x as the second
 * operand read at p + off.
 *
 *   y[p] = sum over segments s covering p of
 *          Av[s.base + (p - s.lo)] * x[p + s.off]
 *
 * One segment per diagonal  -> dense-DIA behavior (no padding read:
 *   values are packed per diagonal, position = min(row,col)).
 * Segments split at interior zero runs -> zero-skip, SAME kernel:
 *   the zeros between runs stay in Av but are never touched, so
 *   traffic (not storage) shrinks; the plan is built once per
 *   operator and amortized over the run.
 * NV vectors (column-major X[v*n+i]) -> fused real/imag apply reads
 *   each matrix value once for both components.
 * Atomic-free: each block owns a disjoint position range of y.
 * ============================================================ */
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

struct DSeg {
    size_t base;   /* index into Av of the segment's first value   */
    int    off;    /* diagonal offset (col - row)                  */
    int    lo;     /* first row (= y position) the segment covers  */
    int    len;    /* number of consecutive rows covered           */
};

template <int MAXD, int ILP, int NV>
__global__ void spmv_gather_kernel(
    const float* __restrict__ Av,
    const DSeg*  __restrict__ segs, int nseg,
    const float* __restrict__ X,   /* NV vectors, X[v*n + i] */
    int n,
    float* __restrict__ Y)         /* Y[v*n + i] */
{
    __shared__ size_t sb[MAXD];
    __shared__ int    soff[MAXD], slo[MAXD], slen[MAXD];
    const int p0 = blockIdx.x * blockDim.x * ILP;

    float acc[ILP][NV];
#pragma unroll
    for (int r = 0; r < ILP; ++r)
#pragma unroll
        for (int v = 0; v < NV; ++v) acc[r][v] = 0.f;

    for (int c0 = 0; c0 < nseg; c0 += MAXD) {
        int nc = nseg - c0; if (nc > MAXD) nc = MAXD;
        __syncthreads();                       /* guard smem before reuse */
        for (int q = threadIdx.x; q < nc; q += blockDim.x) {
            DSeg s = segs[c0 + q];
            sb[q] = s.base; soff[q] = s.off; slo[q] = s.lo; slen[q] = s.len;
        }
        __syncthreads();
        for (int q = 0; q < nc; ++q) {
            const size_t b = sb[q];
            const int off = soff[q], lo = slo[q], len = slen[q];
#pragma unroll
            for (int r = 0; r < ILP; ++r) {
                const int p = p0 + r * blockDim.x + threadIdx.x;
                const int t = p - lo;
                if (t < 0 || t >= len) continue;   /* p<n implied: lo+len<=n */
                const float a = Av[b + t];
                const int   c = p + off;
#pragma unroll
                for (int v = 0; v < NV; ++v)
                    acc[r][v] += a * X[(size_t)v * n + c];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ILP; ++r) {
        const int p = p0 + r * blockDim.x + threadIdx.x;
        if (p < n)
#pragma unroll
            for (int v = 0; v < NV; ++v) Y[(size_t)v * n + p] = acc[r][v];
    }
}

/* ============================================================
 * Pattern plan: zeros skipped by ARITHMETIC, not metadata.
 *
 * Pauli structure places a diagonal's nonzeros on periodic runs:
 * positions t with ((t-c) mod P) < R  (P a power of two).  One
 * descriptor per diagonal replaces per-nonzero indices entirely;
 * values are repacked to only the true nonzeros, so the matrix
 * stream is 4 B per TRUE nonzero and stays diagonal-contiguous.
 * A dense diagonal is the degenerate descriptor P=1,R=1,c=0.
 * ============================================================ */
struct DPat {
    size_t base;   /* into repacked values                       */
    int off, lo, len;
    int c;         /* phase: first nonzero position              */
    int Pm;        /* P-1 (P power of two)                       */
    int logP;
    int R;         /* run length (<= P)                          */
};

template <int MAXD, int ILP, int NV>
__global__ void spmv_gather_pat_kernel(
    const float* __restrict__ Av,
    const DPat*  __restrict__ pats, int npat,
    const float* __restrict__ X, int n,
    float* __restrict__ Y)
{
    __shared__ size_t sb[MAXD];
    __shared__ int soff[MAXD], slo[MAXD], slen[MAXD], sc[MAXD], sPm[MAXD], slg[MAXD], sR[MAXD];
    const int p0 = blockIdx.x * blockDim.x * ILP;

    float acc[ILP][NV];
#pragma unroll
    for (int r = 0; r < ILP; ++r)
#pragma unroll
        for (int v = 0; v < NV; ++v) acc[r][v] = 0.f;

    for (int c0 = 0; c0 < npat; c0 += MAXD) {
        int nc = npat - c0; if (nc > MAXD) nc = MAXD;
        __syncthreads();
        for (int q = threadIdx.x; q < nc; q += blockDim.x) {
            DPat s = pats[c0 + q];
            sb[q]=s.base; soff[q]=s.off; slo[q]=s.lo; slen[q]=s.len;
            sc[q]=s.c; sPm[q]=s.Pm; slg[q]=s.logP; sR[q]=s.R;
        }
        __syncthreads();
        for (int q = 0; q < nc; ++q) {
            const size_t b = sb[q];
            const int off=soff[q], lo=slo[q], len=slen[q],
                      c=sc[q], Pm=sPm[q], lg=slg[q], R=sR[q];
#pragma unroll
            for (int r = 0; r < ILP; ++r) {
                const int p = p0 + r * blockDim.x + threadIdx.x;
                const int t = p - lo;
                if ((unsigned)t >= (unsigned)len) continue;
                const int tt = t - c;
                if (tt < 0) continue;
                const int q0 = tt & Pm;
                if (q0 >= R) continue;
                const int j = ((tt >> lg) * R) + q0;
#pragma unroll
                for (int v = 0; v < NV; ++v)
                    acc[r][v] += Av[b + j] * X[(size_t)v * n + (p + off)];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ILP; ++r) {
        const int p = p0 + r * blockDim.x + threadIdx.x;
        if (p < n)
#pragma unroll
            for (int v = 0; v < NV; ++v) Y[(size_t)v * n + p] = acc[r][v];
    }
}

/* ---- host plan builders --------------------------------------------------
 * Both take diagonals as (offset, start-into-values, length) with the
 * position-p = min(row,col) packing (DiaHost / SpMSpM output layout).
 * lo (first covered ROW) is max(0, -off); value index t = p - lo.       */

/* one segment per diagonal — dense plan */
inline std::vector<DSeg> build_dense_plan(
    const std::vector<int>& offsets, const std::vector<size_t>& starts,
    const std::vector<int>& lengths)
{
    std::vector<DSeg> segs(offsets.size());
    for (size_t k = 0; k < offsets.size(); ++k) {
        int off = offsets[k];
        segs[k] = DSeg{ starts[k], off, off < 0 ? -off : 0, lengths[k] };
    }
    return segs;
}

/* Detect the periodic-run pattern of each diagonal and repack values.
 * For each diagonal: find first nonzero c, run length R, period P
 * (power of two) from the next run start; accept if every nonzero
 * position lies in {t >= c : ((t-c) mod P) < R} (explicit zeros inside
 * a predicted run are packed too, so indexing stays exact).  Fallback:
 * dense descriptor (P=1,R=1,c=0).  All-zero diagonals are dropped.
 * Returns the plan and the repacked value array it indexes.          */
inline std::vector<DPat> build_pattern_plan(
    const std::vector<int>& offsets, const std::vector<size_t>& starts,
    const std::vector<int>& lengths, const std::vector<float>& values,
    std::vector<float>& packed, size_t* touched_out = nullptr)
{
    std::vector<DPat> plan; packed.clear();
    size_t touched = 0;
    for (size_t k = 0; k < offsets.size(); ++k) {
        const int off = offsets[k], len = lengths[k];
        const int lo  = off < 0 ? -off : 0;
        const size_t s = starts[k];
        int c = -1;
        for (int t = 0; t < len; ++t) if (values[s+t] != 0.f) { c = t; break; }
        if (c < 0) continue;                       /* all-zero diagonal */
        int R = 0; while (c + R < len && values[s + c + R] != 0.f) ++R;
        int nxt = c + R; while (nxt < len && values[s + nxt] == 0.f) ++nxt;
        int P, logP;
        if (nxt >= len) { P = 1 << 30; }           /* single run */
        else            { P = nxt - c; }
        bool pow2 = (P & (P - 1)) == 0 && P >= R;
        bool ok = pow2;
        if (ok && P < (1 << 30)) {                 /* verify all nonzeros in pattern */
            for (int t = 0; t < len && ok; ++t) {
                bool in = t >= c && ((t - c) & (P - 1)) < R;
                if (!in && values[s + t] != 0.f) ok = false;
            }
        }
        DPat d;
        d.off = off; d.lo = lo; d.len = len; d.base = packed.size();
        if (ok) {
            d.c = c; d.Pm = P - 1; d.R = R;
            logP = 0; while ((1 << logP) < P) ++logP; d.logP = logP;
            for (int t = c; t < len; ++t)
                if (((t - c) & (P - 1)) < R) packed.push_back(values[s + t]);
        } else {                                   /* dense fallback */
            d.c = 0; d.Pm = 0; d.logP = 0; d.R = 1;
            for (int t = 0; t < len; ++t) packed.push_back(values[s + t]);
        }
        touched += packed.size() - d.base;
        plan.push_back(d);
    }
    if (touched_out) *touched_out = touched;
    return plan;
}

/* split each diagonal at interior zero runs, IF splitting pays.
 * A diagonal is split only when its nonzero fill is below `fill_cut`
 * (there are enough zeros to skip) AND its average run length is at
 * least `min_avg_run` (meta per segment stays amortized); otherwise it
 * stays one dense segment — same adaptive spirit as ZEROSKIP=auto.   */
inline std::vector<DSeg> build_zskip_plan(
    const std::vector<int>& offsets, const std::vector<size_t>& starts,
    const std::vector<int>& lengths, const std::vector<float>& values,
    float fill_cut = 0.75f, int min_avg_run = 4)
{
    std::vector<DSeg> segs;
    for (size_t k = 0; k < offsets.size(); ++k) {
        const int off = offsets[k], len = lengths[k];
        const int lo  = off < 0 ? -off : 0;
        const size_t s = starts[k];
        /* runs of nonzeros */
        int nnz = 0, nruns = 0;
        for (int t = 0; t < len; ) {
            if (values[s + t] != 0.f) { ++nruns; while (t < len && values[s + t] != 0.f) { ++nnz; ++t; } }
            else ++t;
        }
        const bool split = nnz > 0 && nruns > 0 &&
                           (float)nnz < fill_cut * (float)len &&
                           nnz / nruns >= min_avg_run;
        if (!split) { segs.push_back(DSeg{ s, off, lo, len }); continue; }
        for (int t = 0; t < len; ) {
            if (values[s + t] == 0.f) { ++t; continue; }
            int r0 = t; while (t < len && values[s + t] != 0.f) ++t;
            segs.push_back(DSeg{ s + r0, off, lo + r0, t - r0 });
        }
    }
    return segs;
}
