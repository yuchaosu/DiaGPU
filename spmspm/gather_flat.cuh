/* ============================================================
 * gather_flat.cuh — flattened, precomputed-pair diagonal SpGEMM.
 *
 * C = A*B in diagonal (HM) layout. Beats both the in-block-resolve
 * gather_meta and the HM atomic kernel at small/mid n by:
 *   (1) flattening to a 1-D grid of uniform TILE-sized tiles over all
 *       (C-diagonal, position) work  -> no wave-quantization tail;
 *   (2) precomputing the per-C-diagonal pair lists on the host
 *       (pairPtr -> GPair[]) -> no in-block atomics / resolve-sync.
 * Adaptive ILP (positions/thread): small n -> 1 (many thin blocks),
 * large n -> 4 (amortize per-block smem load+sync). See beat_hm/NOTES.md.
 *
 * NOTE: the host build (build_flat_plan) is cheap for a one-shot product
 * but recurs each step of a matrix-power chain where A grows; time it.
 * ============================================================ */
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

struct GPair { size_t ab, bb; int ash, bsh, al, bl; };

template <int MAXP, int ILP>
__global__ void gather_flat_kernel(
    const float*  __restrict__ Av,
    const float*  __restrict__ Bv,
    const int2*   __restrict__ tiles,
    const int*    __restrict__ pairPtr,
    const GPair*  __restrict__ pairs,
    float*        __restrict__ Cv,
    const size_t* __restrict__ C_starts,
    const int*    __restrict__ C_len)
{
    const int bid    = blockIdx.x;
    const int k      = tiles[bid].x;
    const int tstart = tiles[bid].y;
    const int p0 = pairPtr[k];
    const int np = pairPtr[k + 1] - p0;

    __shared__ size_t sAb[MAXP], sBb[MAXP];
    __shared__ int    sAsh[MAXP], sBsh[MAXP], sAl[MAXP], sBl[MAXP];
    for (int q = threadIdx.x; q < np; q += blockDim.x) {
        GPair g = pairs[p0 + q];
        sAb[q]=g.ab; sBb[q]=g.bb; sAsh[q]=g.ash; sBsh[q]=g.bsh; sAl[q]=g.al; sBl[q]=g.bl;
    }
    __syncthreads();

    const int lenC = C_len[k];
    const size_t cbase = C_starts[k];
    #pragma unroll
    for (int r = 0; r < ILP; ++r) {
        const int p = tstart + r * blockDim.x + threadIdx.x;
        if (p >= lenC) break;
        float acc = 0.0f;
        for (int q = 0; q < np; ++q) {
            const int pa = p + sAsh[q];
            const int pb = p + sBsh[q];
            if (pa < 0 || pa >= sAl[q]) continue;
            if (pb < 0 || pb >= sBl[q]) continue;
            acc += Av[sAb[q] + pa] * Bv[sBb[q] + pb];
        }
        Cv[cbase + p] = acc;
    }
}

/* ---- hybrid: meta's implicit 2-D grid (blockIdx.y = C-diag, x = position tiles)
 * but with HOST-PRECOMPUTED pairs (no in-block atomic resolve). Nothing is
 * materialized per-tile, so only pairPtr+pairs upload (small, ~C_ndiag*|B|) —
 * unlike gather_flat which also uploads a per-tile work list that explodes in a
 * fill-in chain. ILP is set by grid.x sizing + the grid-stride loop (same as meta). */
template <int MAXP>
__global__ void gather_hyb_kernel(
    const float*  __restrict__ Av,
    const float*  __restrict__ Bv,
    const int*    __restrict__ pairPtr,
    const GPair*  __restrict__ pairs,
    float*        __restrict__ Cv,
    const size_t* __restrict__ C_starts,
    const int*    __restrict__ C_len,
    int C_ndiag, int n)
{
    const int k = blockIdx.y;
    if (k >= C_ndiag) return;
    const int lenC = C_len[k];
    const size_t cbase = C_starts[k];
    const int p0 = pairPtr[k];
    const int np = pairPtr[k + 1] - p0;

    __shared__ size_t sAb[MAXP], sBb[MAXP];
    __shared__ int    sAsh[MAXP], sBsh[MAXP], sAl[MAXP], sBl[MAXP];
    for (int q = threadIdx.x; q < np; q += blockDim.x) {
        GPair g = pairs[p0 + q];
        sAb[q]=g.ab; sBb[q]=g.bb; sAsh[q]=g.ash; sBsh[q]=g.bsh; sAl[q]=g.al; sBl[q]=g.bl;
    }
    __syncthreads();

    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < lenC; p += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int q = 0; q < np; ++q) {
            const int pa = p + sAsh[q];
            const int pb = p + sBsh[q];
            if (pa < 0 || pa >= sAl[q]) continue;
            if (pb < 0 || pb >= sBl[q]) continue;
            acc += Av[sAb[q] + pa] * Bv[sBb[q] + pb];
        }
        Cv[cbase + p] = acc;
    }
}

/* ---- fully on-device plan build (no host roundtrip) ----
 * For a matrix-power chain A=P_k grows each step; resolving pairs on the host and
 * uploading them per step swamps the sub-ms kernel. These kernels build the plan on
 * the GPU: (1) scatter A's signed offsets into a lookup, (2) per C-diag, iterate the
 * NARROW B(=H) and look up A, writing a fixed-stride pair block + a per-diag count.
 * Fixed stride = |B| (max pairs per C-diag) avoids a prefix-sum. The consumer is
 * gather_hyb2_kernel below. This also fixes meta's cost: meta loops A_ndiag (WIDE, up
 * to ~2n-1 as P fills in) inside every block; here we resolve over |B| once per C-diag. */
__global__ void scatter_lookup_kernel(const int* __restrict__ Aoff, int An, int n, int* __restrict__ Alk){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if(i<An) Alk[Aoff[i]+(n-1)] = i;
}
__global__ void build_plan_kernel(
    int n, int Cn, int Bn, int stride,
    const int*    __restrict__ Coff,
    const int*    __restrict__ Alk,
    const int*    __restrict__ Alen, const size_t* __restrict__ Astarts,
    const int*    __restrict__ Boff, const int* __restrict__ Blen, const size_t* __restrict__ Bstarts,
    GPair*        __restrict__ pairs, int* __restrict__ count)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x; if(k>=Cn) return;
    int dc = Coff[k], minc = dc<0?dc:0, slot=0; size_t base=(size_t)k*stride;
    for(int bi=0;bi<Bn;++bi){
        int db=Boff[bi], da=dc-db; if(da<=-n||da>=n) continue;
        int ai=Alk[da+(n-1)]; if(ai<0) continue;
        GPair g; g.ab=Astarts[ai]; g.bb=Bstarts[bi];
        g.ash=(da<0?da:0)-minc; g.bsh=da+(db<0?db:0)-minc; g.al=Alen[ai]; g.bl=Blen[bi];
        pairs[base+slot]=g; slot++;
    }
    count[k]=slot;
}
/* plan build via BINARY SEARCH over sorted A (no dAlk lookup table → drops the
 * memset+scatter, 2 fewer launches/step). One thread per C-diag, loops narrow B. */
__global__ void build_plan_bs_kernel(
    int n, int Cn, int Bn, int stride,
    const int*    __restrict__ Coff,
    const int*    __restrict__ Aoff /*sorted*/, const int* __restrict__ Alen, const size_t* __restrict__ Astarts, int An,
    const int*    __restrict__ Boff, const int* __restrict__ Blen, const size_t* __restrict__ Bstarts,
    GPair*        __restrict__ pairs, int* __restrict__ count)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x; if(k>=Cn) return;
    int dc=Coff[k], minc=dc<0?dc:0, slot=0; size_t base=(size_t)k*stride;
    for(int bi=0;bi<Bn;++bi){
        int db=Boff[bi], da=dc-db; if(da<=-n||da>=n) continue;
        int lo=0,hi=An-1,ai=-1;
        while(lo<=hi){ int mid=(lo+hi)>>1, v=Aoff[mid];
            if(v==da){ai=mid;break;} else if(v<da)lo=mid+1; else hi=mid-1; }
        if(ai<0) continue;
        GPair g; g.ab=Astarts[ai]; g.bb=Bstarts[bi];
        g.ash=(da<0?da:0)-minc; g.bsh=da+(db<0?db:0)-minc; g.al=Alen[ai]; g.bl=Blen[bi];
        pairs[base+slot]=g; slot++;
    }
    count[k]=slot;
}
/* consumer: implicit 2-D grid, fixed-stride pair blocks + per-diag count. */
template <int MAXP>
__global__ void gather_hyb2_kernel(
    const float* __restrict__ Av, const float* __restrict__ Bv,
    const GPair* __restrict__ pairs, const int* __restrict__ count, int stride,
    float* __restrict__ Cv, const size_t* __restrict__ C_starts, const int* __restrict__ C_len,
    int C_ndiag, int n)
{
    const int k = blockIdx.y; if(k>=C_ndiag) return;
    const int lenC = C_len[k]; const size_t cbase = C_starts[k];
    const int np = count[k]; const size_t pbase = (size_t)k*stride;
    __shared__ size_t sAb[MAXP], sBb[MAXP];
    __shared__ int    sAsh[MAXP], sBsh[MAXP], sAl[MAXP], sBl[MAXP];
    for(int q=threadIdx.x;q<np;q+=blockDim.x){ GPair g=pairs[pbase+q];
        sAb[q]=g.ab; sBb[q]=g.bb; sAsh[q]=g.ash; sBsh[q]=g.bsh; sAl[q]=g.al; sBl[q]=g.bl; }
    __syncthreads();
    for(int p=blockIdx.x*blockDim.x+threadIdx.x; p<lenC; p+=gridDim.x*blockDim.x){
        float acc=0.0f;
        for(int q=0;q<np;++q){ int pa=p+sAsh[q], pb=p+sBsh[q];
            if(pa<0||pa>=sAl[q])continue; if(pb<0||pb>=sBl[q])continue;
            acc+=Av[sAb[q]+pa]*Bv[sBb[q]+pb]; }
        Cv[cbase+p]=acc;
    }
}

/* ---- single-launch "bsearch" kernel: meta's structure, but the in-block resolve
 * loops the NARROW B (=H, <= |H|) and finds the matching A-diagonal by BINARY SEARCH
 * over A's sorted signed offsets (make_C emits them ascending). No 2n-1 lookup table,
 * no separate plan kernels — ONE launch per step like meta, but the resolve cost is
 * O(|B|·log|A|) instead of meta's O(|A|) over a wide, fill-in A. Wins at every n:
 * same launch count as meta at small n, far less resolve work at large n.
 * Requires A_off ascending-sorted (P from make_C is; wrap the initial H sorted). */
template <int MAXP>
__global__ void gather_bsearch_kernel(
    const float* __restrict__ Av, const size_t* __restrict__ A_starts,
    const int* __restrict__ A_off /*sorted signed*/, const int* __restrict__ A_len, int A_ndiag,
    const float* __restrict__ Bv, const size_t* __restrict__ B_starts,
    const int* __restrict__ B_off /*signed*/, const int* __restrict__ B_len, int B_ndiag,
    float* __restrict__ Cv, const size_t* __restrict__ C_starts,
    const int* __restrict__ C_off, const int* __restrict__ C_len, int C_ndiag, int n)
{
    const int k = blockIdx.y; if(k>=C_ndiag) return;
    const int dc=C_off[k], lenC=C_len[k], minc=dc<0?dc:0; const size_t cbase=C_starts[k];
    __shared__ size_t sAb[MAXP], sBb[MAXP];
    __shared__ int sAsh[MAXP], sBsh[MAXP], sAl[MAXP], sBl[MAXP], sNP;
    if(threadIdx.x==0) sNP=0; __syncthreads();
    for(int bi=threadIdx.x; bi<B_ndiag; bi+=blockDim.x){
        int db=B_off[bi], da=dc-db; if(da<=-n||da>=n) continue;
        int lo=0, hi=A_ndiag-1, ai=-1;          // binary search sorted A_off for da
        while(lo<=hi){ int mid=(lo+hi)>>1, v=A_off[mid];
            if(v==da){ai=mid;break;} else if(v<da) lo=mid+1; else hi=mid-1; }
        if(ai<0) continue;
        int slot=atomicAdd(&sNP,1);
        sAb[slot]=A_starts[ai]; sBb[slot]=B_starts[bi];
        sAsh[slot]=(da<0?da:0)-minc; sBsh[slot]=da+(db<0?db:0)-minc;
        sAl[slot]=A_len[ai]; sBl[slot]=B_len[bi];
    }
    __syncthreads(); const int np=sNP;
    for(int p=blockIdx.x*blockDim.x+threadIdx.x; p<lenC; p+=gridDim.x*blockDim.x){
        float acc=0.0f;
        for(int q=0;q<np;++q){ int pa=p+sAsh[q], pb=p+sBsh[q];
            if(pa<0||pa>=sAl[q])continue; if(pb<0||pb>=sBl[q])continue;
            acc+=Av[sAb[q]+pa]*Bv[sBb[q]+pb]; }
        Cv[cbase+p]=acc;
    }
}

/* lean plan: pairPtr + pairs only (no tiles) — for the hybrid kernel. */
struct PairPlan { std::vector<int> pairPtr; std::vector<GPair> pairs; };
inline PairPlan build_pairs_only(
    int n,
    const std::vector<int>& A_len, const std::vector<size_t>& A_starts,
    const std::vector<int>& B_off, const std::vector<int>& B_len, const std::vector<size_t>& B_starts,
    const std::vector<int>& Alookup,
    const std::vector<int>& C_off)
{
    PairPlan pp; int Cn=(int)C_off.size(), Bn=(int)B_off.size();
    pp.pairPtr.assign(Cn+1,0);
    for(int k=0;k<Cn;++k){
        int dc=C_off[k], minc=dc<0?dc:0;
        pp.pairPtr[k]=(int)pp.pairs.size();
        for(int bi=0;bi<Bn;++bi){
            int db=B_off[bi], da=dc-db; if(da<=-n||da>=n) continue;
            int ai=Alookup[da+(n-1)]; if(ai<0) continue;
            GPair g; g.ab=A_starts[ai]; g.bb=B_starts[bi];
            g.ash=(da<0?da:0)-minc; g.bsh=da+(db<0?db:0)-minc;
            g.al=A_len[ai]; g.bl=B_len[bi];
            pp.pairs.push_back(g);
        }
    }
    pp.pairPtr[Cn]=(int)pp.pairs.size();
    return pp;
}

/* Host pair/tile plan for C = A*B.
 *  A_off/A_len/A_starts  : A diagonals (signed offsets, lengths, value-array starts)
 *  Blookup[db+(n-1)]     : B diagonal index or -1   (B narrow -> iterate B per C-diag)
 *  B_off/B_len/B_starts  : B diagonals
 *  C_off/C_len           : C diagonals (signed), lengths
 * Pairs per C-diag are found by iterating B's offsets (narrow) and looking up A,
 * so cost is C_ndiag * B_ndiag, not C_ndiag * A_ndiag. */
struct FlatPlan { std::vector<int> pairPtr; std::vector<GPair> pairs; std::vector<int2> tiles; int nTiles; };

inline FlatPlan build_flat_plan(
    int n,
    const std::vector<int>& A_off, const std::vector<int>& A_len, const std::vector<size_t>& A_starts,
    const std::vector<int>& B_off, const std::vector<int>& B_len, const std::vector<size_t>& B_starts,
    const std::vector<int>& Alookup,   // db+(n-1) -> A diag idx, -1
    const std::vector<int>& C_off, const std::vector<int>& C_len,
    int POS /* = TILE*ILP */)
{
    FlatPlan fp; int Cn=(int)C_off.size(), Bn=(int)B_off.size();
    fp.pairPtr.assign(Cn+1,0);
    for(int k=0;k<Cn;++k){
        int dc=C_off[k], minc=dc<0?dc:0;
        fp.pairPtr[k]=(int)fp.pairs.size();
        for(int bi=0;bi<Bn;++bi){
            int db=B_off[bi], da=dc-db; if(da<=-n||da>=n) continue;
            int ai=Alookup[da+(n-1)]; if(ai<0) continue;
            GPair g; g.ab=A_starts[ai]; g.bb=B_starts[bi];
            g.ash=(da<0?da:0)-minc; g.bsh=da+(db<0?db:0)-minc;
            g.al=A_len[ai]; g.bl=B_len[bi];
            fp.pairs.push_back(g);
        }
        for(int ts=0;ts<C_len[k];ts+=POS) fp.tiles.push_back(make_int2(k,ts));
    }
    fp.pairPtr[Cn]=(int)fp.pairs.size();
    fp.nTiles=(int)fp.tiles.size();
    return fp;
}
