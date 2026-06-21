/* ============================================================
 * gather_flat_sym.cuh — flattened, precomputed-pair SYMMETRIC SpMSpM.
 *
 * The symmetric store-half/compute-half product (see gather_sym.cuh) with
 * gather_flat's scheduling grafted on:
 *   (1) flatten to a 1-D grid of uniform TILE-sized tiles over all
 *       (upper-C-diagonal, position) work  -> no wave-quantization tail;
 *   (2) precompute the per-C-diagonal pair lists on the HOST
 *       (pairPtr -> GPair[]) -> NO in-block atomicAdd / resolve-sync.
 * gather_meta_sym builds its pair list at runtime with a serializing
 * atomicAdd(&sNP,1)+__syncthreads inside every block; that resolve is the
 * SM<65% ceiling. Here the device kernel is a pure pair-consumer.
 *
 * The symmetric pair logic — A,B stored UPPER only (offsets>=0, values by
 * |offset|); for each C-diagonal dc>=0 each upper A-offset a unfolds into
 * da=+/-a (skip the -0 duplicate), db=dc-da, B looked up via Babs[|db|];
 * shifts ash=min(0,da), bsh=da+min(0,db) (minc=0 since dc>=0) — is done ONCE
 * on the host in build_flat_sym_plan, identical math to gather_meta_sym.
 *
 * NOTE: the host build is cheap for a one-shot C=H*H but recurs each step of
 * a matrix-power chain where A grows; time it there.
 * ============================================================ */
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

struct GPair { size_t ab, bb; int ash, bsh, al, bl; };

/* Pure pair-consumer: uniform 1-D tiles over (C-diag, position). No atomics,
 * no resolve-sync — the block's pair list is loaded into smem once, then ILP
 * positions/thread accumulate. Mirrors the archived gather_flat_kernel; the
 * "symmetric" knowledge is entirely in the host plan it consumes. */
template <int MAXP, int ILP>
__global__ void gather_flat_sym_kernel(
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

/* Host pair/tile plan for the SYMMETRIC C = A*B (A=B=upper H).
 *  A_off            : A UPPER offsets (>=0)
 *  A_len, A_starts  : A upper diagonal lengths, value-array starts
 *  B_len, B_starts  : B upper diagonal lengths, value-array starts
 *  Babs[|off|]      : B upper-diagonal index, or -1 (size n)
 *  C_off, C_len     : C UPPER diagonals (dc>=0), lengths
 *  POS = TILE*ILP   : positions covered per block
 * Per C-diagonal dc>=0 (minc=0), each upper A-offset a unfolds into da=+/-a;
 * db=dc-da; B via Babs[|db|]; ash=min(0,da), bsh=da+min(0,db). Same math as
 * gather_meta_sym, just precomputed. */
struct FlatSymPlan { std::vector<int> pairPtr; std::vector<GPair> pairs; std::vector<int2> tiles; int nTiles; int maxPairs; };

inline FlatSymPlan build_flat_sym_plan(
    int n,
    const std::vector<int>& A_off,
    const std::vector<int>& A_len, const std::vector<size_t>& A_starts,
    const std::vector<int>& B_len, const std::vector<size_t>& B_starts,
    const std::vector<int>& Babs,
    const std::vector<int>& C_off, const std::vector<int>& C_len,
    int POS /* = TILE*ILP */)
{
    FlatSymPlan fp; int Cn=(int)C_off.size(), An=(int)A_off.size();
    fp.pairPtr.assign(Cn+1,0); fp.maxPairs=0;
    for(int k=0;k<Cn;++k){
        int dc=C_off[k];                 // dc>=0 -> minc=0
        fp.pairPtr[k]=(int)fp.pairs.size();
        for(int ai=0;ai<An;++ai){
            int a=A_off[ai];
            for(int s=0;s<2;++s){
                if(s==1 && a==0) continue;          // offset 0 counted once
                int da=(s==0)?a:-a;
                int db=dc-da; int adb=db<0?-db:db;
                if(adb>=n) continue;
                int bi=Babs[adb]; if(bi<0) continue;
                GPair g; g.ab=A_starts[ai]; g.bb=B_starts[bi];
                g.ash=(da<0?da:0); g.bsh=da+(db<0?db:0);
                g.al=A_len[ai]; g.bl=B_len[bi];
                fp.pairs.push_back(g);
            }
        }
        int np=(int)fp.pairs.size()-fp.pairPtr[k]; if(np>fp.maxPairs) fp.maxPairs=np;
        for(int ts=0;ts<C_len[k];ts+=POS) fp.tiles.push_back(make_int2(k,ts));
    }
    fp.pairPtr[Cn]=(int)fp.pairs.size();
    fp.nTiles=(int)fp.tiles.size();
    return fp;
}
