/* ============================================================
 * gather_sym.cuh — symmetric diagonal SpMSpM (store-half + compute-half).
 *
 * For a symmetric operator H (Hermitian Hamiltonians), C = A*B is itself
 * symmetric, so only the dc>=0 diagonals of C are computed and only the
 * upper (|offset|) diagonals of A,B are stored. Roughly halves both storage
 * and work versus the full gather.
 *
 * Layout: A,B store one entry per distinct |offset| (Aoff/Boff >= 0). Babs
 * maps |offset| -> upper-diagonal index (size n, -1 if absent). C stores the
 * dc>=0 diagonals only. Each block (blockIdx.y = C-diagonal k) builds the
 * contributing (A,B) pair list in shared memory once, then the threads sweep
 * the positions of that C-diagonal.
 * ============================================================ */
#pragma once
#include <cstddef>

// Symmetric gather: A,B upper-only (values by |offset|), C upper-only (dc>=0).
template<int DMAX>
__global__ void gather_meta_sym(
    const float* __restrict__ Aval, const size_t* __restrict__ Astart,
    const int* __restrict__ Aoff /*>=0*/, const int* __restrict__ Alen, int A_und,
    const float* __restrict__ Bval, const size_t* __restrict__ Bstart,
    const int* __restrict__ Blen, const int* __restrict__ Babs /*|off|->upper idx, size n, -1 absent*/,
    float* __restrict__ Cval, const size_t* __restrict__ Cstart,
    const int* __restrict__ Coff /*>=0*/, const int* __restrict__ Clen, int C_und, int n)
{
    int k=blockIdx.y; if(k>=C_und) return;
    int dc=Coff[k]; int lenC=Clen[k]; size_t cbase=Cstart[k];   // dc>=0 -> minc=0
    __shared__ size_t sAb[DMAX], sBb[DMAX];
    __shared__ int sAsh[DMAX], sBsh[DMAX], sAl[DMAX], sBl[DMAX], sNP;
    if(threadIdx.x==0) sNP=0; __syncthreads();
    for(int ai=threadIdx.x; ai<A_und; ai+=blockDim.x){
        int a=Aoff[ai];
        #pragma unroll
        for(int s=0;s<2;++s){
            if(s==1 && a==0) continue;          // offset 0 counted once
            int da = (s==0)? a : -a;
            int db = dc - da; int adb = db<0?-db:db;
            if(adb>=n) continue;
            int bi = Babs[adb]; if(bi<0) continue;   // |db| not a B diagonal
            int slot=atomicAdd(&sNP,1); if(slot>=DMAX) continue;
            sAb[slot]=Astart[ai]; sBb[slot]=Bstart[bi];
            sAsh[slot]=(da<0?da:0);              // min(0,da)
            sBsh[slot]=da+(db<0?db:0);           // da+min(0,db)
            sAl[slot]=Alen[ai]; sBl[slot]=Blen[bi];
        }
    }
    __syncthreads(); int nP=sNP; if(nP>DMAX)nP=DMAX;
    for(int p=blockIdx.x*blockDim.x+threadIdx.x; p<lenC; p+=gridDim.x*blockDim.x){
        float acc=0;
        for(int q=0;q<nP;++q){ int pa=p+sAsh[q], pb=p+sBsh[q];
            if(pa<0||pa>=sAl[q])continue; if(pb<0||pb>=sBl[q])continue;
            acc += Aval[sAb[q]+pa]*Bval[sBb[q]+pb]; }
        Cval[cbase+p]=acc;
    }
}
