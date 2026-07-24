/* ============================================================
 * tc_spmv_sym.cuh — SYMMETRIC (store-half) reconstruction + SpMV.
 *
 * For a real-symmetric H (the Hamiltonians here), the d<0 diagonals are
 * redundant: H[r,r-d] = H[r-d,r]. So we reconstruct only the d>=0 diagonals
 * (half the K x N recon) and recover the full y as
 *
 *   y[r] = v0[r]*x[r]                                          (d=0, main)
 *        + sum_{d>0} ( Rh[d, r+d]*x[r+d]   (upper)
 *                    + Rh[d, r  ]*x[r-d] ) (lower, SAME half value)
 *
 * where Rh[k,c] = H[c-d_k, c], d_k >= 0 (identical layout to the full recon,
 * just the non-negative rows).  This HALVES the recon operand -> the GEMM
 * K-dimension halves and the recon fits for the large-q matrices whose full
 * recon OOMs.  It does NOT reduce the MMA count in this straightforward
 * mapping (the lower term is a second gather, so ~K MMAs either way); the
 * open speedup is to route the lower term through the TC's wasted off-diagonal
 * accumulator lanes — see NOTE at the symmetric TC kernel.
 *
 * Provides:
 *   build_recon_sym(...)             -> half recon (host)
 *   sym_spmv_cuda<<<>>>              -> CUDA-core symmetric SpMV (exact)
 *   launch_tc_spmv_regdirect_sym(...)-> tensor-core symmetric SpMV (dual-gather)
 * ============================================================ */
#pragma once
#include "dia_reconstruct.cuh"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <functional>

/* ---- host: build the d>=0 half recon from a full DiaMatrix-like input ----
 * offsets_sym: the non-negative offsets, DESCENDING (main d=0 is last).
 * values_sym : num_sym * cols, Rh[k,c] = H[c-d_k,c]. Returns Ksym. ---------- */
struct ReconSym {
    int rows, cols, num_sym;
    std::vector<int>   offsets;   /* d>=0, descending; last entry may be 0 */
    std::vector<float> values;    /* num_sym * cols */
};

/* Build from the compact host arrays (offsets ascending, lengths, starts,
 * values with position p=min(row,col)) — same fields as sim/dia_io DiaHost. */
inline ReconSym build_recon_sym(int n, const std::vector<int>& offsets,
                                const std::vector<int>& lengths,
                                const std::vector<size_t>& starts,
                                const std::vector<float>& values)
{
    ReconSym R; R.rows=n; R.cols=n;
    for (int o : offsets) if (o >= 0) R.offsets.push_back(o);
    std::sort(R.offsets.begin(), R.offsets.end(), std::greater<int>());
    R.num_sym = (int)R.offsets.size();
    R.values.assign((size_t)R.num_sym * n, 0.f);
    // map offset -> its index in the compact input
    for (int k = 0; k < R.num_sym; ++k) {
        int d = R.offsets[k], di = -1;
        for (int i = 0; i < (int)offsets.size(); ++i) if (offsets[i]==d){di=i;break;}
        if (di < 0) continue;
        int len = lengths[di]; size_t base = starts[di];
        // d>=0: stored position p = row = col-d ; recon column c = d + j
        for (int j = 0; j < len; ++j) R.values[(size_t)k*n + (d + j)] = values[base + j];
    }
    return R;
}

/* ---- CUDA-core symmetric SpMV: one thread/row, reads the half recon. ---- */
__global__ void sym_spmv_cuda(int n, int K, const int* __restrict__ off,
                              const float* __restrict__ Rh,
                              const float* __restrict__ x, float* __restrict__ y)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int d = off[k];
        const int cu = r + d;                       // main (d=0) + upper (d>0)
        if (cu < n) acc += Rh[(size_t)k*n + cu] * x[cu];
        if (d > 0) { const int cl = r - d;          // lower: same half value at col r
            if (cl >= 0) acc += Rh[(size_t)k*n + r] * x[cl]; }
    }
    y[r] = acc;
}

/* ---- tensor-core symmetric SpMV (register-direct, dual gather). ----
 * Same m16n8k8 machinery as tc_spmv_regdirect, but over the d>=0 half recon,
 * accumulating TWO passes into the same accumulator:
 *   UPPER: A[m,k]=Rh[k, tile+m+d_k], B=x[tile+n+d_k]        (all k)
 *   LOWER: A[m,k]=Rh[k, tile+m     ], B=x[tile+n-d_k]        (k with d_k>0)
 * The diagonal of (A_up*B_up + A_lo*B_lo) is the full y.
 *
 * NOTE (open optimization): this still issues ~K MMAs (K/2 upper + K/2 lower).
 * The TC keeps only 16 of 128 accumulator entries; a layout that harvests the
 * lower term from the dead off-diagonal lanes would halve the MMA count. Not
 * done here — this kernel establishes the correct half-operand baseline. */
namespace symtc {
// TF32 MMA is sm_80+. Stub the asm for pre-Ampere device passes so this header
// COMPILES on sm_75 etc. (the TC kernel is never launched there — callers pick
// the CUDA-core sym_spmv_cuda). Behaviour on sm_80+ is unchanged.
__device__ __forceinline__ uint32_t to_tf32(float f){
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    uint32_t u; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(f)); return u;
#else
    return __float_as_uint(f);
#endif
}
__device__ __forceinline__ void mma(float* acc, const uint32_t* A, const uint32_t* B){
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
        : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]));
#endif
}
}

__global__ void tc_spmv_sym_kernel(ReconView R, const float* x, int x_size, float* y)
{
    const int wpb = blockDim.x >> 5;
    const int gw  = (int)blockIdx.x * wpb + (threadIdx.x >> 5);
    const int tile_row = gw * MMA_M;
    if (tile_row >= R.rows) return;
    const int lane = (int)threadIdx.x & 31, g = lane>>2, t = lane&3;
    float acc_top[4]={0,0,0,0}, acc_bot[4]={0,0,0,0};

    for (int dk = 0; dk < R.num_diags; dk += MMA_K) {
        const int di0=dk+t, di4=dk+t+4;
        const bool v0=(di0<R.num_diags), v4=(di4<R.num_diags);
        const int o0 = v0 ? R.diag_offsets[di0] : 0;   // d>=0 (half recon)
        const int o4 = v4 ? R.diag_offsets[di4] : 0;

        // ---- UPPER pass: A col = tile+g(+8)+off, B = x[tile+g(+8)+off] ----
        uint32_t Au[4]={0,0,0,0}, Bt[2]={0,0}, Bb[2]={0,0};
        if (v0){ int c=tile_row+g+o0;   if(c>=0&&c<R.cols)Au[0]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                 c=tile_row+g+8+o0;      if(c>=0&&c<R.cols)Au[1]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                 int xi=tile_row+g+o0;   if(xi>=0&&xi<x_size)Bt[0]=symtc::to_tf32(x[xi]);
                 xi=tile_row+g+8+o0;     if(xi>=0&&xi<x_size)Bb[0]=symtc::to_tf32(x[xi]); }
        if (v4){ int c=tile_row+g+o4;   if(c>=0&&c<R.cols)Au[2]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                 c=tile_row+g+8+o4;      if(c>=0&&c<R.cols)Au[3]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                 int xi=tile_row+g+o4;   if(xi>=0&&xi<x_size)Bt[1]=symtc::to_tf32(x[xi]);
                 xi=tile_row+g+8+o4;     if(xi>=0&&xi<x_size)Bb[1]=symtc::to_tf32(x[xi]); }
        symtc::mma(acc_top,Au,Bt); symtc::mma(acc_bot,Au,Bb);

        // ---- LOWER pass: only d>0. A col = tile+g(+8), B = x[tile+g(+8)-off] ----
        uint32_t Al[4]={0,0,0,0}, Lt[2]={0,0}, Lb[2]={0,0};
        if (v0 && o0>0){ int c=tile_row+g;    if(c>=0&&c<R.cols)Al[0]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                         c=tile_row+g+8;       if(c>=0&&c<R.cols)Al[1]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                         int xi=tile_row+g-o0; if(xi>=0&&xi<x_size)Lt[0]=symtc::to_tf32(x[xi]);
                         xi=tile_row+g+8-o0;   if(xi>=0&&xi<x_size)Lb[0]=symtc::to_tf32(x[xi]); }
        if (v4 && o4>0){ int c=tile_row+g;    if(c>=0&&c<R.cols)Al[2]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                         c=tile_row+g+8;       if(c>=0&&c<R.cols)Al[3]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                         int xi=tile_row+g-o4; if(xi>=0&&xi<x_size)Lt[1]=symtc::to_tf32(x[xi]);
                         xi=tile_row+g+8-o4;   if(xi>=0&&xi<x_size)Lb[1]=symtc::to_tf32(x[xi]); }
        symtc::mma(acc_top,Al,Lt); symtc::mma(acc_bot,Al,Lb);
    }

    // diagonal extraction (identical lane map to the full kernel)
    if (g == 2*t) {
        int rt=tile_row+g, rb=tile_row+8+g;
        if (rt<R.rows) y[rt]=acc_top[0];
        if (rb<R.rows) y[rb]=acc_bot[2];
    } else if (g == 2*t+1) {
        int rt=tile_row+g, rb=tile_row+8+g;
        if (rt<R.rows) y[rt]=acc_top[1];
        if (rb<R.rows) y[rb]=acc_bot[3];
    }
}

inline void launch_tc_spmv_regdirect_sym(ReconView R, const float* d_x, int x_size,
                                         float* d_y, cudaStream_t stream=0)
{
    const int n_tiles=(R.rows+MMA_M-1)/MMA_M; if(!n_tiles) return;
    const int n_blocks=(n_tiles+3)/4;
    tc_spmv_sym_kernel<<<n_blocks,4*32,0,stream>>>(R,d_x,x_size,d_y);
}
