/* ============================================================
 * didx_strip_bench.cu — didx_s: diagonal-major zero-skip SpMV with the
 * OUTPUT strip staged in shared memory (the feasible dual of "put the
 * vector in shared": the input window grows with matrix bandwidth and
 * cannot fit; the output window is a constant strip and always fits).
 *
 * Layout (plan): true nonzeros sorted by (strip, diagonal, row); per run
 * (= one diagonal inside one strip) a header {elemStart, offset}; per
 * element {val 4B, strip-local row 2B}.  Kernel: one block per strip,
 * y-strip zeroed in shared, each WARP claims runs round-robin, lanes
 * stride a run's elements -> x reads are sequential (streamed), y updates
 * are shared-memory atomicAdd (same-run rows are distinct -> no self
 * conflicts), strip written back coalesced.
 *
 * Registered PREDICTION: if O2_20's 74% long-scoreboard stall is the
 * random x gather, expect >=1.5x over didx_w (-> <0.65ms, beating
 * cuSPARSE ALG2's 0.947ms); ~1.1x would instead indict the val stream.
 *
 * out: PREPCSV rows (didx_w ref, didx_s) + SSTAT verify (norm-relative;
 * atomics reorder the sum so bitwise equality does not apply)
 * usage: didx_strip_bench <dia.txt> [iters=50] [strip=4096]
 * ============================================================ */
#include "../dia_io.hpp"
#include "common.cuh"
#include <chrono>
#include <cstdint>
#include <cmath>

using clk2 = std::chrono::steady_clock;
static double msb(clk2::time_point a, clk2::time_point b){
    return std::chrono::duration<double,std::milli>(b-a).count();
}
#ifndef TPB
#define TPB 256
#endif

__global__ void spmv_didx_w_kernel(int n, int D,
    const int* __restrict__ rp, const unsigned short* __restrict__ slot,
    const float* __restrict__ val, const int* __restrict__ offs,
    const float* __restrict__ X, float* __restrict__ Y)
{
    extern __shared__ int soff[];
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offs[k];
    __syncthreads();
    const int w = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (w >= n) return;
    float acc = 0.f;
    const int e = rp[w + 1];
    for (int j = rp[w] + lane; j < e; j += 32)
        acc += val[j] * X[w + soff[slot[j]]];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) Y[w] = acc;
}

__global__ void spmv_strip_kernel(int n, int S,
    const int* __restrict__ sRunPtr,   /* strips+1 -> run range        */
    const int* __restrict__ runElem,   /* nruns+1  -> element start    */
    const int* __restrict__ runOff,    /* nruns    -> diagonal offset  */
    const unsigned short* __restrict__ lrow,
    const float* __restrict__ val,
    const float* __restrict__ X, float* __restrict__ Y)
{
    extern __shared__ float sy[];
    const int s = blockIdx.x;
    const size_t r0 = (size_t)s * S;
    const int rows = min((size_t)S, (size_t)n - r0);
    for (int i = threadIdx.x; i < rows; i += blockDim.x) sy[i] = 0.f;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int rend = sRunPtr[s + 1];
    for (int ri = sRunPtr[s] + warp; ri < rend; ri += nw) {
        const int e0 = runElem[ri], e1 = runElem[ri + 1], off = runOff[ri];
        const float* xb = X + r0 + off;
        for (int j = e0 + lane; j < e1; j += 32) {
            const int lr = lrow[j];
            atomicAdd(&sy[lr], val[j] * xb[lr]);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows; i += blockDim.x) Y[r0 + i] = sy[i];
}

static double maxrel(const std::vector<double>& ref, const std::vector<float>& y){
    double mr = 0, nrm = 0;
    for (size_t i = 0; i < ref.size(); ++i) nrm = std::max(nrm, std::fabs(ref[i]));
    for (size_t i = 0; i < ref.size(); ++i)
        mr = std::max(mr, std::fabs(ref[i] - (double)y[i]));
    return nrm > 0 ? mr / nrm : mr;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s dia.txt [iters] [strip]\n", argv[0]); return 1; }
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    const int S     = argc > 3 ? atoi(argv[3]) : 4096;   /* strip rows; u16 lrow => S<=65536 */
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, D = (int)H.offsets.size();

    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; ++i)
        x[i] = (float)(((i * 1103515245u + 12345u) >> 16 & 0x3ff)) / 1024.f - 0.5f;
    float* dX = dupload(x);
    float *dY1, *dY2;
    CUDA_CHECK(cudaMalloc(&dY1,(size_t)n*4)); CUDA_CHECK(cudaMalloc(&dY2,(size_t)n*4));

    /* base didx plan (row-major) for the reference variant + CPU ref */
    auto t0 = clk2::now();
    DidxPlan R = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
    const size_t nnz = (size_t)R.rp[n];
    std::vector<unsigned short> gk(nnz);
    if (R.wide) gk = R.d16; else gk.assign(R.d8.begin(), R.d8.end());
    auto t1 = clk2::now();

    std::vector<double> yref((size_t)n, 0.0);
    for (int r = 0; r < n; ++r)
        for (int j = R.rp[r]; j < R.rp[r+1]; ++j)
            yref[r] += (double)R.val[j] * (double)x[r + H.offsets[gk[j]]];

    int* drp = dupload(R.rp); int* doff = dupload(std::vector<int>(H.offsets));
    float* dv = dupload(R.val); unsigned short* ds = dupload(gk);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = clk2::now();
    std::vector<float> y((size_t)n);
    double r1, r2;
    {
        const int wblocks = (int)(((size_t)n * 32 + TPB - 1) / TPB);
        auto l=[&](){ spmv_didx_w_kernel<<<wblocks,TPB,(size_t)D*4>>>(n,D,drp,ds,dv,doff,dX,dY1); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("PREPCSV,%s,%d,%d,spmv,didx_w,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,msb(t0,t1),msb(t1,t2),tms(l,10,iters));
        CUDA_CHECK(cudaMemcpy(y.data(),dY1,(size_t)n*4,cudaMemcpyDeviceToHost)); r1=maxrel(yref,y);
    }

    /* strip-diagonal plan: elements sorted (strip, diag, row) — built by
     * iterating diagonals once per strip via per-diagonal row windows */
    auto t3 = clk2::now();
    const int strips = (n + S - 1) / S;
    std::vector<int> sRunPtr(strips + 1, 0), runElem, runOff;
    std::vector<unsigned short> lrow(nnz);
    std::vector<float> sval(nnz);
    size_t w = 0;
    for (int s = 0; s < strips; ++s) {
        const int r0 = s * S, r1_ = std::min(n, r0 + S);
        for (int k = 0; k < D; ++k) {
            const int off = H.offsets[k], len = H.lengths[k];
            /* rows carrying this diagonal: r in [max(0,-off), ...) with p=min(r,r+off) */
            int lo = std::max(r0, off >= 0 ? 0 : -off);
            int hi = std::min(r1_, (off >= 0 ? len : len - off));
            if (lo >= hi) continue;
            const size_t st = H.starts[k];
            int cnt = 0;
            for (int r = lo; r < hi; ++r) {
                const int p = off >= 0 ? r : r + off;
                const float v = H.values[st + p];
                if (v == 0.f) continue;
                sval[w] = v; lrow[w] = (unsigned short)(r - r0); ++w; ++cnt;
            }
            if (cnt) { runElem.push_back((int)(w - cnt)); runOff.push_back(off); }
        }
        sRunPtr[s + 1] = (int)runElem.size();
    }
    runElem.push_back((int)w);
    auto t4 = clk2::now();
    int* dsp = dupload(sRunPtr); int* dre = dupload(runElem); int* dro = dupload(runOff);
    unsigned short* dlr = dupload(lrow); float* dsv = dupload(sval);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t5 = clk2::now();
    {
        auto l=[&](){ spmv_strip_kernel<<<strips,1024,(size_t)S*4>>>(n,S,dsp,dre,dro,dlr,dsv,dX,dY2); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("PREPCSV,%s,%d,%d,spmv,didx_s,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,msb(t3,t4),msb(t4,t5),tms(l,10,iters));
        CUDA_CHECK(cudaMemcpy(y.data(),dY2,(size_t)n*4,cudaMemcpyDeviceToHost)); r2=maxrel(yref,y);
    }
    printf("SSTAT,%s,S=%d,strips=%d,runs=%zu,avg_run=%.1f,relerr_w=%.2e,relerr_s=%.2e,%s\n",
           argv[1], S, strips, runOff.size(), (double)w/std::max((size_t)1,runOff.size()),
           r1, r2, (r1<1e-5 && r2<1e-5) ? "verify_OK" : "verify_FAIL");
    return (r1<1e-5 && r2<1e-5) ? 0 : 2;
}
