/* ============================================================
 * didx_w4_bench.cu — didx_w4: warp-per-row + 4-wide vectorized lanes.
 *
 * ncu on didx_w @ O2_20: long_scoreboard 74%, short_scoreboard 5.5%,
 * DRAM 33%, occupancy 91%  =>  global-latency bound, NOT the smem table.
 * Fix: raise per-lane memory-level parallelism — each lane processes 4
 * consecutive elements per iteration (float4 val + ushort4 slot, 4
 * independent x-gathers).  Rows padded to a multiple of 4 in the plan
 * (pad val=0, slot=0: contributes exactly 0, x read stays in bounds).
 *
 * Registered PREDICTION: 1.2-1.5x over didx_w at O2_20 (0.71-0.88ms),
 * i.e. ties or beats honest cuSPARSE ALG2 (0.947ms) diagonal-natively.
 *
 * out: PREPCSV rows (didx_w ref, didx_w4) + W4STAT verify (norm-relative)
 * usage: didx_w4_bench <dia.txt> [iters=50]
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

__global__ void spmv_didx_w4_kernel(int n, int D,
    const int* __restrict__ rp4,                 /* padded, entries %4 == 0 */
    const ushort4* __restrict__ slot4, const float4* __restrict__ val4,
    const int* __restrict__ offs,
    const float* __restrict__ X, float* __restrict__ Y)
{
    extern __shared__ int soff[];
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offs[k];
    __syncthreads();
    const int w = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (w >= n) return;
    float acc = 0.f;
    const int b = rp4[w] >> 2, e = rp4[w + 1] >> 2;   /* float4 units */
    for (int q = b + lane; q < e; q += 32) {
        const float4  v = val4[q];
        const ushort4 s = slot4[q];
        acc += v.x * X[w + soff[s.x]] + v.y * X[w + soff[s.y]]
             + v.z * X[w + soff[s.z]] + v.w * X[w + soff[s.w]];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) Y[w] = acc;
}

/* w4p: same as w4 + streaming hints on the val/slot streams (__ldcs, evict-
 * first) so the 1.24GB/apply stream stops thrashing x (4MB, should live in
 * L2); host side additionally pins x with an L2 persisting-access window. */
__global__ void spmv_didx_w4p_kernel(int n, int D,
    const int* __restrict__ rp4,
    const ushort4* __restrict__ slot4, const float4* __restrict__ val4,
    const int* __restrict__ offs,
    const float* __restrict__ X, float* __restrict__ Y)
{
    extern __shared__ int soff[];
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offs[k];
    __syncthreads();
    const int w = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (w >= n) return;
    float acc = 0.f;
    const int b = rp4[w] >> 2, e = rp4[w + 1] >> 2;
    for (int q = b + lane; q < e; q += 32) {
        const float4  v = __ldcs(val4 + q);
        const ushort4 s = __ldcs(slot4 + q);
        acc += v.x * X[w + soff[s.x]] + v.y * X[w + soff[s.y]]
             + v.z * X[w + soff[s.z]] + v.w * X[w + soff[s.w]];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) Y[w] = acc;
}

static double maxrel(const std::vector<double>& ref, const std::vector<float>& y){
    double mr = 0, nrm = 0;
    for (size_t i = 0; i < ref.size(); ++i) nrm = std::max(nrm, std::fabs(ref[i]));
    for (size_t i = 0; i < ref.size(); ++i)
        mr = std::max(mr, std::fabs(ref[i] - (double)y[i]));
    return nrm > 0 ? mr / nrm : mr;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s dia.txt [iters]\n", argv[0]); return 1; }
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, D = (int)H.offsets.size();

    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; ++i)
        x[i] = (float)(((i * 1103515245u + 12345u) >> 16 & 0x3ff)) / 1024.f - 0.5f;
    float* dX = dupload(x);
    float *dY1, *dY4;
    CUDA_CHECK(cudaMalloc(&dY1,(size_t)n*4)); CUDA_CHECK(cudaMalloc(&dY4,(size_t)n*4));

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
    const int wblocks = (int)(((size_t)n * 32 + TPB - 1) / TPB);
    std::vector<float> y((size_t)n);
    double r1, r4;

    { auto l=[&](){ spmv_didx_w_kernel<<<wblocks,TPB,(size_t)D*4>>>(n,D,drp,ds,dv,doff,dX,dY1); };
      l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
      printf("PREPCSV,%s,%d,%d,spmv,didx_w,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,msb(t0,t1),msb(t1,t2),tms(l,10,iters));
      CUDA_CHECK(cudaMemcpy(y.data(),dY1,(size_t)n*4,cudaMemcpyDeviceToHost)); r1=maxrel(yref,y); }

    /* padded plan: every row length rounded up to a multiple of 4 */
    auto t3 = clk2::now();
    std::vector<int> rp4(n + 1, 0);
    for (int r = 0; r < n; ++r)
        rp4[r+1] = rp4[r] + ((R.rp[r+1] - R.rp[r] + 3) & ~3);
    const size_t nnz4 = rp4[n];
    std::vector<float> v4(nnz4, 0.f);
    std::vector<unsigned short> s4(nnz4, 0);
    for (int r = 0; r < n; ++r) {
        int dst = rp4[r];
        for (int j = R.rp[r]; j < R.rp[r+1]; ++j, ++dst) { v4[dst]=R.val[j]; s4[dst]=gk[j]; }
    }
    auto t4 = clk2::now();
    int* drp4 = dupload(rp4); float* dv4 = dupload(v4); unsigned short* ds4 = dupload(s4);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t5 = clk2::now();
    { auto l=[&](){ spmv_didx_w4_kernel<<<wblocks,TPB,(size_t)D*4>>>(n,D,drp4,
          (const ushort4*)ds4,(const float4*)dv4,doff,dX,dY4); };
      l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
      printf("PREPCSV,%s,%d,%d,spmv,didx_w4,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,
             msb(t0,t1)+msb(t3,t4), msb(t4,t5), tms(l,10,iters));
      CUDA_CHECK(cudaMemcpy(y.data(),dY4,(size_t)n*4,cudaMemcpyDeviceToHost)); r4=maxrel(yref,y); }

    /* w4p: pin x in a persisting L2 window, val/slot via __ldcs */
    double rp_;
    {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)n * 4);
        cudaStreamAttrValue at{};
        at.accessPolicyWindow.base_ptr  = dX;
        at.accessPolicyWindow.num_bytes = (size_t)n * 4;
        at.accessPolicyWindow.hitRatio  = 1.0f;
        at.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        at.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        CUDA_CHECK(cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &at));
        auto l=[&](){ spmv_didx_w4p_kernel<<<wblocks,TPB,(size_t)D*4>>>(n,D,drp4,
            (const ushort4*)ds4,(const float4*)dv4,doff,dX,dY4); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("PREPCSV,%s,%d,%d,spmv,didx_w4p,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,
               msb(t0,t1)+msb(t3,t4), msb(t4,t5), tms(l,10,iters));
        CUDA_CHECK(cudaMemcpy(y.data(),dY4,(size_t)n*4,cudaMemcpyDeviceToHost)); rp_=maxrel(yref,y);
    }
    printf("W4STAT,%s,pad_overhead=%.3f,relerr_w=%.2e,relerr_w4=%.2e,relerr_w4p=%.2e,%s\n",
           argv[1], (double)nnz4/(double)nnz - 1.0, r1, r4, rp_,
           (r1<1e-5 && r4<1e-5 && rp_<1e-5) ? "verify_OK" : "verify_FAIL");
    return (r1<1e-5 && r4<1e-5 && rp_<1e-5) ? 0 : 2;
}
