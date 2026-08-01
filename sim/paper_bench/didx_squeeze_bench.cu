/* ============================================================
 * didx_squeeze_bench.cu — attack the row-serialization factor on wide bands.
 *
 * O2_20 evidence chain so far: table size ruled out (didx_t), 1B slots +
 * small tables worth +26% (didx_g), remaining 1.9x vs cuSPARSE ALG2 blamed
 * on one-THREAD-per-row over ~205-nonzero rows.  This bench tests warp-per-
 * row on the same true-nonzero data:
 *   didx     scalar reference (continuity with all previous rows)
 *   didx_w   one WARP per row, 2B global slots, global offs table in smem
 *   csr_warp existing spmv_csr_warp kernel on true-nonzero CSR (4B cols)
 * didx_w vs csr_warp isolates index-representation AT equal parallelism.
 *
 * Warp reduction reorders the sum -> verification is vs a CPU double
 * reference (max rel err), not bitwise.
 *
 * out:  PREPCSV rows (variants didx, didx_w, csr_warp) + SQZSTAT,file,relerrs
 * usage: didx_squeeze_bench <dia.txt> [iters=50]
 * ============================================================ */
#include "../dia_io.hpp"
#include "common.cuh"
#include "../../spmv/src/spmv_zeroskip_kernels.cuh"
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

/* max |ref-y| normalized by ||ref||_inf: per-row relative error explodes on
 * cancellation rows (y[r]~0 from ~1-sized terms) and would flag correct
 * kernels; norm-relative is the right criterion for a linear operator apply */
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
    float *dY0, *dY1, *dY2;
    CUDA_CHECK(cudaMalloc(&dY0, (size_t)n*4)); CUDA_CHECK(cudaMalloc(&dY1, (size_t)n*4));
    CUDA_CHECK(cudaMalloc(&dY2, (size_t)n*4));

    /* shared plan build (didx layout is the substrate for all three) */
    auto t0 = clk2::now();
    DidxPlan R = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
    auto t1 = clk2::now();
    const size_t nnz = (size_t)R.rp[n];
    std::vector<unsigned short> gk(nnz);
    if (R.wide) gk = R.d16; else gk.assign(R.d8.begin(), R.d8.end());

    /* CPU double reference */
    std::vector<double> yref((size_t)n, 0.0);
    for (int r = 0; r < n; ++r)
        for (int j = R.rp[r]; j < R.rp[r+1]; ++j)
            yref[r] += (double)R.val[j] * (double)x[r + H.offsets[gk[j]]];

    int* drp = dupload(R.rp); int* doff = dupload(std::vector<int>(H.offsets));
    float* dv = dupload(R.val);
    unsigned short* d16 = dupload(gk);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = clk2::now();
    const double plan_ms = msb(t0,t1), up_ms = msb(t1,t2);
    const int blocks  = (n + TPB - 1) / TPB;
    const int wblocks = (int)(((size_t)n * 32 + TPB - 1) / TPB);
    std::vector<float> y((size_t)n);
    double rr[3];

    /* didx scalar (reference variant; 2B slots for uniformity when wide) */
    {
        float kms;
        if (R.wide){ auto l=[&](){ spmv_didx_kernel<unsigned short,1><<<blocks,TPB,(size_t)D*4>>>(n,D,drp,d16,dv,doff,dX,dY0); };
                     l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kms=tms(l,10,iters); }
        else { unsigned char* d8 = dupload(R.d8);
               auto l=[&](){ spmv_didx_kernel<unsigned char,1><<<blocks,TPB,(size_t)D*4>>>(n,D,drp,d8,dv,doff,dX,dY0); };
               l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kms=tms(l,10,iters); }
        printf("PREPCSV,%s,%d,%d,spmv,didx,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,plan_ms,up_ms,kms);
        CUDA_CHECK(cudaMemcpy(y.data(),dY0,(size_t)n*4,cudaMemcpyDeviceToHost)); rr[0]=maxrel(yref,y);
    }
    /* didx_w: warp per row */
    {
        auto l=[&](){ spmv_didx_w_kernel<<<wblocks,TPB,(size_t)D*4>>>(n,D,drp,d16,dv,doff,dX,dY1); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("PREPCSV,%s,%d,%d,spmv,didx_w,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,plan_ms,up_ms,tms(l,10,iters));
        CUDA_CHECK(cudaMemcpy(y.data(),dY1,(size_t)n*4,cudaMemcpyDeviceToHost)); rr[1]=maxrel(yref,y);
    }
    /* csr_warp: existing kernel, true-nonzero CSR (4B cols) */
    {
        auto t3 = clk2::now();
        std::vector<int> ci(nnz);
        for (int r = 0; r < n; ++r)
            for (int j = R.rp[r]; j < R.rp[r+1]; ++j) ci[j] = r + H.offsets[gk[j]];
        auto t4 = clk2::now();
        int* dci = dupload(ci);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t5 = clk2::now();
        CsrView A{ n, drp, dci, dv };
        auto l=[&](){ spmv_csr_warp<<<wblocks,TPB>>>(A,dX,dY2); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("PREPCSV,%s,%d,%d,spmv,csr_warp,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,
               plan_ms+msb(t3,t4), up_ms+msb(t4,t5), tms(l,10,iters));
        CUDA_CHECK(cudaMemcpy(y.data(),dY2,(size_t)n*4,cudaMemcpyDeviceToHost)); rr[2]=maxrel(yref,y);
    }
    printf("SQZSTAT,%s,relerr_didx=%.2e,relerr_didx_w=%.2e,relerr_csr_warp=%.2e,%s\n",
           argv[1], rr[0], rr[1], rr[2],
           (rr[0]<1e-5 && rr[1]<1e-5 && rr[2]<1e-5) ? "verify_OK" : "verify_FAIL");
    return (rr[0]<1e-5 && rr[1]<1e-5 && rr[2]<1e-5) ? 0 : 2;
}
