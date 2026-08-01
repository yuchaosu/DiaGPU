/* ============================================================
 * didx_dtile_bench.cu — didx_g: DIAGONAL-dimension tiling (user proposal).
 *
 * Row-tiling (didx_t) failed to shrink slots at O2_20: 256 rows still touch
 * 839 distinct diagonals.  Here the D diagonals are cut into groups of 256,
 * so the per-element slot is (k & 255) = 1 byte BY CONSTRUCTION for any D,
 * and the "table" for group g is just offsets[g*256 .. ), loaded slice by
 * slice into one 1KB shared buffer inside a group loop.  Per-row per-group
 * element ranges come from a group-major pointer array
 *     rpg[g*n + r] = first j >= rp[r] whose diagonal id is in group >= g,
 * rpg[G*n + r] = rp[r+1]  ((G+1)*n ints; ~60MB at O2_20 — fine).
 * Element order is untouched (ascending diagonal id), so output must match
 * didx bitwise.
 *
 * Registered PREDICTION (before measurement): O2_20 is ~6x off the bandwidth
 * bound -> latency-limited, so the 1B slot saving should be worth only a few
 * percent there; real target is the O2_16 class + design completeness.
 *
 * out:  PREPCSV rows (didx ref, didx_g) + DTILESTAT,file,D,G,verify
 * usage: didx_dtile_bench <dia.txt> [iters=50]
 * ============================================================ */
#include "../dia_io.hpp"
#include "common.cuh"
#include <chrono>
#include <cstdint>

using clk2 = std::chrono::steady_clock;
static double msb(clk2::time_point a, clk2::time_point b){
    return std::chrono::duration<double,std::milli>(b-a).count();
}
#ifndef TPB
#define TPB 256
#endif
#define GSZ 256   /* diagonals per group; slot = k & (GSZ-1) fits 1 byte */

__global__ void spmv_didx_g_kernel(int n, int D, int G,
    const unsigned char* __restrict__ slot, const float* __restrict__ val,
    const int* __restrict__ rpg, const int* __restrict__ offs,
    const float* __restrict__ X, float* __restrict__ Y)
{
    __shared__ int soff[GSZ];
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    for (int g = 0; g < G; ++g) {
        __syncthreads();
        const int base = g * GSZ, cnt = min(GSZ, D - base);
        for (int k = threadIdx.x; k < cnt; k += blockDim.x) soff[k] = offs[base + k];
        __syncthreads();
        if (r < n) {
            const int e = rpg[(size_t)(g + 1) * n + r];
            for (int j = rpg[(size_t)g * n + r]; j < e; ++j)
                acc += val[j] * X[r + soff[slot[j]]];
        }
    }
    if (r < n) Y[r] = acc;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s dia.txt [iters]\n", argv[0]); return 1; }
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, D = (int)H.offsets.size(), G = (D + GSZ - 1) / GSZ;

    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; ++i)
        x[i] = (float)(((i * 1103515245u + 12345u) >> 16 & 0x3ff)) / 1024.f - 0.5f;
    float *dX = dupload(x), *dY, *dYg;
    CUDA_CHECK(cudaMalloc(&dY,  (size_t)n * 4));
    CUDA_CHECK(cudaMalloc(&dYg, (size_t)n * 4));
    const int blocks = (n + TPB - 1) / TPB;

    /* ---- reference didx ---- */
    auto t0 = clk2::now();
    DidxPlan R = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
    auto t1 = clk2::now();
    int* drp = dupload(R.rp); int* doff = dupload(std::vector<int>(H.offsets));
    float* dv = dupload(R.val);
    unsigned char* r8 = nullptr; unsigned short* r16 = nullptr;
    if (R.wide) r16 = dupload(R.d16); else r8 = dupload(R.d8);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = clk2::now();
    float kms;
    if (R.wide){ auto l=[&](){ spmv_didx_kernel<unsigned short,1><<<blocks,TPB,(size_t)D*4>>>(n,D,drp,r16,dv,doff,dX,dY); };
                 l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kms = tms(l,10,iters); }
    else       { auto l=[&](){ spmv_didx_kernel<unsigned char ,1><<<blocks,TPB,(size_t)D*4>>>(n,D,drp,r8 ,dv,doff,dX,dY); };
                 l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kms = tms(l,10,iters); }
    printf("PREPCSV,%s,%d,%d,spmv,didx,%.3f,%.3f,%.6f,-1\n", argv[1], n, D,
           msb(t0,t1), msb(t1,t2), kms);

    /* ---- didx_g: 1B slots + group-major range pointers ---- */
    auto t3 = clk2::now();
    const size_t nnz = (size_t)R.rp[n];
    std::vector<unsigned char> s1(nnz);
    std::vector<unsigned short> gk(nnz);
    if (R.wide) gk = R.d16; else gk.assign(R.d8.begin(), R.d8.end());
    for (size_t j = 0; j < nnz; ++j) s1[j] = (unsigned char)(gk[j] & (GSZ - 1));
    std::vector<int> rpg((size_t)(G + 1) * n);
    for (int r = 0; r < n; ++r) {
        int j = R.rp[r]; const int e = R.rp[r + 1];
        for (int g = 0; g <= G; ++g) {
            while (j < e && (gk[j] >> 8) < g) ++j;   /* GSZ==256 */
            rpg[(size_t)g * n + r] = j;
        }
        rpg[(size_t)G * n + r] = e;
    }
    auto t4 = clk2::now();
    unsigned char* ds1 = dupload(s1); int* drpg = dupload(rpg);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t5 = clk2::now();
    auto lg = [&](){ spmv_didx_g_kernel<<<blocks,TPB>>>(n, D, G, ds1, dv, drpg, doff, dX, dYg); };
    lg(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    printf("PREPCSV,%s,%d,%d,spmv,didx_g,%.3f,%.3f,%.6f,-1\n", argv[1], n, D,
           msb(t3,t4), msb(t4,t5), tms(lg,10,iters));

    std::vector<float> y((size_t)n), yg((size_t)n);
    CUDA_CHECK(cudaMemcpy(y.data(),  dY,  (size_t)n*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yg.data(), dYg, (size_t)n*4, cudaMemcpyDeviceToHost));
    int bad = 0; for (int i = 0; i < n; ++i) if (y[i] != yg[i]) ++bad;
    printf("DTILESTAT,%s,%d,%d,verify_%s\n", argv[1], D, G, bad ? "FAIL" : "BITEXACT");
    return bad ? 2 : 0;
}
