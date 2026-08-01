/* ============================================================
 * didx_tiled_bench.cu — didx_t: per-row-tile LOCAL diagonal-offset tables.
 *
 * didx keeps one global offs[D] table in shared memory (D*4 bytes/block) and
 * needs 2-byte slots once D>256.  didx_t re-encodes slots per tile of TPB
 * rows: each block loads only the offsets its rows actually touch (interior
 * sparsity makes this set << D on wide-band molecules), so slots return to
 * 1 byte whenever every tile touches <=256 distinct diagonals and the shared
 * table shrinks from D*4 (13.4KB at O2_20's D=3359) to maxLocalD*4.
 *
 * Elements keep the exact didx order (only the slot ENCODING changes), so
 * didx and didx_t sum in the same order -> outputs must match bitwise; the
 * bench verifies that and emits both rows for A/B.
 *
 * out:  TILESTAT,file,D,maxLocalD,slotBytes_didx,slotBytes_t,smemB_didx,smemB_t
 *       PREPCSV,file,n,D,spmv,didx,plan_ms,upload_ms,kernel_ms,-1   (fresh ref)
 *       PREPCSV,file,n,D,spmv,didx_t,plan_ms,upload_ms,kernel_ms,-1
 * usage: didx_tiled_bench <dia.txt> [iters=50]
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

template <typename IT>
__global__ void spmv_didx_tiled_kernel(int n,
    const int* __restrict__ rp, const IT* __restrict__ slot,
    const float* __restrict__ val,
    const int* __restrict__ tilePtr, const int* __restrict__ tileOff,
    const float* __restrict__ X, float* __restrict__ Y)
{
    extern __shared__ int soff[];
    const int t0 = tilePtr[blockIdx.x], tn = tilePtr[blockIdx.x + 1] - t0;
    for (int k = threadIdx.x; k < tn; k += blockDim.x) soff[k] = tileOff[t0 + k];
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float acc = 0.f;
    const int e = rp[r + 1];
    for (int j = rp[r]; j < e; ++j)
        acc += val[j] * X[r + soff[slot[j]]];
    Y[r] = acc;
}

struct TiledPlan {
    std::vector<int> rp, tilePtr, tileOff;
    std::vector<unsigned char>  s8;
    std::vector<unsigned short> s16;
    std::vector<float> val;
    bool wide = false;      /* some tile touches >256 distinct diagonals */
    int  maxLocalD = 0;
};

static TiledPlan build_tiled(int n, const std::vector<int>& offsets,
    const std::vector<size_t>& starts, const std::vector<int>& lengths,
    const std::vector<float>& values)
{
    const int D = (int)offsets.size();
    TiledPlan P;
    P.rp.assign(n + 1, 0);
    for (int k = 0; k < D; ++k) {
        int off = offsets[k]; size_t s = starts[k]; int len = lengths[k];
        for (int p = 0; p < len; ++p)
            if (values[s + p] != 0.f) ++P.rp[(off >= 0 ? p : p - off) + 1];
    }
    for (int r = 0; r < n; ++r) P.rp[r + 1] += P.rp[r];
    const size_t nnz = P.rp[n];
    P.val.resize(nnz);
    std::vector<unsigned short> gk(nnz);          /* global diagonal id */
    std::vector<int> cur(P.rp.begin(), P.rp.end() - 1);
    for (int k = 0; k < D; ++k) {
        int off = offsets[k]; size_t s = starts[k]; int len = lengths[k];
        for (int p = 0; p < len; ++p) {
            float v = values[s + p]; if (v == 0.f) continue;
            int r = off >= 0 ? p : p - off;
            int j = cur[r]++;
            P.val[j] = v; gk[j] = (unsigned short)k;
        }
    }
    /* per-tile re-encode: first-seen order, local table holds offset VALUES */
    const int ntiles = (n + TPB - 1) / TPB;
    P.tilePtr.assign(ntiles + 1, 0);
    std::vector<unsigned short> local(nnz);
    std::vector<int> stamp(D, -1), lid(D, 0);
    for (int t = 0; t < ntiles; ++t) {
        const int rlo = t * TPB, rhi = std::min(n, rlo + TPB);
        int nloc = 0;
        for (int j = P.rp[rlo]; j < P.rp[rhi]; ++j) {
            int k = gk[j];
            if (stamp[k] != t) { stamp[k] = t; lid[k] = nloc++; P.tileOff.push_back(offsets[k]); }
            local[j] = (unsigned short)lid[k];
        }
        P.tilePtr[t + 1] = (int)P.tileOff.size();
        P.maxLocalD = std::max(P.maxLocalD, nloc);
    }
    P.wide = P.maxLocalD > 256;
    if (P.wide) P.s16.assign(local.begin(), local.end());
    else        P.s8.assign(local.begin(),  local.end());
    return P;
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s dia.txt [iters]\n", argv[0]); return 1; }
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, D = (int)H.offsets.size();

    /* deterministic non-constant x: constant x would mask wrong-column bugs */
    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; ++i)
        x[i] = (float)(((i * 1103515245u + 12345u) >> 16 & 0x3ff)) / 1024.f - 0.5f;
    float *dX = dupload(x), *dY, *dYt;
    CUDA_CHECK(cudaMalloc(&dY,  (size_t)n * 4));
    CUDA_CHECK(cudaMalloc(&dYt, (size_t)n * 4));
    const int blocks = (n + TPB - 1) / TPB;

    /* ---- reference didx (fresh, same methodology) ---- */
    auto t0 = clk2::now();
    DidxPlan R = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
    auto t1 = clk2::now();
    int* drp = dupload(R.rp); int* doff = dupload(std::vector<int>(H.offsets));
    float* dv = dupload(R.val);
    unsigned char* r8 = nullptr; unsigned short* r16 = nullptr;
    if (R.wide) r16 = dupload(R.d16); else r8 = dupload(R.d8);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = clk2::now();
    const size_t smem_g = (size_t)D * 4;
    float kms;
    if (R.wide){ auto l=[&](){ spmv_didx_kernel<unsigned short,1><<<blocks,TPB,smem_g>>>(n,D,drp,r16,dv,doff,dX,dY); };
                 l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kms = tms(l,10,iters); }
    else       { auto l=[&](){ spmv_didx_kernel<unsigned char ,1><<<blocks,TPB,smem_g>>>(n,D,drp,r8 ,dv,doff,dX,dY); };
                 l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kms = tms(l,10,iters); }
    printf("PREPCSV,%s,%d,%d,spmv,didx,%.3f,%.3f,%.6f,-1\n", argv[1], n, D,
           msb(t0,t1), msb(t1,t2), kms);

    /* ---- didx_t ---- */
    auto t3 = clk2::now();
    TiledPlan T = build_tiled(n, H.offsets, H.starts, H.lengths, H.values);
    auto t4 = clk2::now();
    int* trp = dupload(T.rp); int* dtp = dupload(T.tilePtr); int* dto = dupload(T.tileOff);
    float* tv = dupload(T.val);
    unsigned char* s8 = nullptr; unsigned short* s16 = nullptr;
    if (T.wide) s16 = dupload(T.s16); else s8 = dupload(T.s8);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t5 = clk2::now();
    const size_t smem_t = (size_t)T.maxLocalD * 4;
    float kmt;
    if (T.wide){ auto l=[&](){ spmv_didx_tiled_kernel<unsigned short><<<blocks,TPB,smem_t>>>(n,trp,s16,tv,dtp,dto,dX,dYt); };
                 l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kmt = tms(l,10,iters); }
    else       { auto l=[&](){ spmv_didx_tiled_kernel<unsigned char ><<<blocks,TPB,smem_t>>>(n,trp,s8 ,tv,dtp,dto,dX,dYt); };
                 l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kmt = tms(l,10,iters); }
    printf("PREPCSV,%s,%d,%d,spmv,didx_t,%.3f,%.3f,%.6f,-1\n", argv[1], n, D,
           msb(t3,t4), msb(t4,t5), kmt);

    /* ---- verify: same summation order -> must match bitwise ---- */
    std::vector<float> y((size_t)n), yt((size_t)n);
    CUDA_CHECK(cudaMemcpy(y.data(),  dY,  (size_t)n*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yt.data(), dYt, (size_t)n*4, cudaMemcpyDeviceToHost));
    int bad = 0; for (int i = 0; i < n; ++i) if (y[i] != yt[i]) ++bad;
    printf("TILESTAT,%s,%d,%d,%d,%d,%zu,%zu,verify_%s\n", argv[1], D, T.maxLocalD,
           R.wide?2:1, T.wide?2:1, smem_g, smem_t, bad ? "FAIL" : "BITEXACT");
    if (bad){ fprintf(stderr, "VERIFY FAIL: %d rows differ\n", bad); return 2; }
    return 0;
}
