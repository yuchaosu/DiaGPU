/* ============================================================
 * bench.cu — diagonal-direct gather SpMV vs existing kernels.
 *
 * Compares, kernel-only, device-resident, correctness vs fp64 CPU:
 *   dense-DIA row kernel  cuda_spmv_dia   (padded recon, NV=1)
 *   fused row SpMM<2>     cuda_spmm_dia<2>
 *   gather dense plan     spmv_gather (1 seg/diagonal), NV=1 and NV=2
 *   gather zskip plan     spmv_gather (segments at zero runs), NV=1,2
 *   CSR scalar zero-skip  spmv_csr_scalar (reference zero-skip)
 * Prints the plan statistics (fill, #segments, avg run) that decide
 * whether segment splitting pays on this operator.
 *
 * usage: bench <dia_file> [iters=200]
 * ============================================================ */
#include "../dia_io.hpp"
#include "../../spmv/src/cuda_dia_kernels.cuh"
#include "../../spmv/src/spmv_zeroskip_kernels.cuh"
#include "../../spmv/src/spmv_gather.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

/* Per-nonzero diagonal-index plan: iterate only true nonzeros; the column is
 * COMPUTED as r + offset[didx] (offset table in smem), never stored. didx is
 * 1 B for D<=256 diagonals, 2 B otherwise. NV vectors share each matrix read. */
template <typename IT, int NV>
__global__ void spmv_didx_kernel(int n, int D,
    const int* __restrict__ rp, const IT* __restrict__ didx,
    const float* __restrict__ val, const int* __restrict__ offs,
    const float* __restrict__ X, float* __restrict__ Y)
{
    extern __shared__ int soff[];
    for (int k = threadIdx.x; k < D; k += blockDim.x) soff[k] = offs[k];
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float acc[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) acc[v] = 0.f;
    const int e = rp[r + 1];
    for (int j = rp[r]; j < e; ++j) {
        const int c = r + soff[didx[j]];
        const float a = val[j];
#pragma unroll
        for (int v = 0; v < NV; ++v) acc[v] += a * X[(size_t)v * n + c];
    }
#pragma unroll
    for (int v = 0; v < NV; ++v) Y[(size_t)v * n + r] = acc[v];
}

template<class F> static float tms(F f, int wu, int it){
    for (int i = 0; i < wu; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < it; ++i) f();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / it;
}

template<typename T> static T* dupload(const std::vector<T>& h){
    T* d; CUDA_CHECK(cudaMalloc(&d, h.size()*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static double rel_l2(const std::vector<double>& ref, const float* got, int n){
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i){ double d = ref[i]-(double)got[i]; num += d*d; den += ref[i]*ref[i]; }
    return den > 0 ? std::sqrt(num/den) : std::sqrt(num);
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s <dia_file> [iters=200]\n", argv[0]); return 1; }
    int iters = argc > 2 ? atoi(argv[2]) : 200;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();

    /* true-nonzero stats */
    size_t nnz_true = 0;
    for (float v : H.values) if (v != 0.f) ++nnz_true;
    printf("=== %s  n=%d  diags=%d  stored=%zu  nnz=%zu  fill=%.1f%% ===\n",
           argv[1], n, nd, H.nnz, nnz_true, 100.0*nnz_true/H.nnz);

    /* x (two components), fp64 reference */
    std::vector<float> X(2*(size_t)n);
    srand(42);
    for (size_t i = 0; i < X.size(); ++i) X[i] = (float)rand()/RAND_MAX - 0.5f;
    std::vector<double> yr(n, 0.0), yi(n, 0.0);
    for (int k = 0; k < nd; ++k){
        int off = H.offsets[k]; size_t s = H.starts[k]; int len = H.lengths[k];
        for (int p = 0; p < len; ++p){
            int row = off >= 0 ? p : p - off, col = off >= 0 ? p + off : p;
            double v = (double)H.values[s + p];
            yr[row] += v * (double)X[col];
            yi[row] += v * (double)X[n + col];
        }
    }

    float *dX = dupload(X), *dY;
    CUDA_CHECK(cudaMalloc(&dY, 2*(size_t)n*4));
    std::vector<float> out(2*(size_t)n);
    const int TPB = 256;

    /* ---------------- dense-DIA row kernel (padded recon) ---------------- */
    {
        std::vector<float> pad((size_t)nd * n, 0.f);
        for (int k = 0; k < nd; ++k){
            int off = H.offsets[k]; size_t s = H.starts[k]; int len = H.lengths[k];
            int c0 = off >= 0 ? off : 0;
            for (int p = 0; p < len; ++p) pad[(size_t)k * n + c0 + p] = H.values[s + p];
        }
        float* dV = dupload(pad);
        int* dOff = dupload(std::vector<int>(H.offsets));
        ReconView R{ n, n, nd, dOff, dV };
        int blocks = (n + TPB - 1)/TPB;
        auto l1 = [&](){ cuda_spmv_dia<<<blocks,TPB>>>(R, dX, n, dY); };
        auto l2 = [&](){ cuda_spmm_dia<2><<<blocks,TPB>>>(R, dX, n, dY); };
        l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dY, (size_t)n*4, cudaMemcpyDeviceToHost));
        printf("  row dense NV1    : %9.5f ms   relerr %.2e\n", tms(l1,10,iters), rel_l2(yr,out.data(),n));
        l2(); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dY, 2*(size_t)n*4, cudaMemcpyDeviceToHost));
        printf("  row dense NV2    : %9.5f ms   relerr %.2e\n", tms(l2,10,iters),
               std::max(rel_l2(yr,out.data(),n), rel_l2(yi,out.data()+n,n)));
        cudaFree(dV); cudaFree(dOff);
    }

    /* ---------------- gather kernel, dense and zskip plans ---------------- */
    float* dAv = dupload(H.values);
    auto run_gather = [&](const std::vector<DSeg>& plan, const char* tag){
        DSeg* dS = dupload(plan);
        int nseg = (int)plan.size();
        const int ILP = n >= 65536 ? 4 : 1;
        int blocks1 = (n + TPB*1 - 1)/(TPB*1), blocks4 = (n + TPB*4 - 1)/(TPB*4);
        auto l1 = [&](){
            if (ILP==4) spmv_gather_kernel<128,4,1><<<blocks4,TPB>>>(dAv,dS,nseg,dX,n,dY);
            else        spmv_gather_kernel<128,1,1><<<blocks1,TPB>>>(dAv,dS,nseg,dX,n,dY);
        };
        auto l2 = [&](){
            if (ILP==4) spmv_gather_kernel<128,4,2><<<blocks4,TPB>>>(dAv,dS,nseg,dX,n,dY);
            else        spmv_gather_kernel<128,1,2><<<blocks1,TPB>>>(dAv,dS,nseg,dX,n,dY);
        };
        l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dY, (size_t)n*4, cudaMemcpyDeviceToHost));
        printf("  %s NV1 : %9.5f ms   relerr %.2e   (%d segs, ILP=%d)\n",
               tag, tms(l1,10,iters), rel_l2(yr,out.data(),n), nseg, ILP);
        l2(); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dY, 2*(size_t)n*4, cudaMemcpyDeviceToHost));
        printf("  %s NV2 : %9.5f ms   relerr %.2e\n",
               tag, tms(l2,10,iters), std::max(rel_l2(yr,out.data(),n), rel_l2(yi,out.data()+n,n)));
        cudaFree(dS);
    };
    run_gather(build_dense_plan(H.offsets, H.starts, H.lengths), "gather dense");
    {
        auto plan = build_zskip_plan(H.offsets, H.starts, H.lengths, H.values);
        double touched = 0; for (const DSeg& s : plan) touched += s.len;
        printf("  [zskip plan] %zu segs, touches %.1f%% of stored band\n",
               plan.size(), 100.0*touched/H.nnz);
        run_gather(plan, "gather zskip");
    }

    /* ---------------- pattern plan: arithmetic zero-skip ---------------- */
    {
        std::vector<float> packed; size_t touched = 0;
        auto plan = build_pattern_plan(H.offsets, H.starts, H.lengths, H.values, packed, &touched);
        int nfall = 0; for (auto& d : plan) if (d.Pm == 0 && d.R == 1 && d.c == 0) ++nfall;
        printf("  [pattern plan] %zu diags (%d dense-fallback), packed %.1f%% of stored (true %.1f%%)\n",
               plan.size(), nfall, 100.0*packed.size()/H.nnz, 100.0*nnz_true/H.nnz);
        float* dP = dupload(packed);
        DPat* dS = dupload(plan);
        int npat = (int)plan.size();
        const int ILP = n >= 65536 ? 4 : 1;
        int blocks1 = (n + TPB*1 - 1)/(TPB*1), blocks4 = (n + TPB*4 - 1)/(TPB*4);
        auto l1 = [&](){
            if (ILP==4) spmv_gather_pat_kernel<128,4,1><<<blocks4,TPB>>>(dP,dS,npat,dX,n,dY);
            else        spmv_gather_pat_kernel<128,1,1><<<blocks1,TPB>>>(dP,dS,npat,dX,n,dY);
        };
        auto l2 = [&](){
            if (ILP==4) spmv_gather_pat_kernel<128,4,2><<<blocks4,TPB>>>(dP,dS,npat,dX,n,dY);
            else        spmv_gather_pat_kernel<128,1,2><<<blocks1,TPB>>>(dP,dS,npat,dX,n,dY);
        };
        l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dY, (size_t)n*4, cudaMemcpyDeviceToHost));
        printf("  gather pat  NV1  : %9.5f ms   relerr %.2e\n", tms(l1,10,iters), rel_l2(yr,out.data(),n));
        l2(); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dY, 2*(size_t)n*4, cudaMemcpyDeviceToHost));
        printf("  gather pat  NV2  : %9.5f ms   relerr %.2e\n", tms(l2,10,iters),
               std::max(rel_l2(yr,out.data(),n), rel_l2(yi,out.data()+n,n)));
        cudaFree(dP); cudaFree(dS);
    }

    /* ---------------- per-nonzero diagonal-index plan (didx) ---------------- */
    {
        std::vector<int> rp(n+1,0);
        std::vector<unsigned short> dx16; std::vector<unsigned char> dx8;
        std::vector<float> dv;
        std::vector<std::vector<std::pair<int,float>>> rows(n);
        for (int k = 0; k < nd; ++k){
            int off = H.offsets[k]; size_t s = H.starts[k]; int len = H.lengths[k];
            for (int p = 0; p < len; ++p){
                float v = H.values[s + p]; if (v == 0.f) continue;
                int row = off >= 0 ? p : p - off;
                rows[row].push_back({k, v});
            }
        }
        for (int r = 0; r < n; ++r){ rp[r+1] = rp[r] + (int)rows[r].size();
            for (auto& e : rows[r]){
                if (nd <= 256) dx8.push_back((unsigned char)e.first);
                else           dx16.push_back((unsigned short)e.first);
                dv.push_back(e.second); } }
        int *drp = dupload(rp), *doff = dupload(std::vector<int>(H.offsets));
        float *ddv = dupload(dv);
        int blocks = (n + TPB - 1)/TPB;
        size_t smem = (size_t)nd * 4;
        auto run_didx = [&](auto* ddx){
            auto l1 = [&](){ spmv_didx_kernel<std::remove_pointer_t<decltype(ddx)>,1><<<blocks,TPB,smem>>>(n,nd,drp,ddx,ddv,doff,dX,dY); };
            auto l2 = [&](){ spmv_didx_kernel<std::remove_pointer_t<decltype(ddx)>,2><<<blocks,TPB,smem>>>(n,nd,drp,ddx,ddv,doff,dX,dY); };
            l1(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(out.data(), dY, (size_t)n*4, cudaMemcpyDeviceToHost));
            printf("  didx (%zuB) NV1  : %9.5f ms   relerr %.2e\n", sizeof(*ddx), tms(l1,10,iters), rel_l2(yr,out.data(),n));
            l2(); CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(out.data(), dY, 2*(size_t)n*4, cudaMemcpyDeviceToHost));
            printf("  didx (%zuB) NV2  : %9.5f ms   relerr %.2e\n", sizeof(*ddx), tms(l2,10,iters),
                   std::max(rel_l2(yr,out.data(),n), rel_l2(yi,out.data()+n,n)));
        };
        if (nd <= 256){ unsigned char*  d8  = dupload(dx8);  run_didx(d8);  cudaFree(d8); }
        else          { unsigned short* d16 = dupload(dx16); run_didx(d16); cudaFree(d16); }
        cudaFree(drp); cudaFree(doff); cudaFree(ddv);
    }

    /* ---------------- CSR scalar zero-skip (reference) ---------------- */
    {
        std::vector<int> rp(n+1,0), ci; std::vector<float> cv;
        std::vector<std::vector<std::pair<int,float>>> rows(n);
        for (int k = 0; k < nd; ++k){
            int off = H.offsets[k]; size_t s = H.starts[k]; int len = H.lengths[k];
            for (int p = 0; p < len; ++p){
                float v = H.values[s + p]; if (v == 0.f) continue;
                int row = off >= 0 ? p : p - off, col = off >= 0 ? p + off : p;
                rows[row].push_back({col, v});
            }
        }
        for (int r = 0; r < n; ++r){ rp[r+1] = rp[r] + (int)rows[r].size();
            for (auto& e : rows[r]){ ci.push_back(e.first); cv.push_back(e.second); } }
        int *drp = dupload(rp), *dci = dupload(ci); float *dcv = dupload(cv);
        CsrView A{ n, drp, dci, dcv };
        int blocks = (n + TPB - 1)/TPB;
        auto l = [&](){ spmv_csr_scalar<<<blocks,TPB>>>(A, dX, dY); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dY, (size_t)n*4, cudaMemcpyDeviceToHost));
        printf("  csr zskip NV1    : %9.5f ms   relerr %.2e\n", tms(l,10,iters), rel_l2(yr,out.data(),n));
        cudaFree(drp); cudaFree(dci); cudaFree(dcv);
    }
    return 0;
}
