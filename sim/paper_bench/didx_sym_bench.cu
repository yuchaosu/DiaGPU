/* ============================================================
 * didx_sym_bench.cu — didx_sym: SYMMETRIC zero-skip diagonal SpMV.
 *
 * H symmetric => diagonal -d equals diagonal +d elementwise, so only d>=0
 * value arrays are kept and each row reads the shared array at TWO
 * positions: +d term v_d[r]*x[r+d], mirror term v_d[r-d]*x[r-d].  Values
 * are NOT copied row-major (that would re-double the bytes): the kernel
 * addresses the compact upper diagonal array directly; per-element
 * metadata is 1 byte = 7-bit upper-diagonal slot + 1-bit side.  Register
 * accumulation only — no scatter, no atomics.
 *
 * Byte account per true nonzero (upper stored once, read by rows p and
 * p+d): 4B val shared by 2 rows (cache-reachable for |d| within L2 span,
 * measured 90-100% of nnz for HamLib q16-20) + 1B meta  ->  ~3B vs didx's
 * 5B.  Registered PREDICTION: 1.5-1.8x over didx in the win regime.
 *
 * out: PREPCSV rows (didx ref, didx_sym) + SYMSTAT verify (norm-relative
 * vs CPU double; summation order differs from didx by design)
 * usage: didx_sym_bench <dia.txt> [iters=50]     (numerically symmetric H!)
 * ============================================================ */
#include "../dia_io.hpp"
#include "common.cuh"
#include <chrono>
#include <cstdint>
#include <cmath>
#include <unordered_map>

using clk2 = std::chrono::steady_clock;
static double msb(clk2::time_point a, clk2::time_point b){
    return std::chrono::duration<double,std::milli>(b-a).count();
}
#ifndef TPB
#define TPB 256
#endif

/* kernel now lives in common.cuh (mainline) */

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

    /* symmetry precheck: H[-d] must equal H[+d] elementwise */
    { std::unordered_map<int,int> ix; for (int i=0;i<D;++i) ix[H.offsets[i]]=i;
      double md=0, vmax=0;
      for (float v : H.values) vmax = std::max(vmax, (double)std::fabs(v));
      for (int i=0;i<D;++i){ int d=H.offsets[i]; if (d<=0) continue;
          auto it=ix.find(-d); if (it==ix.end()){ fprintf(stderr,"NOT SYMMETRIC (missing -%d)\n",d); return 3; }
          const float* a=&H.values[H.starts[i]]; const float* b=&H.values[H.starts[it->second]];
          for (int p=0;p<H.lengths[i];++p) md=std::max(md,(double)std::fabs(a[p]-b[p])); }
      /* rounding-tolerant: <=1e-6 relative is below the fp32 kernel's eps */
      if (md > 1e-6*vmax){ fprintf(stderr,"NOT NUMERICALLY SYMMETRIC (maxdiff %.3e)\n",md); return 3; } }

    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; ++i)
        x[i] = (float)(((i * 1103515245u + 12345u) >> 16 & 0x3ff)) / 1024.f - 0.5f;
    float* dX = dupload(x);
    float *dY0, *dY1; CUDA_CHECK(cudaMalloc(&dY0,(size_t)n*4)); CUDA_CHECK(cudaMalloc(&dY1,(size_t)n*4));
    const int blocks = (n + TPB - 1) / TPB;
    std::vector<float> y((size_t)n);
    double r0, r1;

    /* ---- reference didx (full) ---- */
    auto t0 = clk2::now();
    DidxPlan R = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
    auto t1 = clk2::now();
    std::vector<unsigned short> gk((size_t)R.rp[n]);
    if (R.wide) gk = R.d16; else gk.assign(R.d8.begin(), R.d8.end());
    std::vector<double> yref((size_t)n, 0.0);
    for (int r = 0; r < n; ++r)
        for (int j = R.rp[r]; j < R.rp[r+1]; ++j)
            yref[r] += (double)R.val[j] * (double)x[r + H.offsets[gk[j]]];
    {
        int* drp = dupload(R.rp); int* doff = dupload(std::vector<int>(H.offsets)); float* dv = dupload(R.val);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = clk2::now(); float kms;
        if (R.wide){ unsigned short* dsl=dupload(R.d16);
            auto l=[&](){ spmv_didx_kernel<unsigned short,1><<<blocks,TPB,(size_t)D*4>>>(n,D,drp,dsl,dv,doff,dX,dY0); };
            l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kms=tms(l,10,iters); }
        else { unsigned char* dsl=dupload(R.d8);
            auto l=[&](){ spmv_didx_kernel<unsigned char,1><<<blocks,TPB,(size_t)D*4>>>(n,D,drp,dsl,dv,doff,dX,dY0); };
            l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); kms=tms(l,10,iters); }
        printf("PREPCSV,%s,%d,%d,spmv,didx,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,msb(t0,t1),msb(t1,t2),kms);
        CUDA_CHECK(cudaMemcpy(y.data(),dY0,(size_t)n*4,cudaMemcpyDeviceToHost)); r0=maxrel(yref,y);
    }

    /* ---- didx_sym plan: upper diagonals only, 1B meta = slot|side ---- */
    auto t3 = clk2::now();
    std::vector<int> upOff; std::vector<long long> upSt; std::vector<int> upLen; std::vector<int> upIdx;
    for (int i = 0; i < D; ++i) if (H.offsets[i] >= 0){
        upIdx.push_back(i); upOff.push_back(H.offsets[i]);
        upSt.push_back((long long)H.starts[i]); upLen.push_back(H.lengths[i]); }
    const int Dup = (int)upOff.size();
    if (Dup > 128){ fprintf(stderr,"Dup=%d > 128 (7-bit slot)\n",Dup); return 4; }
    std::vector<int> rp2(n + 1, 0);
    for (int s = 0; s < Dup; ++s){
        const int d = upOff[s]; const size_t st = H.starts[upIdx[s]]; const int len = upLen[s];
        for (int p = 0; p < len; ++p){
            if (H.values[st + p] == 0.f) continue;
            ++rp2[p + 1];                          /* +d term: row p           */
            if (d > 0) ++rp2[p + d + 1];           /* mirror : row p+d         */
        }
    }
    for (int r = 0; r < n; ++r) rp2[r+1] += rp2[r];
    std::vector<unsigned char> meta((size_t)rp2[n]);
    { std::vector<int> cur(rp2.begin(), rp2.end() - 1);
      for (int s = 0; s < Dup; ++s){
        const int d = upOff[s]; const size_t st = H.starts[upIdx[s]]; const int len = upLen[s];
        for (int p = 0; p < len; ++p){
            if (H.values[st + p] == 0.f) continue;
            meta[cur[p]++]         = (unsigned char)s;
            if (d > 0) meta[cur[p+d]++] = (unsigned char)(s | 128);
        }
      }
    }
    auto t4 = clk2::now();
    int* drp2 = dupload(rp2); unsigned char* dm = dupload(meta);
    int* dou = dupload(upOff); long long* dsu = dupload(upSt);
    float* dvfull = dupload(H.values);             /* full array; kernel reads d>=0 region */
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t5 = clk2::now();
    {
        const size_t smem = (size_t)Dup * 12;
        auto l=[&](){ spmv_didx_sym_kernel<<<blocks,TPB,smem>>>(n,Dup,drp2,dm,dou,dsu,dvfull,dX,dY1); };
        l(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("PREPCSV,%s,%d,%d,spmv,didx_sym,%.3f,%.3f,%.6f,-1\n", argv[1],n,D,
               msb(t3,t4), msb(t4,t5), tms(l,10,iters));
        CUDA_CHECK(cudaMemcpy(y.data(),dY1,(size_t)n*4,cudaMemcpyDeviceToHost)); r1=maxrel(yref,y);
    }
    printf("SYMSTAT,%s,Dup=%d,relerr_didx=%.2e,relerr_sym=%.2e,%s\n", argv[1], Dup, r0, r1,
           (r0<1e-5 && r1<1e-5) ? "verify_OK" : "verify_FAIL");
    return (r0<1e-5 && r1<1e-5) ? 0 : 2;
}
