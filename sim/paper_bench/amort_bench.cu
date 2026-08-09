/* ============================================================
 * amort_bench.cu — MEASURED end-to-end amortization curve:
 *   T(N) = preprocessing + upload + N x kernel   (wall clock, one process,
 *   fresh build per point — validates additivity of the table columns)
 * variants: didx (plan+upload+kernel), cusparse (dia_to_csr+setup+SpMV),
 *           diaq fp32 (upload-only prep + fused kernel)
 * out: AMORTCSV,file,variant,N,total_ms
 * usage: amort_bench <dia.txt>            N in {1,10,100,1000,10000}
 * ============================================================ */
#include "../dia_io.hpp"
#include "common.cuh"
#include <cusparse.h>
#include <unordered_map>
#include <chrono>
#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)
using clk2 = std::chrono::steady_clock;
static double msb(clk2::time_point a, clk2::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
#ifndef TPB
#define TPB 256
#endif

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr,"usage: %s dia.txt\n", argv[0]); return 1; }
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, D = (int)H.offsets.size();
    std::vector<float> x((size_t)n, 1.0f);
    float *dX = dupload(x), *dY; CUDA_CHECK(cudaMalloc(&dY,(size_t)n*4));
    const int blocks = (n + TPB - 1)/TPB;
    const long Ns[] = {1, 10, 100, 1000, 10000};
    /* one throwaway kernel to absorb context/clock ramp before ANY timing */
    { DidxPlan W = build_didx_plan(n,H.offsets,H.starts,H.lengths,H.values);
      int* rp=dupload(W.rp); int* of=dupload(std::vector<int>(H.offsets)); float* v=dupload(W.val);
      unsigned char* s8=nullptr; unsigned short* s16=nullptr;
      if (W.wide){ s16=dupload(W.d16); for(int i=0;i<50;++i) spmv_didx_kernel<unsigned short,1><<<blocks,TPB,(size_t)D*4>>>(n,D,rp,s16,v,of,dX,dY); }
      else       { s8 =dupload(W.d8);  for(int i=0;i<50;++i) spmv_didx_kernel<unsigned char ,1><<<blocks,TPB,(size_t)D*4>>>(n,D,rp,s8 ,v,of,dX,dY); }
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaFree(rp);cudaFree(of);cudaFree(v); if(s8)cudaFree(s8); if(s16)cudaFree(s16); }

    if (!getenv("AMORT_SYM_ONLY"))
    for (long N : Ns) {                                   /* ---- didx ---- */
        auto t0 = clk2::now();
        DidxPlan P = build_didx_plan(n, H.offsets, H.starts, H.lengths, H.values);
        int* rp=dupload(P.rp); int* of=dupload(std::vector<int>(H.offsets)); float* v=dupload(P.val);
        unsigned char* s8=nullptr; unsigned short* s16=nullptr;
        if (P.wide) s16=dupload(P.d16); else s8=dupload(P.d8);
        for (long i = 0; i < N; ++i){
            if (P.wide) spmv_didx_kernel<unsigned short,1><<<blocks,TPB,(size_t)D*4>>>(n,D,rp,s16,v,of,dX,dY);
            else        spmv_didx_kernel<unsigned char ,1><<<blocks,TPB,(size_t)D*4>>>(n,D,rp,s8 ,v,of,dX,dY);
        }
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("AMORTCSV,%s,didx,%ld,%.3f\n", argv[1], N, msb(t0, clk2::now()));
        cudaFree(rp);cudaFree(of);cudaFree(v); if(s8)cudaFree(s8); if(s16)cudaFree(s16);
    }

    /* ---- didx_sym (mainline symmetric mode): plan+upload+N*kernel ---- */
    {
        bool symok = true; int Dup = 0;
        /* rounding-tolerant check, same as prep_driver (BH_20_7: 7e-16 diff) */
        { std::unordered_map<int,int> ix; for (int i=0;i<D;++i) ix[H.offsets[i]]=i;
          double smd = 0, svmax = 0;
          for (float v : H.values) svmax = std::max(svmax, (double)std::fabs(v));
          for (int i=0;i<D && symok;++i){ int d=H.offsets[i]; if (d<0) continue;
              if (d>0){ auto it=ix.find(-d); if (it==ix.end()){ symok=false; break; }
                  const float* a=&H.values[H.starts[i]]; const float* b=&H.values[H.starts[it->second]];
                  for (int p=0;p<H.lengths[i];++p) smd=std::max(smd,(double)std::fabs(a[p]-b[p])); }
              ++Dup; }
          if (symok && smd > 1e-6*svmax) symok = false; }
        if (symok && Dup > 128) symok = false;
        if (!symok) fprintf(stderr, "# didx_sym arm skipped (non-symmetric or Dup>128)\n");
        else for (long N : Ns) {
            auto t0 = clk2::now();
            std::vector<int> upOff; std::vector<long long> upSt; std::vector<int> upLen;
            for (int i=0;i<D;++i) if (H.offsets[i]>=0){ upOff.push_back(H.offsets[i]);
                upSt.push_back((long long)H.starts[i]); upLen.push_back(H.lengths[i]); }
            std::vector<int> rp2(n+1,0);
            for (int s2=0;s2<Dup;++s2){ int d=upOff[s2]; size_t st=(size_t)upSt[s2]; int len=upLen[s2];
                for (int p=0;p<len;++p){ if (H.values[st+p]==0.f) continue; ++rp2[p+1]; if (d>0) ++rp2[p+d+1]; } }
            for (int r=0;r<n;++r) rp2[r+1]+=rp2[r];
            std::vector<unsigned char> mt((size_t)rp2[n]);
            { std::vector<int> cur(rp2.begin(), rp2.end()-1);
              for (int s2=0;s2<Dup;++s2){ int d=upOff[s2]; size_t st=(size_t)upSt[s2]; int len=upLen[s2];
                for (int p=0;p<len;++p){ if (H.values[st+p]==0.f) continue;
                    mt[cur[p]++]=(unsigned char)s2; if (d>0) mt[cur[p+d]++]=(unsigned char)(s2|128); } } }
            int* drp2=dupload(rp2); unsigned char* dm=dupload(mt);
            int* dou=dupload(upOff); long long* dsu=dupload(upSt); float* dvf=dupload(H.values);
            for (long i=0;i<N;++i)
                spmv_didx_sym_kernel<<<blocks,TPB,(size_t)Dup*12>>>(n,Dup,drp2,dm,dou,dsu,dvf,dX,dY);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            printf("AMORTCSV,%s,didx_sym,%ld,%.3f\n", argv[1], N, msb(t0, clk2::now()));
            cudaFree(drp2);cudaFree(dm);cudaFree(dou);cudaFree(dsu);cudaFree(dvf);
        }
    }

    if (getenv("AMORT_SYM_ONLY") || getenv("OURS_ONLY")) return 0;   /* baselines below */
    for (long N : Ns) {                                   /* ---- cusparse ---- */
        auto t0 = clk2::now();
        CsrHost csr = dia_to_csr(H);
        int *drp=dupload(csr.row_ptr), *dci=dupload(csr.col_idx); float* dv=dupload(csr.vals);
        cusparseHandle_t h; CUSP_CHECK(cusparseCreate(&h));
        cusparseSpMatDescr_t mH; cusparseDnVecDescr_t vi, vo;
        CUSP_CHECK(cusparseCreateCsr(&mH,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateDnVec(&vi,n,dX,CUDA_R_32F)); CUSP_CHECK(cusparseCreateDnVec(&vo,n,dY,CUDA_R_32F));
        const float a1=1.f,b0=0.f; size_t bsz=0; void* buf=nullptr;
        CUSP_CHECK(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vi,&b0,vo,CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,&bsz));
        if (bsz) CUDA_CHECK(cudaMalloc(&buf,bsz));
        for (long i = 0; i < N; ++i)
            CUSP_CHECK(cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&a1,mH,vi,&b0,vo,CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,buf));
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("AMORTCSV,%s,cusparse,%ld,%.3f\n", argv[1], N, msb(t0, clk2::now()));
        if (buf) cudaFree(buf);
        cusparseDestroySpMat(mH); cusparseDestroyDnVec(vi); cusparseDestroyDnVec(vo); cusparseDestroy(h);
        cudaFree(drp);cudaFree(dci);cudaFree(dv);
    }
    for (long N : Ns) {                                   /* ---- diaq fp32 ---- */
        auto t0 = clk2::now();
        std::vector<unsigned> qo(D), ql(D);
        for (int i = 0; i < D; ++i){ qo[i]=(unsigned)H.starts[i]; ql[i]=(unsigned)H.lengths[i]; }
        int* dqi = dupload(std::vector<int>(H.offsets));
        unsigned *dqo = dupload(qo), *dql = dupload(ql);
        float* dqa = dupload(H.values);
        float* dqz; CUDA_CHECK(cudaMalloc(&dqz,(size_t)H.nnz*4)); CUDA_CHECK(cudaMemset(dqz,0,(size_t)H.nnz*4));
        float* dXi; CUDA_CHECK(cudaMalloc(&dXi,(size_t)n*4)); CUDA_CHECK(cudaMemset(dXi,0,(size_t)n*4));
        float* dYi; CUDA_CHECK(cudaMalloc(&dYi,(size_t)n*4));
        for (long i = 0; i < N; ++i)
            diaq_spmv_row_kernel<float><<<blocks,TPB>>>((unsigned)n,(unsigned)n,(unsigned)D,dqi,dqo,dql,dqa,dqz,dX,dXi,dY,dYi);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        printf("AMORTCSV,%s,diaq_fp32,%ld,%.3f\n", argv[1], N, msb(t0, clk2::now()));
        cudaFree(dqi);cudaFree(dqo);cudaFree(dql);cudaFree(dqa);cudaFree(dqz);cudaFree(dXi);cudaFree(dYi);
    }
    return 0;
}
