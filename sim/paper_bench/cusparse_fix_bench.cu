/* ============================================================
 * cusparse_fix_bench.cu — honest cuSPARSE CSR SpMV baseline.
 *
 * Before 2026-08-01 every cuSPARSE row was measured on a CSR that kept the
 * DIA diagonals' interior zeros (dia_to_csr had no zero check): the baseline
 * did 1/fill extra work and hit the 32I wall at STORED nnz.  This tool
 * re-measures with textbook (true-nonzero) CSR: 32I when true nnz fits,
 * 64I otherwise.  Methodology identical to prep_driver's cusparse block
 * (SPMV_CSR_ALG2, tms(l,10,iters)); conversion/upload reported separately.
 *
 * out:  CSRNNZ,file,stored,true,fill
 *       PREPCSV,file,n,D,spmv,cusparse_csr[_64i],conv_ms,upload_ms,kernel_ms,-1
 * usage: cusparse_fix_bench <dia.txt> [iters=50]
 * ============================================================ */
#include "../dia_io.hpp"
#include "common.cuh"
#include <cusparse.h>
#include <chrono>
#include <cstdint>

#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)
using clk = std::chrono::steady_clock;
static double ms_between(clk::time_point a, clk::time_point b){
    return std::chrono::duration<double,std::milli>(b-a).count();
}

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr,"usage: %s dia.txt [iters]\n", argv[0]); return 1; }
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n, nd = (int)H.offsets.size();

    auto t0 = clk::now();
    CsrHost csr = dia_to_csr(H);                    /* drop_zeros = true */
    auto t1 = clk::now();
    printf("CSRNNZ,%s,%lld,%lld,%.4f\n", argv[1], (long long)H.nnz,
           (long long)csr.nnz, (double)csr.nnz / (double)H.nnz);

    std::vector<float> x((size_t)n, 1.0f);
    float *dX = dupload(x), *dY;
    CUDA_CHECK(cudaMalloc(&dY, (size_t)n * 4));
    const bool i64 = csr.nnz > (int64_t)INT32_MAX;

    void *drp, *dci;
    if (i64) {
        std::vector<int64_t> ci64(csr.col_idx.begin(), csr.col_idx.end());
        drp = dupload(csr.row_ptr64); dci = dupload(ci64);
    } else {
        drp = dupload(csr.row_ptr);   dci = dupload(csr.col_idx);
    }
    float* dv = dupload(csr.vals);
    cusparseHandle_t h; CUSP_CHECK(cusparseCreate(&h));
    cusparseSpMatDescr_t mH; cusparseDnVecDescr_t vIn, vOut;
    const cusparseIndexType_t it = i64 ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
    CUSP_CHECK(cusparseCreateCsr(&mH, n, n, csr.nnz, drp, dci, dv,
        it, it, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSP_CHECK(cusparseCreateDnVec(&vIn,  n, dX, CUDA_R_32F));
    CUSP_CHECK(cusparseCreateDnVec(&vOut, n, dY, CUDA_R_32F));
    const float a1 = 1.f, b0 = 0.f; size_t bsz = 0; void* dbuf = nullptr;
    CUSP_CHECK(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a1, mH, vIn, &b0, vOut, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bsz));
    if (bsz) CUDA_CHECK(cudaMalloc(&dbuf, bsz));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = clk::now();

    auto l = [&](){ CUSP_CHECK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a1, mH, vIn, &b0, vOut, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, dbuf)); };
    l(); CUDA_CHECK(cudaDeviceSynchronize());
    printf("PREPCSV,%s,%d,%d,spmv,%s,%.3f,%.3f,%.6f,-1\n", argv[1], n, nd,
           i64 ? "cusparse_csr_64i" : "cusparse_csr",
           ms_between(t0,t1), ms_between(t1,t2), tms(l, 10, iters));
    return 0;
}
