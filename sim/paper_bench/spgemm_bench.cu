/* spgemm_bench — cuSPARSE SpGEMM (fp32, true-nonzero CSR) in isolation, so
 * ncu can capture its internal kernels without wading through our drivers'
 * launches.  Methodology mirrors spmspm_driver's cusparse block
 * (SPGEMM_DEFAULT, workEstimation + compute + copy per iteration).
 *   usage: spgemm_bench <dia.txt> [iters=3] */
#include "../dia_io.hpp"
#include "common.cuh"
#include <cusparse.h>
#define CUSP_CHECK(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

int main(int argc, char** argv){
    if (argc < 2){ fprintf(stderr, "usage: %s dia.txt [iters]\n", argv[0]); return 1; }
    const int iters = argc > 2 ? atoi(argv[2]) : 3;
    DiaHost H = load_dia(argv[1]);
    const int n = H.n;
    CsrHost csr = dia_to_csr(H);                       /* true nonzeros */
    int *drp = dupload(csr.row_ptr), *dci = dupload(csr.col_idx);
    float* dv = dupload(csr.vals);
    cusparseHandle_t hd; CUSP_CHECK(cusparseCreate(&hd));
    const float al = 1.f, be = 0.f;
    const auto OP = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const auto AL = CUSPARSE_SPGEMM_DEFAULT;
    for (int it = 0; it < iters; ++it) {
        cusparseSpMatDescr_t mA, mB, mC;
        CUSP_CHECK(cusparseCreateCsr(&mA,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateCsr(&mB,n,n,csr.nnz,drp,dci,dv,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        CUSP_CHECK(cusparseCreateCsr(&mC,n,n,0,nullptr,nullptr,nullptr,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
        cusparseSpGEMMDescr_t sd; CUSP_CHECK(cusparseSpGEMM_createDescr(&sd));
        size_t b1 = 0, b2 = 0; void *d1 = nullptr, *d2 = nullptr;
        CUSP_CHECK(cusparseSpGEMM_workEstimation(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b1,nullptr));
        CUDA_CHECK(cudaMalloc(&d1,b1));
        CUSP_CHECK(cusparseSpGEMM_workEstimation(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b1,d1));
        CUSP_CHECK(cusparseSpGEMM_compute(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b2,nullptr));
        CUDA_CHECK(cudaMalloc(&d2,b2));
        CUSP_CHECK(cusparseSpGEMM_compute(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd,&b2,d2));
        int64_t cr, cc, cn; CUSP_CHECK(cusparseSpMatGetSize(mC,&cr,&cc,&cn));
        int *crp, *cci; float* cv;
        CUDA_CHECK(cudaMalloc(&crp,(n+1)*4)); CUDA_CHECK(cudaMalloc(&cci,cn*4)); CUDA_CHECK(cudaMalloc(&cv,cn*4));
        CUSP_CHECK(cusparseCsrSetPointers(mC,crp,cci,cv));
        CUSP_CHECK(cusparseSpGEMM_copy(hd,OP,OP,&al,mA,mB,&be,mC,CUDA_R_32F,AL,sd));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (it == 0) printf("SPGEMM,%s,%d,%lld,%lld\n", argv[1], n, (long long)csr.nnz, (long long)cn);
        cusparseSpGEMM_destroyDescr(sd);
        cusparseDestroySpMat(mA); cusparseDestroySpMat(mB); cusparseDestroySpMat(mC);
        cudaFree(d1); cudaFree(d2); cudaFree(crp); cudaFree(cci); cudaFree(cv);
    }
    return 0;
}
