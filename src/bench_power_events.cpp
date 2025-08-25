// src/bench_power_events.cpp
#include "DiaGPU.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <vector>
#include <unordered_map>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
// #include <mkl_spblas.h>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <fstream>


#define CUDA_OK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); std::exit(1);} }while(0)
#define CUSPARSE_OK(call) do { cusparseStatus_t _s=(call); if(_s!=CUSPARSE_STATUS_SUCCESS){ \
  fprintf(stderr,"cuSPARSE %s:%d code=%d\n",__FILE__,__LINE__,(int)_s); std::exit(1);} }while(0)

// ----------------- small helpers (host) -----------------
static inline int start_row_h(int offset){ return (offset>=0?0:-offset); }
static inline int end_row_h(int N,int offset){ return (offset>=0?N-offset:N); }

// Build a random diagonal list at chosen offsets (for quick testing)
static DiagListF32 build_random_diaglist(int N, const std::vector<int>& offsets, unsigned seed=123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    DiagListF32 A; A.N = N;
    A.offsets = offsets; std::sort(A.offsets.begin(), A.offsets.end());
    A.ptr.assign(A.offsets.size()+1, 0);
    for (size_t d=0; d<A.offsets.size(); ++d) A.ptr[d+1] = A.ptr[d] + diag_len(N, A.offsets[d]);
    A.vals.assign(A.ptr.back(), 0.0f);
    for (size_t d=0; d<A.offsets.size(); ++d) {
        int off = A.offsets[d];
        int s = start_row_h(off), e = end_row_h(N, off);
        for (int i=s; i<e; ++i) A.vals[A.ptr[d] + (i-s)] = dist(rng);
    }
    return A;
}

// Diagonal list -> CSR (for cuSPARSE)
static void diaglist_to_csr(const DiagListF32& A,
                            std::vector<int>& rowPtr,
                            std::vector<int>& colInd,
                            std::vector<float>& vals)
{
    const int N = A.N;
    rowPtr.assign(N+1, 0);
    int nnz=0;
    for (size_t d=0; d<A.offsets.size(); ++d) {
        int off=A.offsets[d], s=start_row_h(off), e=end_row_h(N,off);
        nnz += (e-s);
        for (int i=s;i<e;++i) rowPtr[i+1]++;
    }
    for (int i=0;i<N;++i) rowPtr[i+1]+=rowPtr[i];
    colInd.assign(nnz,0); vals.assign(nnz,0.0f);
    std::vector<int> cur=rowPtr;
    for (size_t d=0; d<A.offsets.size(); ++d) {
        int off=A.offsets[d], s=start_row_h(off), e=end_row_h(N,off), base=A.ptr[d];
        for (int i=s;i<e;++i){
            int t=i-s, j=i+off, p=cur[i]++;
            colInd[p]=j; vals[p]=A.vals[base+t];
        }
    }
}

// CSR -> diagonal list (for verification)
static DiagListF32 csr_to_diaglist(int N,
                                   const std::vector<int>& rp,
                                   const std::vector<int>& ci,
                                   const std::vector<float>& va)
{
    std::vector<int> offs; offs.reserve(N*2);
    for (int i=0;i<N;++i) for (int p=rp[i]; p<rp[i+1]; ++p) offs.push_back(ci[p]-i);
    std::sort(offs.begin(),offs.end()); offs.erase(std::unique(offs.begin(),offs.end()),offs.end());

    DiagListF32 C; C.N=N; C.offsets=offs; C.ptr.assign(offs.size()+1,0);
    for (size_t d=0; d<offs.size(); ++d) C.ptr[d+1]=C.ptr[d]+diag_len(N,offs[d]);
    C.vals.assign(C.ptr.back(), 0.0f);

    for (int i=0;i<N;++i){
        for (int p=rp[i]; p<rp[i+1]; ++p){
            int j=ci[p], off=j-i, s=start_row_h(off);
            int d=int(std::lower_bound(C.offsets.begin(),C.offsets.end(),off)-C.offsets.begin());
            int t=i-s;
            C.vals[C.ptr[d]+t] += va[p];
        }
    }
    return C;
}

static void compare_diaglists(const DiagListF32& X, const DiagListF32& Y, float tol=1e-4f){
    std::unordered_map<int,int> ix,iy;
    for (int d=0; d<(int)X.offsets.size(); ++d) ix[X.offsets[d]]=d;
    for (int d=0; d<(int)Y.offsets.size(); ++d) iy[Y.offsets[d]]=d;
    std::vector<int> all; all.reserve(X.offsets.size()+Y.offsets.size());
    all.insert(all.end(), X.offsets.begin(), X.offsets.end());
    all.insert(all.end(), Y.offsets.begin(), Y.offsets.end());
    std::sort(all.begin(), all.end()); all.erase(std::unique(all.begin(),all.end()), all.end());
    double diff2=0.0, ref2=0.0; float maxabs=0.f;
    for (int off:all){
        int L=diag_len(X.N,off);
        for (int t=0;t<L;++t){
            float xv=0.f, yv=0.f;
            auto itx=ix.find(off); if (itx!=ix.end()) xv=X.vals[X.ptr[itx->second]+t];
            auto ity=iy.find(off); if (ity!=iy.end()) yv=Y.vals[Y.ptr[ity->second]+t];
            float d=xv-yv; diff2+=double(d)*d; ref2+=double(yv)*yv; if (std::fabs(d)>maxabs) maxabs=std::fabs(d);
        }
    }
    float rel = (ref2>0.0)? float(std::sqrt(diff2/ref2)) : float(std::sqrt(diff2));
    printf("[VERIFY] max_abs=%.3g rel_fro=%.3g\n", maxabs, rel);
}

// ----------------- cuSPARSE: power A^p with CUDA-event timing -----------------
static void cusparse_power_right_events(int N,
                                        const std::vector<int>& rpA0,
                                        const std::vector<int>& ciA0,
                                        const std::vector<float>& vaA0,
                                        int power,
                                        float* total_compute_ms_out,  // compute+copy only
                                        DiagListF32* C_diag_out)      // final result (optional)
{
    if (total_compute_ms_out) *total_compute_ms_out = 0.0f;
    if (power <= 1) {
        if (C_diag_out) {
            // Materialize A^1 = A
            *C_diag_out = csr_to_diaglist(N, rpA0, ciA0, vaA0);
        }
        return;
    }

    cusparseHandle_t h; CUSPARSE_OK(cusparseCreate(&h));
    // Scalars must be valid pointers (host mode)
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_OK(cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST));

    const cusparseIndexType_t it = CUSPARSE_INDEX_32I;
    const cudaDataType        dt = CUDA_R_32F;

    // -------- B = A (constant on device) --------
    int nnzB = (int)ciA0.size();
    int *d_rpB=nullptr, *d_ciB=nullptr; float *d_vaB=nullptr;
    CUDA_OK(cudaMalloc((void**)&d_rpB, (N+1)*sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_ciB, nnzB*sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_vaB, nnzB*sizeof(float)));
    CUDA_OK(cudaMemcpy(d_rpB, rpA0.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ciB, ciA0.data(), nnzB*sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_vaB, vaA0.data(), nnzB*sizeof(float),cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t B_csr;
    CUSPARSE_OK(cusparseCreateCsr(&B_csr, N, N, nnzB,
                                  d_rpB, d_ciB, d_vaB,
                                  it, it, CUSPARSE_INDEX_BASE_ZERO, dt));

    // -------- Prev = A (mutable on device) --------
    int nnzPrev = (int)ciA0.size();
    int *d_rpPrev=nullptr, *d_ciPrev=nullptr; float *d_vaPrev=nullptr;
    CUDA_OK(cudaMalloc((void**)&d_rpPrev, (N+1)*sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_ciPrev, nnzPrev*sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_vaPrev, nnzPrev*sizeof(float)));
    CUDA_OK(cudaMemcpy(d_rpPrev, rpA0.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ciPrev, ciA0.data(), nnzPrev*sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_vaPrev, vaA0.data(), nnzPrev*sizeof(float),cudaMemcpyHostToDevice));

    float total_ms = 0.0f;

    for (int k = 2; k <= power; ++k) {
        // Descriptors for Prev and empty C
        cusparseSpMatDescr_t Prev_csr, C_csr;
        CUSPARSE_OK(cusparseCreateCsr(&Prev_csr, N, N, nnzPrev,
                                      d_rpPrev, d_ciPrev, d_vaPrev,
                                      it, it, CUSPARSE_INDEX_BASE_ZERO, dt));
        CUSPARSE_OK(cusparseCreateCsr(&C_csr, N, N, 0,
                                      nullptr, nullptr, nullptr,
                                      it, it, CUSPARSE_INDEX_BASE_ZERO, dt));

        cusparseSpGEMMDescr_t desc; CUSPARSE_OK(cusparseSpGEMM_createDescr(&desc));
        size_t buf1 = 0, buf2 = 0; void *dBuf1 = nullptr, *dBuf2 = nullptr;

        // --- Structure phase (NOT timed): workEstimation + first compute ---
        CUSPARSE_OK(cusparseSpGEMM_workEstimation(
            h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, Prev_csr, B_csr, &beta, C_csr, dt,
            CUSPARSE_SPGEMM_DEFAULT, desc, &buf1, nullptr));
        CUDA_OK(cudaMalloc(&dBuf1, buf1));
        CUSPARSE_OK(cusparseSpGEMM_workEstimation(
            h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, Prev_csr, B_csr, &beta, C_csr, dt,
            CUSPARSE_SPGEMM_DEFAULT, desc, &buf1, dBuf1));

        CUSPARSE_OK(cusparseSpGEMM_compute(
            h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, Prev_csr, B_csr, &beta, C_csr, dt,
            CUSPARSE_SPGEMM_DEFAULT, desc, &buf2, nullptr));
        CUDA_OK(cudaMalloc(&dBuf2, buf2));

        // --- TIMED REGION: second compute + copy ---
        cudaEvent_t e0, e1; CUDA_OK(cudaEventCreate(&e0)); CUDA_OK(cudaEventCreate(&e1));
        CUDA_OK(cudaEventRecord(e0, 0));

        // Second compute (numeric)
        CUSPARSE_OK(cusparseSpGEMM_compute(
            h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, Prev_csr, B_csr, &beta, C_csr, dt,
            CUSPARSE_SPGEMM_DEFAULT, desc, &buf2, dBuf2));

        // Obtain nnz(C), allocate CSR buffers for C, bind to C_csr
        int64_t rows, cols, cNnz64;
        CUSPARSE_OK(cusparseSpMatGetSize(C_csr, &rows, &cols, &cNnz64));
        int nnzC = (int)cNnz64;
        int *d_rpC=nullptr, *d_ciC=nullptr; float *d_vaC=nullptr;
        CUDA_OK(cudaMalloc((void**)&d_rpC, (N+1)*sizeof(int)));
        CUDA_OK(cudaMalloc((void**)&d_ciC, nnzC*sizeof(int)));
        CUDA_OK(cudaMalloc((void**)&d_vaC, nnzC*sizeof(float)));
        CUSPARSE_OK(cusparseCsrSetPointers(C_csr, d_rpC, d_ciC, d_vaC));

        // Copy values into C
        CUSPARSE_OK(cusparseSpGEMM_copy(
            h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, Prev_csr, B_csr, &beta, C_csr, dt,
            CUSPARSE_SPGEMM_DEFAULT, desc));

        CUDA_OK(cudaEventRecord(e1, 0));
        CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0.0f; CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_OK(cudaEventDestroy(e0)); CUDA_OK(cudaEventDestroy(e1));
        total_ms += ms;

        // --- Cleanup this step and promote C -> Prev ---
        cudaFree(dBuf1); cudaFree(dBuf2);
        cusparseSpGEMM_destroyDescr(desc);
        cusparseDestroySpMat(Prev_csr);
        // free old Prev arrays, then promote C
        cudaFree(d_rpPrev); cudaFree(d_ciPrev); cudaFree(d_vaPrev);
        d_rpPrev = d_rpC; d_ciPrev = d_ciC; d_vaPrev = d_vaC; nnzPrev = nnzC;
        cusparseDestroySpMat(C_csr);
    }

    if (total_compute_ms_out) *total_compute_ms_out = total_ms;

    // Materialize final C if requested
    if (C_diag_out) {
        std::vector<int>    h_rp(N+1), h_ci(nnzPrev);
        std::vector<float>  h_va(nnzPrev);
        CUDA_OK(cudaMemcpy(h_rp.data(), d_rpPrev, (N+1)*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_ci.data(), d_ciPrev, nnzPrev*sizeof(int),  cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_va.data(), d_vaPrev, nnzPrev*sizeof(float),cudaMemcpyDeviceToHost));
        *C_diag_out = csr_to_diaglist(N, h_rp, h_ci, h_va);
    }

    // Final cleanup
    cusparseDestroySpMat(B_csr);
    cudaFree(d_rpB); cudaFree(d_ciB); cudaFree(d_vaB);
    cudaFree(d_rpPrev); cudaFree(d_ciPrev); cudaFree(d_vaPrev);
    cusparseDestroy(h);
}

// static void mkl_power_right_events(int N,
//                                    const std::vector<int>& rpA0,
//                                    const std::vector<int>& ciA0,
//                                    const std::vector<float>& vaA0,
//                                    int power,
//                                    float* total_ms_out,         // total compute time in ms
//                                    DiagListF32* C_diag_out)     // final result (optional)
// {
//     if (total_ms_out) *total_ms_out = 0.0f;
//     if (power <= 1) {
//         if (C_diag_out) *C_diag_out = csr_to_diaglist(N, rpA0, ciA0, vaA0);
//         return;
//     }

//     // Helpers to create/export MKL CSR handles (single precision)
//     auto create_csr = [&](const std::vector<int>& rp,
//                           const std::vector<int>& ci,
//                           const std::vector<float>& va) -> sparse_matrix_t {
//         sparse_matrix_t H{};
//         // MKL_INT is 32-bit on LP64 builds; safe to cast from int.
//         sparse_status_t st = mkl_sparse_s_create_csr(
//             &H, SPARSE_INDEX_BASE_ZERO,
//             (MKL_INT)N, (MKL_INT)N,
//             const_cast<MKL_INT*>(reinterpret_cast<const MKL_INT*>(rp.data())),
//             const_cast<MKL_INT*>(reinterpret_cast<const MKL_INT*>(rp.data())) + 1,
//             const_cast<MKL_INT*>(reinterpret_cast<const MKL_INT*>(ci.data())),
//             const_cast<float*>(va.data())
//         );
//         if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl create_csr failed");
//         return H;
//     };

//     auto export_csr = [&](sparse_matrix_t H,
//                           std::vector<int>& rp,
//                           std::vector<int>& ci,
//                           std::vector<float>& va) {
//         sparse_index_base_t base;
//         MKL_INT nr, nc;
//         MKL_INT *rpB, *rpE, *ciP; float *vv;
//         if (mkl_sparse_s_export_csr(H, &base, &nr, &nc, &rpB, &rpE, &ciP, &vv) != SPARSE_STATUS_SUCCESS)
//             throw std::runtime_error("mkl export_csr failed");
//         rp.resize(nr+1);
//         rp[0] = rpB[0];
//         for (MKL_INT i=0; i<nr; ++i) rp[i+1] = rpE[i];
//         const MKL_INT nnz = rp.back();
//         ci.assign(ciP, ciP + nnz);
//         va.assign(vv,  vv + nnz);
//     };

//     auto make_I = [&]()->sparse_matrix_t{
//         std::vector<int> rp(N+1), ci(N); std::vector<float> va(N, 1.0f);
//         for (int i=0;i<=N;++i) rp[i]=i;
//         for (int i=0;i<N;++i)  ci[i]=i;
//         return create_csr(rp, ci, va);
//     };

//     auto sp2m_full = [&](sparse_matrix_t A, sparse_matrix_t B)->sparse_matrix_t{
//         matrix_descr d{}; d.type = SPARSE_MATRIX_TYPE_GENERAL;
//         sparse_matrix_t C{};
//         sparse_status_t st = mkl_sparse_sp2m(
//             SPARSE_OPERATION_NON_TRANSPOSE, d, A,
//             SPARSE_OPERATION_NON_TRANSPOSE, d, B,
//             SPARSE_STAGE_FULL_MULT, &C);
//         if (st != SPARSE_STATUS_SUCCESS) throw std::runtime_error("mkl sp2m failed");
//         // Sort for deterministic order (outside timing to mirror cuSPARSE copy semantics)
//         mkl_sparse_order(C);
//         return C;
//     };

//     // Build B and Prev from input CSR
//     sparse_matrix_t B = create_csr(rpA0, ciA0, vaA0);
//     sparse_matrix_t Prev = create_csr(rpA0, ciA0, vaA0);

//     double total_ms = 0.0;
//     for (int k=2; k<=power; ++k) {
//         // Time just the multiplication (structure+numeric)
//         auto t0 = std::chrono::high_resolution_clock::now();
//         sparse_matrix_t C = sp2m_full(Prev, B);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

//         mkl_sparse_destroy(Prev);
//         Prev = C;
//         // Optional: printf per-step
//         // printf("[MKL] step %d compute=%.3f ms\n", k, std::chrono::duration<double, std::milli>(t1 - t0).count());
//     }

//     if (total_ms_out) *total_ms_out = (float)total_ms;

//     if (C_diag_out) {
//         std::vector<int> rp, ci; std::vector<float> va;
//         export_csr(Prev, rp, ci, va);
//         *C_diag_out = csr_to_diaglist(N, rp, ci, va);
//     }

//     mkl_sparse_destroy(Prev);
//     mkl_sparse_destroy(B);
// }

int main(int argc, char** argv){
    int N=4096, power=5;
    if (argc>=2) N = std::atoi(argv[1]);
    if (argc>=3) power = std::atoi(argv[2]);
    printf("A^p benchmark with CUDA events: N=%d, power=%d\n", N, power);

    // Build test A (choose offsets you like)
    // std::vector<int> offs = { 0 };
    // std::vector<int> offs = { -8, -4, -1, 0, 3, 5 };
    // std::vector<int> offs = {-21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21};
    std::vector<int> offs = {-31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    DiagListF32 A = build_random_diaglist(N, offs, 123);

    // -------- Your kernel path (reuse B; kernel-only & kernel+copy per multiply) --------
    float total_kernel_ms = 0.0f, total_total_ms = 0.0f;
    float eps = 0.0f;

    DiaBPlan* planB = create_B_plan(A);   // upload B once and reuse
    DiagListF32 Ck = A;

    for (int k = 2; k <= power; ++k) {
        DiagListF32 next;
        float ms_kernel = 0.0f, ms_total = 0.0f; // kernel-only, kernel+device-copy
        multiply_sparse_noPad_with_timing_copy_plan(Ck, planB, eps, next, &ms_kernel, &ms_total);
        total_kernel_ms += ms_kernel;
        total_total_ms  += ms_total;
        Ck = std::move(next);
        printf("[Yours] step %d  kernel=%.3f ms  kernel+copy=%.3f ms\n", k, ms_kernel, ms_total);
    }

    printf("[Yours] totals: kernel=%.3f ms  kernel+copy=%.3f ms\n", total_kernel_ms, total_total_ms);
    destroy_B_plan(planB);

    // -------- cuSPARSE path (compute+copy per multiply) --------
    std::vector<int> rpA, ciA; std::vector<float> vaA;
    diaglist_to_csr(A, rpA, ciA, vaA);

    float total_cus_ms = 0.0f;
    DiagListF32 Cc;
    cusparse_power_right_events(N, rpA, ciA, vaA, power, &total_cus_ms, &Cc);
    printf("[cuSPARSE] total compute+copy: %.3f ms\n", total_cus_ms);

    // Verify final
    compare_diaglists(Ck, Cc, 1e-4f);

    // float total_mkl_ms = 0.0f;
    // DiagListF32 Cm;
    // mkl_power_right_events(N, rpA, ciA, vaA, power, &total_mkl_ms, &Cm);
    // printf("[MKL] total compute: %.3f ms\n", total_mkl_ms);

    // // Verify MKL vs Ours (and optionally vs cuSPARSE)
    // compare_diaglists(Ck, Cm, 1e-4f);

    // Updated summary
    printf("\nSummary over power=%d\n", power);
    printf("Matrix Size: %d\n", N);
    printf("Diagonal Size: %d\n", (int)offs.size());
    printf("  Yours (kernel-only):        %.3f ms\n", total_kernel_ms);
    printf("  Yours (kernel+copy):        %.3f ms\n", total_total_ms);
    printf("  cuSPARSE (compute+copy):    %.3f ms\n", total_cus_ms);
    // printf("  MKL (CPU compute only):     %.3f ms\n", total_mkl_ms);
    printf("  Ratio (cuSPARSE / Ours):    %.3fx\n", total_cus_ms / total_total_ms);
    // printf("  Ratio (MKL / Ours-total):   %.3fx\n", total_mkl_ms / total_total_ms);

    std::ofstream fout("output/summary.txt", std::ios::app);
    fout << "Summary over power=" << power << "\n";
    fout << "Matrix Size: " << N << "\n";
    fout << "Diagonal Size: " << (int)offs.size() << "\n";
    fout << "  Yours (kernel-only):        " << total_kernel_ms << " ms\n";
    fout << "  Yours (kernel+copy):        " << total_total_ms << " ms\n";
    fout << "  cuSPARSE (compute+copy):    " << total_cus_ms << " ms\n";
    // fout << "  MKL (CPU compute only):     " << total_mkl_ms << " ms\n";
    fout << "  Ratio (cuSPARSE / Ours):    " << (total_cus_ms / total_total_ms) << "x\n";
    // fout << "  Ratio (MKL / Ours-total):   " << (total_mkl_ms / total_total_ms) << "x\n";
    fout << "===========================================================================\n";
    fout.close();
    return 0;
}
