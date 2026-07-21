/* ============================================================
 * spmv_dia_vs_tc.cu — one matrix, all SpMV methods on the SAME recon/nonzeros:
 *   tensor-core (tc_spmv_regdirect), dense CUDA-core (cuda_spmv_dia),
 *   DIA-native zero-skip (ours), and cuSPARSE SpMV (CSR ALG2) baseline.
 * Drawloom is run separately by the orchestrator (needs a .mtx).
 *
 * DIA-based zero-skip only (per project design): the compact operand keeps the
 * diagonal structure — (didx, val) with col = row + offset[didx] — using a
 * 1-byte didx for D<=256, 2-byte for D>256, and a 64-bit row pointer so nnz can
 * exceed 2^31 (heis_28 etc.). Never a 4-byte-col CSR gather for OUR kernel;
 * cuSPARSE gets a real CSR (its native format) built from the same nonzeros.
 *
 * Memory is phased so large matrices fit the 80 GB A100: the dense recon (for
 * TC + dense-CUDA) is built/timed/freed before the compact operands are
 * uploaded. Matrices whose recon won't fit host/device run zero-skip + cuSPARSE
 * only (verified against cuSPARSE), with TC/dense reported N/A.
 *
 * build: nvcc -O3 -std=c++17 -arch=sm_80 spmv_dia_vs_tc.cu \
 *        ../spmv/src/tc_spmv_regdirect_kernel.cu -o spmv_dia_vs_tc -lcusparse
 * run:   ./spmv_dia_vs_tc <dia.txt> [iters] [--csv]
 * ============================================================ */
#include "dia_io.hpp"
#include "../spmv/src/dia_reconstruct.cuh"      // ReconView + launch_tc_spmv_regdirect
#include "../spmv/src/cuda_dia_kernels.cuh"     // cuda_spmv_dia (dense CUDA)
#include "../spmv/src/spmv_zeroskip_kernels.cuh"// spmv_dia_zeroskip{,_u16,_i64,_u16_i64}
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CKS(x) do{cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){fprintf(stderr,"cuSPARSE %s:%d status=%d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

static const double RECON_DEV_MAX = 65e9;   // leave headroom on 80 GB for x,y
static const double HOST_MAX      = 110e9;  // node has 125 GB

template<class F> static float tms(F f,int wu,int it){
    for(int i=0;i<wu;++i)f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for(int i=0;i<it;++i)f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <dia.txt> [iters] [--csv]\n",argv[0]); return 1; }
    bool csv=false; int iters=200;
    for(int i=2;i<argc;++i){ std::string a=argv[i]; if(a=="--csv")csv=true; else iters=atoi(argv[i]); }

    DiaHost H=load_dia(argv[1]); const int n=H.n; const int D=(int)H.offsets.size();
    const double recon_bytes=(double)D*n*4.0;
    const double band_bytes =(double)H.nnz*4.0;              // H.values size (stored band)
    const bool use_u16 = D>256;

    // ---- descending offsets (as build_recon) + offset->desc-index map ----
    std::vector<int> off_desc(H.offsets); std::sort(off_desc.begin(),off_desc.end(),std::greater<int>());
    std::vector<int> idxOf(2*(size_t)n,-1); for(int k=0;k<D;++k) idxOf[off_desc[k]+n]=k;

    // ---- zero-skip nonzeros (drop stored zeros) as flat CSR-like arrays. These feed
    // BOTH our DIA zero-skip (didx) and cuSPARSE (col); val + row64 shared. Two-pass
    // counting (no vector<vector>). p=min(row,col): off>=0 r=j; off<0 r=j-off. ----
    std::vector<int64_t> row64(n+1,0);
    for(int i=0;i<D;++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i];
        for(int j=0;j<len;++j){ if(H.values[base+j]==0.f) continue; int r=(off>=0)?j:j-off; row64[r+1]++; } }
    for(int r=0;r<n;++r) row64[r+1]+=row64[r];
    const int64_t nnz=row64[n];
    std::vector<unsigned char>  didx8;  std::vector<unsigned short> didx16;
    if(use_u16) didx16.resize(nnz); else didx8.resize(nnz);
    std::vector<int>   col(nnz);
    std::vector<float> cv(nnz);
    { std::vector<int64_t> cur(row64.begin(),row64.end()-1);   // write cursor per row
      for(int i=0;i<D;++i){ int off=H.offsets[i],len=H.lengths[i]; size_t base=H.starts[i]; int k=idxOf[off+n];
        for(int j=0;j<len;++j){ float v=H.values[base+j]; if(v==0.f) continue;
            int r=(off>=0)?j:j-off; int c=r+off; int64_t p=cur[r]++;
            if(use_u16) didx16[p]=(unsigned short)k; else didx8[p]=(unsigned char)k;
            col[p]=c; cv[p]=v; } } }
    const double fill=100.0*(double)nnz/(double)H.nnz;

    // ---- what fits? recon (dense band) is needed only for TC + dense-CUDA ----
    const bool recon_dev_ok  = recon_bytes <= RECON_DEV_MAX;
    const bool recon_host_ok = (band_bytes + recon_bytes) <= HOST_MAX;  // H.values + recon coexist
    const bool do_recon = recon_dev_ok && recon_host_ok;
    const char* skip_reason = !recon_dev_ok ? "recon>65GB_dev" : (!recon_host_ok ? "recon+band>host" : "-");

    // deterministic x
    std::vector<float> hx(n); for(int i=0;i<n;++i) hx[i]=(float)((i*2654435761u)&1023)/1024.f-0.5f;
    float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*4)); CK(cudaMemcpy(dx,hx.data(),(size_t)n*4,cudaMemcpyHostToDevice)); CK(cudaMalloc(&dy,(size_t)n*4));
    int* dOff; CK(cudaMalloc(&dOff,D*4)); CK(cudaMemcpy(dOff,off_desc.data(),D*4,cudaMemcpyHostToDevice));

    float tc_ms=-1,dense_ms=-1,zs_ms=-1,csp_ms=-1, tc_relerr=-1, zs_err=-1, csp_err=-1, ynorm=0;
    std::vector<float> yref(n,0), ytmp(n);

    // ---- Phase A: dense recon -> dense-CUDA (reference) + TC ----
    if(do_recon){
        std::vector<float> recon((size_t)D*n,0.f);
        for(int i=0;i<D;++i){ int off=H.offsets[i],sc=(off>=0)?off:0,len=H.lengths[i]; size_t base=H.starts[i];
            int k=idxOf[off+n]; for(int j=0;j<len;++j) recon[(size_t)k*n+sc+j]=H.values[base+j]; }
        float* dRec; CK(cudaMalloc(&dRec,(size_t)D*n*4)); CK(cudaMemcpy(dRec,recon.data(),(size_t)D*n*4,cudaMemcpyHostToDevice));
        std::vector<float>().swap(recon);
        ReconView R{n,n,D,dOff,dRec};
        auto k_dense=[&](){ cuda_spmv_dia<<<(n+255)/256,256>>>(R,dx,n,dy); };
        auto k_tc   =[&](){ launch_tc_spmv_regdirect(R,dx,n,dy); };
        CK(cudaMemset(dy,0,(size_t)n*4)); k_dense(); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(yref.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
        CK(cudaMemset(dy,0,(size_t)n*4)); k_tc();    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ytmp.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<n;++i){ ynorm=std::max(ynorm,std::fabs(yref[i])); tc_relerr=std::max(tc_relerr,std::fabs(ytmp[i]-yref[i])); }
        dense_ms=tms(k_dense,10,iters); tc_ms=tms(k_tc,10,iters);
        CK(cudaFree(dRec));
    }

    // free the host band now (biggest host user) before uploading compact operands
    std::vector<float>().swap(H.values);

    // ---- Phase B: DIA zero-skip (ours), int64 row-ptr always ----
    { int64_t *dRp; CK(cudaMalloc(&dRp,(size_t)(n+1)*8)); CK(cudaMemcpy(dRp,row64.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
      float* dCv; CK(cudaMalloc(&dCv,(size_t)nnz*4)); CK(cudaMemcpy(dCv,cv.data(),(size_t)nnz*4,cudaMemcpyHostToDevice));
      int shb=D*sizeof(int); int blk=(n+255)/256;
      std::function<void()> k_zs;
      unsigned char* dD8=nullptr; unsigned short* dD16=nullptr;
      const long long* dRpll=(const long long*)dRp;
      if(use_u16){ CK(cudaMalloc(&dD16,(size_t)nnz*2)); CK(cudaMemcpy(dD16,didx16.data(),(size_t)nnz*2,cudaMemcpyHostToDevice));
                   k_zs=[=](){ spmv_dia_zeroskip_u16_i64<<<blk,256,shb>>>((long long)n,D,dRpll,dD16,dCv,dOff,dx,dy); }; }
      else       { CK(cudaMalloc(&dD8 ,(size_t)nnz  )); CK(cudaMemcpy(dD8 ,didx8.data(),(size_t)nnz  ,cudaMemcpyHostToDevice));
                   k_zs=[=](){ spmv_dia_zeroskip_i64    <<<blk,256,shb>>>((long long)n,D,dRpll,dD8 ,dCv,dOff,dx,dy); }; }
      CK(cudaMemset(dy,0,(size_t)n*4)); k_zs(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(ytmp.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
      zs_ms=tms(k_zs,10,iters);
      if(dD8) CK(cudaFree(dD8)); if(dD16) CK(cudaFree(dD16)); CK(cudaFree(dRp)); CK(cudaFree(dCv));
      if(do_recon){ for(int i=0;i<n;++i) zs_err=std::max(zs_err,std::fabs(ytmp[i]-yref[i])); }
      else { yref=ytmp; }   // no recon reference -> zero-skip becomes the reference for cuSPARSE
    }

    // ---- Phase C: cuSPARSE SpMV (CSR ALG2), real CSR from same nonzeros.
    // cuSPARSE needs matching index types: 32I/32I when nnz<=2^31, else 64I/64I. ----
    { const bool big = nnz > 2147483647LL;
      float* dVal; CK(cudaMalloc(&dVal,(size_t)nnz*4)); CK(cudaMemcpy(dVal,cv.data(),(size_t)nnz*4,cudaMemcpyHostToDevice));
      void *dRp=nullptr,*dCol=nullptr;
      cusparseIndexType_t IT = big?CUSPARSE_INDEX_64I:CUSPARSE_INDEX_32I;
      if(big){
        CK(cudaMalloc(&dRp,(size_t)(n+1)*8)); CK(cudaMemcpy(dRp,row64.data(),(size_t)(n+1)*8,cudaMemcpyHostToDevice));
        std::vector<int64_t> col64(col.begin(),col.end());
        CK(cudaMalloc(&dCol,(size_t)nnz*8)); CK(cudaMemcpy(dCol,col64.data(),(size_t)nnz*8,cudaMemcpyHostToDevice));
      } else {
        std::vector<int> rp32(n+1); for(int r=0;r<=n;++r) rp32[r]=(int)row64[r];
        CK(cudaMalloc(&dRp,(size_t)(n+1)*4)); CK(cudaMemcpy(dRp,rp32.data(),(size_t)(n+1)*4,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dCol,(size_t)nnz*4)); CK(cudaMemcpy(dCol,col.data(),(size_t)nnz*4,cudaMemcpyHostToDevice));
      }
      cusparseHandle_t h; CKS(cusparseCreate(&h));
      cusparseSpMatDescr_t A; cusparseDnVecDescr_t vX,vY;
      CKS(cusparseCreateCsr(&A,n,n,nnz,dRp,dCol,dVal,IT,IT,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
      CKS(cusparseCreateDnVec(&vX,n,dx,CUDA_R_32F)); CKS(cusparseCreateDnVec(&vY,n,dy,CUDA_R_32F));
      float alpha=1.f,beta=0.f; size_t bsz=0;
      CKS(cusparseSpMV_bufferSize(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,A,vX,&beta,vY,CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,&bsz));
      void* buf=nullptr; if(bsz) CK(cudaMalloc(&buf,bsz));
      auto k_csp=[&](){ cusparseSpMV(h,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,A,vX,&beta,vY,CUDA_R_32F,CUSPARSE_SPMV_CSR_ALG2,buf); };
      CK(cudaMemset(dy,0,(size_t)n*4)); k_csp(); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(ytmp.data(),dy,(size_t)n*4,cudaMemcpyDeviceToHost));
      csp_ms=tms(k_csp,10,iters);
      for(int i=0;i<n;++i){ ynorm=std::max(ynorm,std::fabs(yref[i])); csp_err=std::max(csp_err,std::fabs(ytmp[i]-yref[i])); }
      if(buf) CK(cudaFree(buf)); cusparseDestroySpMat(A); cusparseDestroyDnVec(vX); cusparseDestroyDnVec(vY); cusparseDestroy(h);
      CK(cudaFree(dRp)); CK(cudaFree(dCol)); CK(cudaFree(dVal));
    }

    // zero-skip operand traffic for the effective-bandwidth column
    double zs_bytes=(double)nnz*(4+(use_u16?2:1))+(double)(n+1)*8;
    double eff_bw=zs_bytes/((double)zs_ms*1e-3)/1e9;
    const char* zsv=use_u16?"dia2B":"dia1B";
    auto spd=[&](float a,float b){ return (a>0&&b>0)? a/b : -1.f; };

    if(!csv){
        printf("=== SpMV  file=%s  n=%d  D=%d  fill=%.1f%%  nnz=%lld  variant=%s%s ===\n",
               argv[1],n,D,fill,(long long)nnz,zsv, do_recon?"":" [recon N/A]");
        printf("  tensor-core       : %s ms\n", tc_ms>0?std::to_string(tc_ms).c_str():"N/A");
        printf("  dense CUDA        : %s ms\n", dense_ms>0?std::to_string(dense_ms).c_str():"N/A");
        printf("  DIA zero-skip     : %.5f ms   (%.2fx vs TC, %.2fx vs dense, %.2fx vs cuSPARSE)\n",
               zs_ms,spd(tc_ms,zs_ms),spd(dense_ms,zs_ms),spd(csp_ms,zs_ms));
        printf("  cuSPARSE CSR ALG2 : %.5f ms\n",csp_ms);
        printf("  [verify] TC relerr=%.2e  zeroskip_err=%.2e  cusparse_err=%.2e  eff_bw=%.1f GB/s\n",
               (ynorm>0?tc_relerr/ynorm:-1.f),zs_err,csp_err,eff_bw);
    } else {
        // CSV: file,n,D,fill,nnz,zsv,tc_ms,dense_ms,zs_ms,csp_ms,zs_vs_tc,zs_vs_dense,zs_vs_csp,tc_relerr,eff_bw,skip
        printf("CSV,%s,%d,%d,%.3f,%lld,%s,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.3e,%.2f,%s\n",
               argv[1],n,D,fill,(long long)nnz,zsv,tc_ms,dense_ms,zs_ms,csp_ms,
               spd(tc_ms,zs_ms),spd(dense_ms,zs_ms),spd(csp_ms,zs_ms),
               (ynorm>0?tc_relerr/ynorm:-1.f),eff_bw,skip_reason);
    }
    CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(dOff));
    return 0;
}
