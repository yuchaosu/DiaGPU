/* ============================================================
 * tc_spmv_ztile.cuh — zero-tile-skip tensor-core SpMV (+ sym+ztile).
 *
 * The full TC kernel (tc_spmv_regdirect) walks EVERY diagonal batch for every
 * 16-row tile, MMA-ing the interior structural zeros of the band. Here we tile
 * the reconstruction and DROP the pure-zero tiles: per 16-row tile-row we keep
 * only the diagonals that have a nonzero somewhere in those 16 rows, compacted
 * into MMA_K(8)-wide batches. A warp (= one tile-row) then MMAs only its
 * surviving diagonals. Front/back padding is already skipped by the offset
 * gather; this removes the INTERIOR all-zero 16-row tiles -> block-structured
 * bands win, uniformly-dense bands keep ~100% of tiles (no change).
 *
 *   build_tile_plan(...)              -> per-tile-row nonzero-diagonal lists
 *   tc_spmv_ztile_kernel             -> zero-tile-skip TC (full recon)
 *   tc_spmv_symztile_kernel          -> symmetric half-recon + zero-tile-skip
 * Reuses symtc::to_tf32 / symtc::mma / build_recon_sym from tc_spmv_sym.cuh.
 * ============================================================ */
#pragma once
#include "tc_spmv_sym.cuh"      // ReconView, MMA helpers (symtc::), build_recon_sym
#include <vector>

/* ---- host tile plan: per tile-row, the diagonal indices with a nonzero. ----
 * recon is (num_diags x cols), values[di*cols + c] = H[c-off_di, c] (descending
 * offsets, same as build_recon). A (tile-row t, diagonal di) tile is nonzero if
 * any Recon[di, t*16+m+off_di] != 0 for m in [0,16), col in range. Each
 * tile-row's kept-diagonal list is padded to a multiple of 8 with sentinel
 * `num_diags` (-> gathered as 0). tiles_kept_pct = kept/(T*num_diags). */
struct TilePlan { std::vector<int> ptr; std::vector<int> diags; int n_tiles; double kept_pct; };

inline TilePlan build_tile_plan(int rows, int cols, int num_diags,
                                const std::vector<int>& off_desc,
                                const std::vector<float>& recon)
{
    TilePlan P; const int T=(rows+15)/16; P.n_tiles=T; P.ptr.assign(T+1,0);
    long long total=0, kept=0;
    for(int t=0;t<T;++t){
        P.ptr[t]=(int)P.diags.size();
        for(int di=0; di<num_diags; ++di){
            const int d=off_desc[di]; bool nz=false;
            for(int m=0;m<16;++m){ const int r=t*16+m; if(r>=rows) break;
                const long long c=(long long)r+d; if(c<0||c>=cols) continue;
                if(recon[(size_t)di*cols + c]!=0.f){ nz=true; break; } }
            ++total; if(nz){ P.diags.push_back(di); ++kept; }
        }
        while((int)(P.diags.size()-P.ptr[t])%8 != 0) P.diags.push_back(num_diags); // sentinel
    }
    P.ptr[T]=(int)P.diags.size();
    P.kept_pct = total? 100.0*(double)kept/(double)total : 100.0;
    return P;
}

/* ---- zero-tile-skip TC: warp=tile-row iterates only its kept diagonals. ---- */
__global__ void tc_spmv_ztile_kernel(ReconView R,
    const int* __restrict__ tileDiagPtr, const int* __restrict__ tileDiags,
    const float* __restrict__ x, int x_size, float* __restrict__ y)
{
    const int wpb=blockDim.x>>5, gw=(int)blockIdx.x*wpb+(threadIdx.x>>5);
    const int tile_row=gw*MMA_M; if(tile_row>=R.rows) return;
    const int lane=(int)threadIdx.x&31, g=lane>>2, t=lane&3;
    float acc_top[4]={0,0,0,0}, acc_bot[4]={0,0,0,0};
    const int base=tileDiagPtr[gw], end=tileDiagPtr[gw+1];
    for(int jj=base; jj<end; jj+=MMA_K){
        const int di0=tileDiags[jj+t], di4=tileDiags[jj+t+4];
        const bool v0=di0<R.num_diags, v4=di4<R.num_diags;
        const int o0=v0?R.diag_offsets[di0]:0, o4=v4?R.diag_offsets[di4]:0;
        uint32_t A[4]={0,0,0,0}, Bt[2]={0,0}, Bb[2]={0,0};
        if(v0){ int c=tile_row+g+o0;   if(c>=0&&c<R.cols)A[0]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                c=tile_row+g+8+o0;      if(c>=0&&c<R.cols)A[1]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                int xi=tile_row+g+o0;   if(xi>=0&&xi<x_size)Bt[0]=symtc::to_tf32(x[xi]);
                xi=tile_row+g+8+o0;     if(xi>=0&&xi<x_size)Bb[0]=symtc::to_tf32(x[xi]); }
        if(v4){ int c=tile_row+g+o4;   if(c>=0&&c<R.cols)A[2]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                c=tile_row+g+8+o4;      if(c>=0&&c<R.cols)A[3]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                int xi=tile_row+g+o4;   if(xi>=0&&xi<x_size)Bt[1]=symtc::to_tf32(x[xi]);
                xi=tile_row+g+8+o4;     if(xi>=0&&xi<x_size)Bb[1]=symtc::to_tf32(x[xi]); }
        symtc::mma(acc_top,A,Bt); symtc::mma(acc_bot,A,Bb);
    }
    if(g==2*t){ int rt=tile_row+g,rb=tile_row+8+g; if(rt<R.rows)y[rt]=acc_top[0]; if(rb<R.rows)y[rb]=acc_bot[2]; }
    else if(g==2*t+1){ int rt=tile_row+g,rb=tile_row+8+g; if(rt<R.rows)y[rt]=acc_top[1]; if(rb<R.rows)y[rb]=acc_bot[3]; }
}

inline void launch_tc_spmv_ztile(ReconView R,const int* dPtr,const int* dDiags,
                                 const float* dx,int x_size,float* dy,cudaStream_t s=0){
    const int nt=(R.rows+MMA_M-1)/MMA_M; if(!nt)return; const int nb=(nt+3)/4;
    tc_spmv_ztile_kernel<<<nb,4*32,0,s>>>(R,dPtr,dDiags,dx,x_size,dy);
}

/* ---- sym tile plan with SEPARATE upper/lower lists (the fix): a diagonal is
 * kept in the UPPER list for a tile only where its upper tile Rh[di,r+d] has a
 * nonzero, and in the LOWER list (d>0) only where Rh[di,r] has a nonzero. Each
 * gather is then done ONLY where it contributes — no wasted dual-gather. ---- */
struct TilePlanSym {
    std::vector<int> ptrU, diagsU, ptrL, diagsL;   // upper / lower per-tile lists
    int n_tiles; double kept_pct;                  // (keptU+keptL)/(2*T*num_sym) effective
};
inline TilePlanSym build_tile_plan_sym(int rows,int cols,int num_sym,
                                       const std::vector<int>& off_sym /*d>=0 desc*/,
                                       const std::vector<float>& rh)
{
    TilePlanSym P; const int T=(rows+15)/16; P.n_tiles=T;
    P.ptrU.assign(T+1,0); P.ptrL.assign(T+1,0);
    long long totalU=0,keptU=0,totalL=0,keptL=0;
    for(int t=0;t<T;++t){
        P.ptrU[t]=(int)P.diagsU.size(); P.ptrL[t]=(int)P.diagsL.size();
        for(int di=0; di<num_sym; ++di){
            const int d=off_sym[di]; bool nzU=false,nzL=false;
            for(int m=0;m<16 && (!nzU||!nzL);++m){ const int r=t*16+m; if(r>=rows) break;
                const long long cu=(long long)r+d; if(!nzU && cu<cols && rh[(size_t)di*cols+cu]!=0.f) nzU=true;
                if(d>0 && !nzL && r-d>=0 && rh[(size_t)di*cols+r]!=0.f) nzL=true; }
            ++totalU; if(nzU){P.diagsU.push_back(di);++keptU;}
            if(d>0){ ++totalL; if(nzL){P.diagsL.push_back(di);++keptL;} }
        }
        while((int)(P.diagsU.size()-P.ptrU[t])%8) P.diagsU.push_back(num_sym);
        while((int)(P.diagsL.size()-P.ptrL[t])%8) P.diagsL.push_back(num_sym);
    }
    P.ptrU[T]=(int)P.diagsU.size(); P.ptrL[T]=(int)P.diagsL.size();
    long long tot=totalU+totalL, kep=keptU+keptL;
    P.kept_pct = tot? 100.0*(double)kep/(double)tot : 100.0;
    return P;
}

/* ---- symmetric half-recon + zero-tile-skip: upper loop over the upper list,
 * lower loop over the lower list — each gather only where nonzero. ---- */
__global__ void tc_spmv_symztile_kernel(ReconView R /*half recon, d>=0*/,
    const int* __restrict__ ptrU, const int* __restrict__ diagsU,
    const int* __restrict__ ptrL, const int* __restrict__ diagsL,
    const float* __restrict__ x, int x_size, float* __restrict__ y)
{
    const int wpb=blockDim.x>>5, gw=(int)blockIdx.x*wpb+(threadIdx.x>>5);
    const int tile_row=gw*MMA_M; if(tile_row>=R.rows) return;
    const int lane=(int)threadIdx.x&31, g=lane>>2, t=lane&3;
    float acc_top[4]={0,0,0,0}, acc_bot[4]={0,0,0,0};
    // UPPER: A col = tile+g(+8)+o, B = x[tile+g(+8)+o]
    for(int jj=ptrU[gw]; jj<ptrU[gw+1]; jj+=MMA_K){
        const int di0=diagsU[jj+t], di4=diagsU[jj+t+4];
        const bool v0=di0<R.num_diags, v4=di4<R.num_diags;
        const int o0=v0?R.diag_offsets[di0]:0, o4=v4?R.diag_offsets[di4]:0;
        uint32_t A[4]={0,0,0,0}, Bt[2]={0,0}, Bb[2]={0,0};
        if(v0){ int c=tile_row+g+o0;  if(c>=0&&c<R.cols)A[0]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                c=tile_row+g+8+o0;     if(c>=0&&c<R.cols)A[1]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                int xi=tile_row+g+o0;  if(xi>=0&&xi<x_size)Bt[0]=symtc::to_tf32(x[xi]);
                xi=tile_row+g+8+o0;    if(xi>=0&&xi<x_size)Bb[0]=symtc::to_tf32(x[xi]); }
        if(v4){ int c=tile_row+g+o4;  if(c>=0&&c<R.cols)A[2]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                c=tile_row+g+8+o4;     if(c>=0&&c<R.cols)A[3]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                int xi=tile_row+g+o4;  if(xi>=0&&xi<x_size)Bt[1]=symtc::to_tf32(x[xi]);
                xi=tile_row+g+8+o4;    if(xi>=0&&xi<x_size)Bb[1]=symtc::to_tf32(x[xi]); }
        symtc::mma(acc_top,A,Bt); symtc::mma(acc_bot,A,Bb);
    }
    // LOWER (d>0): A col = tile+g(+8), B = x[tile+g(+8)-o]
    for(int jj=ptrL[gw]; jj<ptrL[gw+1]; jj+=MMA_K){
        const int di0=diagsL[jj+t], di4=diagsL[jj+t+4];
        const bool v0=di0<R.num_diags, v4=di4<R.num_diags;
        const int o0=v0?R.diag_offsets[di0]:0, o4=v4?R.diag_offsets[di4]:0;
        uint32_t A[4]={0,0,0,0}, Lt[2]={0,0}, Lb[2]={0,0};
        if(v0){ int c=tile_row+g;    if(c>=0&&c<R.cols)A[0]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                c=tile_row+g+8;       if(c>=0&&c<R.cols)A[1]=symtc::to_tf32(R.values[(size_t)di0*R.cols+c]);
                int xi=tile_row+g-o0; if(xi>=0&&xi<x_size)Lt[0]=symtc::to_tf32(x[xi]);
                xi=tile_row+g+8-o0;   if(xi>=0&&xi<x_size)Lb[0]=symtc::to_tf32(x[xi]); }
        if(v4){ int c=tile_row+g;    if(c>=0&&c<R.cols)A[2]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                c=tile_row+g+8;       if(c>=0&&c<R.cols)A[3]=symtc::to_tf32(R.values[(size_t)di4*R.cols+c]);
                int xi=tile_row+g-o4; if(xi>=0&&xi<x_size)Lt[1]=symtc::to_tf32(x[xi]);
                xi=tile_row+g+8-o4;   if(xi>=0&&xi<x_size)Lb[1]=symtc::to_tf32(x[xi]); }
        symtc::mma(acc_top,A,Lt); symtc::mma(acc_bot,A,Lb);
    }
    if(g==2*t){ int rt=tile_row+g,rb=tile_row+8+g; if(rt<R.rows)y[rt]=acc_top[0]; if(rb<R.rows)y[rb]=acc_bot[2]; }
    else if(g==2*t+1){ int rt=tile_row+g,rb=tile_row+8+g; if(rt<R.rows)y[rt]=acc_top[1]; if(rb<R.rows)y[rb]=acc_bot[3]; }
}

inline void launch_tc_spmv_symztile(ReconView R,const int* dPtrU,const int* dDiU,
                                    const int* dPtrL,const int* dDiL,
                                    const float* dx,int x_size,float* dy,cudaStream_t s=0){
    const int nt=(R.rows+MMA_M-1)/MMA_M; if(!nt)return; const int nb=(nt+3)/4;
    tc_spmv_symztile_kernel<<<nb,4*32,0,s>>>(R,dPtrU,dDiU,dPtrL,dDiL,dx,x_size,dy);
}
