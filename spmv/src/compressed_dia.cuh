#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include "../../sim/dia_io.hpp"

// ============================================================================
// Compressed-DIA: remove the "inside zeros" of Pauli-Hamiltonian diagonals.
//
// Measured facts (sim/cuda_vs_tc/NOTES.md):
//   - most diagonals have a SINGLE distinct |value| -> store 1 magnitude, not nnz
//   - nonzero positions are a fixed (often periodic) pattern -> store a bitmap;
//     where periodic with power-of-2 period P, store just one period.
//
// Per diagonal:
//   STRUCTURED (single |value|): magnitude + period mask (P-1) + posbits[P] +
//                                signbits[P]   (value = +/- magnitude)
//   DENSE      (multi-valued)  : the raw len floats
//
// Reconstruction is bit-exact (value is literally +/- magnitude; dense is verbatim).
// Stored index along diagonal d for row i is pos = (d>=0)? i : i+d; the only bound
// check needed is 0 <= c < n with c = i+d  (pos in [0,len) follows automatically).
// ============================================================================

struct CompressedDiaHost {
    int n = 0, ndiag = 0;
    std::vector<int>      off;        // [ndiag] signed offset
    std::vector<uint8_t>  is_dense;   // [ndiag] 1 = dense fallback
    std::vector<float>    mag;        // [ndiag] magnitude (structured)
    std::vector<int>      pmask;      // [ndiag] P-1, P a power of 2 (structured)
    std::vector<int>      wbase;      // [ndiag] word offset into posbits/signbits (structured)
    std::vector<int>      dbase;      // [ndiag] float offset into dense pool (dense)
    std::vector<uint32_t> posbits;    // packed nonzero indicators
    std::vector<uint32_t> signbits;   // packed sign (1 = negative)
    std::vector<float>    dense;       // raw values for dense diagonals
    // bookkeeping
    size_t struct_bytes() const {
        return posbits.size()*4 + signbits.size()*4 + dense.size()*4
             + (size_t)ndiag*(4+1+4+4+4);
    }
};

static inline int next_pow2(int x){ int p=1; while(p<x) p<<=1; return p; }

// signed code per slot: 0 zero, 1 +, -1 -  ; find smallest power-of-2 period
static int signed_period(const std::vector<int>& code){
    const int L=(int)code.size();
    for(int P=1; P<L; P<<=1){
        bool ok=true;
        for(int j=0;j<L && ok;++j) if(code[j]!=code[j%P]) ok=false;
        if(ok) return P;
    }
    return next_pow2(L);
}

static CompressedDiaHost build_compressed(const DiaHost& H){
    CompressedDiaHost C; C.n=H.n; C.ndiag=(int)H.offsets.size();
    for(int k=0;k<C.ndiag;++k){
        const int d=H.offsets[k], len=H.lengths[k];
        const float* v=&H.values[H.starts[k]];
        // classify: are all nonzero |values| identical?
        float m=0.f; bool single=true; int nnz=0;
        for(int j=0;j<len;++j){ if(v[j]!=0.f){ ++nnz; float a=std::fabs(v[j]);
            if(m==0.f) m=a; else if(a!=m){ single=false; break; } } }
        C.off.push_back(d);
        if(!single || nnz==0){                       // dense fallback (also for all-zero, rare)
            C.is_dense.push_back(1); C.mag.push_back(0.f); C.pmask.push_back(0);
            C.wbase.push_back(0); C.dbase.push_back((int)C.dense.size());
            C.dense.insert(C.dense.end(), v, v+len);
            continue;
        }
        // structured: signed code -> period -> pack one period
        std::vector<int> code(len);
        for(int j=0;j<len;++j) code[j] = (v[j]==0.f)?0:(v[j]<0.f?-1:1);
        int P=signed_period(code);
        int words=(P+31)/32, wb=(int)C.posbits.size();
        C.posbits.resize(wb+words,0u); C.signbits.resize(wb+words,0u);
        for(int j=0;j<P;++j){
            int c = code[j % len];   // for non-periodic P>=len padded region maps via %len harmlessly
            if(c!=0){ C.posbits[wb + (j>>5)] |= (1u<<(j&31));
                      if(c<0) C.signbits[wb + (j>>5)] |= (1u<<(j&31)); }
        }
        C.is_dense.push_back(0); C.mag.push_back(m); C.pmask.push_back(P-1);
        C.wbase.push_back(wb); C.dbase.push_back(0);
    }
    return C;
}

// ---- device view ----
struct CompressedDiaView {
    int n, ndiag;
    const int*      off;
    const uint8_t*  is_dense;
    const float*    mag;
    const int*      pmask;
    const int*      wbase;
    const int*      dbase;
    const uint32_t* posbits;
    const uint32_t* signbits;
    const float*    dense;
};

// one thread per row; decode compressed diagonals, fp32 accumulate.
__global__ void comp_spmv(CompressedDiaView C, const float* __restrict__ x, float* __restrict__ y){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=C.n) return;
    const int nd=C.ndiag, n=C.n;
    float acc=0.f;
    for(int k=0;k<nd;++k){
        const int d=C.off[k]; const int c=i+d;
        if(c<0||c>=n) continue;
        const int pos=(d>=0)?i:c;                 // stored index along diagonal
        if(C.is_dense[k]){
            acc += C.dense[C.dbase[k]+pos]*x[c];
        } else {
            const int idx = pos & C.pmask[k];
            const int w   = C.wbase[k] + (idx>>5);
            const uint32_t bit = 1u<<(idx&31);
            if(C.posbits[w]&bit){
                float v = C.mag[k];
                if(C.signbits[w]&bit) v=-v;
                acc += v*x[c];
            }
        }
    }
    y[i]=acc;
}
