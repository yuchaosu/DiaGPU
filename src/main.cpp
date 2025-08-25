#include "DiaGPU.hpp"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <cstdio>
#include <cmath>

using DiagMapF32 = std::unordered_map<int, std::vector<float>>;

static inline int start_row_h(int offset){ return (offset>=0?0:-offset); }
static inline int end_row_h(int N,int offset){ return (offset>=0?N-offset:N); }


// ---- map -> list (so you can author matrices easily) ----
static DiagListF32 map_to_list(int N, const DiagMapF32& M) {
    DiagListF32 L; L.N = N;
    L.offsets.reserve(M.size());
    for (auto &kv : M) L.offsets.push_back(kv.first);
    std::sort(L.offsets.begin(), L.offsets.end());
    L.ptr.assign(L.offsets.size() + 1, 0);
    for (size_t d=0; d<L.offsets.size(); ++d) L.ptr[d+1] = L.ptr[d] + diag_len(N, L.offsets[d]);
    L.vals.assign(L.ptr.back(), 0.0f);
    for (size_t d=0; d<L.offsets.size(); ++d) {
        int offset = L.offsets[d];
        const auto &v = M.at(offset);
        int need = diag_len(N, offset);
        if ((int)v.size() != need) {
            fprintf(stderr, "[map_to_list] length mismatch at offset=%d: need %d, got %zu\n", offset, need, v.size());
            std::exit(1);
        }
        std::copy(v.begin(), v.end(), L.vals.begin() + L.ptr[d]);
    }
    //print
    dump_list("map_to_list", L);
    return L;
}

// ---- CPU reference multiply (list) for verification ----
// ---- CPU reference multiply (list) for verification ----
static void multiply_sparse_noPad_cpu(const DiagListF32& A,
                                      const DiagListF32& B,
                                      float eps,
                                      DiagListF32& C)
{
    const int N = A.N;
    auto diag_len_h = [](int N, int offset){ int a=(offset>=0?offset:-offset); int L=N-a; return L>0?L:0; };

    if (A.offsets.empty() || B.offsets.empty()) { C = {N, {}, {0}, {}}; return; }

    // ---- Build C offsets and contributing pairs (host) ----
    std::map<int, std::vector<std::pair<int,int>>> buckets; // key: offsetC (sorted)
    int bmin = B.offsets.front(), bmax = B.offsets.back();
    std::vector<int> Bindex(bmax-bmin+1, -1);
    for (int i=0;i<(int)B.offsets.size();++i) Bindex[B.offsets[i]-bmin] = i;

    for (int dA=0; dA<(int)A.offsets.size(); ++dA) {
        int offsetA = A.offsets[dA];
        int clamp_min = std::max(offsetA + B.offsets.front(), -(N-1));
        int clamp_max = std::min(offsetA + B.offsets.back(),   (N-1));
        for (int offsetC = clamp_min; offsetC <= clamp_max; ++offsetC) {
            int offsetB = offsetC - offsetA;
            if (offsetB < bmin || offsetB > bmax) continue;
            int dB = Bindex[offsetB - bmin]; if (dB < 0) continue;
            if (diag_len_h(N, offsetC) <= 0) continue;
            buckets[offsetC].emplace_back(dA, dB);
        }
    }

    std::vector<int> offsetsC, lenC, cPairPtr, cPairA, cPairB;
    cPairPtr.push_back(0);
    for (auto &kv : buckets) {
        offsetsC.push_back(kv.first);
        lenC.push_back(diag_len_h(N, kv.first));
        for (auto &pr : kv.second) {           // append ALL pairs for this offsetC
            cPairA.push_back(pr.first);
            cPairB.push_back(pr.second);
        }
        cPairPtr.push_back((int)cPairA.size()); // <-- push ONCE per diagonal
    }

    // ---- Numeric accumulate ----
    std::vector<int> ptrC(offsetsC.size()+1,0);
    for (size_t i=0;i<offsetsC.size();++i) ptrC[i+1] = ptrC[i] + lenC[i];
    std::vector<float> valsC(ptrC.back(), 0.0f), absSumC(offsetsC.size(), 0.0f);

    auto start_row_h = [](int offset){ return (offset>=0?0:-offset); };
    auto end_row_h   = [](int N,int offset){ return (offset>=0?N-offset:N); };

    for (size_t cIdx=0; cIdx<offsetsC.size(); ++cIdx) {
        int offsetC = offsetsC[cIdx];
        for (int p=cPairPtr[cIdx]; p<cPairPtr[cIdx+1]; ++p) {
            int dA = cPairA[p], dB = cPairB[p];
            int offsetA = A.offsets[dA], offsetB = B.offsets[dB];

            int i_min = start_row_h(offsetA);
            i_min = std::max(i_min, start_row_h(offsetC));
            i_min = std::max(i_min, start_row_h(offsetB) - offsetA);

            int i_max = end_row_h(N, offsetA);
            i_max = std::min(i_max, end_row_h(N, offsetC));
            i_max = std::min(i_max, end_row_h(N, offsetB) - offsetA);
            if (i_min >= i_max) continue;

            for (int i=i_min; i<i_max; ++i) {
                int t  = i - start_row_h(offsetC);
                int tA = i - start_row_h(offsetA);
                int tB = (i + offsetA) - start_row_h(offsetB);
                valsC[ptrC[cIdx] + t] += A.vals[A.ptr[dA] + tA] * B.vals[B.ptr[dB] + tB];
            }
        }
        for (int t=0; t<lenC[cIdx]; ++t) absSumC[cIdx] += std::fabs(valsC[ptrC[cIdx] + t]);
    }

    // ---- Simple host compaction (drop all-zero diagonals) ----
    C.N = N;
    C.offsets.clear();
    C.ptr.clear();
    C.vals.clear();
    C.ptr.push_back(0);

    for (size_t c=0; c<offsetsC.size(); ++c) {
        float s = absSumC[c];
        bool keep = (eps==0.0f) ? (s != 0.0f) : (std::fabs(s) > eps);
        if (!keep) continue;

        C.offsets.push_back(offsetsC[c]);
        int L  = lenC[c];
        int p0 = ptrC[c];
        C.vals.insert(C.vals.end(), valsC.begin()+p0, valsC.begin()+p0+L);
        C.ptr.push_back(C.ptr.back() + L);
    }
}


static DiagListF32 power_repeat_right_cpu(const DiagListF32& A, int power, float eps){
    if (power<=1) return A;
    DiagListF32 C=A, next;
    for (int k=2;k<=power;++k){ multiply_sparse_noPad_cpu(C, A, eps, next); C = std::move(next); }
    return C;
}

static void compare_and_report(const DiagListF32& G, const DiagListF32& R, float tol){
    std::map<int,int> idxG, idxR;
    for (int i=0;i<(int)G.offsets.size();++i) idxG[G.offsets[i]]=i;
    for (int i=0;i<(int)R.offsets.size();++i) idxR[R.offsets[i]]=i;
    std::set<int> offs; for (auto&o:G.offsets) offs.insert(o); for (auto&o:R.offsets) offs.insert(o);

    double diff_sq=0.0, ref_sq=0.0; float max_abs=0.f; int missG=0, missR=0;
    for (int offset:offs){
        auto ig=idxG.find(offset), ir=idxR.find(offset);
        if (ig!=idxG.end() && ir!=idxR.end()){
            int dG=ig->second, dR=ir->second, L=G.ptr[dG+1]-G.ptr[dG];
            for (int t=0;t<L;++t){ float g=G.vals[G.ptr[dG]+t], r=R.vals[R.ptr[dR]+t], d=g-r;
                diff_sq+=double(d)*d; ref_sq+=double(r)*r; max_abs=std::max(max_abs,std::fabs(d)); }
        } else if (ig!=idxG.end()){
            int dG=ig->second, L=G.ptr[dG+1]-G.ptr[dG]; float sabs=0; double loc=0;
            for (int t=0;t<L;++t){ float g=G.vals[G.ptr[dG]+t]; sabs+=std::fabs(g); loc+=double(g)*g; max_abs=std::max(max_abs,std::fabs(g)); }
            if (sabs>tol){ diff_sq+=loc; missR++; }
        } else {
            int dR=ir->second, L=R.ptr[dR+1]-R.ptr[dR]; float sabs=0; double loc=0;
            for (int t=0;t<L;++t){ float r=R.vals[R.ptr[dR]+t]; sabs+=std::fabs(r); loc+=double(r)*r; ref_sq+=double(r)*r; max_abs=std::max(max_abs,std::fabs(r)); }
            if (sabs>tol){ diff_sq+=loc; missG++; }
        }
    }
    float rel = (ref_sq>0.0)? float(std::sqrt(diff_sq/ref_sq)) : float(std::sqrt(diff_sq));
    printf("\n[VERIFY] max_abs_diff=%.6g, rel_fro=%.6g, missing_in_gpu=%d, missing_in_ref=%d\n",
           max_abs, rel, missG, missR);
}

int main(){
    int N = 5, power = 3; float eps = 0.0f;

    // Author in map form
    DiagMapF32 Amap = {
        {  1, std::vector<float>{1,2,3,4} },
        {  0, std::vector<float>{1,2,3,4,5} },
        { -1, std::vector<float>{1,2,3,4}   }
    };
    // Convert to list for the GPU API
    DiagListF32 A = map_to_list(N, Amap);

    // GPU power
    DiagListF32 A_gpu = power_repeat_right(A, power, eps);
    dump_list("A^GPU", A_gpu);

    // CPU reference
    DiagListF32 A_cpu = power_repeat_right_cpu(A, power, eps);
    dump_list("A^CPU", A_cpu);

    compare_and_report(A_gpu, A_cpu, 1e-6f);
    return 0;
}
