#include "DiaGPU.hpp"
#include <unordered_map>
#include <vector>
#include <map>
#include <set>
#include <cstdio>
#include <cmath>
#include <algorithm>

static inline int diag_len(int N, int o) { int a=(o>=0?o:-o); int L=N-a; return L>0?L:0; }
static inline int s_of(int o){ return (o>=0?0:-o); }
static inline int e_of(int N,int o){ return (o>=0?N-o:N); }

static void dump_map(const char* name, int N, const DiagMapF32& M) {
    // print in sorted offset order for readability
    std::map<int,const std::vector<float>*> sorted;
    for (auto &kv : M) sorted[kv.first] = &kv.second;
    printf("%s (N=%d): K=%zu\n", name, N, sorted.size());
    for (auto &kv : sorted) {
        printf("  %d: [", kv.first);
        const auto &v = *kv.second;
        for (size_t i=0;i<v.size();++i) printf("%g%s", v[i], (i+1<v.size()? ", ":""));
        printf("]\n");
    }
}

// --- CPU map-based multiply & power for verification ---
static void multiply_sparse_noPad_cpu(int N,
                                      const DiagMapF32& A,
                                      const DiagMapF32& B,
                                      float eps,
                                      DiagMapF32& C)
{
    C.clear();
    // accumulate per q in a temporary std::map to keep order stable
    std::map<int, std::vector<float>> acc;

    for (auto &a : A) {
        int oA = a.first; const auto &vA = a.second;
        for (auto &b : B) {
            int oB = b.first; const auto &vB = b.second;
            int q = oA + oB; int Lq = diag_len(N, q);
            if (Lq <= 0) continue;

            int i_min = s_of(oA);
            i_min = std::max(i_min, s_of(q));
            i_min = std::max(i_min, s_of(oB) - oA);

            int i_max = e_of(N, oA);
            i_max = std::min(i_max, e_of(N, q));
            i_max = std::min(i_max, e_of(N, oB) - oA);

            if (i_min >= i_max) continue;

            auto &vq = acc[q];
            if ((int)vq.size() != Lq) vq.assign(Lq, 0.0f);

            for (int i = i_min; i < i_max; ++i) {
                int t  = i - s_of(q);
                int tA = i - s_of(oA);
                int tB = (i + oA) - s_of(oB);
                vq[t] += vA[tA] * vB[tB];
            }
        }
    }

    // prune all-zero diagonals
    for (auto it = acc.begin(); it != acc.end(); ) {
        float s = 0.0f; for (float x : it->second) s += std::fabs(x);
        if ((eps==0.0f ? s==0.0f : s<=eps)) it = acc.erase(it);
        else { ++it; }
    }
    C.clear();
    for (auto &kv : acc) C.emplace(kv.first, std::move(kv.second));
}

static DiagMapF32 power_repeat_right_cpu(int N, const DiagMapF32& A, int power, float eps) {
    if (power <= 1) return A;
    DiagMapF32 C = A;
    for (int k=2;k<=power;++k){ DiagMapF32 nxt; multiply_sparse_noPad_cpu(N, C, A, eps, nxt); C.swap(nxt); }
    return C;
}

struct VerifyReport {
    float max_abs_diff = 0.f;
    float rel_fro_error = 0.f;
    bool offsets_equal = true;
    int missing_in_ref = 0;
    int missing_in_gpu = 0;
};

static VerifyReport compare_maps(int N, const DiagMapF32& G, const DiagMapF32& R, float tol) {
    VerifyReport rep;
    std::set<int> offs;
    for (auto &kv : G) offs.insert(kv.first);
    for (auto &kv : R) offs.insert(kv.first);
    rep.offsets_equal = (offs.size() == G.size() && offs.size() == R.size());

    double diff_sq = 0.0, ref_sq = 0.0;
    float  max_abs = 0.0f;

    for (int q : offs) {
        auto ig = G.find(q), ir = R.find(q);
        if (ig != G.end() && ir != R.end()) {
            const auto &vg = ig->second, &vr = ir->second;
            int L = diag_len(N, q);
            for (int t=0;t<L;++t){
                float d = vg[t] - vr[t];
                diff_sq += double(d)*d;
                ref_sq  += double(vr[t])*vr[t];
                max_abs = std::max(max_abs, std::fabs(d));
            }
        } else if (ig != G.end()) {
            const auto &vg = ig->second;
            double loc_sq = 0.0; float sabs = 0.0f;
            for (float g : vg){ loc_sq += double(g)*g; sabs += std::fabs(g); max_abs = std::max(max_abs, std::fabs(g)); }
            if (sabs > tol) { diff_sq += loc_sq; rep.missing_in_ref++; }
        } else {
            const auto &vr = ir->second;
            double loc_sq = 0.0; float sabs = 0.0f;
            for (float r : vr){ loc_sq += double(r)*r; ref_sq += double(r)*r; sabs += std::fabs(r); max_abs = std::max(max_abs, std::fabs(r)); }
            if (sabs > tol) { diff_sq += loc_sq; rep.missing_in_gpu++; }
        }
    }
    rep.max_abs_diff = max_abs;
    rep.rel_fro_error = (ref_sq>0.0) ? float(std::sqrt(diff_sq/ref_sq)) : float(std::sqrt(diff_sq));
    return rep;
}

int main() {
    int N = 5, power = 3; float eps = 0.0f;

    // Build A in the new map format
    DiagMapF32 A = {
        {  0, {1,2,3,4,5} },
        { -1, {1,2,3,4}   }
    };

    dump_map("A", N, A);

    // GPU power (map API)
    DiagMapF32 A_gpu = power_repeat_right(N, A, power, eps);
    dump_map("A^GPU", N, A_gpu);

    // CPU reference
    DiagMapF32 A_cpu = power_repeat_right_cpu(N, A, power, eps);
    dump_map("A^CPU", N, A_cpu);

    // Verify
    float tol = 1e-6f;
    auto rep = compare_maps(N, A_gpu, A_cpu, tol);
    printf("\n[VERIFY] offsets_equal=%s, missing_in_ref=%d, missing_in_gpu=%d\n",
           rep.offsets_equal ? "true" : "false", rep.missing_in_ref, rep.missing_in_gpu);
    printf("[VERIFY] max_abs_diff = %.6g, rel_fro_error = %.6g\n",
           rep.max_abs_diff, rep.rel_fro_error);
    bool ok = rep.offsets_equal && rep.missing_in_ref==0 && rep.missing_in_gpu==0 &&
              (rep.max_abs_diff <= tol || rep.rel_fro_error <= tol);
    printf("[VERIFY] %s\n", ok ? "✅ PASS" : "❌ FAIL");
    return ok ? 0 : 1;
}
