/* mtx_to_dia — MatrixMarket coordinate (real, general|symmetric) -> compact
 * DIA text ("N n D d" + ascending "offset: v v ..." dense-diagonal lines),
 * the exact format sim/dia_io.hpp::load_dia expects.  Absent elements inside
 * a stored diagonal become explicit zeros (dense-diagonal contract).
 * Prints the diagonal census to stderr for screening.
 *   usage: mtx_to_dia in.mtx out.txt          g++ -O2 -o mtx_to_dia mtx_to_dia.cpp */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>
#include <string>

int main(int argc, char** argv){
    if (argc < 3){ fprintf(stderr, "usage: %s in.mtx out.txt\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "r");
    if (!f){ perror(argv[1]); return 1; }
    char line[512];
    if (!fgets(line, sizeof line, f)) return 1;
    const bool sym = strstr(line, "symmetric") != nullptr;
    if (!strstr(line, "coordinate") || !strstr(line, "real")){
        fprintf(stderr, "only coordinate real supported: %s", line); return 1;
    }
    long n = 0, m = 0, nnz = 0;
    while (fgets(line, sizeof line, f))
        if (line[0] != '%'){ sscanf(line, "%ld %ld %ld", &n, &m, &nnz); break; }
    if (n != m){ fprintf(stderr, "not square (%ld x %ld)\n", n, m); return 1; }

    std::map<int, std::vector<float>> diag;   // offset -> dense values (len n-|off|)
    auto put = [&](long r, long c, double v){
        int off = (int)(c - r);
        auto& vec = diag[off];
        if (vec.empty()) vec.assign(n - std::labs(off), 0.f);
        vec[off >= 0 ? r : c] = (float)v;     // p = min(r, c)
    };
    long r, c; double v; long cnt = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%ld %ld %lf", &r, &c, &v) != 3) continue;
        --r; --c; ++cnt;
        put(r, c, v);
        if (sym && r != c) put(c, r, v);
    }
    fclose(f);
    if (cnt != nnz) fprintf(stderr, "warn: header nnz=%ld, read %ld entries\n", nnz, cnt);

    FILE* o = fopen(argv[2], "w");
    if (!o){ perror(argv[2]); return 1; }
    fprintf(o, "N %ld D %zu\n", n, diag.size());
    size_t stored = 0;
    for (auto& kv : diag) {                   // std::map -> ascending offsets
        fprintf(o, "%d:", kv.first);
        for (float x : kv.second) fprintf(o, " %.9g", (double)x);
        fprintf(o, "\n");
        stored += kv.second.size();
    }
    fclose(o);
    fprintf(stderr, "%s: n=%ld D=%zu stored=%zu true=%ld fill=%.3f sym_in=%d\n",
            argv[2], n, diag.size(), stored, sym ? 2*cnt - n : cnt,
            (double)(sym ? 2*cnt - n : cnt) / (double)stored, (int)sym);
    return 0;
}
