/* ============================================================
 * dia_io.hpp
 *
 * Loader for the diagonal-format Hamiltonian files produced by
 * fetchham_sparse_dia.py / fetchham_dia.py:
 *
 *     Line 1:           N <n> D <num_diagonals>
 *     Per diagonal:     <offset>: v0 v1 v2 ...   (length n-|offset|)
 *
 * The value at index p of diagonal `offset` is the matrix entry at
 * (row, col) with p = min(row, col):
 *     offset >= 0 :  row = p,        col = p + offset
 *     offset <  0 :  row = p - offset, col = p
 *
 * H here is REAL (the Python side takes .real). The complex part of
 * the time evolution lives in the Taylor coefficients, not in H, so
 * everything stored here is float.
 *
 * Provides: load a file into compact diagonal arrays, and convert to
 * CSR (ascending columns per row) for cuSPARSE.
 * ============================================================ */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

struct DiaHost {
    int                 n = 0;
    std::vector<int>    offsets;   // signed, ascending after load
    std::vector<int>    lengths;   // n - |offset|
    std::vector<size_t> starts;    // start index of each diagonal in `values`
    std::vector<float>  values;    // concatenated, position p = min(row,col)
    size_t              nnz = 0;
};

// Read whole file into a heap buffer (caller frees). Returns size in *out_sz.
static char* slurp(const char* path, size_t* out_sz) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    char* buf = (char*)std::malloc((size_t)sz + 1);
    size_t rd = std::fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    std::fclose(f);
    *out_sz = rd;
    return buf;
}

// Load a diagonal-format file. Parses with strtof advancing a cursor
// (fast enough for the ~80 MB q=20 files; this read time is itself one
// of the profiled phases in the driver).
static DiaHost load_dia(const char* path) {
    size_t sz = 0;
    char* buf = slurp(path, &sz);
    char* p = buf;

    DiaHost H;
    // header: "N <n> D <d>"
    // (skip the leading "N ")
    while (*p && *p != ' ') ++p;            // past 'N'
    H.n = (int)std::strtol(p, &p, 10);
    while (*p && *p != 'D') ++p;            // to 'D'
    ++p;                                    // past 'D'
    int d = (int)std::strtol(p, &p, 10);
    while (*p && *p != '\n') ++p;
    if (*p == '\n') ++p;

    struct Tmp { int off; std::vector<float> vals; };
    std::vector<Tmp> tmp;
    tmp.reserve(d);

    for (int i = 0; i < d; ++i) {
        // "<offset>: v0 v1 ..."
        int off = (int)std::strtol(p, &p, 10);
        if (*p == ':') ++p;
        int len = H.n - std::abs(off);
        Tmp t; t.off = off; t.vals.resize(len);
        for (int j = 0; j < len; ++j) t.vals[j] = std::strtof(p, &p);
        while (*p == ' ' || *p == '\r') ++p;
        if (*p == '\n') ++p;
        tmp.push_back(std::move(t));
    }
    std::free(buf);

    // sort diagonals by ascending offset (CSR wants ascending columns/row)
    std::sort(tmp.begin(), tmp.end(),
              [](const Tmp& a, const Tmp& b){ return a.off < b.off; });

    // DIA_TRIM=<tau>: relative magnitude trim at load (ablation tier A0,
    // 2026-08-01). Zeroes |v| < tau*max|H| and drops emptied diagonals; a
    // matrix with nothing below tau loads bit-identically. Every consumer
    // (ours AND the baselines via dia_to_csr true-nonzero) sees the same
    // trimmed operator, so the comparison stays level.
    if (const char* ts = std::getenv("DIA_TRIM")) {
        const double tau = std::atof(ts);
        if (tau > 0) {
            float vmax = 0.f;
            for (auto& t : tmp) for (float v : t.vals) vmax = std::max(vmax, std::fabs(v));
            const float thr = (float)(tau * vmax);
            std::vector<Tmp> kept;
            for (auto& t : tmp) {
                bool any = false;
                for (auto& v : t.vals) { if (std::fabs(v) < thr) v = 0.f; else any = true; }
                if (any) kept.push_back(std::move(t));
            }
            if (kept.size() != tmp.size())
                std::fprintf(stderr, "# DIA_TRIM %g: D %zu -> %zu\n", tau, tmp.size(), kept.size());
            tmp = std::move(kept);
        }
    }

    size_t off = 0;
    for (auto& t : tmp) {
        H.offsets.push_back(t.off);
        H.lengths.push_back((int)t.vals.size());
        H.starts.push_back(off);
        H.values.insert(H.values.end(), t.vals.begin(), t.vals.end());
        off += t.vals.size();
    }
    H.nnz = off;
    return H;
}

// ---- Hermitian "store half" ----------------------------------------------
// For real-symmetric H (H=H^T, the real part of a Hermitian Hamiltonian), the
// -d diagonal is element-wise IDENTICAL to the +d diagonal (both indexed by
// p=min(row,col)):  (-d)[p] = H[p+d,p] = H[p,p+d] = (+d)[p].  So storing only
// offsets >= 0 loses nothing; the lower triangle is rebuilt by mirroring.
static void save_dia_half(const DiaHost& H, const char* path) {
    FILE* f = std::fopen(path, "w");
    int nd = 0; for (int o : H.offsets) if (o >= 0) ++nd;
    std::fprintf(f, "N %d D %d\n", H.n, nd);
    for (size_t i = 0; i < H.offsets.size(); ++i) {
        if (H.offsets[i] < 0) continue;
        std::fprintf(f, "%d:", H.offsets[i]);
        for (int j = 0; j < H.lengths[i]; ++j)
            std::fprintf(f, " %.15g", H.values[H.starts[i] + j]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

// Load a half file (offsets >= 0) and rebuild the FULL symmetric matrix by
// mirroring each +d diagonal to -d (same values). Output offsets ascending.
static DiaHost load_dia_half_rebuild(const char* path) {
    DiaHost h = load_dia(path);          // all offsets >= 0
    struct T { int off; std::vector<float> v; };
    std::vector<T> ts;
    for (size_t i = 0; i < h.offsets.size(); ++i) {
        int o = h.offsets[i], len = h.lengths[i];
        std::vector<float> v(h.values.begin()+h.starts[i], h.values.begin()+h.starts[i]+len);
        ts.push_back({o, v});
        if (o > 0) ts.push_back({-o, v});      // mirror (identical values)
    }
    std::sort(ts.begin(), ts.end(), [](const T&a,const T&b){ return a.off < b.off; });
    DiaHost F; F.n = h.n; size_t off = 0;
    for (auto& t : ts) {
        F.offsets.push_back(t.off); F.lengths.push_back((int)t.v.size()); F.starts.push_back(off);
        F.values.insert(F.values.end(), t.v.begin(), t.v.end()); off += t.v.size();
    }
    F.nnz = off; return F;
}

struct CsrHost {
    int                  n = 0;
    int64_t              nnz = 0;
    std::vector<int>     row_ptr;     // n+1  (exact only while nnz <= INT32_MAX)
    std::vector<int64_t> row_ptr64;   // n+1  (always exact; needed once nnz > INT32_MAX)
    std::vector<int>     col_idx;      // nnz  (individual col values are < n, so int is safe)
    std::vector<float>   vals;         // nnz
};

// Convert compact diagonals to CSR. Offsets must be ascending (load_dia
// guarantees this), so each row's columns come out ascending too.
// drop_zeros (default): textbook CSR = true nonzeros only.  Passing false
// keeps the diagonals' interior zeros as explicit entries — that was the
// (unfair-to-CSR-baselines) behavior of every run before 2026-08-01: it
// inflates baseline work by 1/fill and moves the int32 wall to STORED nnz.
static CsrHost dia_to_csr(const DiaHost& H, bool drop_zeros = true) {
    const int n = H.n;
    CsrHost csr; csr.n = n;
    csr.row_ptr.assign(n + 1, 0);
    csr.row_ptr64.assign(n + 1, 0);
    csr.col_idx.reserve(H.nnz);
    csr.vals.reserve(H.nnz);
    const int nd = (int)H.offsets.size();
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < nd; ++k) {
            const int d = H.offsets[k];
            const int col = i + d;
            if (col < 0 || col >= n) continue;
            const int pos = (d >= 0) ? i : (i + d);   // p = min(row,col)
            const float v = H.values[H.starts[k] + pos];
            if (drop_zeros && v == 0.f) continue;
            csr.col_idx.push_back(col);
            csr.vals.push_back(v);
        }
        csr.row_ptr64[i + 1] = (int64_t)csr.col_idx.size();
        csr.row_ptr[i + 1]   = (int)csr.col_idx.size();   // exact while nnz <= INT32_MAX
    }
    csr.nnz = (int64_t)csr.col_idx.size();
    return csr;
}
