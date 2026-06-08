/* ============================================================
 * gen_banded_mtx.cpp
 *
 * Emit the SAME banded matrix used by bench_cusparse.cu as a
 * Matrix Market coordinate file, so external baselines (Drawloom,
 * DASP) can be run on the identical structure.
 *
 *   usage: gen_banded_mtx N out.mtx
 *   offsets: {-9,-5,-2,-1,0,1,2,5,9}  (matches the benchmark)
 *
 * Build: g++ -O2 -std=c++17 gen_banded_mtx.cpp -o gen_banded_mtx
 * ============================================================ */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static inline float hval(int r, int d)
{
    unsigned h = (unsigned)r * 2654435761u ^ (unsigned)(d + 1024) * 40503u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (float)(h & 0xFFFF) / 65536.0f * 2.0f - 1.0f;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: %s N out.mtx\n", argv[0]); return 1; }
    const int N = std::atoi(argv[1]);
    std::vector<int> offs = {-9, -5, -2, -1, 0, 1, 2, 5, 9};
    std::sort(offs.begin(), offs.end());

    /* count nnz */
    long nnz = 0;
    for (int r = 0; r < N; ++r)
        for (int d : offs) { int c = r + d; if (c >= 0 && c < N) ++nnz; }

    FILE* f = std::fopen(argv[2], "w");
    if (!f) { std::perror("fopen"); return 1; }
    std::fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    std::fprintf(f, "%% banded, offsets {-9,-5,-2,-1,0,1,2,5,9}\n");
    std::fprintf(f, "%d %d %ld\n", N, N, nnz);
    for (int r = 0; r < N; ++r)
        for (int d : offs) {
            int c = r + d;
            if (c >= 0 && c < N)
                std::fprintf(f, "%d %d %.7g\n", r + 1, c + 1, (double)hval(r, d));
        }
    std::fclose(f);
    std::fprintf(stderr, "wrote %s : %d x %d, nnz=%ld\n", argv[2], N, N, nnz);
    return 0;
}
