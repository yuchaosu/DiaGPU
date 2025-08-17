#pragma once
#include <vector>
#include <cstdio>

struct DiagListF32 {
    int N = 0;                        // matrix size
    std::vector<int>   offsets;       // sorted offsets (size = number of stored diagonals)
    std::vector<int>   ptr;           // size = offsets.size() + 1; diagonal d => [ptr[d], ptr[d+1])
    std::vector<float> vals;          // concatenated values across all diagonals
};

inline int diag_len(int N, int offset) {
    int a = (offset >= 0 ? offset : -offset);
    int L = N - a;
    return L > 0 ? L : 0;
}

static void dump_list(const char* name, const DiagListF32& M) {
    printf("%s (N=%d): diagonals=%d\n", name, M.N, (int)M.offsets.size());
    for (size_t d=0; d<M.offsets.size(); ++d) {
        int offset = M.offsets[d];
        int L = M.ptr[d+1]-M.ptr[d];
        printf("  %d: [", offset);
        for (int t=0; t<L; ++t) {
            float x = M.vals[M.ptr[d]+t];
            printf("%g%s", x, (t+1<L? ", ":""));
        }
        printf("]\n");
    }
}

// C = A * B (drop any all-zero output diagonal by eps; eps=0 → drop exactly-zero)
void multiply_sparse_noPad(const DiagListF32& A,
                           const DiagListF32& B,
                           float eps,
                           DiagListF32& C_out);

// A^power using repeated right-multiply by the original A
DiagListF32 power_repeat_right(const DiagListF32& A, int power, float eps = 0.0f);
