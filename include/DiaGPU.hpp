#pragma once
#include <vector>
#include <unordered_map>

using DiagMapF32 = std::unordered_map<int, std::vector<float>>;

// Length of a diagonal offset o (unpadded)
static inline int diag_len(int N, int o) {
    int s = (o >= 0 ? o : -o);
    int L = N - s;
    return L > 0 ? L : 0;
}

// GPU SpMSpM with compaction: C = A * B (B can be reused across steps)
// Drops any output diagonal whose absolute-sum <= eps (set eps=0 for exact zeros)
void multiply_sparse_noPad(const DiagListF32& A,
                           const DiagListF32& B,
                           float eps,
                           DiagListF32& C_out);

// Power by repeated right-multiply with static right factor B=A:
// C_1=A; for k=2..power: C_k = C_{k-1} * A; returns C_power.
// Uses multiply_sparse_noPad internally (so it also prunes zero diagonals).
DiagListF32 power_repeat_right(const DiagListF32& A, int power, float eps=0.0f);
