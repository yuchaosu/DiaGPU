/* ============================================================
 * paper_hm.cuh
 *
 * Host-side types and helpers for the HM (Hossain–Mahmud)
 * diagonal storage scheme, plus kernel declaration.
 *
 * Extracted from paper_algorithms.cu for use as a library.
 * ============================================================ */
#pragma once

#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include <cuda_runtime.h>

/* ============================================================
 * HMMatrix: Hossain–Mahmud Diagonal Storage (Section 3.1)
 *
 * Diagonal offset d:
 *   d > 0: superdiagonal of length n - d
 *   d = 0: main diagonal of length n
 *   d < 0: subdiagonal of length n - |d|
 *
 * Element at (i, j) with d = j - i:
 *   position within diagonal = min(i, j)
 * ============================================================ */
struct HMMatrix {
    int n;
    int num_diags;
    std::vector<int>   diag_offsets;   /* DsHM: diagonal offsets */
    std::vector<int>   diag_starts;    /* StartDg: start index in values[] */
    std::vector<int>   diag_lengths;   /* length of each diagonal */
    std::vector<float> values;         /* flat concatenation */
    int                total_nz;       /* total nonzeros */
};

/* Build an HM matrix from a dense matrix, given a list of
 * diagonal offsets that are nonzero. */
inline HMMatrix
dense_to_hm(const float* dense, int n, const std::vector<int>& offsets)
{
    HMMatrix hm;
    hm.n = n;
    hm.num_diags = static_cast<int>(offsets.size());
    hm.diag_offsets = offsets;

    int val_off = 0;
    for (int i = 0; i < hm.num_diags; ++i) {
        int d = offsets[i];
        int len = n - std::abs(d);
        hm.diag_starts.push_back(val_off);
        hm.diag_lengths.push_back(len);

        int sr = (d >= 0) ? 0 : -d;
        int sc = (d >= 0) ? d :  0;
        for (int p = 0; p < len; ++p) {
            hm.values.push_back(dense[(sr + p) * n + (sc + p)]);
        }
        val_off += len;
    }
    hm.total_nz = val_off;
    return hm;
}

/* Convert HM back to dense. */
inline void
hm_to_dense(const HMMatrix& hm, float* dense)
{
    int n = hm.n;
    memset(dense, 0, n * n * sizeof(float));
    for (int i = 0; i < hm.num_diags; ++i) {
        int d   = hm.diag_offsets[i];
        int sr  = (d >= 0) ? 0 : -d;
        int sc  = (d >= 0) ? d :  0;
        int len = hm.diag_lengths[i];
        int st  = hm.diag_starts[i];
        for (int p = 0; p < len; ++p) {
            dense[(sr + p) * n + (sc + p)] = hm.values[st + p];
        }
    }
}

/* Compute all possible C diagonal offsets from A and B,
 * and allocate the output HMMatrix structure. */
inline HMMatrix
compute_c_hm_structure(const HMMatrix& A, const HMMatrix& B, int n)
{
    std::vector<bool> present(2 * n - 1, false);
    for (int ai = 0; ai < A.num_diags; ++ai) {
        for (int bi = 0; bi < B.num_diags; ++bi) {
            int d_c = A.diag_offsets[ai] + B.diag_offsets[bi];
            if (d_c >= -(n - 1) && d_c <= (n - 1)) {
                present[d_c + (n - 1)] = true;
            }
        }
    }

    HMMatrix C;
    C.n = n;
    int val_off = 0;

    for (int d = -(n - 1); d <= (n - 1); ++d) {
        if (!present[d + (n - 1)]) continue;
        int len = n - std::abs(d);
        if (len <= 0) continue;
        C.diag_offsets.push_back(d);
        C.diag_starts.push_back(val_off);
        C.diag_lengths.push_back(len);
        val_off += len;
    }
    C.num_diags = static_cast<int>(C.diag_offsets.size());
    C.total_nz  = val_off;
    C.values.assign(val_off, 0.0f);
    return C;
}

/* Build C diagonal lookup: offset d -> index in C's diagonal list.
 * Array size = 2*n - 1, indexed as lookup[d + (n-1)]. */
inline std::vector<int>
build_c_diag_lookup(const HMMatrix& C, int n)
{
    std::vector<int> lookup(2 * n - 1, -1);
    for (int i = 0; i < C.num_diags; ++i) {
        int d = C.diag_offsets[i];
        lookup[d + (n - 1)] = i;
    }
    return lookup;
}

/* ============================================================
 * Kernel declaration (definition in paper_hm_kernel.cu)
 * ============================================================ */
__global__ void
hm_structured_sparse_matmul_kernel(
    const float* __restrict__ A_vals,
    const int*   __restrict__ A_offsets,
    const int*   __restrict__ A_starts,
    const int*   __restrict__ A_lengths,
    int                       A_num_diags,
    const float* __restrict__ B_vals,
    const int*   __restrict__ B_offsets,
    const int*   __restrict__ B_starts,
    const int*   __restrict__ B_lengths,
    int                       B_num_diags,
    float*       __restrict__ C_vals,
    const int*   __restrict__ C_offsets,
    const int*   __restrict__ C_starts,
    const int*   __restrict__ C_lengths,
    int                       C_num_diags,
    const int*   __restrict__ C_diag_lookup,
    int                       nzA,
    int                       n);
