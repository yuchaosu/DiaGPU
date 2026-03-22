/* ============================================================
 * diag_types.cuh
 * Core data structures and compile-time constants for
 * diagonal sparse matrix multiplication (DiagSpMM).
 * ============================================================ */
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

/* ============================================================
 * Compile-time constants
 * ============================================================ */
constexpr int WARP_SIZE       = 32;

constexpr int BLOCK_SIZE_MED  = 128;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE_MED / WARP_SIZE;  // 4

constexpr int BLOCK_SIZE_LIGHT = 128;
constexpr int TASKS_PER_CTA_LIGHT = BLOCK_SIZE_LIGHT / WARP_SIZE; // 4

constexpr int BLOCK_SIZE_HEAVY = 256;
constexpr int WARPS_PER_BLOCK_HEAVY = BLOCK_SIZE_HEAVY / WARP_SIZE; // 8

constexpr int TILE_SIZE       = BLOCK_SIZE_MED;               // 128
constexpr int TILE_SIZE_LIGHT = WARP_SIZE;                    // 32
constexpr int TILE_SIZE_HEAVY = BLOCK_SIZE_HEAVY;             // 256

constexpr int WIDE_TILE_SIZE        = 512;
constexpr int WIDE_BLOCK_SIZE       = BLOCK_SIZE_MED;            // 128
constexpr int WIDE_ELEMS_PER_THREAD = WIDE_TILE_SIZE / WIDE_BLOCK_SIZE; // 4

constexpr int LIGHT_WORK_MAX  = 128;
constexpr int MEDIUM_WORK_MAX = 4096;

constexpr int ADAPTIVE_LARGE_DIAG_THRESH = 1024;
constexpr int ADAPTIVE_HUGE_DIAG_THRESH  = 4096;

/* ============================================================
 * DiagMatrix  (host-side, owning)
 * ============================================================ */
struct DiagMatrix {
    int rows;
    int cols;
    int num_diags;
    std::vector<int>   offsets;       // size = num_diags
    std::vector<float> values;        // concatenated diagonal values
    std::vector<int>   diag_starts;   // start index into values[]
    std::vector<int>   diag_lengths;  // length of each diagonal

    static inline int diag_start_row(int d) { return d >= 0 ? 0 : -d; }
    static inline int diag_start_col(int d) { return d >= 0 ? d :  0; }
    static inline int diag_length(int rows, int cols, int d) {
        int sr  = diag_start_row(d);
        int sc  = diag_start_col(d);
        int len = std::min(rows - sr, cols - sc);
        return len > 0 ? len : 0;
    }
};

/* ============================================================
 * DiagMatrixDev  (device-side, non-owning POD mirror)
 * ============================================================ */
struct DiagMatrixDev {
    int          rows;
    int          cols;
    int          num_diags;
    const int*   offsets;
    const float* values;
    const int*   diag_starts;
    const int*   diag_lengths;
};

/* ============================================================
 * Task
 *
 * One output tile: a contiguous segment of one output diagonal.
 * One CTA exclusively owns and writes back the tile.
 *
 * The kernel computes B indices on the fly from d_c and the
 * A/B diagonal metadata — no Group or PairMeta indirection.
 * ============================================================ */
struct Task {
    int c_diag_idx;     // index into OutputDiag table
    int c_offset;       // actual diagonal offset d_c
    int p_begin;        // first position in the output diagonal
    int p_len;          // tile length  (<= TILE_SIZE)
    int work_est;       // estimated work (sum of overlaps)
    int bucket;         // 0=LIGHT, 1=MEDIUM, 2=HEAVY, 3=WIDE
};

/* ============================================================
 * OutputDiag
 *
 * Describes one diagonal of the output matrix C.
 * ============================================================ */
struct OutputDiag {
    int offset;         // d_c
    int start_row;
    int start_col;
    int length;         // number of elements on this diagonal
    int values_start;   // offset into C_values[]
};

/* ============================================================
 * KernelArgs  (device-side, non-owning POD)
 *
 * Bundles everything the kernel needs.  Passed by value →
 * lands in constant memory (fast broadcast to all threads).
 *
 * The kernel iterates A diagonals and uses B_diag_lookup[]
 * (size 2n-1) to find the matching B diagonal for each
 * (d_c, d_a) pair.  All metadata arrays are tiny and fit
 * in L1 cache.
 * ============================================================ */
struct KernelArgs {
    /* Task scheduling */
    const Task*       tasks;
    const int*        task_ids;
    int               num_tasks;

    /* Output */
    const OutputDiag* c_diags;
    float*            C_values;

    /* A matrix (diagonal format, device pointers) */
    const float*      A_values;
    const int*        A_offsets;      // A.offsets[ai] = d_a
    const int*        A_starts;       // A.diag_starts[ai]
    const int*        A_lengths;      // A.diag_lengths[ai]
    int               A_num_diags;

    /* B matrix (diagonal format, device pointers) */
    const float*      B_values;
    const int*        B_starts;       // B.diag_starts[bi]
    const int*        B_lengths;      // B.diag_lengths[bi]
    int               B_num_diags;

    /* B diagonal lookup: B_diag_lookup[d_b + (n-1)] = bi or -1.
     * Size = 2n-1.  Fits in L1 cache (~8 KB for n=1024). */
    const int*        B_diag_lookup;
    int               n;

    /* B offset range — used to narrow the A-diagonal loop.
     * For output diagonal d_c, valid d_a is in
     *   [d_c - B_offset_max, d_c - B_offset_min].
     * Binary search A_offsets[] (sorted) for this range
     * to skip non-matching A diags entirely.              */
    int               B_offset_min;   // min(B.offsets[])
    int               B_offset_max;   // max(B.offsets[])
};

/* ============================================================
 * Host-only preprocessing scratch types (not uploaded to GPU)
 * ============================================================ */

struct ContributorPair {
    int a_diag_idx;
    int a_offset;        // d_a
    int b_diag_idx;
    int b_offset;        // d_b
    int valid_begin;     // global valid position range along c diagonal
    int valid_end;
};

struct OutputDiagInfo {
    int c_offset;        // d_c
    int c_start_row;
    int c_start_col;
    int c_length;
    std::vector<ContributorPair> contributors;
};
