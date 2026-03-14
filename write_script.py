/* ============================================================
 * diag_types.cuh
 * Core data structures and compile-time constants for
 * diagonal sparse matrix multiplication (DiagSpMM).
 *
 * Design principle:  all structs are POD so they can be
 * trivially copied to device memory.  Host-owning containers
 * live separately in the preprocessing layer.
 * ============================================================ */
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

/* ============================================================
 * Compile-time constants
 * ============================================================ */
constexpr int WARP_SIZE       = 32;

/* Medium kernel block configuration:
 *   128 threads = 4 warps
 *   Each warp owns 32 contiguous output positions.
 *   Warp 0 -> [0..31], Warp 1 -> [32..63], etc.            */
constexpr int BLOCK_SIZE_MED  = 128;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE_MED / WARP_SIZE;  // 4

/* Light kernel: same 128 threads, but packs 4 tasks per CTA.
 *   Each warp independently handles one task (32 output positions). */
constexpr int BLOCK_SIZE_LIGHT = 128;
constexpr int TASKS_PER_CTA_LIGHT = BLOCK_SIZE_LIGHT / WARP_SIZE; // 4

/* Heavy kernel: 256 threads = 8 warps for more compute throughput.
 *   Double-buffered smemA for overlapping load and compute.     */
constexpr int BLOCK_SIZE_HEAVY = 256;
constexpr int WARPS_PER_BLOCK_HEAVY = BLOCK_SIZE_HEAVY / WARP_SIZE; // 8

/* Output tile size = one element per thread in the medium kernel.
 * For the light kernel, each warp works on up to WARP_SIZE positions. */
constexpr int TILE_SIZE       = BLOCK_SIZE_MED;               // 128

/* Default tile size for light tasks (one warp = 32 positions). */
constexpr int TILE_SIZE_LIGHT = WARP_SIZE;                    // 32

/* Tile size for heavy kernel (256 positions, one per thread). */
constexpr int TILE_SIZE_HEAVY = BLOCK_SIZE_HEAVY;             // 256

/* ---- 15.5  Wide (multi-output) kernel --------------------------------
 *   128 threads, each thread owns 4 output positions.
 *   Tile covers 512 output elements (WIDE_TILE_SIZE).
 *   Inner B/smemA loop iterates in WIDE_BLOCK_SIZE-element chunks
 *   so each warp still reads 32 consecutive floats → coalesced.    */
constexpr int WIDE_TILE_SIZE        = 512;
constexpr int WIDE_BLOCK_SIZE       = BLOCK_SIZE_MED;            // 128
constexpr int WIDE_ELEMS_PER_THREAD = WIDE_TILE_SIZE / WIDE_BLOCK_SIZE; // 4

/* Bucket thresholds (tunable).
 *   work(task) = sum of overlap_len across all pairs in the task.
 *   LIGHT  : work <= LIGHT_WORK_MAX   -> multi-task CTA (15.1)
 *   MEDIUM : work <= MEDIUM_WORK_MAX  -> 4-warp CTA (original)
 *   HEAVY  : work >  MEDIUM_WORK_MAX  -> double-buffer CTA (15.2)
 *   WIDE   :  opt-in via build_all_adaptive  -> multi-output (15.5) */
constexpr int LIGHT_WORK_MAX  = 128;
constexpr int MEDIUM_WORK_MAX = 4096;

/* ---- 15.6  Adaptive tile-size thresholds ----------------------------
 *   If the maximum output diagonal length exceeds these values, the
 *   adaptive preprocessor (build_all_adaptive) upgrades the tile
 *   size to reduce task-launch overhead for large matrices.
 *     > ADAPTIVE_LARGE → use TILE_SIZE_HEAVY  (256)
 *     > ADAPTIVE_HUGE  → use WIDE_TILE_SIZE   (512)               */
constexpr int ADAPTIVE_LARGE_DIAG_THRESH = 1024;
constexpr int ADAPTIVE_HUGE_DIAG_THRESH  = 4096;

/* ============================================================
 * DiagMatrix  (host-side, owning)
 *
 * Stores a sparse matrix in diagonal (DIA) format.
 * Each non-zero diagonal is identified by its integer offset:
 *   offset > 0  -> super-diagonal
 *   offset = 0  -> main diagonal
 *   offset < 0  -> sub-diagonal
 *
 * For an M x N matrix with diagonal offset d:
 *   start_row = max(0, -d)
 *   start_col = max(0,  d)
 *   length    = min(M - start_row, N - start_col)
 *   Element p along the diagonal sits at
 *       (start_row + p,  start_col + p)  for p in [0, length).
 *
 * `values` is a flat concatenation of all diagonal arrays;
 * `diag_starts[i]` gives the offset into `values` where
 *  diagonal i begins, and `diag_lengths[i]` gives its length.
 * ============================================================ */
struct DiagMatrix {
    int rows;
    int cols;
    int num_diags;
    std::vector<int>   offsets;       // size = num_diags
    std::vector<float> values;        // concatenated diagonal values
    std::vector<int>   diag_starts;   // start index into values[]
    std::vector<int>   diag_lengths;  // length of each diagonal

    /* ---------- static helpers (usable without an instance) ---------- */
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
 *
 * Points into device global memory allocated by the host.
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
 * Represents a single output tile: a contiguous segment of
 * one output diagonal.  Exactly ONE CTA is responsible for
 * computing and writing back the entire tile.
 *
 *   Output positions covered:
 *     C_diag[c_offset], positions [p_begin, p_begin + p_len)
 *
 *   All contributor information is encoded in the range
 *     groups[ group_begin .. group_begin + group_count )
 * ============================================================ */
struct Task {
    int c_diag_idx;     // index into OutputDiag table
    int c_offset;       // actual diagonal offset d_c
    int p_begin;        // first position in the output diagonal
    int p_len;          // tile length  (<= TILE_SIZE)
    int group_begin;    // first index into Group table
    int group_count;    // number of groups
    int work_est;       // estimated work (sum of overlaps)
    int bucket;         // 0=LIGHT, 1=MEDIUM, 2=HEAVY, 3=WIDE
};

/* ============================================================
 * Group
 *
 * Within a Task, contributors are grouped by their A diagonal
 * (a_diag_idx).  All pairs in a group share the same A slice,
 * which is loaded into shared memory ONCE and reused across
 * every pair in the group  -- the "A-stationary" principle.
 *
 * The mapping from tile-local position q  (0 <= q < tile_len)
 * to the A diagonal index is:
 *     p_a = a_map_offset + q
 * Valid when  0 <= p_a < a_diag_len.
 * ============================================================ */
struct Group {
    int a_diag_idx;      // index into A's diagonal list
    int a_offset;        // diagonal offset d_a
    int a_global_start;  // = A.diag_starts[a_diag_idx]
    int a_diag_len;      // length of the A diagonal
    int a_map_offset;    // offset: tile-local pos q -> A diagonal pos
                         //   = c_start_row + task.p_begin - a_start_row
    int pair_begin;      // first index into PairMeta table
    int pair_count;      // number of (a,b) pairs in this group
};

/* ============================================================
 * PairMeta
 *
 * Metadata for one (a_diag, b_diag) contributor pair, already
 * resolved to a specific task / tile.
 *
 * out_valid_begin / out_valid_end give the tile-local range
 * where both A and B have valid data.  Outside this range the
 * zero-padded smemA / packedB guarantee a zero product, so
 * the kernel may skip the validity check for simplicity
 * (the padded zeros handle it automatically).
 *
 * packedB_offset points into the flat packedB array where
 * this pair's warp-major packed B data begins.
 * ============================================================ */
struct PairMeta {
    int b_diag_idx;       // index into B's diagonal list
    int b_offset;         // diagonal offset d_b
    int out_valid_begin;  // first valid tile-local position (inclusive)
    int out_valid_end;    // last  valid tile-local position (exclusive)
    int a_base;           // stored for diagnostics / debugging
    int b_base;           // b_map_offset (stored for diagnostics)
    int packedB_offset;   // offset into packedB[] for this pair's data
};

/* ============================================================
 * OutputDiag
 *
 * Describes one diagonal of the output matrix C.
 * values_start gives the offset into the flat C_values array.
 * ============================================================ */
struct OutputDiag {
    int offset;         // d_c
    int start_row;
    int start_col;
    int length;         // number of elements on this diagonal
    int values_start;   // offset into C_values[]
};

/* ============================================================
 * KernelParams  (device-side, non-owning)
 *
 * Bundles every pointer the kernel needs into one struct
 * for cleaner launch signatures.
 * ============================================================ */
struct KernelParams {
    const Task*       tasks;
    const Group*      groups;
    const PairMeta*   pairs;
    const float*      A_values;
    const float*      packedB;
    const OutputDiag* c_diags;
    float*            C_values;
    int               num_tasks;
};
