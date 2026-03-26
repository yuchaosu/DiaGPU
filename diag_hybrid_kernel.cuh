/* ============================================================
 * diag_hybrid_kernel.cuh
 *
 * Hybrid two-stage diagonal SpMM kernel.
 *
 * Splits output diagonals into two categories:
 *
 *   CORNER (few contributing pairs, <= HYBRID_CORNER_THRESH):
 *     One CTA owns one (C diagonal, segment).
 *     Iterates all (dA, dB) pairs, stages A/B tiles through
 *     shared memory, accumulates in registers, writes directly.
 *
 *   HEAVY (many contributing pairs, > HYBRID_CORNER_THRESH):
 *     Stage 1 — One CTA owns one (C diagonal, segment, pair-subset).
 *                Loads HYBRID_PAIRS_PER_PART A+B tile pairs into
 *                shared memory, accumulates partial sums, writes
 *                to an exclusive slot in partial_buf.
 *     Stage 2 — One CTA owns one (C diagonal, segment).
 *                Scalar-reduces all partial slots for that
 *                segment and writes final output to C.
 *
 * All three kernels are PERSISTENT: a fixed occupancy-saturated
 * grid (num_SMs × target_blocks_per_SM) is launched once, and
 * each CTA iterates its task slice via a grid-stride loop.
 * This keeps kernel launch overhead at O(1) regardless of the
 * number of tasks (corner, s1, or s2).
 *
 * Design invariants:
 *   ZERO atomic operations
 *   One CTA == exclusive output region (no write conflicts)
 *   Shared memory staging for coalesced A/B loads
 *   Register accumulation, single final writeback
 * ============================================================ */
#pragma once

#include "diag_types.cuh"     // DiagMatrix, OutputDiagInfo, etc.
#include "diag_host_preprocess.cuh"
#include <cuda_runtime.h>
#include <vector>

/* ============================================================
 * Compile-time configuration
 * ============================================================ */
constexpr int HYBRID_TILE_CORNER    = 128;   // segment length, corner kernel
constexpr int HYBRID_BLOCK_CORNER   = 128;   // block size, corner kernel
constexpr int HYBRID_TILE_HEAVY     = 256;   // segment length, heavy kernels
constexpr int HYBRID_BLOCK_HEAVY_S1 = 256;   // block size, stage-1 kernel
constexpr int HYBRID_BLOCK_HEAVY_S2 = 256;   // block size, stage-2 kernel
constexpr int HYBRID_PAIRS_PER_PART =   8;   // pairs per stage-1 CTA
constexpr int HYBRID_CORNER_THRESH  =  16;   // <= this → corner, else heavy

/* ============================================================
 * Device-side data structures
 * ============================================================ */

/* One (ai, bi) contributor pair for a heavy output diagonal. */
struct HybridPair {
    int ai;   // index into A diagonal list
    int bi;   // index into B diagonal list
};

/* Metadata for one output diagonal. */
struct HybridCDiag {
    int c_offset;       // dC
    int c_sr;           // start row: max(0, -dC)
    int length;         // number of elements
    int values_start;   // offset into C_vals[]
};

/* Corner task: CTA owns (c_diag, segment). */
struct HybridCornerTask {
    int c_idx;          // index into c_diags[]
    int c_offset;       // dC
    int p_begin;        // segment start position
    int p_len;          // segment length (<= HYBRID_TILE_CORNER)
    int ai_begin;       // valid A-diagonal range (precomputed on host)
    int ai_end;
};

/* Heavy stage-1 task: CTA owns (c_diag, segment, pair partition). */
struct HybridS1Task {
    int c_idx;
    int c_offset;
    int p_begin;
    int p_len;          // <= HYBRID_TILE_HEAVY
    int pair_begin;     // index into global pairs[] array
    int pair_count;     // number of pairs to process (<= HYBRID_PAIRS_PER_PART)
    int partial_offset; // element offset into partial_buf[] for this slot
    int s2_task_idx;    // which s2 task this partition belongs to
                        // (used by fused kernel to signal readiness)
};

/* Heavy stage-2 task: CTA owns (c_diag, segment) and reduces partials. */
struct HybridS2Task {
    int c_idx;
    int p_begin;
    int p_len;
    int partial_offset; // start of partial slots for this (c_diag, segment)
    int num_partials;   // number of partial slots to sum
};

/* ============================================================
 * FusedKernelCtrl — device-side control state for the pipelined
 * fused kernel.  Allocated once on the device and reused across
 * calls (reset by the host before each launch).
 *
 * Layout:
 *   [0]          : global s1 task counter  (atomicAdd)
 *   [1]          : global s2 claim counter (atomicAdd)
 *   [2 .. 2+n_s2): pending[i] = number of s1 partitions still
 *                  outstanding for s2 task i.  Initialised by
 *                  the host to num_partitions[i] before launch.
 * ============================================================ */
struct FusedKernelCtrl {
    int s1_next;             // next s1 task to claim
    int s2_claim;            // next s2 task to claim (in readiness order)
    // pending[] follows in the same allocation: ctrl + 2 + n_s2 ints
};

/* ============================================================
 * KernelArgs — passed by value to all kernels (lands in
 * constant memory for fast broadcast).
 * ============================================================ */
struct HybridKernelArgs {
    /* Corner tasks */
    const HybridCornerTask* corner_tasks;
    int                     n_corner;

    /* Heavy stage-1 tasks */
    const HybridS1Task*     s1_tasks;
    int                     n_s1;

    /* Heavy stage-2 tasks */
    const HybridS2Task*     s2_tasks;
    int                     n_s2;

    /* Output diagonal metadata */
    const HybridCDiag*      c_diags;

    /* A matrix (sorted by offset, required for binary-search range) */
    const float* A_vals;
    const int*   A_offsets;      // sorted ascending
    const int*   A_starts;       // A_vals base index per diagonal
    const int*   A_lengths;
    int          A_num_diags;

    /* B matrix */
    const float* B_vals;
    const int*   B_offsets;      // B_offsets[bi] = d_b (needed for b_sr)
    const int*   B_starts;
    const int*   B_lengths;
    const int*   B_lookup;       // size 2n-1, maps d_b+(n-1) → bi, or -1
    int          n;
    int          B_offset_min;
    int          B_offset_max;

    /* Pair list (for stage-1 tasks) */
    const HybridPair* pairs;

    /* Buffers */
    float* C_vals;
    float* partial_buf;           // layout: see build_hybrid_plan()
};

/* ============================================================
 * Host-side plan (result of build_hybrid_plan)
 * ============================================================ */
struct HybridPlan {
    std::vector<HybridCDiag>      c_diags;
    std::vector<HybridCornerTask> corner_tasks;
    std::vector<HybridS1Task>     s1_tasks;
    std::vector<HybridS2Task>     s2_tasks;
    std::vector<HybridPair>       pairs;      // global pair list for heavy tasks
    int total_c_values;                       // total floats needed for C_vals
    int partial_buf_size;                     // total floats needed for partial_buf
};

/* ============================================================
 * Host-side preprocessing
 * ============================================================ */

/* Build the full hybrid plan from two diagonal matrices.
 * A must have its offsets sorted ascending (call
 * sort_diag_matrix_by_offset from diag_host_preprocess.cuh). */
HybridPlan build_hybrid_plan(const DiagMatrix& A,
                              const DiagMatrix& B,
                              int M, int K, int N);

/* ============================================================
 * Kernel declarations
 * ============================================================ */

__global__ void __launch_bounds__(HYBRID_BLOCK_CORNER, 8)
hybrid_corner_kernel(HybridKernelArgs args);

__global__ void __launch_bounds__(HYBRID_BLOCK_HEAVY_S1, 4)
hybrid_heavy_s1_kernel(HybridKernelArgs args);

__global__ void __launch_bounds__(HYBRID_BLOCK_HEAVY_S2, 4)
hybrid_heavy_s2_reduce_kernel(HybridKernelArgs args);

/* Pipelined fused kernel: s1 and s2 overlap within a single launch.
 * CTAs interleave s1 work and s2 readiness checks so that s2 for
 * segment S starts the moment all s1 partitions for S have committed
 * their partial sums — without waiting for other segments' s1 work.
 *
 * Requires ctrl to be allocated on the device (see launch_hybrid_pipelined).
 * ctrl layout: [s1_next | s2_claim | pending[0] ... pending[n_s2-1]]
 * The host must initialise ctrl before each launch via
 * init_fused_ctrl(). */
__global__ void __launch_bounds__(HYBRID_BLOCK_HEAVY_S1, 4)
hybrid_heavy_fused_kernel(HybridKernelArgs args, int* ctrl);

/* ============================================================
 * Host-side launch wrappers
 * ============================================================ */

/* Two-kernel sequential launch (3 kernels total with corner). */
void launch_hybrid(HybridKernelArgs args, cudaStream_t stream = 0);

/* Pipelined launch: corner kernel + one fused s1/s2 kernel.
 * ctrl must be a device allocation of size
 *   (2 + args.n_s2) * sizeof(int)
 * initialised by init_fused_ctrl() before each call.         */
void launch_hybrid_pipelined(HybridKernelArgs args,
                              int*             ctrl,
                              cudaStream_t     stream = 0);

/* Reset ctrl on the device (async, runs on stream). */
void init_fused_ctrl(HybridKernelArgs args,
                     int*             ctrl,
                     cudaStream_t     stream = 0);
