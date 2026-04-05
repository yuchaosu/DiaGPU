/* ============================================================
 * diag_hybrid_kernel.cuh
 *
 * Unified C-centric diagonal SpMM kernel (C = A × B in DIA format).
 *
 * Each CTA owns a GROUP of K consecutive output C diagonals.
 * It tiles by position (p_begin), preloads all contributing
 * A diagonal chunks into shared memory, then accumulates with
 * B from registers.  Each thread holds K independent accumulators.
 *
 * Corner vs Heavy split (smem capacity based):
 *
 *   CORNER group — all contributing A diagonals fit in smem
 *     in one pass.  CTA writes acc[] directly to C_vals.
 *
 *   HEAVY group — too many A diagonals for smem.  A contributors
 *     are split into partitions.  Each partition is a separate
 *     task that runs the SAME computation kernel and writes to
 *     partial_buf.  A reduction kernel sums partials into C_vals.
 *
 * Design invariants:
 *   ZERO atomic operations in computation kernel
 *   One CTA == exclusive output region (no write conflicts)
 *   A staged in shared memory, B in registers
 *   K local accumulators per thread, single final writeback
 *
 * Tuned for H100:
 *   228 KB smem/SM, 65536 regs/SM, 132 SMs.
 *   Target 4 blocks/SM → 57 KB smem/block, 128 regs/thread.
 * ============================================================ */
#pragma once

#include "diag_types.cuh"
#include "diag_host_preprocess.cuh"
#include <cuda_runtime.h>
#include <vector>

/* ============================================================
 * Compile-time configuration (H100 tuned)
 * ============================================================ */
constexpr int HYBRID_TILE          = 128;   // positions per tile = threads/block
constexpr int HYBRID_BLOCK         = 128;   // threads per block
constexpr int HYBRID_DIAGS_PER_CTA =   8;   // C diagonals per group (K)
constexpr int HYBRID_BLOCKS_PER_SM =   4;   // target occupancy

/* smem budget per block: 228 KB / 4 = 57 KB.
 * Double buffering halves effective budget: 57 / 2 = 28.5 KB per buffer.
 * Chunk padded to 4-float alignment for float4 loads:
 *   raw = TILE + K - 1 = 135, padded -> 136.
 * Max A diags per partition = 28.5 KB / (136 * 4) = 52.           */
constexpr int HYBRID_SMEM_BUDGET   = 57 * 1024;  // total bytes per block
constexpr int HYBRID_MAX_CHUNK_RAW = HYBRID_TILE + HYBRID_DIAGS_PER_CTA - 1;
constexpr int HYBRID_MAX_CHUNK     = (HYBRID_MAX_CHUNK_RAW + 3) & ~3;  // 136
constexpr int HYBRID_A_UNROLL      = 4;   // A diags processed simultaneously for B ILP
/* Each buffer holds a_count * chunk floats; two buffers needed. */
constexpr int HYBRID_MAX_A_PER_PART =
    HYBRID_SMEM_BUDGET / (2 * HYBRID_MAX_CHUNK * static_cast<int>(sizeof(float)));

/* ============================================================
 * Device-side data structures
 * ============================================================ */

/* Metadata for one output diagonal (unchanged). */
struct HybridCDiag {
    int c_offset;       // dC
    int c_sr;           // start row: max(0, -dC)
    int length;         // number of elements
    int values_start;   // offset into C_vals[]
};

/* Unified task: one (group, A-partition) assignment for a CTA.
 *
 * Corner group  → 1 task,  out_offset = c_vals offset (direct write)
 * Heavy partition → P tasks, out_offset = partial_buf offset
 *
 * The computation kernel is identical for both; only the output
 * pointer differs (C_vals vs partial_buf), selected by is_direct. */
struct HybridTask {
    int c_begin;       // first index into c_diags[]
    int c_count;       // C diags in this group (≤ HYBRID_DIAGS_PER_CTA)
    int min_c_sr;      // min c_sr across group
    int spread;        // max_c_sr - min_c_sr (≤ K-1)
    int max_c_len;     // max C diagonal length in group (for tile loop bound)
    int a_begin;       // index into a_contrib[]
    int a_count;       // number of A diags in this partition
    int is_direct;     // 1 = write to C_vals, 0 = write to partial_buf
    int out_offset;    // base offset into partial_buf (heavy only)
};

/* Reduction task: sums partial_buf partitions into C_vals.
 * One per (group, tile position) for heavy groups. */
struct HybridReduceTask {
    int c_begin;       // first index into c_diags[]
    int c_count;       // C diags in group
    int min_c_sr;      // min c_sr across group
    int spread;        // max_c_sr - min_c_sr
    int max_c_len;     // for tile loop bound
    int partial_base;  // start in partial_buf for this group
    int num_partials;  // number of partitions to sum
};

/* ============================================================
 * KernelArgs — passed by value to all kernels.
 * ============================================================ */
struct HybridKernelArgs {
    /* Corner tasks (DIRECT=true kernel) */
    const HybridTask*       tasks;      // used by kernel (set per launch)
    int                     n_tasks;    // used by kernel (set per launch)
    const HybridTask*       corner_tasks;
    int                     n_corner;
    /* Heavy tasks (DIRECT=false kernel) */
    const HybridTask*       heavy_tasks;
    int                     n_heavy;
    /* Smem sizes for each launch */
    int                     corner_max_smem;
    int                     heavy_max_smem;

    /* Reduce tasks */
    const HybridReduceTask* reduce_tasks;
    int                     n_reduce;

    /* Output diagonal metadata */
    const HybridCDiag*      c_diags;
    int                     n_c_diags;

    /* Contributing A diagonal indices per task (flat array) */
    const int*              a_contrib;

    /* A matrix (sorted by offset) */
    const float* A_vals;
    const int*   A_offsets;
    const int*   A_starts;
    const int*   A_lengths;
    int          A_num_diags;

    /* B matrix */
    const float* B_vals;
    const int*   B_offsets;
    const int*   B_starts;
    const int*   B_lengths;
    const int*   B_lookup;       // extended: size 4n-3, maps d_b+b_lookup_base → bi or -1
    int          b_lookup_base;  // = 2*(n-1), so any d_b in [-(2n-2), 2n-2] is safe
    int          n;

    /* Buffers */
    float* C_vals;
    float* partial_buf;
};

/* ============================================================
 * Host-side plan
 * ============================================================ */
struct HybridPlan {
    std::vector<HybridCDiag>       c_diags;
    std::vector<HybridTask>        corner_tasks; // direct-write tasks
    std::vector<HybridTask>        heavy_tasks;  // partial_buf tasks
    std::vector<HybridReduceTask>  reduce_tasks; // heavy groups only
    std::vector<int>               a_contrib;    // flat A-diag indices per task
    int total_c_values;
    int partial_buf_size;
    int corner_max_smem;  // max dynamic smem across corner tasks
    int heavy_max_smem;   // max dynamic smem across heavy tasks
};

/* ============================================================
 * Host-side preprocessing
 * ============================================================ */
HybridPlan build_hybrid_plan(const DiagMatrix& A,
                              const DiagMatrix& B,
                              int M, int K, int N);

/* ============================================================
 * Kernel declarations
 * ============================================================ */

/* Computation kernel — template-specialized for corner (direct write)
 * and heavy (partial_buf write).  No runtime branch in Phase 3.
 * Dynamic shared memory: 2 * task.a_count * chunk * sizeof(float). */
template <bool DIRECT>
__global__ void __launch_bounds__(HYBRID_BLOCK, HYBRID_BLOCKS_PER_SM)
hybrid_compute_kernel(HybridKernelArgs args);

/* Reduction kernel — sums partial_buf → C_vals for heavy groups. */
__global__ void __launch_bounds__(HYBRID_BLOCK, HYBRID_BLOCKS_PER_SM)
hybrid_reduce_kernel(HybridKernelArgs args);

/* ============================================================
 * Host-side launch wrapper
 * ============================================================ */
void launch_hybrid(HybridKernelArgs args, cudaStream_t stream = 0);
