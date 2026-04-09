/* diag_hybrid_kernel — C-centric diagonal SpMM (C = A × B in DIA format).
 *
 * Each group of K consecutive C diagonals is split into TILE-position tasks.
 * Each CTA owns one exclusive tile [tile_p_begin, +TILE); no write conflicts.
 * A streams through smem in batches; B is loaded into smem once per task.
 *
 * Tuned for H100: 228 KB smem/SM, target 4 blocks/SM → 57 KB/block. */
#pragma once

#include "diag_types.cuh"
#include "diag_host_preprocess.cuh"
#include <cuda_runtime.h>
#include <vector>

constexpr int HYBRID_TILE          = 128;   // positions per tile = threads/block
constexpr int HYBRID_BLOCK         = 128;   // threads per block
constexpr int HYBRID_DIAGS_PER_CTA =   8;   // C diagonals per group (K)
constexpr int HYBRID_BLOCKS_PER_SM =   4;   // target occupancy

/* smem budget per block: 228 KB / 4 blocks per SM = 57 KB. */
constexpr int HYBRID_SMEM_BUDGET = 57 * 1024;
constexpr int HYBRID_A_UNROLL   = 4;   // A diags unrolled per accumulate step

struct HybridCDiag {
    int c_offset;       // dC
    int c_sr;           // start row: max(0, -dC)
    int length;         // number of elements
    int values_start;   // offset into C_vals[]
};

/* One task = one CTA owns one exclusive output position tile.
 * Each group of K consecutive C diagonals is split into n_tiles tasks,
 * one per [tile_p_begin, tile_p_begin + TILE) output range.
 * A is streamed through smem in batches of a_smem_cap diagonals.
 * B is loaded into smem once per task and reused across all A batches. */
struct HybridTask {
    int c_begin;       // first index into c_diags[]
    int c_count;       // C diags in this group (≤ HYBRID_DIAGS_PER_CTA)
    int min_c_sr;      // min c_sr across group
    int spread;        // max_c_sr - min_c_sr
    int max_c_len;     // max C diagonal length in group
    int a_begin;       // index into a_contrib[]
    int a_count;       // total A diagonals for this group
    int a_smem_cap;    // max A diagonals per smem batch
    /* B smem */
    int min_c_sc;      // min start column across C diagonals
    int spread_sc;     // max_c_sc - min_c_sc
    int b_begin;       // index into b_contrib[]
    int b_count;       // unique B diagonals needed
    int b_d_min;       // min d_b among staged B diagonals
    int b_d_range;     // b_d_max - b_d_min + 1
    /* Exclusive output tile */
    int tile_p_begin;  // starting position for this CTA's output tile
};

struct HybridKernelArgs {
    const HybridTask*  tasks;
    int                n_tasks;
    int                max_smem;   // dynamic smem per block

    /* Output diagonal metadata */
    const HybridCDiag* c_diags;
    int                n_c_diags;

    /* Flat contributor index arrays */
    const int*         a_contrib;
    const int*         b_contrib;

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

    /* Output */
    float* C_vals;
};

struct HybridPlan {
    std::vector<HybridCDiag>  c_diags;
    std::vector<HybridTask>   tasks;
    std::vector<int>          a_contrib;
    std::vector<int>          b_contrib;
    int total_c_values;
    int max_smem;   // max dynamic smem across all tasks
};

HybridPlan build_hybrid_plan(const DiagMatrix& A,
                              const DiagMatrix& B,
                              int M, int K, int N);

__global__ void __launch_bounds__(HYBRID_BLOCK, HYBRID_BLOCKS_PER_SM)
hybrid_kernel(HybridKernelArgs args);

void launch_hybrid(HybridKernelArgs args, cudaStream_t stream = 0);
