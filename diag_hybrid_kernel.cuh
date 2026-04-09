/* diag_hybrid_kernel — C-centric diagonal SpMM (C = A × B in DIA format).
 *
 * Each group of K consecutive C diagonals is split into TILE-position tasks.
 * Each CTA owns one exclusive tile [tile_p_begin, +TILE); no write conflicts.
 * A streams through smem in batches; B is loaded into smem once per A-partition within each task.
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
constexpr int HYBRID_PARTITION_SIZE = 53;  // A (and ~B) diags per smem partition

constexpr int HYBRID_A_UNROLL   = 4;   // A diags unrolled per accumulate step

/* Per-partition B metadata.  Stored in a flat array on device;
 * task.part_b_base indexes into it. */
struct PartBMeta {
    int b_begin;    // offset into b_contrib[] relative to task.b_begin
    int b_count;    // B diagonals in this partition
    int b_d_min;    // min d_b among this partition's B contributors
    int b_d_range;  // b_d_max - b_d_min + 1  (lookup table width)
};

struct HybridCDiag {
    int c_offset;       // dC
    int c_sr;           // start row: max(0, -dC)
    int length;         // number of elements
    int values_start;   // offset into C_vals[]
};

struct HybridTask {
    int c_begin;
    int c_count;
    int min_c_sr;
    int spread;
    int max_c_len;
    int a_begin;
    int a_count;
    /* B smem — column side */
    int min_c_sc;
    int spread_sc;
    int b_begin;       // start of this group's b_contrib entries
    /* Partition metadata */
    int part_b_base;   // index into part_b_meta[]
    int n_parts;       // number of A partitions
    /* Output tile */
    int tile_p_begin;
};

struct HybridKernelArgs {
    const HybridTask*   tasks;
    int                 n_tasks;
    int                 max_smem;
    const HybridCDiag*  c_diags;
    int                 n_c_diags;
    const int*          a_contrib;
    const int*          b_contrib;
    const PartBMeta*    part_b_meta;
    const float*        A_vals;
    const int*          A_offsets;
    const int*          A_starts;
    const int*          A_lengths;
    int                 A_num_diags;
    const float*        B_vals;
    const int*          B_offsets;
    const int*          B_starts;
    const int*          B_lengths;
    float*              C_vals;
};

struct HybridPlan {
    std::vector<HybridCDiag>  c_diags;
    std::vector<HybridTask>   tasks;
    std::vector<int>          a_contrib;
    std::vector<int>          b_contrib;
    std::vector<PartBMeta>    part_b_meta;
    int total_c_values;
    int max_smem;
};

HybridPlan build_hybrid_plan(const DiagMatrix& A,
                              const DiagMatrix& B,
                              int M, int K, int N);

__global__ void __launch_bounds__(HYBRID_BLOCK, HYBRID_BLOCKS_PER_SM)
hybrid_kernel(HybridKernelArgs args);

void launch_hybrid(HybridKernelArgs args, cudaStream_t stream = 0);
