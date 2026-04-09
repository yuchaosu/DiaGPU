/* diag_cluster_kernel — H100 thread-block-cluster diagonal SpMM.
 *
 * Cluster of CLUSTER_SIZE CTAs shares one A smem region (loaded by rank-0,
 * read by all via DSMEM).  Each CTA loads its own B partition independently.
 * A global memory traffic reduced by CLUSTER_SIZE× vs diag_hybrid_kernel.
 *
 * Requires: CUDA 12+, sm_90, cooperative_groups. */
#pragma once

#include "diag_hybrid_kernel.cuh"   // HybridCDiag, PartBMeta, DiagMatrix
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int CLUSTER_SIZE           =  8;  // CTAs per cluster (H100 GPC limit)
constexpr int CLUSTER_DIAGS_PER_CTA  =  8;  // K: C diagonals per CTA
constexpr int CLUSTER_TILE           = 128; // positions per tile = threads/block
constexpr int CLUSTER_BLOCK          = 128; // threads per block
constexpr int CLUSTER_BLOCKS_PER_SM  =  3;  // occupancy (smem larger than hybrid)
constexpr int CLUSTER_PARTITION_SIZE = 53;  // A diagonals per smem partition
constexpr int CLUSTER_A_UNROLL       =  4;  // A diagonals unrolled per accumulate step

/* Per-cluster metadata: same tile position, shared A set.
 * cluster_meta[cluster_abs_idx] where cluster_abs_idx = blockIdx.x / CLUSTER_SIZE */
struct ClusterMeta {
    int tile_p_begin;   // shared output tile position for all CTAs in cluster
    int a_begin;        // first index into a_contrib[] for this cluster's A set
    int a_count;        // |A| for this cluster (union across all CLUSTER_SIZE groups)
    int min_c_sr_all;   // min start-row across ALL K*CLUSTER_SIZE C diags in cluster
                        // rank-0 loads A starting here; others offset by (c_sr - min_c_sr_all)
    int spread_all;     // max_c_sr_all - min_c_sr_all (determines A chunk width)
    int max_c_len_all;  // longest C diagonal in cluster (tile loop upper bound)
};

/* Per-CTA metadata: one K-group of C diagonals, independent B partition info.
 * tasks[blockIdx.x] where blockIdx.x = cluster_abs_idx * CLUSTER_SIZE + rank */
struct ClusterTask {
    int cluster_idx;    // index into cluster_meta[]
    int c_begin;        // first index into c_diags[] for this CTA's K group
    int c_count;        // number of C diagonals (<= CLUSTER_DIAGS_PER_CTA)
    int min_c_sr;       // min start-row for this CTA's K diags (for bounds)
    int spread;         // max_c_sr - min_c_sr for this CTA's K diags
    int max_c_len;      // longest C diagonal in this CTA's group
    int min_c_sc;       // min start-col for this CTA's K diags (B indexing)
    int spread_sc;      // max_c_sc - min_c_sc (determines B chunk width)
    int b_begin;        // first index into b_contrib[] for this CTA's B set
    int part_b_base;    // first index into part_b_meta[] for (this CTA, partition 0)
                        // part_b_meta[part_b_base + p_idx] = PartBMeta for A-partition p
};

struct ClusterKernelArgs {
    const ClusterMeta*  cluster_meta;
    int                 n_clusters;
    int                 max_smem;      // dynamic smem bytes per block
    const ClusterTask*  tasks;         // n_clusters * CLUSTER_SIZE entries
    const HybridCDiag*  c_diags;
    int                 n_c_diags;
    const int*          a_contrib;     // flat: cluster-wide A diagonal indices
    const int*          b_contrib;     // flat: per-CTA B diagonal indices
    const PartBMeta*    part_b_meta;   // indexed by task.part_b_base + p_idx
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

struct ClusterPlan {
    std::vector<HybridCDiag>  c_diags;
    std::vector<ClusterMeta>  cluster_meta;
    std::vector<ClusterTask>  tasks;          // cluster_meta.size() * CLUSTER_SIZE entries
    std::vector<int>          a_contrib;      // cluster-wide A sets, concatenated
    std::vector<int>          b_contrib;      // per-CTA B sets, concatenated
    std::vector<PartBMeta>    part_b_meta;    // per (CTA-group, A-partition)
    int total_c_values;
    int max_smem;
};

ClusterPlan build_cluster_plan(const DiagMatrix& A, const DiagMatrix& B,
                                int M, int K, int N);

__global__ void __launch_bounds__(CLUSTER_BLOCK, CLUSTER_BLOCKS_PER_SM)
cluster_kernel(ClusterKernelArgs args);

void launch_cluster(ClusterKernelArgs args, cudaStream_t stream = 0);
