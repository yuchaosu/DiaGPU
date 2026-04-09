#include "diag_cluster_kernel.cuh"
#include "diag_host_preprocess.cuh"
#include <algorithm>
#include <climits>
#include <unordered_set>
#include <vector>

/* ── build_cluster_plan ─────────────────────────────────────────── */
ClusterPlan build_cluster_plan(const DiagMatrix& A, const DiagMatrix& B,
                                int M, int /*K_dim*/, int N)
{
    ClusterPlan plan;
    plan.total_c_values = 0;
    plan.max_smem       = 0;

    /* 1. Enumerate output C diagonals (same logic as build_hybrid_plan). */
    {
        std::unordered_set<int> c_offsets_seen;
        for (int ai = 0; ai < A.num_diags; ++ai)
            for (int bi = 0; bi < B.num_diags; ++bi)
                c_offsets_seen.insert(A.offsets[ai] + B.offsets[bi]);

        std::vector<int> c_offs(c_offsets_seen.begin(), c_offsets_seen.end());
        std::sort(c_offs.begin(), c_offs.end());

        for (int d_c : c_offs) {
            HybridCDiag cd;
            cd.c_offset    = d_c;
            cd.c_sr        = std::max(0, -d_c);
            const int c_sc = std::max(0,  d_c);
            cd.length      = std::min(M - cd.c_sr, N - c_sc);
            if (cd.length <= 0) continue;
            cd.values_start = plan.total_c_values;
            plan.total_c_values += cd.length;
            plan.c_diags.push_back(cd);
        }
    }

    const int n_c_diags        = static_cast<int>(plan.c_diags.size());
    const int super_group_size = CLUSTER_DIAGS_PER_CTA * CLUSTER_SIZE; // 64

    /* 2. Process super-groups of 64 consecutive C diagonals. */
    for (int sg_base = 0; sg_base < n_c_diags; sg_base += super_group_size) {
        const int sg_count = std::min(super_group_size, n_c_diags - sg_base);

        /* a. Cluster-wide geometry. */
        int min_c_sr_all = INT_MAX, max_c_sr_all = 0, max_c_len_all = 0;
        for (int i = 0; i < sg_count; ++i) {
            const auto& cd = plan.c_diags[sg_base + i];
            if (cd.c_sr < min_c_sr_all) min_c_sr_all = cd.c_sr;
            if (cd.c_sr > max_c_sr_all) max_c_sr_all = cd.c_sr;
            if (cd.length > max_c_len_all) max_c_len_all = cd.length;
        }
        const int spread_all = max_c_sr_all - min_c_sr_all;

        /* b. Collect cluster-wide A contributor set (union across all sg_count C diags). */
        std::vector<int> cluster_a_indices;
        {
            std::unordered_set<int> a_set;
            for (int i = 0; i < sg_count; ++i) {
                const int d_c = plan.c_diags[sg_base + i].c_offset;
                for (int bi = 0; bi < B.num_diags; ++bi) {
                    const int d_a_needed = d_c - B.offsets[bi];
                    for (int ai = 0; ai < A.num_diags; ++ai) {
                        if (A.offsets[ai] == d_a_needed) { a_set.insert(ai); break; }
                    }
                }
            }
            cluster_a_indices.assign(a_set.begin(), a_set.end());
        }
        /* Sort A by d_a ascending — guarantees contiguous B ranges per partition. */
        std::sort(cluster_a_indices.begin(), cluster_a_indices.end(),
                  [&](int a, int b){ return A.offsets[a] < A.offsets[b]; });

        const int total_a   = static_cast<int>(cluster_a_indices.size());
        if (total_a == 0) continue;

        const int a_base_sg = static_cast<int>(plan.a_contrib.size());
        for (int ai : cluster_a_indices) plan.a_contrib.push_back(ai);

        const int n_parts        = (total_a + CLUSTER_PARTITION_SIZE - 1) / CLUSTER_PARTITION_SIZE;
        const int part_b_base_sg = static_cast<int>(plan.part_b_meta.size());

        /* c. Per-rank geometry and B metadata.
         *    part_b_meta layout: [rank0_p0..rank0_pN, rank1_p0..rank1_pN, ...] */
        struct RankGeom {
            int min_c_sr, spread, max_c_len;
            int min_c_sc, spread_sc;
            int b_begin;   // global start of this rank's b_contrib entries
        };
        std::vector<RankGeom> rank_geom(CLUSTER_SIZE);

        for (int rank = 0; rank < CLUSTER_SIZE; ++rank) {
            const int g_base  = sg_base + rank * CLUSTER_DIAGS_PER_CTA;
            const int g_count = std::min(CLUSTER_DIAGS_PER_CTA, n_c_diags - g_base);

            if (g_count <= 0) {
                /* Pad with empty PartBMeta so part_b_base + p_idx arithmetic stays valid. */
                rank_geom[rank] = {0, 0, 0, 0, 0, static_cast<int>(plan.b_contrib.size())};
                for (int p = 0; p < n_parts; ++p)
                    plan.part_b_meta.push_back({0, 0, 0, 0});
                continue;
            }

            /* Group geometry. */
            int min_c_sr_g = INT_MAX, max_c_sr_g = 0;
            int min_c_sc_g = INT_MAX, max_c_sc_g = 0;
            int max_c_len_g = 0;
            int min_d_c = INT_MAX, max_d_c = INT_MIN;
            for (int ki = 0; ki < g_count; ++ki) {
                const auto& cd = plan.c_diags[g_base + ki];
                const int c_sc = std::max(0, cd.c_offset);
                if (cd.c_sr  < min_c_sr_g) min_c_sr_g = cd.c_sr;
                if (cd.c_sr  > max_c_sr_g) max_c_sr_g = cd.c_sr;
                if (c_sc     < min_c_sc_g) min_c_sc_g = c_sc;
                if (c_sc     > max_c_sc_g) max_c_sc_g = c_sc;
                if (cd.length > max_c_len_g) max_c_len_g = cd.length;
                if (cd.c_offset < min_d_c) min_d_c = cd.c_offset;
                if (cd.c_offset > max_d_c) max_d_c = cd.c_offset;
            }

            rank_geom[rank] = {
                min_c_sr_g, max_c_sr_g - min_c_sr_g, max_c_len_g,
                min_c_sc_g, max_c_sc_g - min_c_sc_g,
                static_cast<int>(plan.b_contrib.size())
            };

            const int b_base_rank = rank_geom[rank].b_begin;

            /* Per-A-partition B metadata for this rank. */
            int max_b_d_range_rank = 0;  // track actual max lookup size across partitions
            for (int p = 0; p < n_parts; ++p) {
                const int a_p_begin = p * CLUSTER_PARTITION_SIZE;
                const int a_p_end   = std::min(a_p_begin + CLUSTER_PARTITION_SIZE, total_a);
                const int d_a_min_p = A.offsets[cluster_a_indices[a_p_begin]];
                const int d_a_max_p = A.offsets[cluster_a_indices[a_p_end - 1]];

                const int d_b_lo = min_d_c - d_a_max_p;
                const int d_b_hi = max_d_c - d_a_min_p;

                std::vector<int> part_b;
                for (int bi = 0; bi < B.num_diags; ++bi) {
                    const int d_b = B.offsets[bi];
                    if (d_b >= d_b_lo && d_b <= d_b_hi) part_b.push_back(bi);
                }
                std::sort(part_b.begin(), part_b.end(),
                          [&](int a, int b){ return B.offsets[a] < B.offsets[b]; });

                PartBMeta meta;
                meta.b_begin   = static_cast<int>(plan.b_contrib.size()) - b_base_rank;
                meta.b_count   = static_cast<int>(part_b.size());
                meta.b_d_min   = part_b.empty() ? 0 : B.offsets[part_b.front()];
                meta.b_d_range = part_b.empty() ? 0
                               : (B.offsets[part_b.back()] - meta.b_d_min + 1);
                if (meta.b_d_range > max_b_d_range_rank) max_b_d_range_rank = meta.b_d_range;
                plan.part_b_meta.push_back(meta);
                for (int bi : part_b) plan.b_contrib.push_back(bi);
            }

            /* Update max_smem using actual b_d_range for the lookup table.
             * lpad must cover the widest b_d_range seen across any partition of this rank,
             * not just max_b_per_part (which bounds the count, not the offset span). */
            const int chunk_a  = (CLUSTER_TILE + spread_all                  + 3) & ~3;
            const int chunk_b  = (CLUSTER_TILE + rank_geom[rank].spread_sc   + 3) & ~3;
            const int max_b_pp = CLUSTER_PARTITION_SIZE + CLUSTER_DIAGS_PER_CTA - 1; // 60
            const int lpad     = (max_b_d_range_rank + 3) & ~3;
            const int smem_needed = static_cast<int>(sizeof(float))
                                  * (CLUSTER_PARTITION_SIZE * chunk_a
                                     + max_b_pp * chunk_b + lpad);
            if (smem_needed > plan.max_smem) plan.max_smem = smem_needed;
        }

        /* d. Emit one ClusterMeta + CLUSTER_SIZE ClusterTasks per tile. */
        for (int tile = 0; tile * CLUSTER_TILE < max_c_len_all; ++tile) {
            const int tile_p_begin = tile * CLUSTER_TILE;
            const int cluster_idx  = static_cast<int>(plan.cluster_meta.size());

            ClusterMeta cm;
            cm.tile_p_begin  = tile_p_begin;
            cm.a_begin       = a_base_sg;
            cm.a_count       = total_a;
            cm.min_c_sr_all  = min_c_sr_all;
            cm.spread_all    = spread_all;
            cm.max_c_len_all = max_c_len_all;
            plan.cluster_meta.push_back(cm);

            for (int rank = 0; rank < CLUSTER_SIZE; ++rank) {
                const int g_base  = sg_base + rank * CLUSTER_DIAGS_PER_CTA;
                const int g_count = std::min(CLUSTER_DIAGS_PER_CTA, n_c_diags - g_base);
                const auto& rg    = rank_geom[rank];

                ClusterTask t;
                t.cluster_idx  = cluster_idx;
                t.c_begin      = (g_count > 0) ? g_base : 0;
                t.c_count      = std::max(0, g_count);
                t.min_c_sr     = rg.min_c_sr;
                t.spread       = rg.spread;
                t.max_c_len    = rg.max_c_len;
                t.min_c_sc     = rg.min_c_sc;
                t.spread_sc    = rg.spread_sc;
                t.b_begin      = rg.b_begin;
                t.part_b_base  = part_b_base_sg + rank * n_parts;
                plan.tasks.push_back(t);
            }
        }
    } /* super-group loop */

    return plan;
}

/* ── cluster_kernel ─────────────────────────────────────────────── */
namespace cg = cooperative_groups;

__cluster_dims__(CLUSTER_SIZE, 1, 1)
__global__ void __launch_bounds__(CLUSTER_BLOCK, CLUSTER_BLOCKS_PER_SM)
cluster_kernel(ClusterKernelArgs args)
{
    auto cluster       = cg::this_cluster();
    const int rank     = static_cast<int>(cluster.block_rank());
    const int tid      = threadIdx.x;

    const int cluster_abs_idx = blockIdx.x / CLUSTER_SIZE;
    const ClusterMeta cmeta   = args.cluster_meta[cluster_abs_idx];
    const ClusterTask task    = args.tasks[blockIdx.x];

    /* All CTAs must reach every cluster.sync() regardless of c_count.
     * Empty ranks (c_count == 0) participate in syncs but do no accumulation or output.
     * active[] is all-false for empty ranks, guarding all writes. */

    /* Shared memory layout (dynamic):
     *   smem_A        : CLUSTER_PARTITION_SIZE × chunk   floats  (rank-0 fills)
     *   smem_B        : max_b_per_part × chunk_b         floats  (per-CTA)
     *   smem_B_lookup : max_b_per_part                   ints    (per-CTA) */
    constexpr int max_b_per_part = CLUSTER_PARTITION_SIZE + CLUSTER_DIAGS_PER_CTA - 1;
    const int chunk   = (CLUSTER_TILE + cmeta.spread_all  + 3) & ~3;
    const int chunk_b = (CLUSTER_TILE + task.spread_sc    + 3) & ~3;
    const int chunk4  = chunk   >> 2;
    const int cb4     = chunk_b >> 2;

    extern __shared__ float smem[];
    float* smem_A        = smem;
    float* smem_B        = smem_A + CLUSTER_PARTITION_SIZE * chunk;
    int*   smem_B_lookup = reinterpret_cast<int*>(smem_B + max_b_per_part * chunk_b);

    /* Load C diagonal metadata for this CTA's K group. */
    int c_offset[CLUSTER_DIAGS_PER_CTA];
    int c_sr    [CLUSTER_DIAGS_PER_CTA];
    int c_sc    [CLUSTER_DIAGS_PER_CTA];
    int c_len   [CLUSTER_DIAGS_PER_CTA];
    int c_start [CLUSTER_DIAGS_PER_CTA];
    #pragma unroll
    for (int ki = 0; ki < CLUSTER_DIAGS_PER_CTA; ++ki) {
        if (ki < task.c_count) {
            const HybridCDiag cd = args.c_diags[task.c_begin + ki];
            c_offset[ki] = cd.c_offset;
            c_sr    [ki] = cd.c_sr;
            c_sc    [ki] = (cd.c_offset > 0) ? cd.c_offset : 0;
            c_len   [ki] = cd.length;
            c_start [ki] = cd.values_start;
        } else {
            c_offset[ki] = c_sr[ki] = c_sc[ki] = c_len[ki] = c_start[ki] = 0;
        }
    }

    const int p_begin = cmeta.tile_p_begin;

    bool active[CLUSTER_DIAGS_PER_CTA];
    #pragma unroll
    for (int ki = 0; ki < CLUSTER_DIAGS_PER_CTA; ++ki)
        active[ki] = (ki < task.c_count) && (p_begin + tid < c_len[ki]);

    float acc[CLUSTER_DIAGS_PER_CTA];
    #pragma unroll
    for (int ki = 0; ki < CLUSTER_DIAGS_PER_CTA; ++ki) acc[ki] = 0.0f;

    const int total_a = cmeta.a_count;
    const int a_begin = cmeta.a_begin;

    for (int a_off = 0; a_off < total_a; a_off += CLUSTER_PARTITION_SIZE) {
        const int a_batch = min(CLUSTER_PARTITION_SIZE, total_a - a_off);
        const int p_idx   = a_off / CLUSTER_PARTITION_SIZE;

        /* ── Per-partition B metadata for this CTA ── */
        const PartBMeta pmeta        = args.part_b_meta[task.part_b_base + p_idx];
        const int part_b_count       = pmeta.b_count;
        const int part_b_d_min       = pmeta.b_d_min;
        const int part_b_d_range     = pmeta.b_d_range;
        const int part_b_d_range_pad = (part_b_d_range + 3) & ~3;

        /* ── B lookup init ── */
        for (int i = tid; i < part_b_d_range_pad; i += CLUSTER_BLOCK)
            smem_B_lookup[i] = -1;
        __syncthreads();

        /* ── B lookup fill ── */
        for (int sb = tid; sb < part_b_count; sb += CLUSTER_BLOCK) {
            const int bi = args.b_contrib[task.b_begin + pmeta.b_begin + sb];
            smem_B_lookup[args.B_offsets[bi] - part_b_d_min] = sb;
        }
        __syncthreads();

        /* ── B values load (vectorised float4) ── */
        for (int sb = 0; sb < part_b_count; ++sb) {
            const int bi    = args.b_contrib[task.b_begin + pmeta.b_begin + sb];
            const int d_b   = args.B_offsets[bi];
            const int b_len = args.B_lengths[bi];
            const int b_st  = args.B_starts[bi];
            /* Position in B diagonal corresponding to col = task.min_c_sc at tile start. */
            const int b_sc  = (d_b > 0) ? d_b : 0;
            const int bp_min = task.min_c_sc + p_begin - b_sc;
            float* dst = smem_B + sb * chunk_b;
            for (int j = tid; j < cb4; j += CLUSTER_BLOCK) {
                const int i0 = j << 2;
                float4 v;
                int p0 = bp_min + i0;
                v.x = (p0   >= 0 && p0   < b_len) ? args.B_vals[b_st + p0]   : 0.f;
                v.y = (p0+1 >= 0 && p0+1 < b_len) ? args.B_vals[b_st + p0+1] : 0.f;
                v.z = (p0+2 >= 0 && p0+2 < b_len) ? args.B_vals[b_st + p0+2] : 0.f;
                v.w = (p0+3 >= 0 && p0+3 < b_len) ? args.B_vals[b_st + p0+3] : 0.f;
                *reinterpret_cast<float4*>(dst + i0) = v;
            }
        }
        __syncthreads();

        /* ── A load: rank-0 only, into its own smem_A ── */
        if (rank == 0) {
            for (int s = 0; s < a_batch; ++s) {
                const int ai    = args.a_contrib[a_begin + a_off + s];
                const int d_a   = args.A_offsets[ai];
                const int a_sr  = (d_a < 0) ? -d_a : 0;
                const int a_len = args.A_lengths[ai];
                const int a_st  = args.A_starts[ai];
                /* Chunk starts at position: p_begin + min_c_sr_all - a_sr along diagonal. */
                const int ap_min = p_begin + cmeta.min_c_sr_all - a_sr;
                float* dst = smem_A + s * chunk;
                for (int j = tid; j < chunk4; j += CLUSTER_BLOCK) {
                    const int i0 = j << 2;
                    float4 v;
                    int p0 = ap_min + i0;
                    v.x = (p0   >= 0 && p0   < a_len) ? args.A_vals[a_st + p0]   : 0.f;
                    v.y = (p0+1 >= 0 && p0+1 < a_len) ? args.A_vals[a_st + p0+1] : 0.f;
                    v.z = (p0+2 >= 0 && p0+2 < a_len) ? args.A_vals[a_st + p0+2] : 0.f;
                    v.w = (p0+3 >= 0 && p0+3 < a_len) ? args.A_vals[a_st + p0+3] : 0.f;
                    *reinterpret_cast<float4*>(dst + i0) = v;
                }
            }
        }
        /* All CTAs wait for rank-0's A-load to finish. */
        cluster.sync();

        /* All CTAs read rank-0's smem_A via DSMEM. */
        float* shared_A = reinterpret_cast<float*>(
            cluster.map_shared_rank(smem_A, 0));

        /* ── Accumulate ── */
        const int a_main = (a_batch / CLUSTER_A_UNROLL) * CLUSTER_A_UNROLL;

        for (int s = 0; s < a_main; s += CLUSTER_A_UNROLL) {
            float av_g[CLUSTER_A_UNROLL][CLUSTER_DIAGS_PER_CTA];
            int   da_g[CLUSTER_A_UNROLL];
            #pragma unroll
            for (int u = 0; u < CLUSTER_A_UNROLL; ++u) {
                da_g[u] = args.A_offsets[args.a_contrib[a_begin + a_off + s + u]];
                #pragma unroll
                for (int ki = 0; ki < CLUSTER_DIAGS_PER_CTA; ++ki) {
                    /* Row offset into A chunk: c_sr[ki] - min_c_sr_all + tid */
                    av_g[u][ki] = shared_A[(s + u) * chunk
                                           + (c_sr[ki] - cmeta.min_c_sr_all) + tid];
                }
            }
            #pragma unroll
            for (int ki = 0; ki < CLUSTER_DIAGS_PER_CTA; ++ki) {
                if (!active[ki]) continue;
                #pragma unroll
                for (int u = 0; u < CLUSTER_A_UNROLL; ++u) {
                    const int d_b = c_offset[ki] - da_g[u];
                    const int rel = d_b - part_b_d_min;
                    if ((unsigned)rel >= (unsigned)part_b_d_range) continue;
                    const int sb = smem_B_lookup[rel];
                    if (sb < 0) continue;
                    /* Col offset into B chunk: c_sc[ki] - min_c_sc + tid */
                    acc[ki] += av_g[u][ki]
                             * smem_B[sb * chunk_b + (c_sc[ki] - task.min_c_sc) + tid];
                }
            }
        }
        /* Scalar tail for remaining A diagonals. */
        for (int s = a_main; s < a_batch; ++s) {
            const int da = args.A_offsets[args.a_contrib[a_begin + a_off + s]];
            float av[CLUSTER_DIAGS_PER_CTA];
            #pragma unroll
            for (int ki = 0; ki < CLUSTER_DIAGS_PER_CTA; ++ki)
                av[ki] = shared_A[s * chunk + (c_sr[ki] - cmeta.min_c_sr_all) + tid];
            #pragma unroll
            for (int ki = 0; ki < CLUSTER_DIAGS_PER_CTA; ++ki) {
                if (!active[ki]) continue;
                const int d_b = c_offset[ki] - da;
                const int rel = d_b - part_b_d_min;
                if ((unsigned)rel >= (unsigned)part_b_d_range) continue;
                const int sb = smem_B_lookup[rel];
                if (sb < 0) continue;
                acc[ki] += av[ki]
                         * smem_B[sb * chunk_b + (c_sc[ki] - task.min_c_sc) + tid];
            }
        }

        /* smem_A free for rank-0's next load; smem_B free for next B-lookup init. */
        cluster.sync();
    } /* partition loop */

    /* ── Write output ── */
    #pragma unroll
    for (int ki = 0; ki < CLUSTER_DIAGS_PER_CTA; ++ki)
        if (active[ki])
            args.C_vals[c_start[ki] + p_begin + tid] = acc[ki];
}

/* ── launch_cluster ─────────────────────────────────────────────── */
void launch_cluster(ClusterKernelArgs args, cudaStream_t stream)
{
    if (args.n_clusters == 0) return;
    const int n_blocks = args.n_clusters * CLUSTER_SIZE;

    cudaLaunchConfig_t cfg   = {};
    cfg.gridDim              = dim3(n_blocks);
    cfg.blockDim             = dim3(CLUSTER_BLOCK);
    cfg.dynamicSmemBytes     = static_cast<size_t>(args.max_smem);
    cfg.stream               = stream;

    cudaLaunchAttribute attr;
    attr.id                      = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x        = CLUSTER_SIZE;
    attr.val.clusterDim.y        = 1;
    attr.val.clusterDim.z        = 1;
    cfg.attrs    = &attr;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, cluster_kernel, args);
}
