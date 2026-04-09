/* ============================================================
 * diag_hybrid_kernel.cu
 *
 * Unified C-centric diagonal SpMM kernel implementation.
 * See diag_hybrid_kernel.cuh for the full design description.
 *
 * Index arithmetic:
 *
 *   For output diagonal dC, position k along the diagonal:
 *     row r  = c_sr + k        where c_sr = max(0, -dC)
 *     col c  = r + dC
 *
 *   For A diagonal dA at row r:
 *     a_sr   = max(0, -dA)
 *     a_pos  = r - a_sr
 *
 *   For B diagonal dB at inner dimension m = r + dA:
 *     b_sr   = max(0, -dB)
 *     b_pos  = (r + dA) - b_sr
 *
 * ============================================================ */

#include "diag_hybrid_kernel.cuh"

#include <algorithm>
#include <cassert>
#include <climits>
#include <map>
#include <numeric>
#include <set>
#include <vector>

/* ============================================================
 * HOST HELPERS
 * ============================================================ */

static int hybrid_get_sm_count()
{
    static int cached = 0;
    if (cached == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached,
                               cudaDevAttrMultiProcessorCount, dev);
    }
    return cached;
}


/* Each CTA owns one exclusive tile [tile_p_begin, +TILE).
 * A streams through smem in batches; B loaded once per task into smem. */
__global__ void __launch_bounds__(HYBRID_BLOCK, HYBRID_BLOCKS_PER_SM)
hybrid_kernel(HybridKernelArgs args)
{
    const int tid = static_cast<int>(threadIdx.x);

    extern __shared__ float smem[];

    for (int task_id = static_cast<int>(blockIdx.x);
         task_id < args.n_tasks;
         task_id += static_cast<int>(gridDim.x))
    {
        const HybridTask task = args.tasks[task_id];

        const int chunk      = ((HYBRID_TILE + task.spread) + 3) & ~3;
        const int a_smem_cap = task.a_smem_cap;
        const int chunk_b    = (HYBRID_TILE + task.spread_sc + 3) & ~3;

        float* smem_A        = smem;
        float* smem_B        = smem + a_smem_cap * chunk;
        int*   smem_B_lookup = reinterpret_cast<int*>(smem_B + task.b_count * chunk_b);

        int c_offset[HYBRID_DIAGS_PER_CTA];
        int c_sr[HYBRID_DIAGS_PER_CTA];
        int c_len[HYBRID_DIAGS_PER_CTA];
        int c_start[HYBRID_DIAGS_PER_CTA];
        int c_sc[HYBRID_DIAGS_PER_CTA];

        #pragma unroll
        for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
            if (ki < task.c_count) {
                const HybridCDiag cd = args.c_diags[task.c_begin + ki];
                c_offset[ki] = cd.c_offset;
                c_sr[ki]     = cd.c_sr;
                c_len[ki]    = cd.length;
                c_start[ki]  = cd.values_start;
                c_sc[ki]     = (cd.c_offset >= 0) ? cd.c_offset : 0;
            }
        }

        for (int i = tid; i < task.b_d_range; i += HYBRID_BLOCK)
            smem_B_lookup[i] = -1;
        __syncthreads();
        for (int sb = tid; sb < task.b_count; sb += HYBRID_BLOCK) {
            const int bi  = args.b_contrib[task.b_begin + sb];
            smem_B_lookup[args.B_offsets[bi] - task.b_d_min] = sb;
        }
        __syncthreads();

        const int p_begin = task.tile_p_begin;

        bool active[HYBRID_DIAGS_PER_CTA];
        #pragma unroll
        for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki)
            active[ki] = (ki < task.c_count) && (p_begin + tid < c_len[ki]);

        float acc[HYBRID_DIAGS_PER_CTA];
        #pragma unroll
        for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) acc[ki] = 0.0f;

        for (int sb = 0; sb < task.b_count; ++sb) {
            const int bi    = args.b_contrib[task.b_begin + sb];
            const int d_b   = args.B_offsets[bi];
            const int b_len = args.B_lengths[bi];
            const int b_st  = args.B_starts[bi];
            const int bp_min = task.min_c_sc + p_begin - max(0, d_b);
            float* dst = smem_B + sb * chunk_b;
            const int cb4 = chunk_b >> 2;
            for (int j = tid; j < cb4; j += HYBRID_BLOCK) {
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

        const int total_a = task.a_count;
        const int a_begin = task.a_begin;
        const int chunk4  = chunk >> 2;

        for (int a_off = 0; a_off < total_a; a_off += a_smem_cap) {
            const int a_batch = min(a_smem_cap, total_a - a_off);

            for (int s = 0; s < a_batch; ++s) {
                const int ai    = args.a_contrib[a_begin + a_off + s];
                const int d_a   = args.A_offsets[ai];
                const int a_sr  = (d_a >= 0) ? 0 : -d_a;
                const int base  = task.min_c_sr - a_sr + p_begin;
                const int a_len = args.A_lengths[ai];
                const int a_st  = args.A_starts[ai];
                float* dst = smem_A + s * chunk;
                for (int j = tid; j < chunk4; j += HYBRID_BLOCK) {
                    const int i0 = j << 2;
                    float4 v;
                    int p0 = base + i0;
                    v.x = (p0   >= 0 && p0   < a_len) ? args.A_vals[a_st + p0]   : 0.f;
                    v.y = (p0+1 >= 0 && p0+1 < a_len) ? args.A_vals[a_st + p0+1] : 0.f;
                    v.z = (p0+2 >= 0 && p0+2 < a_len) ? args.A_vals[a_st + p0+2] : 0.f;
                    v.w = (p0+3 >= 0 && p0+3 < a_len) ? args.A_vals[a_st + p0+3] : 0.f;
                    *reinterpret_cast<float4*>(dst + i0) = v;
                }
            }
            __syncthreads();

            const int a_main = a_batch & ~(HYBRID_A_UNROLL - 1);

            for (int s = 0; s < a_main; s += HYBRID_A_UNROLL) {
                int   da_g[HYBRID_A_UNROLL];
                float av_g[HYBRID_A_UNROLL][HYBRID_DIAGS_PER_CTA];

                #pragma unroll
                for (int u = 0; u < HYBRID_A_UNROLL; ++u) {
                    const int ai = args.a_contrib[a_begin + a_off + s + u];
                    da_g[u] = args.A_offsets[ai];
                    #pragma unroll
                    for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki)
                        av_g[u][ki] = smem_A[(s + u) * chunk
                                             + c_sr[ki] - task.min_c_sr + tid];
                }

                #pragma unroll
                for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
                    if (!active[ki]) continue;
                    #pragma unroll
                    for (int u = 0; u < HYBRID_A_UNROLL; ++u) {
                        const int d_b = c_offset[ki] - da_g[u];
                        const int rel = d_b - task.b_d_min;
                        const int sb  = ((unsigned)rel < (unsigned)task.b_d_range)
                                      ? smem_B_lookup[rel] : -1;
                        if (sb < 0) continue;
                        acc[ki] += av_g[u][ki]
                                 * smem_B[sb * chunk_b
                                          + c_sc[ki] - task.min_c_sc + tid];
                    }
                }
            }

            for (int s = a_main; s < a_batch; ++s) {
                const int ai  = args.a_contrib[a_begin + a_off + s];
                const int d_a = args.A_offsets[ai];
                #pragma unroll
                for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
                    if (!active[ki]) continue;
                    const float a_val = smem_A[s * chunk
                                               + c_sr[ki] - task.min_c_sr + tid];
                    const int d_b = c_offset[ki] - d_a;
                    const int rel = d_b - task.b_d_min;
                    const int sb  = ((unsigned)rel < (unsigned)task.b_d_range)
                                  ? smem_B_lookup[rel] : -1;
                    if (sb < 0) continue;
                    acc[ki] += a_val
                             * smem_B[sb * chunk_b
                                      + c_sc[ki] - task.min_c_sc + tid];
                }
            }

            if (a_off + a_smem_cap < total_a) __syncthreads();
        }

        #pragma unroll
        for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki)
            if (active[ki])
                args.C_vals[c_start[ki] + p_begin + tid] = acc[ki];

        __syncthreads();
    }
}

/* build_hybrid_plan — host-side preprocessing.
 * Groups C diagonals into batches of K; splits each group into
 * TILE-sized tasks, one per output position tile. */
HybridPlan build_hybrid_plan(const DiagMatrix& A,
                              const DiagMatrix& B,
                              int M, int K_dim, int N)
{
    HybridPlan plan;

    auto out_map = build_output_diagonals(A, B, M, N);
    build_contributors(out_map, A, B);
    int c_val_offset = 0;
    for (auto& [d_c, info] : out_map) {
        if (info.contributors.empty()) continue;
        HybridCDiag cd;
        cd.c_offset     = d_c;
        cd.c_sr         = (d_c >= 0) ? 0 : -d_c;
        cd.length       = info.c_length;
        cd.values_start = c_val_offset;
        plan.c_diags.push_back(cd);
        c_val_offset += info.c_length;
    }
    plan.total_c_values = c_val_offset;

    const int n_c = static_cast<int>(plan.c_diags.size());
    const int n_m_1 = N - 1;
    std::vector<int> b_lookup = build_b_diag_lookup(B, N);

    /* Step 3 — group consecutive C diagonals, emit one task per output tile. */
    plan.max_smem = 0;

    for (int g_base = 0; g_base < n_c;
         g_base += HYBRID_DIAGS_PER_CTA)
    {
        const int g_end = std::min(g_base + HYBRID_DIAGS_PER_CTA, n_c);
        const int g_count = g_end - g_base;

        int min_c_sr  = INT_MAX;
        int max_c_sr  = 0;
        int max_c_len = 0;
        for (int ki = 0; ki < g_count; ++ki) {
            const HybridCDiag& cd = plan.c_diags[g_base + ki];
            min_c_sr  = std::min(min_c_sr, cd.c_sr);
            max_c_sr  = std::max(max_c_sr, cd.c_sr);
            max_c_len = std::max(max_c_len, cd.length);
        }
        const int spread = max_c_sr - min_c_sr;
        const int chunk  = ((HYBRID_TILE + spread) + 3) & ~3; /* 4-aligned for float4 */

        int min_c_sc_g = INT_MAX, max_c_sc_g = 0;
        for (int ki = 0; ki < g_count; ++ki) {
            const int d_c = plan.c_diags[g_base + ki].c_offset;
            const int c_sc = (d_c >= 0) ? d_c : 0;
            min_c_sc_g = std::min(min_c_sc_g, c_sc);
            max_c_sc_g = std::max(max_c_sc_g, c_sc);
        }
        const int spread_sc_g = max_c_sc_g - min_c_sc_g;
        const int chunk_b_g   = ((HYBRID_TILE + spread_sc_g) + 3) & ~3;

        std::vector<int> group_a_indices;
        for (int ai = 0; ai < static_cast<int>(A.offsets.size()); ++ai) {
            const int d_a = A.offsets[ai];
            bool used = false;
            for (int ki = 0; ki < g_count && !used; ++ki) {
                const int d_b = plan.c_diags[g_base + ki].c_offset - d_a;
                const int b_idx = d_b + n_m_1;
                if (b_idx >= 0 && b_idx < 2 * N - 1
                    && b_lookup[b_idx] >= 0) {
                    used = true;
                }
            }
            if (used) group_a_indices.push_back(ai);
        }

        const int total_a = static_cast<int>(group_a_indices.size());

        std::vector<int> group_b_indices;
        {
            std::set<int> b_set;
            for (int ai : group_a_indices) {
                const int d_a = A.offsets[ai];
                for (int ki = 0; ki < g_count; ++ki) {
                    const int d_b   = plan.c_diags[g_base + ki].c_offset - d_a;
                    const int b_idx = d_b + n_m_1;
                    if (b_idx >= 0 && b_idx < 2 * N - 1 && b_lookup[b_idx] >= 0)
                        b_set.insert(b_lookup[b_idx]);
                }
            }
            group_b_indices.assign(b_set.begin(), b_set.end());
            std::sort(group_b_indices.begin(), group_b_indices.end(),
                      [&](int a, int b){ return B.offsets[a] < B.offsets[b]; });
        }
        const int b_count_g = static_cast<int>(group_b_indices.size());

        int b_d_min_g = 0, b_d_max_g = 0, b_d_range_g = 0;
        if (b_count_g > 0) {
            b_d_min_g   = B.offsets[group_b_indices.front()];
            b_d_max_g   = B.offsets[group_b_indices.back()];
            b_d_range_g = b_d_max_g - b_d_min_g + 1;
        }
        const int b_d_range_pad_g  = (b_d_range_g + 3) & ~3;
        const int b_smem_bytes_g   = (b_count_g * chunk_b_g + b_d_range_pad_g)
                                   * static_cast<int>(sizeof(float));

        const int b_base_g = static_cast<int>(plan.b_contrib.size());
        for (int bi : group_b_indices)
            plan.b_contrib.push_back(bi);

        const int a_base_g = static_cast<int>(plan.a_contrib.size());
        for (int ai : group_a_indices)
            plan.a_contrib.push_back(ai);

        const int a_smem_cap = std::max(1,
            (HYBRID_SMEM_BUDGET - b_smem_bytes_g)
            / (chunk * static_cast<int>(sizeof(float))));

        const int n_tiles = (max_c_len + HYBRID_TILE - 1) / HYBRID_TILE;
        for (int tile = 0; tile < n_tiles; ++tile) {
            HybridTask t;
            t.c_begin      = g_base;
            t.c_count      = g_count;
            t.min_c_sr     = min_c_sr;
            t.spread       = spread;
            t.max_c_len    = max_c_len;
            t.a_begin      = a_base_g;
            t.a_count      = total_a;
            t.a_smem_cap   = a_smem_cap;
            t.min_c_sc     = min_c_sc_g;
            t.spread_sc    = spread_sc_g;
            t.b_begin      = b_base_g;
            t.b_count      = b_count_g;
            t.b_d_min      = b_d_min_g;
            t.b_d_range    = b_d_range_g;
            t.tile_p_begin = tile * HYBRID_TILE;
            plan.tasks.push_back(t);
        }

        const int smem_needed = (a_smem_cap * chunk
                                 + b_count_g * chunk_b_g + b_d_range_pad_g)
                              * static_cast<int>(sizeof(float));
        plan.max_smem = std::max(plan.max_smem, smem_needed);
    }

    return plan;
}

void launch_hybrid(HybridKernelArgs args, cudaStream_t stream)
{
    if (args.n_tasks == 0) return;

    const int sm = hybrid_get_sm_count();
    const int target_blocks = sm * HYBRID_BLOCKS_PER_SM;

    int dev_max_smem = 0;
    cudaDeviceGetAttribute(&dev_max_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    const int grid = std::min(target_blocks, args.n_tasks);
    const int smem = std::min(args.max_smem, dev_max_smem);
    cudaFuncSetAttribute(hybrid_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    hybrid_kernel<<<grid, HYBRID_BLOCK, smem, stream>>>(args);
}
