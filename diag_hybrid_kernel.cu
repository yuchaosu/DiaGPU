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

/* ============================================================
 * COMPUTATION KERNEL  —  persistent, grid-stride
 *
 * Each CTA claims one HybridTask (= one group × one A-partition)
 * via grid-stride.  It iterates tile positions p_begin over the
 * longest C diagonal in the group.
 *
 * Optimisations implemented:
 *   1. Double-buffered A preload — next tile's A loads overlap
 *      with current tile's computation via cp.async.
 *   2. B load ILP — HYBRID_A_UNROLL (=4) A diagonals are
 *      processed simultaneously so multiple B loads are in
 *      flight, saturating the memory pipeline.
 *   3. Vectorised A loads — float4 (128-bit) loads in Phase 1
 *      reduce load instruction count 4×.  chunk is padded to
 *      a multiple of 4 to guarantee alignment.
 *
 * Future optimisations (not yet implemented):
 *   4. L2 cache residency hints — pin B_vals in L2 with
 *      cudaAccessPropertyPersisting on the stream's access
 *      policy window.  Beneficial when B fits in L2 (50 MB
 *      on H100).
 *   5. Thread block clusters (Hopper) — CTAs in a cluster
 *      share A data across groups via distributed shared
 *      memory (dsmem), eliminating redundant A loads at
 *      group boundaries.
 *   6. Warp shuffle for spread=0 — when all C diagonals in
 *      the group are positive (c_sr=0, spread=0), every
 *      thread reads its own A value.  Shared memory can be
 *      bypassed entirely; A lives in a register.  Check
 *      spread==0 and branch to a fast path.
 *
 * Shared memory layout (dynamic, double-buffered):
 *   buf 0: smem[0 .. a_count*chunk - 1]
 *   buf 1: smem[a_count*chunk .. 2*a_count*chunk - 1]
 *   chunk is 4-aligned for float4 loads.
 *   Total = 2 * task.a_count * chunk floats.
 * ============================================================ */

/* cp.async helpers (Ampere+, sm_80). */
__device__ __forceinline__ void
cp_async_f32(float* __restrict__ dst, const float* __restrict__ src)
{
    /* Copy 4 bytes from global to shared memory asynchronously.
     * Falls back to a regular load on pre-Ampere. */
#if __CUDA_ARCH__ >= 800
    uint32_t dst_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(dst_addr), "l"(src));
#else
    *dst = *src;
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void cp_async_wait_all()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n");
#else
    __threadfence_block();
#endif
}

template <bool DIRECT>
__global__ void __launch_bounds__(HYBRID_BLOCK, HYBRID_BLOCKS_PER_SM)
hybrid_compute_kernel(HybridKernelArgs args)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int blb = args.b_lookup_base; /* extended B_lookup: no bounds check needed */

    extern __shared__ float smem[];

    for (int task_id = static_cast<int>(blockIdx.x);
         task_id < args.n_tasks;
         task_id += static_cast<int>(gridDim.x))
    {
        const HybridTask task = args.tasks[task_id];
        /* chunk is 4-aligned (host ensures this via HYBRID_MAX_CHUNK). */
        const int chunk_raw = HYBRID_TILE + task.spread;
        const int chunk     = (chunk_raw + 3) & ~3;
        const int buf_floats = task.a_count * chunk;
        /* Double-buffer pointers. */
        float* buf[2] = { smem, smem + buf_floats };

        /* Load C diagonal metadata into registers. */
        int c_offset[HYBRID_DIAGS_PER_CTA];
        int c_sr[HYBRID_DIAGS_PER_CTA];
        int c_len[HYBRID_DIAGS_PER_CTA];
        int c_start[HYBRID_DIAGS_PER_CTA];

        #pragma unroll
        for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
            if (ki < task.c_count) {
                const HybridCDiag cd = args.c_diags[task.c_begin + ki];
                c_offset[ki] = cd.c_offset;
                c_sr[ki]     = cd.c_sr;
                c_len[ki]    = cd.length;
                c_start[ki]  = cd.values_start;
            }
        }

        /* ============================================================
         * Macro: issue cp.async loads for one tile into buf[b].
         * Uses float4 (128-bit) loads where possible, scalar for tail.
         * ============================================================ */
        #define LOAD_A_TILE(b, p_begin_val)                              \
        do {                                                             \
            for (int s = 0; s < task.a_count; ++s) {                     \
                const int ai   = args.a_contrib[task.a_begin + s];       \
                const int d_a  = args.A_offsets[ai];                     \
                const int a_sr = (d_a >= 0) ? 0 : -d_a;                 \
                const int base = task.min_c_sr - a_sr + (p_begin_val);   \
                const int a_len = args.A_lengths[ai];                    \
                const int a_start = args.A_starts[ai];                   \
                float* dst_base = buf[(b)] + s * chunk;                  \
                /* Float4 vectorised path (tid indexes float4 slots). */ \
                const int chunk4 = chunk >> 2;                           \
                for (int j = tid; j < chunk4; j += HYBRID_BLOCK) {       \
                    const int i0 = j << 2;                               \
                    float4 v;                                            \
                    int p0 = base + i0;                                  \
                    v.x = (p0   >= 0 && p0   < a_len)                   \
                        ? args.A_vals[a_start + p0]   : 0.0f;           \
                    v.y = (p0+1 >= 0 && p0+1 < a_len)                   \
                        ? args.A_vals[a_start + p0+1] : 0.0f;           \
                    v.z = (p0+2 >= 0 && p0+2 < a_len)                   \
                        ? args.A_vals[a_start + p0+2] : 0.0f;           \
                    v.w = (p0+3 >= 0 && p0+3 < a_len)                   \
                        ? args.A_vals[a_start + p0+3] : 0.0f;           \
                    *reinterpret_cast<float4*>(dst_base + i0) = v;       \
                }                                                        \
            }                                                            \
            cp_async_commit();                                           \
        } while (0)

        /* ============================================================
         * Double-buffered tile loop.
         *
         * Iteration 0: load tile 0 into buf[0], wait, compute tile 0.
         * Iteration i>0: load tile i into buf[cur], compute tile i-1
         *                from buf[1-cur], wait for buf[cur].
         * After loop: compute the last tile.
         * ============================================================ */
        const int n_tiles = (task.max_c_len + HYBRID_TILE - 1) / HYBRID_TILE;
        int cur = 0;

        /* Preload first tile. */
        if (n_tiles > 0) {
            LOAD_A_TILE(0, 0);
            cp_async_wait_all();
            __syncthreads();
        }

        for (int tile = 0; tile < n_tiles; ++tile) {
            const int p_begin = tile * HYBRID_TILE;

            /* Kick off async load for the NEXT tile into the other
             * buffer while we compute the current tile. */
            if (tile + 1 < n_tiles) {
                LOAD_A_TILE(1 - cur, (tile + 1) * HYBRID_TILE);
            }

            /* ---- Phase 2: accumulate (A from smem, B from register).
             *
             * Branch reduction:
             *   - active[ki]: precomputed mask merges c_count + c_len checks
             *   - B_lookup is extended (size 4N-3): no index bounds check
             *   - b_pos is clamped; invalid → multiply by 0 (predicated)
             *
             * Only remaining branch: if (bi < 0) — fundamental sparsity.
             *
             * ILP: HYBRID_A_UNROLL A diagonals processed together so
             * multiple B loads are in flight simultaneously.            */

            /* Precompute active mask for this tile. */
            bool active[HYBRID_DIAGS_PER_CTA];
            #pragma unroll
            for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki)
                active[ki] = (ki < task.c_count)
                           && (p_begin + tid < c_len[ki]);

            float acc[HYBRID_DIAGS_PER_CTA];
            #pragma unroll
            for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki)
                acc[ki] = 0.0f;

            float* cur_buf = buf[cur];
            const int a_count = task.a_count;
            const int a_begin = task.a_begin;
            const int a_main = a_count & ~(HYBRID_A_UNROLL - 1);

            /* Main loop: groups of HYBRID_A_UNROLL A diagonals. */
            for (int s = 0; s < a_main; s += HYBRID_A_UNROLL) {
                int   da_g[HYBRID_A_UNROLL];
                float av_g[HYBRID_A_UNROLL][HYBRID_DIAGS_PER_CTA];

                #pragma unroll
                for (int u = 0; u < HYBRID_A_UNROLL; ++u) {
                    const int ai = args.a_contrib[a_begin + s + u];
                    da_g[u] = args.A_offsets[ai];
                    #pragma unroll
                    for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
                        const int off = c_sr[ki] - task.min_c_sr + tid;
                        av_g[u][ki] = cur_buf[(s + u) * chunk + off];
                    }
                }

                #pragma unroll
                for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
                    if (!active[ki]) continue;  /* single branch */
                    #pragma unroll
                    for (int u = 0; u < HYBRID_A_UNROLL; ++u) {
                        const int bi = args.B_lookup[
                            c_offset[ki] - da_g[u] + blb];
                        if (bi < 0) continue;  /* sparsity check */

                        const int d_b   = c_offset[ki] - da_g[u];
                        const int b_sr  = (d_b >= 0) ? 0 : -d_b;
                        const int b_len = args.B_lengths[bi];
                        const int b_pos =
                            (c_sr[ki] + p_begin + tid + da_g[u])
                            - b_sr;
                        /* Clamp + predicate: no branch for bounds. */
                        const int safe  = max(0, min(b_pos, b_len - 1));
                        const float mask = (b_pos >= 0 && b_pos < b_len)
                                         ? 1.0f : 0.0f;
                        acc[ki] += av_g[u][ki]
                                 * args.B_vals[args.B_starts[bi] + safe]
                                 * mask;
                    }
                }
            }

            /* Remainder: leftover A diagonals (< HYBRID_A_UNROLL). */
            for (int s = a_main; s < a_count; ++s) {
                const int ai  = args.a_contrib[a_begin + s];
                const int d_a = args.A_offsets[ai];

                #pragma unroll
                for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
                    if (!active[ki]) continue;

                    const float a_val = cur_buf[
                        s * chunk + c_sr[ki] - task.min_c_sr + tid];
                    const int bi = args.B_lookup[
                        c_offset[ki] - d_a + blb];
                    if (bi < 0) continue;

                    const int d_b   = c_offset[ki] - d_a;
                    const int b_sr  = (d_b >= 0) ? 0 : -d_b;
                    const int b_len = args.B_lengths[bi];
                    const int b_pos =
                        (c_sr[ki] + p_begin + tid + d_a) - b_sr;
                    const int safe  = max(0, min(b_pos, b_len - 1));
                    const float mask = (b_pos >= 0 && b_pos < b_len)
                                     ? 1.0f : 0.0f;
                    acc[ki] += a_val
                             * args.B_vals[args.B_starts[bi] + safe]
                             * mask;
                }
            }

            /* ---- Phase 3: write (active mask reused).
             * Template-specialized: DIRECT → C_vals, !DIRECT → partial_buf.
             * No runtime branch — compiler eliminates the dead path. ---- */
            if constexpr (DIRECT) {
                #pragma unroll
                for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki)
                    if (active[ki])
                        args.C_vals[c_start[ki] + p_begin + tid] =
                            acc[ki];
            } else {
                const int padded =
                    (task.max_c_len + HYBRID_TILE - 1)
                    / HYBRID_TILE * HYBRID_TILE;
                #pragma unroll
                for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki)
                    if (active[ki])
                        args.partial_buf[task.out_offset
                                         + ki * padded
                                         + p_begin + tid] =
                            acc[ki];
            }

            /* Wait for the next tile's async A loads, then sync. */
            if (tile + 1 < n_tiles) {
                cp_async_wait_all();
                __syncthreads();
            }
            cur = 1 - cur;
        }

        #undef LOAD_A_TILE

        /* Sync before next grid-stride task. */
        __syncthreads();
    }
}

/* Explicit instantiations. */
template __global__ void hybrid_compute_kernel<true>(HybridKernelArgs);
template __global__ void hybrid_compute_kernel<false>(HybridKernelArgs);

/* ============================================================
 * REDUCTION KERNEL  —  persistent, grid-stride
 *
 * Sums partial_buf partitions → C_vals for heavy groups.
 * Each CTA claims one HybridReduceTask and iterates tiles.
 * No shared memory needed — each thread independently reduces
 * its position across all partitions.
 * ============================================================ */
__global__ void __launch_bounds__(HYBRID_BLOCK, HYBRID_BLOCKS_PER_SM)
hybrid_reduce_kernel(HybridKernelArgs args)
{
    const int tid = static_cast<int>(threadIdx.x);

    for (int task_id = static_cast<int>(blockIdx.x);
         task_id < args.n_reduce;
         task_id += static_cast<int>(gridDim.x))
    {
        const HybridReduceTask rt = args.reduce_tasks[task_id];

        /* Load C diagonal metadata. */
        int c_len[HYBRID_DIAGS_PER_CTA];
        int c_start[HYBRID_DIAGS_PER_CTA];

        #pragma unroll
        for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
            if (ki < rt.c_count) {
                const HybridCDiag cd = args.c_diags[rt.c_begin + ki];
                c_len[ki]   = cd.length;
                c_start[ki] = cd.values_start;
            }
        }

        const int padded =
            (rt.max_c_len + HYBRID_TILE - 1) / HYBRID_TILE * HYBRID_TILE;
        const int partition_stride =
            rt.c_count * padded;  // floats per partition

        for (int p_begin = 0; p_begin < rt.max_c_len;
             p_begin += HYBRID_TILE)
        {
            #pragma unroll
            for (int ki = 0; ki < HYBRID_DIAGS_PER_CTA; ++ki) {
                if (ki >= rt.c_count) break;
                if (p_begin + tid >= c_len[ki]) continue;

                float sum = 0.0f;
                for (int p = 0; p < rt.num_partials; ++p) {
                    sum += args.partial_buf[rt.partial_base
                                            + p * partition_stride
                                            + ki * padded
                                            + p_begin + tid];
                }
                args.C_vals[c_start[ki] + p_begin + tid] = sum;
            }
        }
    }
}

/* ============================================================
 * build_hybrid_plan
 *
 * Host-side preprocessing.
 *
 * 1. Enumerate output diagonals and build c_diags[].
 * 2. Group consecutive c_diags into groups of K.
 * 3. For each group, find contributing A diagonals.
 * 4. If all A diags fit in smem → corner (1 task, direct write).
 *    Else → heavy (P partition tasks + 1 reduce task).
 * ============================================================ */
HybridPlan build_hybrid_plan(const DiagMatrix& A,
                              const DiagMatrix& B,
                              int M, int K_dim, int N)
{
    HybridPlan plan;

    /* Step 1 — enumerate output diagonals and contributors. */
    auto out_map = build_output_diagonals(A, B, M, N);
    build_contributors(out_map, A, B);

    /* Step 2 — build c_diags table. */
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
    const int n_m_1 = N - 1;  /* for B_lookup indexing */

    /* Host-side B lookup table: b_lookup[d_b + n_m_1] → bi, or -1. */
    std::vector<int> b_lookup = build_b_diag_lookup(B, N);

    /* Step 3 — group consecutive C diagonals, classify, emit tasks. */
    int partial_offset = 0;
    plan.corner_max_smem = 0;
    plan.heavy_max_smem  = 0;

    for (int g_base = 0; g_base < n_c;
         g_base += HYBRID_DIAGS_PER_CTA)
    {
        const int g_end = std::min(g_base + HYBRID_DIAGS_PER_CTA, n_c);
        const int g_count = g_end - g_base;

        /* Compute group geometry. */
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

        /* Find contributing A diagonals for this group:
         * ai contributes if any C diagonal in the group has a valid
         * B partner for d_b = c_offset[ki] - A_offsets[ai]. */
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
        /* Double-buffered: need 2 × a_count × chunk floats. */
        const int smem_bytes  = 2 * total_a * chunk
                                * static_cast<int>(sizeof(float));

        const bool fits = (smem_bytes <= HYBRID_SMEM_BUDGET);

        if (fits) {
            /* ---- Corner group: 1 task, direct write ---- */
            const int a_base = static_cast<int>(plan.a_contrib.size());
            for (int ai : group_a_indices)
                plan.a_contrib.push_back(ai);

            HybridTask t;
            t.c_begin    = g_base;
            t.c_count    = g_count;
            t.min_c_sr   = min_c_sr;
            t.spread     = spread;
            t.max_c_len  = max_c_len;
            t.a_begin    = a_base;
            t.a_count    = total_a;
            t.is_direct  = 1;
            t.out_offset = 0;  /* unused for direct */
            plan.corner_tasks.push_back(t);

            plan.corner_max_smem = std::max(plan.corner_max_smem, smem_bytes);

        } else {
            /* ---- Heavy group: partition A contributors ---- */
            /* Half the budget per buffer (double-buffered). */
            const int max_a_per_part =
                HYBRID_SMEM_BUDGET / (2 * chunk * static_cast<int>(sizeof(float)));
            const int n_partitions =
                (total_a + max_a_per_part - 1) / max_a_per_part;

            /* partial_buf layout for this group:
             *   partition p, C diagonal ki, position pos →
             *   partial_buf[group_base
             *               + p * (c_count * padded)
             *               + ki * padded + pos]
             * where padded = ceil(max_c_len / TILE) * TILE. */
            const int padded =
                (max_c_len + HYBRID_TILE - 1) / HYBRID_TILE * HYBRID_TILE;
            const int partition_stride = g_count * padded;
            const int group_partial_base = partial_offset;

            for (int part = 0; part < n_partitions; ++part) {
                const int a_off   = part * max_a_per_part;
                const int a_count = std::min(max_a_per_part,
                                              total_a - a_off);
                const int a_base =
                    static_cast<int>(plan.a_contrib.size());
                for (int j = 0; j < a_count; ++j)
                    plan.a_contrib.push_back(group_a_indices[a_off + j]);

                HybridTask t;
                t.c_begin    = g_base;
                t.c_count    = g_count;
                t.min_c_sr   = min_c_sr;
                t.spread     = spread;
                t.max_c_len  = max_c_len;
                t.a_begin    = a_base;
                t.a_count    = a_count;
                t.is_direct  = 0;
                t.out_offset = group_partial_base
                               + part * partition_stride;
                plan.heavy_tasks.push_back(t);

                const int part_smem =
                    2 * a_count * chunk * static_cast<int>(sizeof(float));
                plan.heavy_max_smem = std::max(plan.heavy_max_smem, part_smem);
            }

            partial_offset += n_partitions * partition_stride;

            /* Reduction task for this group. */
            HybridReduceTask rt;
            rt.c_begin      = g_base;
            rt.c_count      = g_count;
            rt.min_c_sr     = min_c_sr;
            rt.spread       = spread;
            rt.max_c_len    = max_c_len;
            rt.partial_base = group_partial_base;
            rt.num_partials = n_partitions;
            plan.reduce_tasks.push_back(rt);
        }
    }

    plan.partial_buf_size = partial_offset;
    return plan;
}

/* ============================================================
 * launch_hybrid
 *
 * Up to three kernel launches on the same stream:
 *   1. Corner kernel  — hybrid_compute_kernel<true>  (direct → C_vals)
 *   2. Heavy kernel   — hybrid_compute_kernel<false> (→ partial_buf)
 *   3. Reduce kernel  — hybrid_reduce_kernel         (partial_buf → C_vals)
 *
 * Separate launches let each use its own smem allocation:
 *   corner gets corner_max_smem (small → more L1 for B loads),
 *   heavy  gets heavy_max_smem  (large → many A diags per partition).
 *
 * Same-stream FIFO ordering guarantees heavy finishes before reduce.
 * Corner and heavy are independent and could run concurrently on
 * separate streams if desired.
 * ============================================================ */
void launch_hybrid(HybridKernelArgs args, cudaStream_t stream)
{
    const int sm = hybrid_get_sm_count();
    const int target_blocks = sm * HYBRID_BLOCKS_PER_SM;

    /* Query device smem limit once. */
    int dev_max_smem = 0;
    cudaDeviceGetAttribute(&dev_max_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    /* ---- Corner kernel (DIRECT=true) ---- */
    if (args.n_corner > 0) {
        const int grid = std::min(target_blocks, args.n_corner);
        const int smem = std::min(args.corner_max_smem, dev_max_smem);
        cudaFuncSetAttribute(hybrid_compute_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        HybridKernelArgs corner_args = args;
        corner_args.tasks   = args.corner_tasks;
        corner_args.n_tasks = args.n_corner;
        hybrid_compute_kernel<true>
            <<<grid, HYBRID_BLOCK, smem, stream>>>(corner_args);
    }

    /* ---- Heavy kernel (DIRECT=false) ---- */
    if (args.n_heavy > 0) {
        const int grid = std::min(target_blocks, args.n_heavy);
        const int smem = std::min(args.heavy_max_smem, dev_max_smem);
        cudaFuncSetAttribute(hybrid_compute_kernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        HybridKernelArgs heavy_args = args;
        heavy_args.tasks   = args.heavy_tasks;
        heavy_args.n_tasks = args.n_heavy;
        hybrid_compute_kernel<false>
            <<<grid, HYBRID_BLOCK, smem, stream>>>(heavy_args);
    }

    /* ---- Reduction kernel (after heavy, same stream) ---- */
    if (args.n_reduce > 0) {
        const int grid = std::min(target_blocks, args.n_reduce);
        hybrid_reduce_kernel
            <<<grid, HYBRID_BLOCK, 0, stream>>>(args);
    }
}
