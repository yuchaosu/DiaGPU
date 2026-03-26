/* ============================================================
 * diag_hybrid_kernel.cu
 *
 * Implementation of the hybrid corner / heavy two-stage
 * diagonal sparse matrix multiplication kernel.
 *
 * See diag_hybrid_kernel.cuh for the full design description.
 *
 * Index arithmetic (shared with diag_kernel.cu):
 *
 *   For output diagonal dC, position k along the diagonal:
 *     row r  = c_sr + k        where c_sr = max(0, -dC)
 *     col c  = r + dC
 *
 *   A[dA] value at position k:
 *     a_sr   = max(0, -dA)
 *     a_pos  = c_sr - a_sr + p_begin + tid   (relative to A diag start)
 *
 *   B[dB] value at position k  (B is accessed at column = row of A_inner):
 *     b_sr   = max(0, -dB)
 *     b_pos  = c_sr + dA - b_sr + p_begin + tid
 *
 * ============================================================ */

#include "diag_hybrid_kernel.cuh"

#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <vector>

/* ============================================================
 * Device helpers
 * ============================================================ */

/* Binary search helpers (same as diag_kernel.cu). */
__device__ __forceinline__ int
hk_lower_bound(const int* __restrict__ arr, int n, int val)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < val) lo = mid + 1;
        else                hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int
hk_upper_bound(const int* __restrict__ arr, int n, int val)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] <= val) lo = mid + 1;
        else                 hi = mid;
    }
    return lo;
}

/* Compute the valid position range [lo, hi) within a tile segment
 * [p_begin, p_begin+p_len) for one (dA, dB) contributor pair.
 *
 * Returns lo >= hi when the pair has no valid contribution to
 * this segment (occurs near matrix corners).
 *
 * Parameters:
 *   c_sr, p_begin, p_len  — tile descriptor
 *   a_sr, a_len           — A diagonal geometry
 *   d_a                   — diagonal offset of A (for b_pos formula)
 *   b_sr, b_len           — B diagonal geometry
 *
 * lo and hi are returned as offsets relative to p_begin (i.e., thread
 * tid is valid when lo <= tid < hi). */
__device__ __forceinline__ void
compute_overlap(int c_sr, int p_begin, int p_len,
                int a_sr, int a_len,
                int d_a,
                int b_sr, int b_len,
                int& lo, int& hi)
{
    /* a_pos = c_sr - a_sr + p_begin + tid  must be in [0, a_len) */
    int a_base = c_sr - a_sr + p_begin;
    int a_lo   = -a_base;                /* tid >= a_lo → a_pos >= 0     */
    int a_hi   = a_len - a_base;         /* tid <  a_hi → a_pos < a_len  */

    /* b_pos = c_sr + d_a - b_sr + p_begin + tid  must be in [0, b_len) */
    int b_base = c_sr + d_a - b_sr + p_begin;
    int b_lo   = -b_base;
    int b_hi   = b_len - b_base;

    lo = max(0,     max(a_lo, b_lo));
    hi = min(p_len, min(a_hi, b_hi));
}

/* ============================================================
 * CORNER KERNEL  —  persistent, grid-stride
 *
 * Grid:   fixed occupancy-saturated size (num_SMs × 8).
 *         Each CTA iterates over its slice of corner_tasks[]
 *         with a grid-stride loop, eliminating per-task launch
 *         overhead regardless of how many corner tasks exist.
 *
 * Block:  HYBRID_BLOCK_CORNER threads (= HYBRID_TILE_CORNER).
 *
 * Each thread owns one output position (tid < p_len).
 * Threads outside the tile boundary participate in cooperative
 * shared-memory loads (keeping __syncthreads valid) but skip
 * the register accumulation and the final write.
 *
 * Shared memory (1 KB):
 *   smem_A[HYBRID_TILE_CORNER]  — staged A diagonal tile
 *   smem_B[HYBRID_TILE_CORNER]  — staged B diagonal tile
 *
 * Data reuse:
 *   A and B tiles for each pair are loaded collaboratively by
 *   all 128 threads in a single, fully coalesced transaction.
 *   Within a tile, each thread reuses its smem_A[tid] across
 *   all B diagonals that map to the same A row (structure-
 *   dependent; grouped automatically by sorted A-offset order).
 * ============================================================ */
__global__ void __launch_bounds__(HYBRID_BLOCK_CORNER, 8)
hybrid_corner_kernel(HybridKernelArgs args)
{
    const int tid    = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x);
    const int n_m_1  = args.n - 1;

    __shared__ float smem_A[HYBRID_TILE_CORNER];
    __shared__ float smem_B[HYBRID_TILE_CORNER];

    /* Grid-stride loop: each CTA processes multiple tasks. */
    for (int bid = static_cast<int>(blockIdx.x);
         bid < args.n_corner;
         bid += stride)
    {
        const HybridCornerTask task = args.corner_tasks[bid];
        const bool active = (tid < task.p_len);

        const int d_c     = task.c_offset;
        const int c_sr    = (d_c >= 0) ? 0 : -d_c;
        const int p_begin = task.p_begin;

        float acc = 0.0f;

        for (int ai = task.ai_begin; ai < task.ai_end; ++ai) {
            const int d_a = args.A_offsets[ai];
            const int d_b = d_c - d_a;

            /* Skip if B does not have this diagonal. */
            const int bi = args.B_lookup[d_b + n_m_1];
            if (bi < 0) continue;

            /* ---- Collaborative load: A tile ---- */
            {
                const int a_sr  = (d_a >= 0) ? 0 : -d_a;
                const int a_pos = c_sr - a_sr + p_begin + tid;
                smem_A[tid] = (a_pos >= 0 && a_pos < args.A_lengths[ai])
                            ? args.A_vals[args.A_starts[ai] + a_pos]
                            : 0.0f;
            }

            /* ---- Collaborative load: B tile ---- */
            {
                const int b_sr  = (d_b >= 0) ? 0 : -d_b;
                const int b_pos = c_sr + d_a - b_sr + p_begin + tid;
                smem_B[tid] = (b_pos >= 0 && b_pos < args.B_lengths[bi])
                            ? args.B_vals[args.B_starts[bi] + b_pos]
                            : 0.0f;
            }

            /* Ensure both tiles are visible to all threads before use. */
            __syncthreads();

            /* Register accumulation — only active threads contribute. */
            if (active) {
                acc += smem_A[tid] * smem_B[tid];
            }

            /* Sync before next iteration overwrites smem. */
            __syncthreads();
        }

        /* Direct write (exclusive tile ownership — no atomics). */
        if (active) {
            const HybridCDiag cd = args.c_diags[task.c_idx];
            args.C_vals[cd.values_start + p_begin + tid] = acc;
        }
        /* No sync needed between grid-stride iterations: smem is
         * fully rewritten at the top of the next task's pair loop. */
    }
}

/* ============================================================
 * HEAVY STAGE-1 KERNEL  —  persistent, grid-stride
 *
 * Grid:   fixed occupancy-saturated size (num_SMs × 4).
 *         Each CTA iterates over its slice of s1_tasks[] with
 *         a grid-stride loop.  All threads in the block must
 *         participate in every task iteration so that the
 *         __syncthreads calls inside the pair loop remain valid.
 *
 * Block:  HYBRID_BLOCK_HEAVY_S1 threads (= HYBRID_TILE_HEAVY).
 *
 * Each block handles one (c_diag, segment, pair-partition) and
 * writes its partial sum to an exclusive slot in partial_buf.
 *
 * Shared memory layout (4 KB with TILE=256):
 *   smem_A[HYBRID_TILE_HEAVY]  — current A tile
 *   smem_B[HYBRID_TILE_HEAVY]  — current B tile
 *
 * Future optimisation: replace the global→smem loads with
 * cp.async (Ampere+) to pipeline pair[i+1] loads with the
 * MAC computation for pair[i].
 * ============================================================ */
__global__ void __launch_bounds__(HYBRID_BLOCK_HEAVY_S1, 4)
hybrid_heavy_s1_kernel(HybridKernelArgs args)
{
    const int tid    = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x);

    __shared__ float smem_A[HYBRID_TILE_HEAVY];
    __shared__ float smem_B[HYBRID_TILE_HEAVY];

    /* Grid-stride loop: each CTA processes multiple s1 tasks.
     * ALL threads enter every iteration so __syncthreads is valid. */
    for (int bid = static_cast<int>(blockIdx.x);
         bid < args.n_s1;
         bid += stride)
    {
        const HybridS1Task task = args.s1_tasks[bid];
        const bool active = (tid < task.p_len);

        const int d_c     = task.c_offset;
        const int c_sr    = (d_c >= 0) ? 0 : -d_c;
        const int p_begin = task.p_begin;

        float acc = 0.0f;

        for (int pi = 0; pi < task.pair_count; ++pi) {
            const HybridPair pe = args.pairs[task.pair_begin + pi];
            const int ai  = pe.ai;
            const int bi  = pe.bi;
            const int d_a = args.A_offsets[ai];
            const int d_b = args.B_offsets[bi];

            /* ---- Collaborative load: A tile ---- */
            {
                const int a_sr  = (d_a >= 0) ? 0 : -d_a;
                const int a_pos = c_sr - a_sr + p_begin + tid;
                smem_A[tid] = (a_pos >= 0 && a_pos < args.A_lengths[ai])
                            ? args.A_vals[args.A_starts[ai] + a_pos]
                            : 0.0f;
            }

            /* ---- Collaborative load: B tile ---- */
            {
                const int b_sr  = (d_b >= 0) ? 0 : -d_b;
                const int b_pos = c_sr + d_a - b_sr + p_begin + tid;
                smem_B[tid] = (b_pos >= 0 && b_pos < args.B_lengths[bi])
                            ? args.B_vals[args.B_starts[bi] + b_pos]
                            : 0.0f;
            }

            __syncthreads();

            if (active) {
                acc += smem_A[tid] * smem_B[tid];
            }

            /* Sync before next pair iteration overwrites smem. */
            __syncthreads();
        }

        /* Write partial sum to exclusive slot (no atomics). */
        if (active) {
            args.partial_buf[task.partial_offset + tid] = acc;
        }
        /* No extra sync needed between grid-stride iterations:
         * smem is fully rewritten at the start of the next task. */
    }
}

/* ============================================================
 * HEAVY STAGE-2 REDUCE KERNEL  —  persistent, grid-stride
 *
 * Grid:   fixed occupancy-saturated size (num_SMs × 4).
 *         Each CTA iterates over its slice of s2_tasks[] with
 *         a grid-stride loop.
 *
 * Block:  HYBRID_BLOCK_HEAVY_S2 threads (= HYBRID_TILE_HEAVY).
 *
 * No shared memory or __syncthreads: each thread independently
 * reduces its scalar position across all partial slots and
 * writes the final value to C_vals.  Threads outside p_len
 * simply skip both the reduction and the write.
 *
 * Ordering guarantee: stage-2 is enqueued on the same CUDA
 * stream as stage-1, so the runtime guarantees all stage-1
 * writes to partial_buf are globally visible before any
 * stage-2 thread reads them.
 * ============================================================ */
__global__ void __launch_bounds__(HYBRID_BLOCK_HEAVY_S2, 4)
hybrid_heavy_s2_reduce_kernel(HybridKernelArgs args)
{
    const int tid    = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x);

    /* Grid-stride loop: no __syncthreads → safe to skip inactive
     * threads individually inside each iteration. */
    for (int bid = static_cast<int>(blockIdx.x);
         bid < args.n_s2;
         bid += stride)
    {
        const HybridS2Task task = args.s2_tasks[bid];

        if (tid < task.p_len) {
            float sum = 0.0f;

            #pragma unroll 4
            for (int p = 0; p < task.num_partials; ++p) {
                sum += args.partial_buf[task.partial_offset
                                        + p * HYBRID_TILE_HEAVY
                                        + tid];
            }

            const HybridCDiag cd = args.c_diags[task.c_idx];
            args.C_vals[cd.values_start + task.p_begin + tid] = sum;
        }
    }
}

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

/* Round up num_blocks to a full GPU wave of min_blocks_per_sm CTAs. */
static int hybrid_pad_to_wave(int num_blocks, int min_blocks_per_sm)
{
    if (num_blocks <= 0) return 0;
    int wave = hybrid_get_sm_count() * min_blocks_per_sm;
    if (wave <= 0) return num_blocks;
    return ((num_blocks + wave - 1) / wave) * wave;
}

/* ============================================================
 * build_hybrid_plan
 *
 * Host-side preprocessing that produces all task tables and
 * pair lists consumed by the three kernels.
 *
 * Requires A.offsets to be sorted ascending
 * (call sort_diag_matrix_by_offset before invoking this).
 *
 * Partial buffer layout:
 *   For heavy output diagonal dC with C segments S0, S1, ...
 *   and num_partitions = ceil(num_pairs / HYBRID_PAIRS_PER_PART):
 *
 *     base[S_i] = sum of (num_partitions * HYBRID_TILE_HEAVY)
 *                 for all segments before S_i (within dC)
 *                 PLUS the base of dC itself in partial_buf.
 *
 *   partial_buf[base[S_i] + p * HYBRID_TILE_HEAVY + tid]
 *     is the partial sum written by stage-1 partition p,
 *     thread tid, for segment S_i.
 * ============================================================ */
HybridPlan build_hybrid_plan(const DiagMatrix& A,
                              const DiagMatrix& B,
                              int M, int K, int N,
                              int corner_thresh,
                              int pairs_per_part)
{
    HybridPlan plan;

    /* Step 1 — enumerate output diagonals and contributor pairs. */
    auto out_map = build_output_diagonals(A, B, M, N);
    build_contributors(out_map, A, B);

    /* Precompute B offset bounds for the A-range binary search. */
    int B_off_min = *std::min_element(B.offsets.begin(), B.offsets.end());
    int B_off_max = *std::max_element(B.offsets.begin(), B.offsets.end());

    /* Resolve auto parameters (-1 = use compile-time defaults). */
    if (corner_thresh  < 0) corner_thresh  = HYBRID_CORNER_THRESH;
    if (pairs_per_part < 0) pairs_per_part = HYBRID_PAIRS_PER_PART;

    /* Step 2 — build c_diags table. */
    int c_val_offset = 0;
    std::map<int, int> c_idx_map;
    for (auto& [d_c, info] : out_map) {
        if (info.contributors.empty()) continue;
        int idx = static_cast<int>(plan.c_diags.size());
        c_idx_map[d_c] = idx;
        HybridCDiag cd;
        cd.c_offset    = d_c;
        cd.c_sr        = (d_c >= 0) ? 0 : -d_c;
        cd.length      = info.c_length;
        cd.values_start = c_val_offset;
        plan.c_diags.push_back(cd);
        c_val_offset += info.c_length;
    }
    plan.total_c_values = c_val_offset;

    /* Step 3 — tile each output diagonal and emit tasks. */
    int partial_offset = 0;

    for (auto& [d_c, info] : out_map) {
        if (info.contributors.empty()) continue;
        int c_idx     = c_idx_map[d_c];
        int num_pairs = static_cast<int>(info.contributors.size());
        /* Route to heavy only when BOTH conditions hold:
         *   1. many contributor pairs  → pair-partition parallelism needed
         *   2. long diagonal           → enough segments to amortise partial_buf
         * Short diagonals go corner regardless of pair count: their total
         * work is bounded by length, so sequential pair looping is cheaper
         * than the partial_buf round-trip.                                  */
        bool is_heavy = (num_pairs > corner_thresh)
                     && (info.c_length > HYBRID_TILE_HEAVY);

        if (!is_heavy) {
            /* ---- Corner path ---- */
            /* Precompute the valid A-diagonal index range once per dC. */
            int d_a_lo  = d_c - B_off_max;
            int d_a_hi  = d_c - B_off_min;
            int ai_begin = static_cast<int>(
                std::lower_bound(A.offsets.begin(), A.offsets.end(), d_a_lo)
                - A.offsets.begin());
            int ai_end   = static_cast<int>(
                std::upper_bound(A.offsets.begin(), A.offsets.end(), d_a_hi)
                - A.offsets.begin());

            for (int p = 0; p < info.c_length; p += HYBRID_TILE_CORNER) {
                int p_len = std::min(HYBRID_TILE_CORNER, info.c_length - p);

                HybridCornerTask task;
                task.c_idx    = c_idx;
                task.c_offset = d_c;
                task.p_begin  = p;
                task.p_len    = p_len;
                task.ai_begin = ai_begin;
                task.ai_end   = ai_end;
                plan.corner_tasks.push_back(task);
            }
        } else {
            /* ---- Heavy path ---- */

            /* Append this diagonal's pairs to the global pair list. */
            int pairs_base = static_cast<int>(plan.pairs.size());
            for (const auto& cp : info.contributors) {
                HybridPair hp;
                hp.ai = cp.a_diag_idx;
                hp.bi = cp.b_diag_idx;
                plan.pairs.push_back(hp);
            }

            int num_partitions = (num_pairs + pairs_per_part - 1)
                                 / pairs_per_part;

            for (int p = 0; p < info.c_length; p += HYBRID_TILE_HEAVY) {
                int p_len = std::min(HYBRID_TILE_HEAVY, info.c_length - p);

                /* Base offset in partial_buf for this (c_diag, segment). */
                int seg_partial_base = partial_offset;

                /* Stage-2 task index (emitted first so s1 tasks can
                 * reference it via s2_task_idx for the fused kernel). */
                int s2_idx = static_cast<int>(plan.s2_tasks.size());

                HybridS2Task t2;
                t2.c_idx          = c_idx;
                t2.p_begin        = p;
                t2.p_len          = p_len;
                t2.partial_offset = seg_partial_base;
                t2.num_partials   = num_partitions;
                plan.s2_tasks.push_back(t2);

                /* Stage-1 task per pair partition. */
                for (int part = 0; part < num_partitions; ++part) {
                    int pairs_begin = pairs_base + part * pairs_per_part;
                    int pairs_count = std::min(pairs_per_part,
                                               num_pairs - part * pairs_per_part);

                    HybridS1Task t1;
                    t1.c_idx           = c_idx;
                    t1.c_offset        = d_c;
                    t1.p_begin         = p;
                    t1.p_len           = p_len;
                    t1.pair_begin      = pairs_begin;
                    t1.pair_count      = pairs_count;
                    t1.partial_offset  = seg_partial_base
                                       + part * HYBRID_TILE_HEAVY;
                    t1.s2_task_idx     = s2_idx;
                    plan.s1_tasks.push_back(t1);
                }

                /* Advance partial_buf pointer past this segment's slots. */
                partial_offset += num_partitions * HYBRID_TILE_HEAVY;
            }
        }
    }

    plan.partial_buf_size = partial_offset;
    return plan;
}

/* ============================================================
 * launch_hybrid  —  persistent-kernel edition
 *
 * Each kernel is launched with a fixed, occupancy-saturated grid
 * (num_SMs × min_blocks_per_SM), regardless of task count.
 * The CTAs iterate through their task slices via grid-stride
 * loops, so there are always exactly 3 kernel launches with
 * constant overhead even for 100 000+ tasks.
 *
 * Ordering guarantee: all three kernels are enqueued on the
 * same stream (in-order FIFO), so stage-1 is guaranteed to
 * complete before stage-2 begins without any explicit event.
 * ============================================================ */
void launch_hybrid(HybridKernelArgs args, cudaStream_t stream)
{
    const int sm = hybrid_get_sm_count();

    /* ---- Corner kernel (persistent) ---- */
    if (args.n_corner > 0) {
        cudaFuncSetAttribute(hybrid_corner_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxL1);
        /* 8 blocks per SM = target occupancy for 128-thread blocks. */
        const int grid = std::min(sm * 8, args.n_corner);
        hybrid_corner_kernel<<<grid, HYBRID_BLOCK_CORNER, 0, stream>>>(args);
    }

    /* ---- Heavy stage-1 kernel (persistent) ---- */
    if (args.n_s1 > 0) {
        cudaFuncSetAttribute(hybrid_heavy_s1_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        /* 4 blocks per SM = target occupancy for 256-thread blocks
         * with 4 KB shared memory each. */
        const int grid = std::min(sm * 4, args.n_s1);
        hybrid_heavy_s1_kernel<<<grid, HYBRID_BLOCK_HEAVY_S1, 0, stream>>>(args);
    }

    /* ---- Heavy stage-2 reduce kernel (persistent, after s1) ---- */
    if (args.n_s2 > 0) {
        cudaFuncSetAttribute(hybrid_heavy_s2_reduce_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxL1);
        const int grid = std::min(sm * 4, args.n_s2);
        hybrid_heavy_s2_reduce_kernel<<<grid, HYBRID_BLOCK_HEAVY_S2, 0, stream>>>(args);
    }
}

/* ============================================================
 * HYBRID HEAVY FUSED KERNEL  —  pipelined s1 + s2
 *
 * Replaces the two separate s1/s2 kernel launches with a single
 * persistent kernel whose CTAs overlap s1 computation and s2
 * reduction at per-segment granularity.
 *
 * Control array layout (ctrl, device memory, 2 + n_s2 ints):
 *   ctrl[0]         : s1_next   — atomic counter, next s1 task to claim
 *   ctrl[1]         : s2_claim  — atomic counter, next s2 task to claim
 *   ctrl[2 + s2_idx]: pending[s2_idx] — countdown from num_partitions to 0;
 *                     when 0, all partial sums for that segment are committed.
 *
 * CTA life-cycle (interleaved, deadlock-free):
 *
 *   while (s2 tasks remain) {
 *       // Try to claim one s1 task (non-blocking: returns immediately)
 *       if (s1 tasks remain) {
 *           s1_idx = atomicAdd(ctrl[s1_next], 1)
 *           if s1_idx < n_s1:
 *               compute partial sum, write to partial_buf
 *               __threadfence()           // commit before signalling
 *               atomicSub(pending[s2_idx], 1)  // signal segment S ready
 *       }
 *
 *       // Scan for a ready, unclaimed s2 task (non-blocking)
 *       s2_idx = atomicAdd(ctrl[s2_claim], 1)  // claim next in order
 *       if s2_idx < n_s2:
 *           spin-wait until pending[s2_idx] == 0
 *           reduce partial slots → C_vals
 *   }
 *
 * Deadlock proof:
 *   A CTA only begins the s2 spin-wait after claiming an s2 slot.
 *   At that point there are fewer unclaimed s1 tasks than before,
 *   so at least one other CTA (or this CTA's next iteration) will
 *   claim and process the remaining s1 work.  Because every s1 task
 *   is eventually claimed and processed, every pending[] counter
 *   eventually reaches 0, and every spinning s2 CTA unblocks.
 *
 *   Liveness holds as long as grid_size <= n_s1 + n_s2 (the
 *   occupancy-saturated grid chosen by launch_hybrid_pipelined
 *   always satisfies this for non-trivial workloads).
 * ============================================================ */
__global__ void __launch_bounds__(HYBRID_BLOCK_HEAVY_S1, 4)
hybrid_heavy_fused_kernel(HybridKernelArgs args, int* ctrl)
{
    const int tid    = static_cast<int>(threadIdx.x);
    const int n_m_1  = args.n - 1;

    /* ctrl layout:  [s1_next | s2_claim | pending[0] ... pending[n_s2-1]] */
    int* const s1_next   = ctrl + 0;
    int* const s2_claim  = ctrl + 1;
    int* const pending   = ctrl + 2;

    __shared__ float smem_A[HYBRID_TILE_HEAVY];
    __shared__ float smem_B[HYBRID_TILE_HEAVY];

    /* Shared scratch so thread 0's atomic results are visible to all. */
    __shared__ int sh_idx;

    /* ---- Phase A: drain all s1 tasks (grid-stride) ---- *
     *
     * Each CTA claims s1 tasks until the global pool is exhausted.
     * All CTAs must complete Phase A before any CTA enters Phase B,
     * otherwise the following deadlock arises:
     *   n_s1 >> n_CTAs  →  only ~n_CTAs/n_s1 pending decrements per
     *   segment after one Phase A pass  →  all CTAs block in Phase B
     *   spin-wait while remaining s1 tasks are never executed.
     *
     * With sequential phases every pending count reaches 0 before the
     * first Phase B spin-wait begins.                                  */
    for (;;) {
        if (tid == 0) sh_idx = atomicAdd(s1_next, 1);
        __syncthreads();
        if (sh_idx >= args.n_s1) break;   /* s1 pool exhausted */

        const HybridS1Task task = args.s1_tasks[sh_idx];
        const bool active = (tid < task.p_len);

        const int d_c     = task.c_offset;
        const int c_sr    = (d_c >= 0) ? 0 : -d_c;
        const int p_begin = task.p_begin;

        float acc = 0.0f;

        for (int pi = 0; pi < task.pair_count; ++pi) {
            const HybridPair pe = args.pairs[task.pair_begin + pi];
            const int ai  = pe.ai;
            const int bi  = pe.bi;
            const int d_a = args.A_offsets[ai];
            const int d_b = args.B_offsets[bi];

            {
                const int a_sr  = (d_a >= 0) ? 0 : -d_a;
                const int a_pos = c_sr - a_sr + p_begin + tid;
                smem_A[tid] = (a_pos >= 0 && a_pos < args.A_lengths[ai])
                            ? args.A_vals[args.A_starts[ai] + a_pos]
                            : 0.0f;
            }
            {
                const int b_sr  = (d_b >= 0) ? 0 : -d_b;
                const int b_pos = c_sr + d_a - b_sr + p_begin + tid;
                smem_B[tid] = (b_pos >= 0 && b_pos < args.B_lengths[bi])
                            ? args.B_vals[args.B_starts[bi] + b_pos]
                            : 0.0f;
            }
            __syncthreads();
            if (active) acc += smem_A[tid] * smem_B[tid];
            __syncthreads();
        }

        if (active) {
            args.partial_buf[task.partial_offset + tid] = acc;
        }
        __syncthreads();

        /* Flush partial_buf to L2/DRAM before signalling s2. */
        if (tid == 0) {
            __threadfence();
            atomicSub(&pending[task.s2_task_idx], 1);
        }
        __syncthreads();
    }

    /* ---- Phase B: drain all s2 tasks (grid-stride) ---- *
     *
     * All s1 tasks are now either done or in-flight on other CTAs.
     * Spin-wait on pending[s2_idx] is guaranteed to terminate because
     * all s1 tasks will complete (the Phase A loop above is bounded).  */
    for (;;) {
        if (tid == 0) sh_idx = atomicAdd(s2_claim, 1);
        __syncthreads();
        const int s2_idx = sh_idx;

        if (s2_idx >= args.n_s2) {
            if (tid == 0) atomicSub(s2_claim, 1);
            __syncthreads();
            break;
        }

        /* Spin until all s1 partitions for this segment are committed. */
        if (tid == 0) {
            while (atomicAdd(&pending[s2_idx], 0) != 0) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
        }
        __syncthreads();

        /* Reduce partial slots → C. */
        const HybridS2Task task = args.s2_tasks[s2_idx];
        if (tid < task.p_len) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int p = 0; p < task.num_partials; ++p) {
                sum += args.partial_buf[task.partial_offset
                                        + p * HYBRID_TILE_HEAVY
                                        + tid];
            }
            const HybridCDiag cd = args.c_diags[task.c_idx];
            args.C_vals[cd.values_start + task.p_begin + tid] = sum;
        }
        __syncthreads();
    }
}

/* ============================================================
 * init_fused_ctrl
 *
 * Resets the device control array before each fused launch.
 * Must be called on the same stream, before hybrid_heavy_fused_kernel.
 * ============================================================ */
__global__ static void
fused_ctrl_init_kernel(int* ctrl, int n_s2,
                        const int* num_partitions_per_s2)
{
    /* ctrl[0] = s1_next, ctrl[1] = s2_claim */
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ctrl[0] = 0;
        ctrl[1] = 0;
    }
    /* ctrl[2 + i] = pending[i] */
    for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_s2;
         i += static_cast<int>(gridDim.x) * blockDim.x)
    {
        ctrl[2 + i] = num_partitions_per_s2[i];
    }
}

void init_fused_ctrl(HybridKernelArgs args,
                     int*             ctrl,
                     cudaStream_t     stream)
{
    /* The host must also provide a device array of num_partitions per s2
     * task.  We derive it from the s2_tasks array already on device.
     * For simplicity this implementation requires a companion device
     * array d_num_partitions[n_s2] pre-filled by the host from
     * plan.s2_tasks[i].num_partials.
     *
     * Alternative: zero ctrl[0..1] with cudaMemsetAsync and fill
     * pending[] with a simple init kernel as shown below. */
    if (args.n_s2 == 0) return;
    int grid = std::min(hybrid_get_sm_count() * 4, args.n_s2);
    /* Caller is responsible for passing d_num_partitions in a
     * side-channel (e.g., stored adjacent to ctrl in the same
     * allocation).  See launch_hybrid_pipelined for the convention. */
    fused_ctrl_init_kernel<<<grid, 256, 0, stream>>>(
        ctrl, args.n_s2,
        /* d_num_partitions: */ ctrl + 2 + args.n_s2);
}

/* ============================================================
 * launch_hybrid_pipelined
 *
 * Single fused kernel for heavy work (s1 + s2 overlapped) plus
 * the persistent corner kernel.  Reduces 3 kernel launches to 2
 * and allows s2 to start as soon as individual segment's partials
 * are committed — hiding s2 latency inside the s1 phase.
 *
 * ctrl allocation (caller responsibility):
 *   cudaMalloc(&ctrl, (2 + 2*n_s2) * sizeof(int));
 *   // [s1_next | s2_claim | pending[0..n_s2) | num_partitions[0..n_s2)]
 *   // Fill num_partitions[i] = plan.s2_tasks[i].num_partials before use.
 *
 * The init kernel (enqueued on the same stream before the fused
 * kernel) copies num_partitions[] into pending[] and zeros the
 * two counters, so ctrl can be reused across repeated calls.
 * ============================================================ */
void launch_hybrid_pipelined(HybridKernelArgs args,
                              int*             ctrl,
                              cudaStream_t     stream)
{
    const int sm = hybrid_get_sm_count();

    /* ---- Corner kernel (unchanged) ---- */
    if (args.n_corner > 0) {
        cudaFuncSetAttribute(hybrid_corner_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxL1);
        const int grid = std::min(sm * 8, args.n_corner);
        hybrid_corner_kernel<<<grid, HYBRID_BLOCK_CORNER, 0, stream>>>(args);
    }

    if (args.n_s1 == 0 && args.n_s2 == 0) return;

    /* ---- Reset ctrl for this invocation (async, same stream) ---- */
    init_fused_ctrl(args, ctrl, stream);

    /* ---- Fused s1 + s2 kernel ----
     *
     * Grid size: enough CTAs to saturate the GPU, but capped at
     * n_s1 + n_s2 since each CTA processes at least one task.
     * The interleaved scheduler ensures CTAs never all pile up in
     * the s2 spin-wait while s1 work remains. */
    cudaFuncSetAttribute(hybrid_heavy_fused_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    const int grid = std::min(sm * 4, args.n_s1 + args.n_s2);
    hybrid_heavy_fused_kernel<<<grid, HYBRID_BLOCK_HEAVY_S1, 0, stream>>>(
        args, ctrl);
}

/* ============================================================
 * Example host driver (illustrates full usage)
 *
 * void run_hybrid_spmm(
 *     const DiagMatrix& A,
 *     DiagMatrix& B,          // will be offset-sorted in place
 *     int M, int K, int N,
 *     float*& d_C_vals,       // device output (allocated here)
 *     cudaStream_t stream)
 * {
 *     // 0. Ensure A offsets are sorted (required for binary search)
 *     sort_diag_matrix_by_offset(A_mutable);
 *
 *     // 1. Host preprocessing
 *     HybridPlan plan = build_hybrid_plan(A, B, M, K, N);
 *
 *     // 2. Upload all arrays to device
 *     // ... (corner_tasks, s1_tasks, s2_tasks, c_diags, pairs,
 *     //      A_vals, A_offsets, A_starts, A_lengths,
 *     //      B_vals, B_offsets, B_starts, B_lengths, B_lookup)
 *
 *     // 3. Allocate device buffers
 *     cudaMalloc(&d_C_vals, plan.total_c_values * sizeof(float));
 *     float* d_partial_buf = nullptr;
 *     if (plan.partial_buf_size > 0)
 *         cudaMalloc(&d_partial_buf, plan.partial_buf_size * sizeof(float));
 *
 *     // 4. Fill HybridKernelArgs and launch
 *     HybridKernelArgs kargs = { ... };
 *     launch_hybrid(kargs, stream);
 *
 *     // 5. Synchronize, download results, free partial_buf
 * }
 * ============================================================ */
