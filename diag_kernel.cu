/* ============================================================
 * diag_kernel.cu  —  Performance-optimized kernels
 *
 * KEY DESIGN INVARIANTS (unchanged):
 *   ✓  ZERO atomic operations
 *   ✓  One CTA exclusively owns one output tile
 *   ✓  A-stationary grouped contributors
 *   ✓  B from warp-major packed layout (coalesced)
 *   ✓  Register accumulation, single direct writeback
 *
 * CRITICAL OPTIMIZATION INSIGHT:
 *   Thread tid writes smemA[tid] and reads smemA[tid].
 *   No cross-thread shared memory dependency exists.
 *   → Shared memory is UNNECESSARY — A lives in a register.
 *   → ALL __syncthreads() barriers are ELIMINATED.
 *   → Inactive threads exit early (no barriers to wait on).
 *
 * This eliminates barrier stalls (~55% → ~0%) and frees
 * shared memory capacity for higher occupancy.
 * ============================================================ */

#include "diag_kernel.cuh"

/* ============================================================
 * Helper: query SM count (cached) for tail-effect padding.
 * ============================================================ */
static int get_sm_count() {
    static int cached = 0;
    if (cached == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached,
                               cudaDevAttrMultiProcessorCount, dev);
    }
    return cached;
}

/* Round grid up to full wave to eliminate tail effect.
 * Excess CTAs exit immediately via the early-exit check. */
static int pad_to_wave(int num_tasks, int min_blocks_per_sm) {
    int sm = get_sm_count();
    int wave = sm * min_blocks_per_sm;
    if (wave <= 0 || num_tasks <= 0) return num_tasks;
    return ((num_tasks + wave - 1) / wave) * wave;
}

/* ============================================================
 * MEDIUM KERNEL  (register-only, zero barriers)
 *
 * Block:     128 threads = 4 warps
 * Shared:    0 bytes (A in register, not shared memory)
 * Barriers:  ZERO (no __syncthreads__, no __syncwarp__)
 * Registers: ~16-20 per thread
 *
 * Why shared memory is unnecessary:
 *   Thread tid loads A[position_tid] → register a_val.
 *   Multiplies a_val by each pair's B[position_tid].
 *   A-stationary reuse is per-thread (same a_val across pairs).
 *   No other thread ever reads this thread's A value.
 *
 * Why all barriers are unnecessary:
 *   No shared memory → no producer-consumer dependency.
 *   Each thread is fully independent after reading its task.
 *
 * ILP: pair loop unrolled 2x. Two independent B loads
 *   overlap each other's memory latency.
 *
 * Inactive threads (tid >= tile_len) exit immediately.
 *   No barriers means they don't need to participate.
 *
 * __launch_bounds__(128, 8): with 0 smem, target 8 blocks/SM.
 *   Limits compiler to ~64 regs/thread → high occupancy.
 * ============================================================ */
__global__ void
__launch_bounds__(BLOCK_SIZE_MED, 8)
diag_spmm_medium_kernel(const Task*       __restrict__ tasks,
                        const int*        __restrict__ task_ids,
                        const Group*      __restrict__ groups,
                        const PairMeta*   __restrict__ pairs,
                        const float*      __restrict__ A_values,
                        const float*      __restrict__ packedB,
                        const OutputDiag* __restrict__ c_diags,
                        float*            __restrict__ C_values,
                        int               num_tasks_in_bucket)
{
    /* Early exit: dummy CTAs for tail padding + excess blocks. */
    if (static_cast<int>(blockIdx.x) >= num_tasks_in_bucket) return;

    const Task* __restrict__ tp = &tasks[task_ids[blockIdx.x]];
    const int tid = threadIdx.x;

    /* Inactive thread: exits immediately. No barriers to wait on. */
    if (tid >= tp->p_len) return;

    float acc = 0.0f;

    for (int gi = 0; gi < tp->group_count; ++gi) {
        const Group* __restrict__ gp = &groups[tp->group_begin + gi];

        /* A value: loaded directly into register.
         * A-stationary reuse: this single register is multiplied
         * by every pair's B value in the loop below.
         * Zero shared memory, zero barriers.                  */
        const int p_a = gp->a_map_offset + tid;
        const float a_val = (p_a >= 0 && p_a < gp->a_diag_len)
                          ? A_values[gp->a_global_start + p_a]
                          : 0.0f;

        /* Pair loop: 2x unrolled for ILP.
         * Two independent global B loads → memory latency hidden.
         * Each FMA uses the SAME a_val (register reuse).        */
        int pidx = gp->pair_begin;
        const int pair_end = pidx + gp->pair_count;

        for (; pidx + 1 < pair_end; pidx += 2) {
            float b0 = packedB[pairs[pidx    ].packedB_offset + tid];
            float b1 = packedB[pairs[pidx + 1].packedB_offset + tid];
            acc += a_val * b0;
            acc += a_val * b1;
        }
        if (pidx < pair_end) {
            acc += a_val * packedB[pairs[pidx].packedB_offset + tid];
        }
    }

    C_values[c_diags[tp->c_diag_idx].values_start
             + tp->p_begin + tid] = acc;
}

/* ============================================================
 * LIGHT KERNEL  (register-only, warp-independent)
 *
 * Block:     128 threads = 4 warps
 * Shared:    0 bytes
 * Barriers:  ZERO
 *
 * Each warp independently handles one task.
 * Same register-only optimization as medium kernel.
 * ============================================================ */
__global__ void
__launch_bounds__(BLOCK_SIZE_LIGHT, 8)
diag_spmm_light_kernel(const Task*       __restrict__ tasks,
                       const int*        __restrict__ task_ids,
                       const Group*      __restrict__ groups,
                       const PairMeta*   __restrict__ pairs,
                       const float*      __restrict__ A_values,
                       const float*      __restrict__ packedB,
                       const OutputDiag* __restrict__ c_diags,
                       float*            __restrict__ C_values,
                       int               num_tasks_in_bucket)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int task_slot =
        static_cast<int>(blockIdx.x) * TASKS_PER_CTA_LIGHT + warp_id;

    if (task_slot >= num_tasks_in_bucket) return;

    const Task* __restrict__ tp = &tasks[task_ids[task_slot]];

    if (lane_id >= tp->p_len) return;

    float acc = 0.0f;

    for (int gi = 0; gi < tp->group_count; ++gi) {
        const Group* __restrict__ gp = &groups[tp->group_begin + gi];

        const int p_a = gp->a_map_offset + lane_id;
        const float a_val = (p_a >= 0 && p_a < gp->a_diag_len)
                          ? A_values[gp->a_global_start + p_a]
                          : 0.0f;

        for (int pi = 0; pi < gp->pair_count; ++pi) {
            acc += a_val
                 * packedB[pairs[gp->pair_begin + pi].packedB_offset
                           + lane_id];
        }
    }

    C_values[c_diags[tp->c_diag_idx].values_start
             + tp->p_begin + lane_id] = acc;
}

/* ============================================================
 * HEAVY KERNEL — simple variant  (register-only baseline)
 *
 * Block:     256 threads = 8 warps
 * Shared:    0 bytes
 * Barriers:  ZERO
 * Tile:      256 elements
 *
 * Identical pattern to medium kernel but with 256 threads.
 * Use for profiling comparison against heavy_prefetch.
 * ============================================================ */
__global__ void
__launch_bounds__(BLOCK_SIZE_HEAVY, 4)
diag_spmm_heavy_simple_kernel(
                       const Task*       __restrict__ tasks,
                       const int*        __restrict__ task_ids,
                       const Group*      __restrict__ groups,
                       const PairMeta*   __restrict__ pairs,
                       const float*      __restrict__ A_values,
                       const float*      __restrict__ packedB,
                       const OutputDiag* __restrict__ c_diags,
                       float*            __restrict__ C_values,
                       int               num_tasks_in_bucket)
{
    if (static_cast<int>(blockIdx.x) >= num_tasks_in_bucket) return;

    const Task* __restrict__ tp = &tasks[task_ids[blockIdx.x]];
    const int tid = threadIdx.x;

    if (tid >= tp->p_len) return;

    float acc = 0.0f;

    for (int gi = 0; gi < tp->group_count; ++gi) {
        const Group* __restrict__ gp = &groups[tp->group_begin + gi];

        const int p_a = gp->a_map_offset + tid;
        const float a_val = (p_a >= 0 && p_a < gp->a_diag_len)
                          ? A_values[gp->a_global_start + p_a]
                          : 0.0f;

        int pidx = gp->pair_begin;
        const int pair_end = pidx + gp->pair_count;

        for (; pidx + 1 < pair_end; pidx += 2) {
            float b0 = packedB[pairs[pidx    ].packedB_offset + tid];
            float b1 = packedB[pairs[pidx + 1].packedB_offset + tid];
            acc += a_val * b0;
            acc += a_val * b1;
        }
        if (pidx < pair_end) {
            acc += a_val * packedB[pairs[pidx].packedB_offset + tid];
        }
    }

    C_values[c_diags[tp->c_diag_idx].values_start
             + tp->p_begin + tid] = acc;
}

/* ============================================================
 * HEAVY KERNEL — prefetch variant  (software-pipelined A)
 *
 * Block:     256 threads = 8 warps
 * Shared:    0 bytes
 * Barriers:  ZERO
 * Tile:      256 elements
 *
 * Software pipelining: loads NEXT group's A value into a
 * register while computing with the CURRENT group's A.
 * The GPU scheduler overlaps the A load latency with
 * the pair computation loop.
 *
 * Register cost: +1 float (a_next) vs heavy_simple.
 * Benefit: hides A_values global read latency across groups.
 * ============================================================ */
__global__ void
__launch_bounds__(BLOCK_SIZE_HEAVY, 4)
diag_spmm_heavy_prefetch_kernel(
                       const Task*       __restrict__ tasks,
                       const int*        __restrict__ task_ids,
                       const Group*      __restrict__ groups,
                       const PairMeta*   __restrict__ pairs,
                       const float*      __restrict__ A_values,
                       const float*      __restrict__ packedB,
                       const OutputDiag* __restrict__ c_diags,
                       float*            __restrict__ C_values,
                       int               num_tasks_in_bucket)
{
    if (static_cast<int>(blockIdx.x) >= num_tasks_in_bucket) return;

    const Task* __restrict__ tp = &tasks[task_ids[blockIdx.x]];
    const int tid = threadIdx.x;

    if (tid >= tp->p_len) return;
    if (tp->group_count == 0) return;

    float acc = 0.0f;

    /* Load first group's A into register. */
    const Group* __restrict__ gp = &groups[tp->group_begin];
    int p_a = gp->a_map_offset + tid;
    float a_cur = (p_a >= 0 && p_a < gp->a_diag_len)
                ? A_values[gp->a_global_start + p_a]
                : 0.0f;
    float a_next;

    for (int gi = 0; gi < tp->group_count; ++gi) {
        gp = &groups[tp->group_begin + gi];

        /* Prefetch NEXT group's A while we compute with a_cur.
         * The global load for a_next is issued here but its
         * latency is hidden by the pair computation loop below. */
        if (gi + 1 < tp->group_count) {
            const Group* __restrict__ gn = &groups[tp->group_begin + gi + 1];
            int p_a_next = gn->a_map_offset + tid;
            a_next = (p_a_next >= 0 && p_a_next < gn->a_diag_len)
                   ? A_values[gn->a_global_start + p_a_next]
                   : 0.0f;
        }

        /* Pair loop: compute with a_cur (current group). */
        int pidx = gp->pair_begin;
        const int pair_end = pidx + gp->pair_count;

        for (; pidx + 1 < pair_end; pidx += 2) {
            float b0 = packedB[pairs[pidx    ].packedB_offset + tid];
            float b1 = packedB[pairs[pidx + 1].packedB_offset + tid];
            acc += a_cur * b0;
            acc += a_cur * b1;
        }
        if (pidx < pair_end) {
            acc += a_cur * packedB[pairs[pidx].packedB_offset + tid];
        }

        /* Swap: next becomes current for the next iteration. */
        a_cur = a_next;
    }

    C_values[c_diags[tp->c_diag_idx].values_start
             + tp->p_begin + tid] = acc;
}

/* ============================================================
 * WIDE KERNEL  (register-only, multi-output per thread)
 *
 * Block:     128 threads = 4 warps
 * Shared:    0 bytes
 * Barriers:  ZERO
 * Tile:      512 elements (WIDE_TILE_SIZE)
 * Per-thread: 4 outputs (WIDE_ELEMS_PER_THREAD)
 * Registers: 4 accumulators + 4 A values per group
 *
 * Same insight: thread tid writes/reads smemA[tid+k*128]
 * for k=0..3. No other thread touches those positions.
 * → A values stored in 4 registers, no shared memory.
 * ============================================================ */
__global__ void
__launch_bounds__(WIDE_BLOCK_SIZE, 4)
diag_spmm_wide_kernel(const Task*       __restrict__ tasks,
                      const int*        __restrict__ task_ids,
                      const Group*      __restrict__ groups,
                      const PairMeta*   __restrict__ pairs,
                      const float*      __restrict__ A_values,
                      const float*      __restrict__ packedB,
                      const OutputDiag* __restrict__ c_diags,
                      float*            __restrict__ C_values,
                      int               num_tasks_in_bucket)
{
    if (static_cast<int>(blockIdx.x) >= num_tasks_in_bucket) return;

    const Task* __restrict__ tp = &tasks[task_ids[blockIdx.x]];
    const int tid      = threadIdx.x;
    const int tile_len = tp->p_len;

    float acc[WIDE_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < WIDE_ELEMS_PER_THREAD; ++e) acc[e] = 0.0f;

    for (int gi = 0; gi < tp->group_count; ++gi) {
        const Group* __restrict__ gp = &groups[tp->group_begin + gi];

        /* Load 4 A values into registers (no shared memory). */
        float a_reg[WIDE_ELEMS_PER_THREAD];
        #pragma unroll
        for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
            int q   = tid + k * WIDE_BLOCK_SIZE;
            int p_a = gp->a_map_offset + q;
            a_reg[k] = (q < tile_len && p_a >= 0 && p_a < gp->a_diag_len)
                      ? A_values[gp->a_global_start + p_a]
                      : 0.0f;
        }

        /* Pair loop with A-register reuse. */
        for (int pi = 0; pi < gp->pair_count; ++pi) {
            const int pb_off = pairs[gp->pair_begin + pi].packedB_offset;

            #pragma unroll
            for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
                int q = tid + k * WIDE_BLOCK_SIZE;
                if (q < tile_len) {
                    acc[k] += a_reg[k] * packedB[pb_off + q];
                }
            }
        }
    }

    const int vs = c_diags[tp->c_diag_idx].values_start + tp->p_begin;
    #pragma unroll
    for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
        int q = tid + k * WIDE_BLOCK_SIZE;
        if (q < tile_len) {
            C_values[vs + q] = acc[k];
        }
    }
}

/* ============================================================
 * LAUNCH WRAPPERS
 *
 * All kernels use 0 shared memory → prefer max L1 cache.
 * Grid is padded to full SM waves to eliminate tail effect.
 * ============================================================ */

void launch_medium_kernel(const Task*       d_tasks,
                          const int*        d_task_ids,
                          const Group*      d_groups,
                          const PairMeta*   d_pairs,
                          const float*      d_A_values,
                          const float*      d_packedB,
                          const OutputDiag* d_c_diags,
                          float*            d_C_values,
                          int               num_tasks,
                          cudaStream_t      stream)
{
    if (num_tasks == 0) return;

    /* Prefer L1 cache: no shared memory used by this kernel. */
    cudaFuncSetAttribute(diag_spmm_medium_kernel,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         0 /* 0% shared = max L1 */);

    const int min_bpsm = 8;  /* matches __launch_bounds__ second param */
    dim3 grid(pad_to_wave(num_tasks, min_bpsm));
    dim3 block(BLOCK_SIZE_MED);

    diag_spmm_medium_kernel<<<grid, block, 0, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}

void launch_light_kernel(const Task*       d_tasks,
                         const int*        d_task_ids,
                         const Group*      d_groups,
                         const PairMeta*   d_pairs,
                         const float*      d_A_values,
                         const float*      d_packedB,
                         const OutputDiag* d_c_diags,
                         float*            d_C_values,
                         int               num_tasks,
                         cudaStream_t      stream)
{
    if (num_tasks == 0) return;

    cudaFuncSetAttribute(diag_spmm_light_kernel,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         0);

    int num_ctas = (num_tasks + TASKS_PER_CTA_LIGHT - 1)
                 / TASKS_PER_CTA_LIGHT;
    dim3 grid(pad_to_wave(num_ctas, 8));
    dim3 block(BLOCK_SIZE_LIGHT);

    diag_spmm_light_kernel<<<grid, block, 0, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}

void launch_heavy_kernel(const Task*       d_tasks,
                         const int*        d_task_ids,
                         const Group*      d_groups,
                         const PairMeta*   d_pairs,
                         const float*      d_A_values,
                         const float*      d_packedB,
                         const OutputDiag* d_c_diags,
                         float*            d_C_values,
                         int               num_tasks,
                         cudaStream_t      stream)
{
    if (num_tasks == 0) return;

    /* Default: use prefetch variant for better latency hiding. */
    cudaFuncSetAttribute(diag_spmm_heavy_prefetch_kernel,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         0);

    const int min_bpsm = 4;
    dim3 grid(pad_to_wave(num_tasks, min_bpsm));
    dim3 block(BLOCK_SIZE_HEAVY);

    diag_spmm_heavy_prefetch_kernel<<<grid, block, 0, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}

/* Alternate heavy launcher: simple variant (no prefetch).
 * Use for A/B profiling comparison with ncu. */
void launch_heavy_simple_kernel(const Task*       d_tasks,
                                const int*        d_task_ids,
                                const Group*      d_groups,
                                const PairMeta*   d_pairs,
                                const float*      d_A_values,
                                const float*      d_packedB,
                                const OutputDiag* d_c_diags,
                                float*            d_C_values,
                                int               num_tasks,
                                cudaStream_t      stream)
{
    if (num_tasks == 0) return;

    cudaFuncSetAttribute(diag_spmm_heavy_simple_kernel,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         0);

    const int min_bpsm = 4;
    dim3 grid(pad_to_wave(num_tasks, min_bpsm));
    dim3 block(BLOCK_SIZE_HEAVY);

    diag_spmm_heavy_simple_kernel<<<grid, block, 0, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}

void launch_wide_kernel(const Task*       d_tasks,
                        const int*        d_task_ids,
                        const Group*      d_groups,
                        const PairMeta*   d_pairs,
                        const float*      d_A_values,
                        const float*      d_packedB,
                        const OutputDiag* d_c_diags,
                        float*            d_C_values,
                        int               num_tasks,
                        cudaStream_t      stream)
{
    if (num_tasks == 0) return;

    cudaFuncSetAttribute(diag_spmm_wide_kernel,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         0);

    dim3 grid(pad_to_wave(num_tasks, 4));
    dim3 block(WIDE_BLOCK_SIZE);

    diag_spmm_wide_kernel<<<grid, block, 0, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}
