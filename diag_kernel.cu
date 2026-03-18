/* ============================================================
 * diag_kernel.cu
 *
 * Optimized kernel implementations for diagonal sparse matrix
 * multiplication.
 *
 * KEY DESIGN INVARIANTS:
 *   ✓  ZERO atomic operations anywhere
 *   ✓  One CTA exclusively owns one output tile
 *   ✓  A is stationary in shared memory, reused across pairs
 *   ✓  B is read from warp-major packed layout (coalesced)
 *   ✓  Accumulation in registers, single direct writeback
 *   ✓  All complex index mapping pre-resolved on host
 *
 * OPTIMIZATION PRINCIPLES (implementation-level):
 *   ✓  Structs accessed via pointer, never copied by value
 *   ✓  Variable lifetimes minimized (b_val, a_val, indices)
 *   ✓  __launch_bounds__ on every kernel for register control
 *   ✓  Inactive threads exit early where possible
 *   ✓  Only A in shared memory; B stays in global (streaming)
 * ============================================================ */

#include "diag_kernel.cuh"

/* ============================================================
 * MEDIUM KERNEL  (lean version — minimal register footprint)
 *
 * Block:     128 threads = 4 warps
 * Shared:    TILE_SIZE floats (smemA, 512 bytes)
 * Registers: 1 float accumulator per thread
 * Target:    ~32 registers/thread
 *
 * Optimizations vs. original:
 *   - Task/Group/PairMeta accessed via const pointer (no copy)
 *   - warp_id/lane_id removed (unused)
 *   - b_val scope minimized to inner loop body
 *   - Single fused branch for active-thread check
 *   - __launch_bounds__(128, 2) hints compiler for occupancy
 * ============================================================ */
__global__ void
__launch_bounds__(BLOCK_SIZE_MED, 2)
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
    if (static_cast<int>(blockIdx.x) >= num_tasks_in_bucket) return;

    /* Access task via pointer — avoid copying 8-int struct to registers. */
    const Task* __restrict__ tp = &tasks[task_ids[blockIdx.x]];

    const int tid      = threadIdx.x;
    const int tile_len = tp->p_len;

    extern __shared__ float smemA[];

    float acc = 0.0f;

    /* Main loop: iterate over groups (A-stationary reuse). */
    for (int gi = 0; gi < tp->group_count; ++gi) {

        const Group* __restrict__ gp = &groups[tp->group_begin + gi];

        /* Step 1: Load A slice into shared memory (coalesced). */
        {
            int p_a = gp->a_map_offset + tid;
            smemA[tid] = (tid < tile_len && p_a >= 0 && p_a < gp->a_diag_len)
                       ? A_values[gp->a_global_start + p_a]
                       : 0.0f;
        }

        __syncthreads();

        /* Step 2: Iterate over pairs — smemA reused P times. */
        if (tid < tile_len) {
            float a_val = smemA[tid];
            for (int pi = 0; pi < gp->pair_count; ++pi) {
                const PairMeta* __restrict__ pp = &pairs[gp->pair_begin + pi];
                acc += a_val * packedB[pp->packedB_offset + tid];
            }
        }

        __syncthreads();
    }

    /* Final writeback — exclusive tile ownership, no atomics. */
    if (tid < tile_len) {
        C_values[c_diags[tp->c_diag_idx].values_start
                 + tp->p_begin + tid] = acc;
    }
}

/* ============================================================
 * LIGHT KERNEL  (multi-task packing: 4 tasks per CTA)
 *
 * Block:     128 threads = 4 warps
 * Shared:    4 * WARP_SIZE floats (independent partitions)
 * Registers: 1 float accumulator per thread
 *
 * Each warp independently processes one task.
 * Warp-level sync only (__syncwarp).
 * ============================================================ */
__global__ void
__launch_bounds__(BLOCK_SIZE_LIGHT, 2)
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

    const int task_slot = static_cast<int>(blockIdx.x) * TASKS_PER_CTA_LIGHT + warp_id;

    extern __shared__ float smem[];
    float* my_smemA = smem + warp_id * WARP_SIZE;

    if (task_slot >= num_tasks_in_bucket) return;

    /* Access task via pointer. */
    const Task* __restrict__ tp = &tasks[task_ids[task_slot]];
    const int tile_len = tp->p_len;

    float acc = 0.0f;

    for (int gi = 0; gi < tp->group_count; ++gi) {
        const Group* __restrict__ gp = &groups[tp->group_begin + gi];

        /* Load A slice into this warp's partition. */
        {
            int p_a = gp->a_map_offset + lane_id;
            my_smemA[lane_id] = (lane_id < tile_len && p_a >= 0 && p_a < gp->a_diag_len)
                              ? A_values[gp->a_global_start + p_a]
                              : 0.0f;
        }

        __syncwarp();

        if (lane_id < tile_len) {
            float a_val = my_smemA[lane_id];
            for (int pi = 0; pi < gp->pair_count; ++pi) {
                const PairMeta* __restrict__ pp = &pairs[gp->pair_begin + pi];
                acc += a_val * packedB[pp->packedB_offset + lane_id];
            }
        }

        __syncwarp();
    }

    if (lane_id < tile_len) {
        C_values[c_diags[tp->c_diag_idx].values_start
                 + tp->p_begin + lane_id] = acc;
    }
}

/* ============================================================
 * HEAVY KERNEL  (double-buffered smemA, 256 threads)
 *
 * Block:     256 threads = 8 warps
 * Shared:    2 * TILE_SIZE_HEAVY floats (ping-pong)
 * Registers: 1 float accumulator per thread
 *
 * Prefetches next group's A slice while computing current.
 * ============================================================ */
__global__ void
__launch_bounds__(BLOCK_SIZE_HEAVY, 1)
diag_spmm_heavy_kernel(const Task*       __restrict__ tasks,
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

    extern __shared__ float smem_heavy[];
    float* smemA_buf[2] = { smem_heavy, smem_heavy + TILE_SIZE_HEAVY };

    float acc = 0.0f;
    int buf = 0;

    if (tp->group_count == 0) return;

    /* Prefetch group 0 into buffer 0. */
    {
        const Group* __restrict__ g0 = &groups[tp->group_begin];
        int p_a = g0->a_map_offset + tid;
        smemA_buf[0][tid] = (tid < tile_len && p_a >= 0 && p_a < g0->a_diag_len)
                          ? A_values[g0->a_global_start + p_a]
                          : 0.0f;
    }
    __syncthreads();

    for (int gi = 0; gi < tp->group_count; ++gi) {
        const Group* __restrict__ gp = &groups[tp->group_begin + gi];

        /* Compute all pairs using smemA_buf[buf]. */
        if (tid < tile_len) {
            float a_val = smemA_buf[buf][tid];
            for (int pi = 0; pi < gp->pair_count; ++pi) {
                const PairMeta* __restrict__ pp = &pairs[gp->pair_begin + pi];
                acc += a_val * packedB[pp->packedB_offset + tid];
            }
        }

        __syncthreads();

        /* Prefetch next group into the other buffer. */
        if (gi + 1 < tp->group_count) {
            const Group* __restrict__ gn = &groups[tp->group_begin + gi + 1];
            int p_a = gn->a_map_offset + tid;
            smemA_buf[1 - buf][tid] = (tid < tile_len && p_a >= 0 && p_a < gn->a_diag_len)
                                    ? A_values[gn->a_global_start + p_a]
                                    : 0.0f;
            __syncthreads();
        }

        buf = 1 - buf;
    }

    if (tid < tile_len) {
        C_values[c_diags[tp->c_diag_idx].values_start
                 + tp->p_begin + tid] = acc;
    }
}

/* ============================================================
 * WIDE KERNEL  (multi-output per thread, tile 512)
 *
 * Block:     128 threads = 4 warps
 * Tile:      512 output positions (WIDE_TILE_SIZE)
 * Per-thread: 4 output positions (WIDE_ELEMS_PER_THREAD)
 * Shared:    WIDE_TILE_SIZE floats (2048 bytes)
 * Registers: 4 float accumulators per thread
 * ============================================================ */
__global__ void
__launch_bounds__(WIDE_BLOCK_SIZE, 1)
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

    extern __shared__ float smemA_wide[];

    float acc[WIDE_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < WIDE_ELEMS_PER_THREAD; ++e) acc[e] = 0.0f;

    for (int gi = 0; gi < tp->group_count; ++gi) {
        const Group* __restrict__ gp = &groups[tp->group_begin + gi];

        /* Load A slice in BLOCK_SIZE-element chunks (coalesced). */
        #pragma unroll
        for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
            int q   = tid + k * WIDE_BLOCK_SIZE;
            int p_a = gp->a_map_offset + q;
            smemA_wide[q] = (q < tile_len && p_a >= 0 && p_a < gp->a_diag_len)
                          ? A_values[gp->a_global_start + p_a]
                          : 0.0f;
        }

        __syncthreads();

        for (int pi = 0; pi < gp->pair_count; ++pi) {
            const PairMeta* __restrict__ pp = &pairs[gp->pair_begin + pi];
            const int pb_off = pp->packedB_offset;

            #pragma unroll
            for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
                int q = tid + k * WIDE_BLOCK_SIZE;
                if (q < tile_len) {
                    acc[k] += smemA_wide[q] * packedB[pb_off + q];
                }
            }
        }

        __syncthreads();
    }

    /* Writeback: each thread writes WIDE_ELEMS_PER_THREAD positions. */
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
 * Launch wrappers
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
    dim3 grid(num_tasks);
    dim3 block(BLOCK_SIZE_MED);
    int  smem_bytes = TILE_SIZE * sizeof(float);

    diag_spmm_medium_kernel<<<grid, block, smem_bytes, stream>>>(
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
    int num_ctas = (num_tasks + TASKS_PER_CTA_LIGHT - 1) / TASKS_PER_CTA_LIGHT;
    dim3 grid(num_ctas);
    dim3 block(BLOCK_SIZE_LIGHT);
    int  smem_bytes = TASKS_PER_CTA_LIGHT * WARP_SIZE * sizeof(float);

    diag_spmm_light_kernel<<<grid, block, smem_bytes, stream>>>(
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
    dim3 grid(num_tasks);
    dim3 block(BLOCK_SIZE_HEAVY);
    int  smem_bytes = 2 * TILE_SIZE_HEAVY * sizeof(float);

    diag_spmm_heavy_kernel<<<grid, block, smem_bytes, stream>>>(
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
    dim3 grid(num_tasks);
    dim3 block(WIDE_BLOCK_SIZE);
    int  smem_bytes = WIDE_TILE_SIZE * sizeof(float);

    diag_spmm_wide_kernel<<<grid, block, smem_bytes, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}
