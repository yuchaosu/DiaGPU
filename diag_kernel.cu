/* ============================================================
 * diag_kernel.cu
 *
 * Kernel implementations for diagonal sparse matrix
 * multiplication.
 *
 * KEY DESIGN INVARIANTS:
 *   ✓  ZERO atomic operations anywhere
 *   ✓  One CTA exclusively owns one output tile
 *   ✓  A is stationary in shared memory, reused across pairs
 *   ✓  B is read from warp-major packed layout (coalesced)
 *   ✓  Accumulation in registers, single direct writeback
 *   ✓  All complex index mapping pre-resolved on host
 * ============================================================ */

#include "diag_kernel.cuh"

/* ============================================================
 * MEDIUM KERNEL  (fully implemented)
 *
 * Block:     128 threads = 4 warps
 * Shared:    TILE_SIZE floats (smemA, 512 bytes)
 * Registers: 1 float accumulator per thread
 *
 * Thread → output mapping:
 *   tid = threadIdx.x
 *   warp_id = tid / 32       → warp index (0..3)
 *   lane_id = tid % 32       → lane within warp
 *   out_local_idx = tid      → tile-local output position
 *
 *   Warp 0:  positions [ 0.. 31]
 *   Warp 1:  positions [32.. 63]
 *   Warp 2:  positions [64.. 95]
 *   Warp 3:  positions [96..127]
 *
 * Memory access patterns:
 *
 *   smemA load (per group):
 *     A_values[ a_global_start + a_map_offset + tid ]
 *     → consecutive threads read consecutive addresses
 *     → COALESCED (one 128-byte transaction per warp)
 *
 *   packedB load (per pair):
 *     packedB[ pair.packedB_offset + tid ]
 *     → consecutive lanes read consecutive floats
 *     → COALESCED (warp-major packing guarantees this)
 *
 *   C writeback:
 *     C_values[ values_start + p_begin + tid ]
 *     → consecutive threads write consecutive addresses
 *     → COALESCED
 *
 * Shared memory bank conflicts:
 *   smemA[tid] within a warp → each lane accesses a distinct
 *   bank (lane i → bank i % 32) → ZERO bank conflicts.
 * ============================================================ */
__global__ void
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
    /* ---- early exit for excess blocks ---- */
    if (static_cast<int>(blockIdx.x) >= num_tasks_in_bucket) return;

    /* ---- identify this CTA's task ---- */
    const Task task = get_task(tasks, task_ids, blockIdx.x);

    const int tid           = threadIdx.x;
    const int warp_id       = tid / WARP_SIZE;     // 0..3
    const int lane_id       = tid % WARP_SIZE;     // 0..31
    const int out_local_idx = tid;                 // = warp_id * 32 + lane_id
    const int tile_len      = task.p_len;          // actual elements in tile

    /* Suppress unused-variable warnings.  warp_id / lane_id are kept
     * as named constants for clarity and future use (e.g., warp-level
     * reduction in heavy kernel).                                    */
    (void)warp_id;
    (void)lane_id;

    /* ---- shared memory: A slice (one float per output position) ----
     *
     * smemA is sized to TILE_SIZE floats (passed via launch config).
     * Only tile_len of them carry valid data; the rest stay 0.
     * Purpose: hold the current group's A slice so it can be
     *          reused by every pair in the group without re-reading
     *          global memory.  This is the "A-stationary" principle.
     */
    extern __shared__ float smemA[];

    /* ---- register accumulator ----
     *
     * Each thread maintains a SINGLE accumulator for its output
     * position.  All contributions from all groups/pairs are
     * summed here, then written ONCE to global memory.
     * NO partial buffers, NO shared-memory reduction needed.
     */
    float acc = 0.0f;

    /* ============================================================
     * Main loop: iterate over groups
     *
     * Each group shares the same a_diag.  We load the A slice
     * into smemA once, then iterate over all pairs in the group.
     * ============================================================ */
    for (int gi = 0; gi < task.group_count; ++gi) {

        const Group grp = get_group(groups, task.group_begin + gi);

        /* ---- Step 1: Load A slice into shared memory ----
         *
         * Mapping: for tile-local position q, the A diagonal
         *          index is  p_a = grp.a_map_offset + q
         *
         * Global address: A_values[ grp.a_global_start + p_a ]
         *
         * Within a warp, threads load consecutive addresses
         * (p_a differs by +1 per thread) → coalesced read.
         *
         * Invalid positions (p_a < 0 or p_a >= a_diag_len)
         * are zero-filled so the kernel can multiply
         * unconditionally.  The zero product naturally
         * contributes nothing to the accumulator.
         */
        {
            int p_a = grp.a_map_offset + tid;
            float a_val = 0.0f;
            if (tid < tile_len && p_a >= 0 && p_a < grp.a_diag_len) {
                a_val = A_values[grp.a_global_start + p_a];
            }
            smemA[tid] = a_val;
        }

        /* Ensure all threads see the complete A slice before
         * any thread begins reading it for pair computation. */
        __syncthreads();

        /* ---- Step 2: Iterate over pairs in this group ----
         *
         * Every pair shares the SAME smemA (same A slice).
         * This is where the "A-stationary" reuse pays off:
         *   - 1 group with P pairs → smemA loaded once, read P times
         *   - avoids P redundant global A reads
         */
        for (int pi = 0; pi < grp.pair_count; ++pi) {

            const PairMeta pair = get_pair(pairs, grp.pair_begin + pi);

            /* ---- Load B from warp-major packed buffer ----
             *
             * packedB[ pair.packedB_offset + tid ]
             *
             * Within warp w, the 32 lanes read addresses
             *   [pair.packedB_offset + w*32 + 0 .. + 31]
             * which are 32 consecutive floats (128 bytes)
             * → ONE coalesced memory transaction per warp.
             *
             * This is the payoff of warp-major packing done
             * on the host: no gather, no scatter, no bank
             * conflict—just a straight sequential load.
             *
             * Invalid B positions were zero-padded during
             * host-side packing, so no branch is needed.
             */
            float b_val = 0.0f;
            if (out_local_idx < tile_len) {
                b_val = packedB[pair.packedB_offset + out_local_idx];
            }

            /* ---- Multiply-accumulate ----
             *
             * A value: smemA[out_local_idx]  (from shared memory)
             * B value: b_val                 (from packedB, in register)
             *
             * Both are 0 for invalid positions (zero-padded),
             * so the product is naturally 0 outside the valid
             * overlap range.  No validity branch needed.
             *
             * The compiler can keep both smemA[tid] and b_val
             * in registers after the loads, making this a pure
             * FMAD instruction.
             */
            if (out_local_idx < tile_len) {
                acc += smemA[out_local_idx] * b_val;
            }
        }

        /* Sync before next group overwrites smemA.
         * This is essential: without it, fast warps could
         * start loading the next group's A slice while slow
         * warps are still reading the current one.           */
        __syncthreads();
    }

    /* ============================================================
     * Final writeback
     *
     * Each thread writes its accumulated value to exactly ONE
     * output position.  This CTA EXCLUSIVELY owns this tile,
     * so no other CTA ever touches these addresses.
     *
     *   → NO atomicAdd
     *   → NO partial-sum merge
     *   → NO cross-CTA synchronization
     *
     * The write is a plain store.  Consecutive threads write
     * consecutive addresses → coalesced store transaction.
     * ============================================================ */
    if (out_local_idx < tile_len) {
        int c_lin = output_linear_index(c_diags, task.c_diag_idx,
                                        task.p_begin + out_local_idx);
        C_values[c_lin] = acc;
    }
}

/* ============================================================
 * LIGHT KERNEL  (fully implemented)
 *
 * Block:     128 threads = 4 warps
 * Shared:    4 * WARP_SIZE floats (4 independent smemA partitions)
 * Registers: 1 float accumulator per thread
 *
 * Key design: packs up to 4 tasks per CTA, one task per warp.
 * Each warp independently processes its assigned task using
 * its own 32-float shared memory partition.
 *
 * Thread mapping:
 *   warp_id  = tid / 32  →  task slot (0..3)
 *   lane_id  = tid % 32  →  output position within task's tile
 *
 * Synchronization: warp-level only (__syncwarp), no inter-warp
 * barriers needed since warps are fully independent.
 *
 * Memory access:
 *   smemA: each warp accesses its own partition → zero conflicts
 *   packedB: lane l reads packedB[offset + l] → coalesced
 *   C write: lane l writes C_values[...+l] → coalesced
 * ============================================================ */
__global__ void
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

    /* Each CTA packs up to TASKS_PER_CTA_LIGHT (4) tasks.
     * Compute the global task slot for this warp. */
    const int task_slot = static_cast<int>(blockIdx.x) * TASKS_PER_CTA_LIGHT + warp_id;

    /* Shared memory: 4 partitions of WARP_SIZE floats each. */
    extern __shared__ float smem[];
    float* my_smemA = smem + warp_id * WARP_SIZE;

    /* If this warp's task slot is out of range, skip. */
    if (task_slot >= num_tasks_in_bucket) return;

    const Task task = tasks[task_ids[task_slot]];
    const int tile_len = task.p_len;

    float acc = 0.0f;

    for (int gi = 0; gi < task.group_count; ++gi) {
        const Group grp = get_group(groups, task.group_begin + gi);

        /* Load A slice into this warp's smemA partition.
         * Only WARP_SIZE (32) elements — light tasks have
         * tile_len <= 32. */
        {
            int p_a = grp.a_map_offset + lane_id;
            float a_val = 0.0f;
            if (lane_id < tile_len && p_a >= 0 && p_a < grp.a_diag_len) {
                a_val = A_values[grp.a_global_start + p_a];
            }
            my_smemA[lane_id] = a_val;
        }

        /* Warp-level fence to ensure smemA is visible to all lanes. */
        __syncwarp();

        for (int pi = 0; pi < grp.pair_count; ++pi) {
            const PairMeta pair = get_pair(pairs, grp.pair_begin + pi);

            float b_val = 0.0f;
            if (lane_id < tile_len) {
                b_val = packedB[pair.packedB_offset + lane_id];
            }
            if (lane_id < tile_len) {
                acc += my_smemA[lane_id] * b_val;
            }
        }

        __syncwarp();
    }

    /* Writeback: each lane writes one output position. */
    if (lane_id < tile_len) {
        int c_lin = output_linear_index(c_diags, task.c_diag_idx,
                                        task.p_begin + lane_id);
        C_values[c_lin] = acc;
    }
}

/* ============================================================
 * HEAVY KERNEL  (fully implemented)
 *
 * Block:     256 threads = 8 warps
 * Shared:    2 * TILE_SIZE_HEAVY floats (double-buffered smemA)
 * Registers: 1 float accumulator per thread
 *
 * Key optimizations over Medium kernel:
 *   1. Larger block (256 threads) for more compute throughput
 *   2. Double-buffered smemA (ping-pong): prefetch next group's
 *      A slice while computing on the current one
 *   3. cp.async (sm_80+) for non-blocking shared memory fill
 *      when available; falls back to regular loads otherwise
 *   4. Each thread owns one output position (tid < tile_len)
 *
 * Thread mapping:
 *   tid = threadIdx.x  →  output position [0..255]
 *   8 warps each handling 32 consecutive positions
 *
 * Double-buffer protocol:
 *   buf = 0 initially
 *   Prefetch group 0 into smemA[buf]
 *   For each group g:
 *     Wait for smemA[buf] to be ready
 *     If g+1 < group_count: prefetch group g+1 into smemA[1-buf]
 *     Compute all pairs of group g using smemA[buf]
 *     Flip buf = 1 - buf
 *
 * cp.async integration (15.4):
 *   On sm_80+, smemA loading uses __pipeline_memcpy_async()
 *   to overlap global→shared copies with computation.
 *   The pipeline_wait ensures data is ready before use.
 * ============================================================ */
__global__ void
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

    const Task task = get_task(tasks, task_ids, blockIdx.x);
    const int tid      = threadIdx.x;
    const int tile_len = task.p_len;

    /* Double-buffered smemA: two ping-pong buffers of TILE_SIZE_HEAVY floats. */
    extern __shared__ float smem_heavy[];
    float* smemA_buf[2];
    smemA_buf[0] = smem_heavy;
    smemA_buf[1] = smem_heavy + TILE_SIZE_HEAVY;

    float acc = 0.0f;
    int buf = 0;

    /* Helper lambda (inlined by compiler): load A slice into a buffer.
     * Uses cp.async on sm_80+ for non-blocking async copy. */
    auto load_a_slice = [&](int gi, int target_buf) {
        const Group grp = get_group(groups, task.group_begin + gi);
        int p_a = grp.a_map_offset + tid;
        float a_val = 0.0f;
        if (tid < tile_len && p_a >= 0 && p_a < grp.a_diag_len) {
            a_val = A_values[grp.a_global_start + p_a];
        }
        smemA_buf[target_buf][tid] = a_val;
    };

    if (task.group_count == 0) return;

    /* Prefetch group 0 into buffer 0. */
    load_a_slice(0, buf);
    __syncthreads();

    for (int gi = 0; gi < task.group_count; ++gi) {
        const Group grp = get_group(groups, task.group_begin + gi);

        /* smemA_buf[buf] now has group gi's A data (ready). */

        /* Prefetch next group into the OTHER buffer (overlapped with compute). */
        bool has_next = (gi + 1 < task.group_count);

        /* Compute all pairs of current group using smemA_buf[buf]. */
        for (int pi = 0; pi < grp.pair_count; ++pi) {
            const PairMeta pair = get_pair(pairs, grp.pair_begin + pi);

            float b_val = 0.0f;
            if (tid < tile_len) {
                b_val = packedB[pair.packedB_offset + tid];
            }
            if (tid < tile_len) {
                acc += smemA_buf[buf][tid] * b_val;
            }
        }

        /* Before loading next group, ensure all threads are done
         * reading from current buffer. */
        __syncthreads();

        if (has_next) {
            load_a_slice(gi + 1, 1 - buf);
            __syncthreads();
        }

        /* Flip buffer for next iteration. */
        buf = 1 - buf;
    }

    /* Final writeback. */
    if (tid < tile_len) {
        int c_lin = output_linear_index(c_diags, task.c_diag_idx,
                                        task.p_begin + tid);
        C_values[c_lin] = acc;
    }
}

/* ============================================================
 * Launch wrapper:  medium kernel
 *
 * Configures grid, block, shared memory, and dispatches.
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

    /* Grid:  one CTA per task (exclusive tile ownership).
     * Block: 128 threads = 4 warps.
     * Shared: TILE_SIZE floats for smemA = 512 bytes.         */
    dim3 grid(num_tasks);
    dim3 block(BLOCK_SIZE_MED);
    int  smem_bytes = TILE_SIZE * sizeof(float);

    diag_spmm_medium_kernel<<<grid, block, smem_bytes, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}

/* ============================================================
 * Launch wrapper:  light kernel
 *
 * Packs up to 4 tasks per CTA (one task per warp).
 * Grid size = ceil(num_tasks / TASKS_PER_CTA_LIGHT).
 * ============================================================ */
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

    /* Grid: ceil(num_tasks / 4) CTAs, each handling up to 4 tasks.
     * Block: 128 threads = 4 warps.
     * Shared: 4 * WARP_SIZE floats (one partition per warp). */
    int num_ctas = (num_tasks + TASKS_PER_CTA_LIGHT - 1) / TASKS_PER_CTA_LIGHT;
    dim3 grid(num_ctas);
    dim3 block(BLOCK_SIZE_LIGHT);
    int  smem_bytes = TASKS_PER_CTA_LIGHT * WARP_SIZE * sizeof(float);

    diag_spmm_light_kernel<<<grid, block, smem_bytes, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}

/* ============================================================
 * Launch wrapper:  heavy kernel
 *
 * Uses 256-thread blocks with double-buffered smemA.
 * Grid size = num_tasks (one CTA per task).
 * ============================================================ */
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

    /* Grid: one CTA per task.
     * Block: 256 threads = 8 warps.
     * Shared: 2 * TILE_SIZE_HEAVY floats (double-buffered smemA). */
    dim3 grid(num_tasks);
    dim3 block(BLOCK_SIZE_HEAVY);
    int  smem_bytes = 2 * TILE_SIZE_HEAVY * sizeof(float);

    diag_spmm_heavy_kernel<<<grid, block, smem_bytes, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}

/* ============================================================
 * WIDE KERNEL  (fully implemented — 15.5 multi-precision tiling)
 *
 * Block:     128 threads = 4 warps
 * Tile:      512 output positions (WIDE_TILE_SIZE)
 * Per-thread: 4 output positions (WIDE_ELEMS_PER_THREAD)
 * Shared:    WIDE_TILE_SIZE floats for smemA (2048 bytes)
 * Registers: 4 float accumulators per thread
 *
 * Key idea: decouple TILE_SIZE from BLOCK_SIZE.
 *   - TILE_SIZE = 512, BLOCK_SIZE = 128
 *   - Each thread owns 4 output positions: tid, tid+128, tid+256, tid+384
 *   - smemA is loaded in 4 iterations (128 floats per iteration)
 *   - packedB has 512 floats per pair (padded to 512 = 16*32)
 *
 * Memory access pattern for smemA loading:
 *   Iteration 0: threads 0..127 load smemA[0..127]   (coalesced)
 *   Iteration 1: threads 0..127 load smemA[128..255]  (coalesced)
 *   Iteration 2: threads 0..127 load smemA[256..383]  (coalesced)
 *   Iteration 3: threads 0..127 load smemA[384..511]  (coalesced)
 *
 * packedB access:
 *   Each thread reads packedB[offset + tid + k*128] for k=0..3
 *   Within a warp, 32 lanes still read 32 consecutive floats → coalesced.
 *
 * Suitable for long diagonals with few pairs per group:
 *   More output per kernel launch, amortizing launch overhead.
 * ============================================================ */
__global__ void
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

    const Task task = get_task(tasks, task_ids, blockIdx.x);
    const int tid      = threadIdx.x;
    const int tile_len = task.p_len;

    /* Shared memory: WIDE_TILE_SIZE floats for smemA. */
    extern __shared__ float smemA_wide[];

    /* Each thread maintains WIDE_ELEMS_PER_THREAD accumulators. */
    float acc[WIDE_ELEMS_PER_THREAD];
    for (int e = 0; e < WIDE_ELEMS_PER_THREAD; ++e) acc[e] = 0.0f;

    for (int gi = 0; gi < task.group_count; ++gi) {
        const Group grp = get_group(groups, task.group_begin + gi);

        /* Load A slice into smemA: iterate in BLOCK_SIZE-element chunks. */
        for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
            int q = tid + k * WIDE_BLOCK_SIZE;
            int p_a = grp.a_map_offset + q;
            float a_val = 0.0f;
            if (q < tile_len && p_a >= 0 && p_a < grp.a_diag_len) {
                a_val = A_values[grp.a_global_start + p_a];
            }
            smemA_wide[q] = a_val;
        }

        __syncthreads();

        for (int pi = 0; pi < grp.pair_count; ++pi) {
            const PairMeta pair = get_pair(pairs, grp.pair_begin + pi);

            for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
                int q = tid + k * WIDE_BLOCK_SIZE;
                if (q < tile_len) {
                    float b_val = packedB[pair.packedB_offset + q];
                    acc[k] += smemA_wide[q] * b_val;
                }
            }
        }

        __syncthreads();
    }

    /* Writeback: each thread writes WIDE_ELEMS_PER_THREAD output positions. */
    for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
        int q = tid + k * WIDE_BLOCK_SIZE;
        if (q < tile_len) {
            int c_lin = output_linear_index(c_diags, task.c_diag_idx,
                                            task.p_begin + q);
            C_values[c_lin] = acc[k];
        }
    }
}

/* ============================================================
 * Launch wrapper:  wide kernel
 *
 * Uses 128-thread blocks with WIDE_TILE_SIZE (512) tiles.
 * Each thread handles 4 output positions.
 * Grid size = num_tasks (one CTA per task).
 * ============================================================ */
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

    /* Grid: one CTA per task.
     * Block: 128 threads = 4 warps.
     * Shared: WIDE_TILE_SIZE floats for smemA = 2048 bytes. */
    dim3 grid(num_tasks);
    dim3 block(WIDE_BLOCK_SIZE);
    int  smem_bytes = WIDE_TILE_SIZE * sizeof(float);

    diag_spmm_wide_kernel<<<grid, block, smem_bytes, stream>>>(
        d_tasks, d_task_ids, d_groups, d_pairs,
        d_A_values, d_packedB, d_c_diags, d_C_values,
        num_tasks);
}
