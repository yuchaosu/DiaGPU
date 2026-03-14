/* ============================================================
 * diag_kernel.cuh
 *
 * Device-side helper functions and kernel declarations for
 * diagonal sparse matrix multiplication.
 *
 * Design invariants enforced by this layer:
 *   - ZERO atomic operations
 *   - One CTA  ↔  one output tile  (exclusive ownership)
 *   - A loaded into shared memory, reused across group pairs
 *   - B read from warp-major packed layout (coalesced loads)
 *   - Accumulation in registers, direct final writeback
 * ============================================================ */
#pragma once

#include "diag_types.cuh"
#include <cuda_runtime.h>

/* ============================================================
 * Device helper:  output_linear_index
 *
 * Maps (c_diag_idx, position_along_diagonal) →
 *      linear index into the flat C_values[] array.
 * ============================================================ */
__device__ __forceinline__
int output_linear_index(const OutputDiag* __restrict__ c_diags,
                        int c_diag_idx, int p)
{
    return c_diags[c_diag_idx].values_start + p;
}

/* ============================================================
 * Device helper:  get_task / get_group / get_pair
 *
 * Thin accessors for readability.
 * ============================================================ */
__device__ __forceinline__
Task get_task(const Task* __restrict__ tasks,
              const int*  __restrict__ task_ids,
              int block_idx)
{
    return tasks[task_ids[block_idx]];
}

__device__ __forceinline__
Group get_group(const Group* __restrict__ groups, int idx)
{
    return groups[idx];
}

__device__ __forceinline__
PairMeta get_pair(const PairMeta* __restrict__ pairs, int idx)
{
    return pairs[idx];
}

/* ============================================================
 * Device helper:  get_packedB_ptr
 *
 * Returns a pointer to the start of a pair's warp-major
 * packed B data.
 * ============================================================ */
__device__ __forceinline__
const float* get_packedB_ptr(const float* __restrict__ packedB,
                             int packedB_offset)
{
    return packedB + packedB_offset;
}

/* ============================================================
 * Kernel declarations
 * ============================================================ */

/* ---- Medium kernel (fully implemented) ----
 *
 * Block config:  128 threads = 4 warps
 * Shared mem:    TILE_SIZE floats for smemA
 *
 * One CTA computes one Task (one output tile).
 *   Warp 0 → output positions [0..31]
 *   Warp 1 → output positions [32..63]
 *   Warp 2 → output positions [64..95]
 *   Warp 3 → output positions [96..127]
 *
 * Execution flow per CTA:
 *   for each Group:
 *     1. Load A slice → smemA  (coalesced global read)
 *     2. __syncthreads()
 *     3. for each Pair in Group:
 *          a. Load B from packedB  (coalesced, warp-major)
 *          b. acc += smemA[tid] * b_val
 *     4. __syncthreads()          (protect smemA before next group)
 *   Final: write acc → C_values  (NO atomic!)
 */
__global__ void
diag_spmm_medium_kernel(const Task*       __restrict__ tasks,
                        const int*        __restrict__ task_ids,
                        const Group*      __restrict__ groups,
                        const PairMeta*   __restrict__ pairs,
                        const float*      __restrict__ A_values,
                        const float*      __restrict__ packedB,
                        const OutputDiag* __restrict__ c_diags,
                        float*            __restrict__ C_values,
                        int               num_tasks_in_bucket);

/* ---- Light kernel (fully implemented) ----
 *
 * One-warp-per-task variant: packs up to 4 tasks per CTA.
 * Each warp independently handles one task with its own
 * 32-float shared memory partition.
 * Ideal for tasks with few pairs and short overlaps.
 */
__global__ void
diag_spmm_light_kernel(const Task*       __restrict__ tasks,
                       const int*        __restrict__ task_ids,
                       const Group*      __restrict__ groups,
                       const PairMeta*   __restrict__ pairs,
                       const float*      __restrict__ A_values,
                       const float*      __restrict__ packedB,
                       const OutputDiag* __restrict__ c_diags,
                       float*            __restrict__ C_values,
                       int               num_tasks_in_bucket);

/* ---- Heavy kernel (fully implemented) ----
 *
 * Warp-specialized heavy-task kernel:
 *   - Larger block (256 threads) for more parallelism
 *   - Double-buffered smemA for overlapping A load and compute
 *   - cp.async integration for non-blocking shared memory fill
 *   - Each thread owns one output position (up to 256 per tile)
 */
__global__ void
diag_spmm_heavy_kernel(const Task*       __restrict__ tasks,
                       const int*        __restrict__ task_ids,
                       const Group*      __restrict__ groups,
                       const PairMeta*   __restrict__ pairs,
                       const float*      __restrict__ A_values,
                       const float*      __restrict__ packedB,
                       const OutputDiag* __restrict__ c_diags,
                       float*            __restrict__ C_values,
                       int               num_tasks_in_bucket);

/* ---- Wide kernel (fully implemented, 15.5) ----
 *
 * Multi-output-per-thread kernel for long diagonals:
 *   - 128 threads, each thread owns 4 output positions
 *   - TILE_SIZE = 512 (decoupled from BLOCK_SIZE = 128)
 *   - smemA loaded in 4 coalesced iterations per group
 *   - Ideal for long diagonals with few pairs
 */
__global__ void
diag_spmm_wide_kernel(const Task*       __restrict__ tasks,
                      const int*        __restrict__ task_ids,
                      const Group*      __restrict__ groups,
                      const PairMeta*   __restrict__ pairs,
                      const float*      __restrict__ A_values,
                      const float*      __restrict__ packedB,
                      const OutputDiag* __restrict__ c_diags,
                      float*            __restrict__ C_values,
                      int               num_tasks_in_bucket);

/* ============================================================
 * Host-side launch wrappers
 * ============================================================ */

/* Launch the medium-bucket kernel.
 * Caller must ensure device pointers are valid. */
void launch_medium_kernel(const Task*       d_tasks,
                          const int*        d_task_ids,
                          const Group*      d_groups,
                          const PairMeta*   d_pairs,
                          const float*      d_A_values,
                          const float*      d_packedB,
                          const OutputDiag* d_c_diags,
                          float*            d_C_values,
                          int               num_tasks,
                          cudaStream_t      stream = 0);

/* Launch the light-bucket kernel.
 * Packs multiple tasks per CTA (one task per warp). */
void launch_light_kernel(const Task*       d_tasks,
                         const int*        d_task_ids,
                         const Group*      d_groups,
                         const PairMeta*   d_pairs,
                         const float*      d_A_values,
                         const float*      d_packedB,
                         const OutputDiag* d_c_diags,
                         float*            d_C_values,
                         int               num_tasks,
                         cudaStream_t      stream = 0);

/* Launch the heavy-bucket kernel.
 * Uses larger blocks with double-buffered smemA. */
void launch_heavy_kernel(const Task*       d_tasks,
                         const int*        d_task_ids,
                         const Group*      d_groups,
                         const PairMeta*   d_pairs,
                         const float*      d_A_values,
                         const float*      d_packedB,
                         const OutputDiag* d_c_diags,
                         float*            d_C_values,
                         int               num_tasks,
                         cudaStream_t      stream = 0);

/* Launch the wide kernel (15.5 multi-output per thread).
 * 128 threads, WIDE_TILE_SIZE tile, 4 outputs per thread. */
void launch_wide_kernel(const Task*       d_tasks,
                        const int*        d_task_ids,
                        const Group*      d_groups,
                        const PairMeta*   d_pairs,
                        const float*      d_A_values,
                        const float*      d_packedB,
                        const OutputDiag* d_c_diags,
                        float*            d_C_values,
                        int               num_tasks,
                        cudaStream_t      stream = 0);
