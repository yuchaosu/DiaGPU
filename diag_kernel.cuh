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
 *
 * Optimization principles:
 *   - Structs accessed via __restrict__ pointer, not copied
 *   - __launch_bounds__ on all kernels
 *   - Minimal register footprint per thread
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
 * Kernel declarations (with __launch_bounds__)
 * ============================================================ */

/* ---- Medium kernel ----
 * Block: 128 threads, 1 CTA per task.
 * __launch_bounds__(128, 2) → target ≥2 blocks/SM. */
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
                        int               num_tasks_in_bucket);

/* ---- Light kernel ----
 * Block: 128 threads, 4 tasks packed per CTA (1 per warp). */
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
                       int               num_tasks_in_bucket);

/* ---- Heavy kernel ----
 * Block: 256 threads, double-buffered smemA.
 * __launch_bounds__(256, 1) → at least 1 block/SM. */
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
                       int               num_tasks_in_bucket);

/* ---- Wide kernel ----
 * Block: 128 threads, tile 512, 4 outputs per thread. */
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
                      int               num_tasks_in_bucket);

/* ============================================================
 * Host-side launch wrappers
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
                          cudaStream_t      stream = 0);

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
