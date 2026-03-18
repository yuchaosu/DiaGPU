/* ============================================================
 * diag_kernel.cuh
 *
 * Kernel declarations for diagonal sparse matrix multiplication.
 *
 * All kernels are register-only (zero shared memory):
 *   Thread tid loads its own A value into a register and
 *   reuses it across all pairs. No cross-thread dependency
 *   → zero barriers, zero shared memory, maximum occupancy.
 *
 * Design invariants:
 *   - ZERO atomic operations
 *   - One CTA ↔ one output tile (exclusive ownership)
 *   - A-stationary reuse (per-thread register)
 *   - B from warp-major packed layout (coalesced)
 *   - Register accumulation, direct final writeback
 * ============================================================ */
#pragma once

#include "diag_types.cuh"
#include <cuda_runtime.h>

/* ============================================================
 * Device helper:  output_linear_index
 * ============================================================ */
__device__ __forceinline__
int output_linear_index(const OutputDiag* __restrict__ c_diags,
                        int c_diag_idx, int p)
{
    return c_diags[c_diag_idx].values_start + p;
}

/* ============================================================
 * Kernel declarations
 * ============================================================ */

/* Medium: 128 threads, register-only, 0 smem, 0 barriers.
 * __launch_bounds__(128, 8) → targets 8 blocks/SM. */
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
                        int               num_tasks_in_bucket);

/* Light: 128 threads, 4 tasks per CTA (1 per warp). */
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
                       int               num_tasks_in_bucket);

/* Heavy simple: 256 threads, register-only baseline.
 * For profiling comparison against heavy_prefetch. */
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
                       int               num_tasks_in_bucket);

/* Heavy prefetch: 256 threads, software-pipelined A loads.
 * Loads next group's A while computing current group. */
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
                       int               num_tasks_in_bucket);

/* Wide: 128 threads, tile 512, 4 outputs per thread. */
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
                      int               num_tasks_in_bucket);

/* ============================================================
 * Host-side launch wrappers
 *
 * All use 0 shared memory, max L1 carveout, and
 * grid padding for tail-effect elimination.
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

/* Default heavy launcher — uses prefetch variant. */
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

/* Alternate heavy launcher — simple variant for profiling. */
void launch_heavy_simple_kernel(const Task*       d_tasks,
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
