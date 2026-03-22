/* ============================================================
 * diag_kernel.cuh
 *
 * Kernel declarations for diagonal sparse matrix multiplication.
 *
 * Zero-metadata design:
 *   The kernel iterates A diagonals and computes B indices
 *   on the fly using B_diag_lookup[].  No Group or PairMeta
 *   arrays — all metadata fits in L1 cache.
 *
 * Design invariants:
 *   - ZERO atomic operations
 *   - One CTA ↔ one output tile (exclusive ownership)
 *   - A and B read directly from original values (zero duplication)
 *   - B_diag_lookup for O(1) diagonal matching
 *   - Register accumulation, direct final writeback
 * ============================================================ */
#pragma once

#include "diag_types.cuh"
#include <cuda_runtime.h>

/* ============================================================
 * Kernel declarations — all take KernelArgs by value
 * ============================================================ */

__global__ void __launch_bounds__(BLOCK_SIZE_MED, 8)
diag_spmm_medium_kernel(KernelArgs args);

__global__ void __launch_bounds__(BLOCK_SIZE_LIGHT, 8)
diag_spmm_light_kernel(KernelArgs args);

__global__ void __launch_bounds__(BLOCK_SIZE_HEAVY, 4)
diag_spmm_heavy_simple_kernel(KernelArgs args);

__global__ void __launch_bounds__(BLOCK_SIZE_HEAVY, 4)
diag_spmm_heavy_prefetch_kernel(KernelArgs args);

__global__ void __launch_bounds__(WIDE_BLOCK_SIZE, 4)
diag_spmm_wide_kernel(KernelArgs args);

__global__ void __launch_bounds__(BLOCK_SIZE_MED, 8)
diag_spmm_unified_kernel(KernelArgs args);

/* ============================================================
 * Host-side launch wrappers — all take KernelArgs + stream
 * ============================================================ */

void launch_medium_kernel(KernelArgs args, cudaStream_t stream = 0);
void launch_light_kernel(KernelArgs args, cudaStream_t stream = 0);
void launch_heavy_kernel(KernelArgs args, cudaStream_t stream = 0);
void launch_heavy_simple_kernel(KernelArgs args, cudaStream_t stream = 0);
void launch_wide_kernel(KernelArgs args, cudaStream_t stream = 0);
void launch_unified_kernel(KernelArgs args, cudaStream_t stream = 0);
