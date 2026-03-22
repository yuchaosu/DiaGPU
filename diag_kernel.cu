/* ============================================================
 * diag_kernel.cu  —  Zero-metadata, range-optimized kernels
 *
 * KEY DESIGN:
 *   ZERO atomic operations
 *   ZERO metadata arrays (no Group, no PairMeta)
 *   ZERO wasted iterations (binary-search A range per task)
 *   One CTA exclusively owns one output tile
 *   A & B read directly from original values
 *   B_diag_lookup[] for O(1) diagonal matching (fits L1)
 *   Register accumulation, single direct writeback
 *
 * Inner loop per thread:
 *   [ai_begin, ai_end) = valid A range for this d_c
 *   for ai in [ai_begin, ai_end):
 *     bi = B_diag_lookup[d_c - A_offsets[ai] + n-1]
 *     if (bi < 0) continue   // rare for non-contiguous B
 *     acc += A[..+p_a] * B[..+p_b]
 *
 * The range [ai_begin, ai_end) is found via binary search on
 * A_offsets[] (sorted), using B_offset_min/max to compute
 * the valid d_a range.  This eliminates ~50% wasted iterations
 * that previously hit `continue`.
 * ============================================================ */

#include "diag_kernel.cuh"

/* ============================================================
 * Host helpers
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

static int pad_to_wave(int num_tasks, int min_blocks_per_sm) {
    int sm = get_sm_count();
    int wave = sm * min_blocks_per_sm;
    if (wave <= 0 || num_tasks <= 0) return num_tasks;
    return ((num_tasks + wave - 1) / wave) * wave;
}

/* ============================================================
 * Device helpers: binary search on sorted A_offsets[]
 *
 * For output diagonal d_c, valid d_a must satisfy:
 *   d_b = d_c - d_a  is in [B_offset_min, B_offset_max]
 *   => d_a is in [d_c - B_offset_max, d_c - B_offset_min]
 *
 * lower_bound: first ai where A_offsets[ai] >= d_a_lo
 * upper_bound: first ai where A_offsets[ai] >  d_a_hi
 * ============================================================ */
__device__ __forceinline__ int
dev_lower_bound(const int* __restrict__ arr, int n, int val)
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
dev_upper_bound(const int* __restrict__ arr, int n, int val)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] <= val) lo = mid + 1;
        else                 hi = mid;
    }
    return lo;
}

/* Compute valid A-diagonal index range for output diagonal d_c. */
__device__ __forceinline__ void
compute_a_range(const KernelArgs& args, int d_c,
                int& ai_begin, int& ai_end)
{
    int d_a_lo = d_c - args.B_offset_max;
    int d_a_hi = d_c - args.B_offset_min;
    ai_begin = dev_lower_bound(args.A_offsets, args.A_num_diags, d_a_lo);
    ai_end   = dev_upper_bound(args.A_offsets, args.A_num_diags, d_a_hi);
}

/* ============================================================
 * Core accumulation loop — shared by all kernel variants.
 *
 * Iterates ONLY the valid A-diagonal range [ai_begin, ai_end),
 * eliminating ~50% wasted iterations on edge output diagonals.
 * ============================================================ */
__device__ __forceinline__ float
compute_tile_element(const KernelArgs& args,
                     int d_c, int c_sr, int p_begin, int pos,
                     int ai_begin, int ai_end)
{
    const int n_m_1 = args.n - 1;
    float acc = 0.0f;

    for (int ai = ai_begin; ai < ai_end; ++ai) {
        const int d_a = args.A_offsets[ai];
        const int d_b = d_c - d_a;

        /* B lookup (still needed for non-contiguous B offsets). */
        const int bi = args.B_diag_lookup[d_b + n_m_1];
        if (bi < 0) continue;

        /* A value */
        const int a_sr = (d_a >= 0) ? 0 : -d_a;
        const int p_a  = c_sr + p_begin + pos - a_sr;
        const float a_val = (p_a >= 0 && p_a < args.A_lengths[ai])
                          ? args.A_values[args.A_starts[ai] + p_a]
                          : 0.0f;

        /* B value */
        const int b_sr = (d_b >= 0) ? 0 : -d_b;
        const int p_b  = c_sr + d_a - b_sr + p_begin + pos;
        const float b_val = (p_b >= 0 && p_b < args.B_lengths[bi])
                          ? args.B_values[args.B_starts[bi] + p_b]
                          : 0.0f;

        acc += a_val * b_val;
    }
    return acc;
}

/* ============================================================
 * MEDIUM KERNEL — 128 threads, 0 smem, 0 barriers
 * ============================================================ */
__global__ void __launch_bounds__(BLOCK_SIZE_MED, 8)
diag_spmm_medium_kernel(KernelArgs args)
{
    if (static_cast<int>(blockIdx.x) >= args.num_tasks) return;

    const Task& tp = args.tasks[args.task_ids[blockIdx.x]];
    const int tid = threadIdx.x;
    if (tid >= tp.p_len) return;

    const int d_c     = tp.c_offset;
    const int c_sr    = (d_c >= 0) ? 0 : -d_c;
    const int p_begin = tp.p_begin;

    int ai_begin, ai_end;
    compute_a_range(args, d_c, ai_begin, ai_end);

    float acc = compute_tile_element(args, d_c, c_sr, p_begin, tid,
                                     ai_begin, ai_end);

    args.C_values[args.c_diags[tp.c_diag_idx].values_start
                  + p_begin + tid] = acc;
}

/* ============================================================
 * LIGHT KERNEL — 128 threads, 4 tasks per CTA (1 per warp)
 * ============================================================ */
__global__ void __launch_bounds__(BLOCK_SIZE_LIGHT, 8)
diag_spmm_light_kernel(KernelArgs args)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int task_slot =
        static_cast<int>(blockIdx.x) * TASKS_PER_CTA_LIGHT + warp_id;
    if (task_slot >= args.num_tasks) return;

    const Task& tp = args.tasks[args.task_ids[task_slot]];
    if (lane_id >= tp.p_len) return;

    const int d_c     = tp.c_offset;
    const int c_sr    = (d_c >= 0) ? 0 : -d_c;
    const int p_begin = tp.p_begin;

    int ai_begin, ai_end;
    compute_a_range(args, d_c, ai_begin, ai_end);

    float acc = compute_tile_element(args, d_c, c_sr, p_begin, lane_id,
                                     ai_begin, ai_end);

    args.C_values[args.c_diags[tp.c_diag_idx].values_start
                  + p_begin + lane_id] = acc;
}

/* ============================================================
 * HEAVY KERNEL — simple, 256 threads
 * ============================================================ */
__global__ void __launch_bounds__(BLOCK_SIZE_HEAVY, 4)
diag_spmm_heavy_simple_kernel(KernelArgs args)
{
    if (static_cast<int>(blockIdx.x) >= args.num_tasks) return;

    const Task& tp = args.tasks[args.task_ids[blockIdx.x]];
    const int tid = threadIdx.x;
    if (tid >= tp.p_len) return;

    const int d_c     = tp.c_offset;
    const int c_sr    = (d_c >= 0) ? 0 : -d_c;
    const int p_begin = tp.p_begin;

    int ai_begin, ai_end;
    compute_a_range(args, d_c, ai_begin, ai_end);

    float acc = compute_tile_element(args, d_c, c_sr, p_begin, tid,
                                     ai_begin, ai_end);

    args.C_values[args.c_diags[tp.c_diag_idx].values_start
                  + p_begin + tid] = acc;
}

/* ============================================================
 * HEAVY KERNEL — prefetch variant
 * ============================================================ */
__global__ void __launch_bounds__(BLOCK_SIZE_HEAVY, 4)
diag_spmm_heavy_prefetch_kernel(KernelArgs args)
{
    if (static_cast<int>(blockIdx.x) >= args.num_tasks) return;

    const Task& tp = args.tasks[args.task_ids[blockIdx.x]];
    const int tid = threadIdx.x;
    if (tid >= tp.p_len) return;

    const int d_c     = tp.c_offset;
    const int c_sr    = (d_c >= 0) ? 0 : -d_c;
    const int p_begin = tp.p_begin;

    int ai_begin, ai_end;
    compute_a_range(args, d_c, ai_begin, ai_end);

    /* Same compute as simple — prefetch is less useful with
     * the narrowed range (fewer iterations to overlap). */
    float acc = compute_tile_element(args, d_c, c_sr, p_begin, tid,
                                     ai_begin, ai_end);

    args.C_values[args.c_diags[tp.c_diag_idx].values_start
                  + p_begin + tid] = acc;
}

/* ============================================================
 * WIDE KERNEL — 128 threads, 4 outputs per thread, tile=512
 * ============================================================ */
__global__ void __launch_bounds__(WIDE_BLOCK_SIZE, 4)
diag_spmm_wide_kernel(KernelArgs args)
{
    if (static_cast<int>(blockIdx.x) >= args.num_tasks) return;

    const Task& tp = args.tasks[args.task_ids[blockIdx.x]];
    const int tid      = threadIdx.x;
    const int tile_len = tp.p_len;
    const int d_c      = tp.c_offset;
    const int c_sr     = (d_c >= 0) ? 0 : -d_c;
    const int p_begin  = tp.p_begin;
    const int n_m_1    = args.n - 1;

    int ai_begin, ai_end;
    compute_a_range(args, d_c, ai_begin, ai_end);

    float acc[WIDE_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < WIDE_ELEMS_PER_THREAD; ++e) acc[e] = 0.0f;

    for (int ai = ai_begin; ai < ai_end; ++ai) {
        const int d_a = args.A_offsets[ai];
        const int d_b = d_c - d_a;
        const int bi = args.B_diag_lookup[d_b + n_m_1];
        if (bi < 0) continue;

        const int a_sr = (d_a >= 0) ? 0 : -d_a;
        const int b_sr = (d_b >= 0) ? 0 : -d_b;

        #pragma unroll
        for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
            int q = tid + k * WIDE_BLOCK_SIZE;
            if (q >= tile_len) break;

            int p_a = c_sr + p_begin + q - a_sr;
            float a_val = (p_a >= 0 && p_a < args.A_lengths[ai])
                        ? args.A_values[args.A_starts[ai] + p_a]
                        : 0.0f;

            int p_b = c_sr + d_a - b_sr + p_begin + q;
            float b_val = (p_b >= 0 && p_b < args.B_lengths[bi])
                        ? args.B_values[args.B_starts[bi] + p_b]
                        : 0.0f;

            acc[k] += a_val * b_val;
        }
    }

    const int vs = args.c_diags[tp.c_diag_idx].values_start + p_begin;
    #pragma unroll
    for (int k = 0; k < WIDE_ELEMS_PER_THREAD; ++k) {
        int q = tid + k * WIDE_BLOCK_SIZE;
        if (q < tile_len) {
            args.C_values[vs + q] = acc[k];
        }
    }
}

/* ============================================================
 * UNIFIED KERNEL — warp-per-task, grid-stride persistent
 * ============================================================ */
__global__ void __launch_bounds__(BLOCK_SIZE_MED, 8)
diag_spmm_unified_kernel(KernelArgs args)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_cta = BLOCK_SIZE_MED / WARP_SIZE;

    const int global_warp = static_cast<int>(blockIdx.x) * warps_per_cta
                          + warp_id;
    const int total_warps = static_cast<int>(gridDim.x) * warps_per_cta;

    for (int wi = global_warp; wi < args.num_tasks; wi += total_warps) {
        const Task& tp = args.tasks[args.task_ids[wi]];
        if (lane_id >= tp.p_len) continue;

        const int d_c     = tp.c_offset;
        const int c_sr    = (d_c >= 0) ? 0 : -d_c;
        const int p_begin = tp.p_begin;

        int ai_begin, ai_end;
        compute_a_range(args, d_c, ai_begin, ai_end);

        float acc = compute_tile_element(args, d_c, c_sr, p_begin, lane_id,
                                         ai_begin, ai_end);

        args.C_values[args.c_diags[tp.c_diag_idx].values_start
                      + p_begin + lane_id] = acc;
    }
}

/* ============================================================
 * LAUNCH WRAPPERS
 * ============================================================ */

void launch_medium_kernel(KernelArgs args, cudaStream_t stream) {
    if (args.num_tasks == 0) return;
    cudaFuncSetAttribute(diag_spmm_medium_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    dim3 grid(pad_to_wave(args.num_tasks, 8));
    diag_spmm_medium_kernel<<<grid, BLOCK_SIZE_MED, 0, stream>>>(args);
}

void launch_light_kernel(KernelArgs args, cudaStream_t stream) {
    if (args.num_tasks == 0) return;
    cudaFuncSetAttribute(diag_spmm_light_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    int num_ctas = (args.num_tasks + TASKS_PER_CTA_LIGHT - 1)
                 / TASKS_PER_CTA_LIGHT;
    dim3 grid(pad_to_wave(num_ctas, 8));
    diag_spmm_light_kernel<<<grid, BLOCK_SIZE_LIGHT, 0, stream>>>(args);
}

void launch_heavy_kernel(KernelArgs args, cudaStream_t stream) {
    if (args.num_tasks == 0) return;
    cudaFuncSetAttribute(diag_spmm_heavy_prefetch_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    dim3 grid(pad_to_wave(args.num_tasks, 4));
    diag_spmm_heavy_prefetch_kernel<<<grid, BLOCK_SIZE_HEAVY, 0, stream>>>(args);
}

void launch_heavy_simple_kernel(KernelArgs args, cudaStream_t stream) {
    if (args.num_tasks == 0) return;
    cudaFuncSetAttribute(diag_spmm_heavy_simple_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    dim3 grid(pad_to_wave(args.num_tasks, 4));
    diag_spmm_heavy_simple_kernel<<<grid, BLOCK_SIZE_HEAVY, 0, stream>>>(args);
}

void launch_wide_kernel(KernelArgs args, cudaStream_t stream) {
    if (args.num_tasks == 0) return;
    cudaFuncSetAttribute(diag_spmm_wide_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    dim3 grid(pad_to_wave(args.num_tasks, 4));
    diag_spmm_wide_kernel<<<grid, WIDE_BLOCK_SIZE, 0, stream>>>(args);
}

void launch_unified_kernel(KernelArgs args, cudaStream_t stream) {
    if (args.num_tasks == 0) return;
    cudaFuncSetAttribute(diag_spmm_unified_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    int sm = get_sm_count();
    int grid_size = sm * 8;
    int min_ctas = (args.num_tasks + 3) / 4;
    if (grid_size > min_ctas) grid_size = min_ctas;
    diag_spmm_unified_kernel<<<grid_size, BLOCK_SIZE_MED, 0, stream>>>(args);
}
