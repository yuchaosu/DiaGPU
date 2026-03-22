/* ============================================================
 * diag_kernel.cu  —  Zero-metadata kernels
 *
 * KEY DESIGN:
 *   ✓  ZERO atomic operations
 *   ✓  ZERO metadata arrays (no Group, no PairMeta)
 *   ✓  One CTA exclusively owns one output tile
 *   ✓  A & B read directly from original values
 *   ✓  B_diag_lookup[] for O(1) diagonal matching (fits L1)
 *   ✓  Register accumulation, single direct writeback
 *
 * Inner loop per thread:
 *   for each A diagonal ai:
 *     d_b = d_c - A_offsets[ai]
 *     bi = B_diag_lookup[d_b + n-1]    // O(1), L1-cached
 *     if (bi < 0) continue
 *     p_a = c_sr + p_begin + tid - a_sr
 *     p_b = c_sr + d_a - b_sr + p_begin + tid
 *     acc += A[..+p_a] * B[..+p_b]     // both coalesced
 *
 * Total metadata read per thread: ~0 bytes from global memory.
 * A_offsets, B_diag_lookup, B_starts, B_lengths are tiny arrays
 * that sit permanently in L1 cache.
 * ============================================================ */

#include "diag_kernel.cuh"

/* ============================================================
 * Helpers
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
    const int n_m_1   = args.n - 1;

    float acc = 0.0f;

    for (int ai = 0; ai < args.A_num_diags; ++ai) {
        const int d_a = args.A_offsets[ai];
        const int d_b = d_c - d_a;

        /* O(1) lookup: does B have diagonal d_b? */
        if (d_b < -n_m_1 || d_b > n_m_1) continue;
        const int bi = args.B_diag_lookup[d_b + n_m_1];
        if (bi < 0) continue;

        /* A index */
        const int a_sr = (d_a >= 0) ? 0 : -d_a;
        const int p_a  = c_sr + p_begin + tid - a_sr;
        const float a_val = (p_a >= 0 && p_a < args.A_lengths[ai])
                          ? args.A_values[args.A_starts[ai] + p_a]
                          : 0.0f;

        /* B index */
        const int b_sr = (d_b >= 0) ? 0 : -d_b;
        const int p_b  = c_sr + d_a - b_sr + p_begin + tid;
        const float b_val = (p_b >= 0 && p_b < args.B_lengths[bi])
                          ? args.B_values[args.B_starts[bi] + p_b]
                          : 0.0f;

        acc += a_val * b_val;
    }

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
    const int n_m_1   = args.n - 1;

    float acc = 0.0f;

    for (int ai = 0; ai < args.A_num_diags; ++ai) {
        const int d_a = args.A_offsets[ai];
        const int d_b = d_c - d_a;
        if (d_b < -n_m_1 || d_b > n_m_1) continue;
        const int bi = args.B_diag_lookup[d_b + n_m_1];
        if (bi < 0) continue;

        const int a_sr = (d_a >= 0) ? 0 : -d_a;
        const int p_a  = c_sr + p_begin + lane_id - a_sr;
        const float a_val = (p_a >= 0 && p_a < args.A_lengths[ai])
                          ? args.A_values[args.A_starts[ai] + p_a]
                          : 0.0f;

        const int b_sr = (d_b >= 0) ? 0 : -d_b;
        const int p_b  = c_sr + d_a - b_sr + p_begin + lane_id;
        const float b_val = (p_b >= 0 && p_b < args.B_lengths[bi])
                          ? args.B_values[args.B_starts[bi] + p_b]
                          : 0.0f;

        acc += a_val * b_val;
    }

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
    const int n_m_1   = args.n - 1;

    float acc = 0.0f;

    for (int ai = 0; ai < args.A_num_diags; ++ai) {
        const int d_a = args.A_offsets[ai];
        const int d_b = d_c - d_a;
        if (d_b < -n_m_1 || d_b > n_m_1) continue;
        const int bi = args.B_diag_lookup[d_b + n_m_1];
        if (bi < 0) continue;

        const int a_sr = (d_a >= 0) ? 0 : -d_a;
        const int p_a  = c_sr + p_begin + tid - a_sr;
        const float a_val = (p_a >= 0 && p_a < args.A_lengths[ai])
                          ? args.A_values[args.A_starts[ai] + p_a]
                          : 0.0f;

        const int b_sr = (d_b >= 0) ? 0 : -d_b;
        const int p_b  = c_sr + d_a - b_sr + p_begin + tid;
        const float b_val = (p_b >= 0 && p_b < args.B_lengths[bi])
                          ? args.B_values[args.B_starts[bi] + p_b]
                          : 0.0f;

        acc += a_val * b_val;
    }

    args.C_values[args.c_diags[tp.c_diag_idx].values_start
                  + p_begin + tid] = acc;
}

/* ============================================================
 * HEAVY KERNEL — prefetch variant (software-pipelined A)
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
    const int n_m_1   = args.n - 1;

    float acc = 0.0f;

    /* Find first valid A diagonal and prefetch its value. */
    int ai_cur = -1;
    float a_cur = 0.0f;
    int bi_cur = -1;

    auto find_next = [&](int start) -> int {
        for (int ai = start; ai < args.A_num_diags; ++ai) {
            int d_b = d_c - args.A_offsets[ai];
            if (d_b < -n_m_1 || d_b > n_m_1) continue;
            if (args.B_diag_lookup[d_b + n_m_1] >= 0) return ai;
        }
        return args.A_num_diags;
    };

    ai_cur = find_next(0);

    while (ai_cur < args.A_num_diags) {
        const int d_a = args.A_offsets[ai_cur];
        const int d_b = d_c - d_a;
        const int bi  = args.B_diag_lookup[d_b + n_m_1];

        const int a_sr = (d_a >= 0) ? 0 : -d_a;
        const int p_a  = c_sr + p_begin + tid - a_sr;
        a_cur = (p_a >= 0 && p_a < args.A_lengths[ai_cur])
              ? args.A_values[args.A_starts[ai_cur] + p_a]
              : 0.0f;

        /* Prefetch next valid A diagonal. */
        int ai_next = find_next(ai_cur + 1);
        float a_next = 0.0f;
        if (ai_next < args.A_num_diags) {
            int d_a_n = args.A_offsets[ai_next];
            int a_sr_n = (d_a_n >= 0) ? 0 : -d_a_n;
            int p_a_n  = c_sr + p_begin + tid - a_sr_n;
            a_next = (p_a_n >= 0 && p_a_n < args.A_lengths[ai_next])
                   ? args.A_values[args.A_starts[ai_next] + p_a_n]
                   : 0.0f;
        }

        /* Compute with a_cur while a_next loads. */
        const int b_sr = (d_b >= 0) ? 0 : -d_b;
        const int p_b  = c_sr + d_a - b_sr + p_begin + tid;
        const float b_val = (p_b >= 0 && p_b < args.B_lengths[bi])
                          ? args.B_values[args.B_starts[bi] + p_b]
                          : 0.0f;

        acc += a_cur * b_val;
        ai_cur = ai_next;
    }

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

    float acc[WIDE_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < WIDE_ELEMS_PER_THREAD; ++e) acc[e] = 0.0f;

    for (int ai = 0; ai < args.A_num_diags; ++ai) {
        const int d_a = args.A_offsets[ai];
        const int d_b = d_c - d_a;
        if (d_b < -n_m_1 || d_b > n_m_1) continue;
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
        const int n_m_1   = args.n - 1;

        float acc = 0.0f;

        for (int ai = 0; ai < args.A_num_diags; ++ai) {
            const int d_a = args.A_offsets[ai];
            const int d_b = d_c - d_a;
            if (d_b < -n_m_1 || d_b > n_m_1) continue;
            const int bi = args.B_diag_lookup[d_b + n_m_1];
            if (bi < 0) continue;

            const int a_sr = (d_a >= 0) ? 0 : -d_a;
            const int p_a  = c_sr + p_begin + lane_id - a_sr;
            const float a_val = (p_a >= 0 && p_a < args.A_lengths[ai])
                              ? args.A_values[args.A_starts[ai] + p_a]
                              : 0.0f;

            const int b_sr = (d_b >= 0) ? 0 : -d_b;
            const int p_b  = c_sr + d_a - b_sr + p_begin + lane_id;
            const float b_val = (p_b >= 0 && p_b < args.B_lengths[bi])
                              ? args.B_values[args.B_starts[bi] + p_b]
                              : 0.0f;

            acc += a_val * b_val;
        }

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
