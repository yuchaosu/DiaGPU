/* ============================================================
 * spmspm.cu
 *
 * Diagonal-format banded SpGEMM (C = A * B) kernel.
 *
 * Idea: store A and B diagonals in shared memory, run a
 * precomputed "pair schedule" that maps (a_diag, b_diag) -> c_diag
 * as shifted vector FMAs, accumulate each C diagonal entirely in
 * shared memory (no atomics), then write C back coalesced.
 *
 * Diagonal offset algebra:
 *   A_{i,j} on diagonal d_a = j-i, B_{j,k} on d_b = k-j
 *   => contributes to C diagonal d_c = d_a + d_b, with
 *      c^{(d_c)}[p_c] += a^{(d_a)}[p_a] * b^{(d_b)}[p_b]
 *   where the HM "position" of an element is min(row,col), giving
 *      a_shift = min(0,d_a) - min(0,d_c)
 *      b_shift = d_a + min(0,d_b) - min(0,d_c)
 *
 * Parallelization: ONE block handles one n-tile x one C-group.
 *   - C diagonals are partitioned into "groups" (<= T diagonals
 *     each) so that all pairs feeding a group's C diagonals are
 *     owned by a single block -> intra-block reduction, no atomics.
 *   - The n dimension is tiled (N_TILE wide); each tile is a
 *     separate block. This is what gives us thousands of blocks
 *     for narrow-band / large-n (without it the grid would only be
 *     ~#C-groups wide and the GPU would sit idle).
 *
 * The metadata arrays (a_count, ..., pairs, c_globals) are indexed
 * by GROUP. block_group[bid] / block_tile[bid] map a block to its
 * (group, n-tile). Offset arrays passed to this kernel store |d|
 * (the absolute offset); all sign information lives in the shifts.
 * ============================================================ */
#pragma once

#include <cuda_runtime.h>

/* One scheduled product: a_diag(local) * b_diag(local) -> c_diag(local),
 * aligned by the precomputed shifts. *_local index into the owning
 * group's A/B/C diagonal lists. */
struct PairEntry {
    int a_local;
    int b_local;
    int c_local;
    int a_shift;
    int b_shift;
};

template <int T, int N_TILE, int THREADS>
__global__ void sym_spgemm_kernel(
    const float*        __restrict__ A_values,
    const size_t*       __restrict__ A_starts,
    const int*          __restrict__ A_offsets,   // |d_a|
    const float*        __restrict__ B_values,
    const size_t*       __restrict__ B_starts,
    const int*          __restrict__ B_offsets,   // |d_b|
    float*              __restrict__ C_values,
    const size_t*       __restrict__ C_starts,
    const int*          __restrict__ C_offsets,   // |d_c|
    int n,
    int a_halo,
    int b_halo,
    const int*          __restrict__ a_count,     // per-group: # A diags
    const int*          __restrict__ b_count,     // per-group: # B diags
    const int*          __restrict__ pair_count,  // per-group: # pairs
    const int*          __restrict__ t_out,       // per-group: # C diags
    const int*          __restrict__ a_globals,   // [group*max_a + d]
    const int*          __restrict__ b_globals,   // [group*max_b + d]
    const PairEntry*    __restrict__ pairs,       // [group*max_pairs + p]
    const int*          __restrict__ c_globals,   // [group*T + c]
    const int*          __restrict__ block_group, // [bid] -> group
    const int*          __restrict__ block_tile,  // [bid] -> tile_start
    int max_a_per_block,
    int max_b_per_block,
    int max_pairs_per_block
) {
    // ====================================================================
    // Shared memory layout: [A_tile | B_tile | C_tile]
    // ====================================================================
    extern __shared__ float smem[];
    const int A_tile_w = N_TILE + 2 * a_halo;
    const int B_tile_w = N_TILE + 2 * b_halo;

    float* A_tile = smem;
    float* B_tile = A_tile + max_a_per_block * A_tile_w;
    float* C_tile = B_tile + max_b_per_block * B_tile_w;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // This block's (group, n-tile).
    const int g          = block_group[bid];
    const int tile_start = block_tile[bid];
    const int tile_len   = min(N_TILE, n - tile_start);
    if (tile_len <= 0) return;

    const int nA   = a_count[g];
    const int nB   = b_count[g];
    const int nP   = pair_count[g];
    const int nOut = t_out[g];

    const int* __restrict__ my_a = a_globals + g * max_a_per_block;
    const int* __restrict__ my_b = b_globals + g * max_b_per_block;
    const PairEntry* __restrict__ my_p = pairs + g * max_pairs_per_block;
    const int* __restrict__ my_c = c_globals + g * T;

    // ================================================================
    // PHASE 1: Load A_tile from global to shared (coalesced)
    // ================================================================
    for (int d = 0; d < nA; ++d) {
        const int ga = my_a[d];
        const size_t base = A_starts[ga];
        const int diag_len = n - A_offsets[ga];

        for (int j = tid; j < A_tile_w; j += THREADS) {
            const int gi = tile_start + j - a_halo;
            A_tile[d * A_tile_w + j] =
                (gi >= 0 && gi < diag_len) ? A_values[base + gi] : 0.0f;
        }
    }

    // ================================================================
    // PHASE 2: Load B_tile from global to shared (coalesced)
    // ================================================================
    for (int d = 0; d < nB; ++d) {
        const int gb = my_b[d];
        const size_t base = B_starts[gb];
        const int diag_len = n - B_offsets[gb];

        for (int j = tid; j < B_tile_w; j += THREADS) {
            const int gi = tile_start + j - b_halo;
            B_tile[d * B_tile_w + j] =
                (gi >= 0 && gi < diag_len) ? B_values[base + gi] : 0.0f;
        }
    }

    // ================================================================
    // PHASE 3: Zero C_tile in shared memory
    // ================================================================
    for (int i = tid; i < nOut * N_TILE; i += THREADS) {
        C_tile[i] = 0.0f;
    }

    __syncthreads();  // All smem loads complete before compute

    // ================================================================
    // PHASE 4: Compute — execute pair schedule
    // ================================================================
    for (int p = 0; p < nP; ++p) {
        const PairEntry pe = my_p[p];
        const float* a_row = A_tile + pe.a_local * A_tile_w;
        const float* b_row = B_tile + pe.b_local * B_tile_w;
        float*       c_row = C_tile + pe.c_local * N_TILE;
        const int a_sh = pe.a_shift;
        const int b_sh = pe.b_shift;

        for (int i = tid; i < tile_len; i += THREADS) {
            c_row[i] += a_row[i + a_sh + a_halo]
                      * b_row[i + b_sh + b_halo];
        }
    }

    __syncthreads();  // All FMAs complete before writeback

    // ================================================================
    // PHASE 5: Write C_tile to global (coalesced, no atomics)
    // ================================================================
    for (int c = 0; c < nOut; ++c) {
        const int gc = my_c[c];
        const size_t base = C_starts[gc];
        const int diag_len = n - C_offsets[gc];
        const int write_len = min(tile_len, diag_len - tile_start);

        if (write_len > 0) {
            float* dst = C_values + base + tile_start;
            const float* src = C_tile + c * N_TILE;
            for (int i = tid; i < write_len; i += THREADS) {
                dst[i] = src[i];
            }
        }
    }
}

/* ============================================================
 * Variant: one thread per C output element ("gather, no-atomic").
 *
 * Same diagonal algebra, but each thread accumulates one output
 * element C^{(d_c)}[p] entirely in a register by gathering the
 * contributing A/B values from global memory (served from L2,
 * since the band is small and heavily reused). No shared memory,
 * no __syncthreads, no atomics -> full occupancy.
 *
 * Grid: blockIdx.y = C diagonal index, x covers positions p.
 * Offset arrays here are SIGNED (we need the sign for the shifts).
 * B_lookup[db + (n-1)] -> B diagonal index, or -1 if absent.
 * ============================================================ */
__global__ void gather_spgemm_kernel(
    const float*  __restrict__ A_values,
    const size_t* __restrict__ A_starts,
    const int*    __restrict__ A_off,     // signed
    const int*    __restrict__ A_len,
    int A_ndiag,
    const float*  __restrict__ B_values,
    const size_t* __restrict__ B_starts,
    const int*    __restrict__ B_len,
    const int*    __restrict__ B_lookup,  // size 2n-1
    float*        __restrict__ C_values,
    const size_t* __restrict__ C_starts,
    const int*    __restrict__ C_off,     // signed
    const int*    __restrict__ C_len,
    int C_ndiag,
    int n)
{
    const int k = blockIdx.y;
    if (k >= C_ndiag) return;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int lenC = C_len[k];
    if (p >= lenC) return;

    const int dc   = C_off[k];
    const int minc = (dc < 0) ? dc : 0;

    float acc = 0.0f;
    for (int ai = 0; ai < A_ndiag; ++ai) {
        const int da = A_off[ai];
        const int db = dc - da;
        if (db <= -n || db >= n) continue;
        const int bi = B_lookup[db + (n - 1)];
        if (bi < 0) continue;

        const int mina = (da < 0) ? da : 0;
        const int minb = (db < 0) ? db : 0;
        const int pa = p + (mina - minc);          // = p + a_shift
        const int pb = p + (da + minb - minc);      // = p + b_shift
        if (pa < 0 || pa >= A_len[ai]) continue;
        if (pb < 0 || pb >= B_len[bi]) continue;

        acc += A_values[A_starts[ai] + pa] * B_values[B_starts[bi] + pb];
    }
    C_values[C_starts[k] + p] = acc;
}

/* ============================================================
 * Variant: gather + per-block shared-memory metadata + ILP.
 *
 * All threads in a block share the same C diagonal (blockIdx.y),
 * hence the same set of contributing pairs. We resolve that pair
 * list ONCE into shared memory (which A/B diagonal, the two shifts,
 * the two start offsets, the two lengths), then every thread just
 * streams A/B values from global. Each thread also handles ILP
 * positions via a grid-stride in x for memory-level parallelism.
 *
 * Still fully general: pairs are resolved through B_lookup, so
 * non-contiguous diagonal offsets (gaps in the band) are fine.
 * A_DMAX is a compile-time upper bound on A's diagonal count.
 * ============================================================ */
template <int A_DMAX>
__global__ void gather_meta_kernel(
    const float*  __restrict__ A_values,
    const size_t* __restrict__ A_starts,
    const int*    __restrict__ A_off,     // signed
    const int*    __restrict__ A_len,
    int A_ndiag,
    const float*  __restrict__ B_values,
    const size_t* __restrict__ B_starts,
    const int*    __restrict__ B_len,
    const int*    __restrict__ B_lookup,  // size 2n-1, -1 if absent
    float*        __restrict__ C_values,
    const size_t* __restrict__ C_starts,
    const int*    __restrict__ C_off,     // signed
    const int*    __restrict__ C_len,
    int C_ndiag,
    int n)
{
    const int k = blockIdx.y;
    if (k >= C_ndiag) return;
    const int dc   = C_off[k];
    const int lenC = C_len[k];
    const int minc = (dc < 0) ? dc : 0;
    const size_t cbase = C_starts[k];

    // --- resolve this diagonal's pair list once, into shared memory ---
    __shared__ size_t sAbase[A_DMAX], sBbase[A_DMAX];
    __shared__ int    sAsh[A_DMAX], sBsh[A_DMAX], sAlen[A_DMAX], sBlen[A_DMAX];
    __shared__ int    sNP;

    if (threadIdx.x == 0) sNP = 0;
    __syncthreads();

    for (int ai = threadIdx.x; ai < A_ndiag; ai += blockDim.x) {
        const int da = A_off[ai];
        const int db = dc - da;
        if (db <= -n || db >= n) continue;
        const int bi = B_lookup[db + (n - 1)];
        if (bi < 0) continue;
        const int slot = atomicAdd(&sNP, 1);   // compact (small, in smem)
        sAbase[slot] = A_starts[ai];
        sBbase[slot] = B_starts[bi];
        sAsh[slot]   = (da < 0 ? da : 0) - minc;            // a_shift
        sBsh[slot]   = da + (db < 0 ? db : 0) - minc;        // b_shift
        sAlen[slot]  = A_len[ai];
        sBlen[slot]  = B_len[bi];
    }
    __syncthreads();
    const int nP = sNP;

    // --- each thread accumulates ILP positions in registers ---
    for (int p = blockIdx.x * blockDim.x + threadIdx.x;
         p < lenC; p += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int q = 0; q < nP; ++q) {
            const int pa = p + sAsh[q];
            const int pb = p + sBsh[q];
            if (pa < 0 || pa >= sAlen[q]) continue;
            if (pb < 0 || pb >= sBlen[q]) continue;
            acc += A_values[sAbase[q] + pa] * B_values[sBbase[q] + pb];
        }
        C_values[cbase + p] = acc;
    }
}
