/*
 * diag_hybrid_simple.cu — Minimal readable version of the hybrid kernel.
 *
 * C = A × B, all in DIA format.
 * Core property: d_a + d_b = d_c.
 *
 * All per-C-diagonal values are explicit scalars (guaranteed registers).
 * Accumulation reads A and B directly from shared memory, no register staging.
 *
 * Smem layout per partition:
 *   smem_A        [53 × chunk]     — one A partition
 *   smem_B        [61 × chunk_b]   — matching B diags + 1 null slot (zeros)
 *   smem_B_lookup [b_d_range]      — d_b offset → smem_B slot
 */

#include "diag_hybrid_kernel.cuh"

/* Accumulate for one C diagonal.
 * Reads A and B directly from smem. Misses land on the zero-filled null slot. */
/*
 * ACCUM: accumulate one A×B product for C diagonal D_C.
 *
 * smem_A indexing:  smem_A[ s * chunk  + (c_sr - min_c_sr) + tid ]
 *   - s * chunk:           jump to the row for A diagonal #s
 *   - c_sr = max(0, -D_C): start row of this C diagonal in the dense matrix
 *   - c_sr - min_c_sr:     offset from the group's lowest start row
 *                           (the chunk was loaded starting at min_c_sr,
 *                            so this shifts to the right row for this D_C)
 *   - tid:                  this thread's position within the tile (0..127)
 *
 * smem_B indexing:  smem_B[ sb * chunk_b + (c_sc - min_c_sc) + tid ]
 *   - sb * chunk_b:         jump to the row for the matched B diagonal
 *   - c_sc = max(0, D_C):  start col of this C diagonal in the dense matrix
 *   - c_sc - min_c_sc:     offset from the group's lowest start col
 *                           (same idea as A, but on the column side)
 *   - tid:                  same position as A — row and col are linked by
 *                           the diagonal structure: col = row + D_C
 */
#define ACCUM(ACC, D_C)                                                    \
{                                                                          \
    int rel = (D_C - d_a) - bdm;  /* d_b - b_d_min: offset into B lookup table */ \
    int sb  = (rel >= 0 && rel < bdr) ? smem_B_lookup[rel] : null_sb;     \
    ACC += smem_A[s * chunk  + (max(0,-(D_C)) - task.min_c_sr) + tid]     \
         * smem_B[sb * chunk_b + (max(0,(D_C)) - task.min_c_sc) + tid];   \
}

__global__ void hybrid_kernel_simple(HybridKernelArgs args)
{
    const int tid = threadIdx.x;   /* 0..127: this thread's position within the tile */

    extern __shared__ float smem[]; /* dynamically-sized shared memory */

    if (blockIdx.x >= args.n_tasks) return;
    const HybridTask task = args.tasks[blockIdx.x]; /* this CTA's task descriptor */

    /* ── Smem geometry ── */
    const int chunk   = ((HYBRID_TILE + task.spread)    + 3) & ~3;  /* floats per A smem row; TILE + row spread across 8 C diags, 4-aligned for float4 */
    const int chunk_b = ((HYBRID_TILE + task.spread_sc) + 3) & ~3;  /* floats per B smem row; TILE + col spread across 8 C diags, 4-aligned for float4 */
    const int chunk4  = chunk   >> 2;  /* float4s per A row (128 threads load chunk4 float4s cooperatively) */
    const int cb4     = chunk_b >> 2;  /* float4s per B row */
    const int max_b   = HYBRID_PARTITION_SIZE + HYBRID_DIAGS_PER_CTA - 1;  /* max B diags per partition: 53 + 8 - 1 = 60 */
    const int null_sb = max_b;         /* slot index 60: always zero-filled, absorbs lookup misses */

    /* ── Smem pointers ── */
    float* smem_A        = smem;                                       /* [53 rows × chunk cols] A diagonal values */
    float* smem_B        = smem_A + HYBRID_PARTITION_SIZE * chunk;     /* [61 rows × chunk_b cols] B diagonal values (slot 60 = null) */
    int*   smem_B_lookup = (int*)(smem_B + (max_b + 1) * chunk_b);    /* [b_d_range] maps (d_b - b_d_min) → smem_B row index */

    /* ── C diagonal metadata — explicit scalars, guaranteed in registers ── */
    const int nc = task.c_count;    /* number of C diagonals this CTA handles (1..8) */
    const int cb = task.c_begin;    /* index of first C diagonal in c_diags[] array */

    /* d_c0..d_c7: diagonal offset for each of the 8 C diagonals.
     * Determines which A-B pairs contribute: d_b = d_c - d_a.
     * Inactive slots (ki >= nc) get d_c0 so they read valid smem (but acc is unused). */
    const int d_c0 = (0<nc) ? args.c_diags[cb+0].c_offset : 0;
    const int d_c1 = (1<nc) ? args.c_diags[cb+1].c_offset : d_c0;
    const int d_c2 = (2<nc) ? args.c_diags[cb+2].c_offset : d_c0;
    const int d_c3 = (3<nc) ? args.c_diags[cb+3].c_offset : d_c0;
    const int d_c4 = (4<nc) ? args.c_diags[cb+4].c_offset : d_c0;
    const int d_c5 = (5<nc) ? args.c_diags[cb+5].c_offset : d_c0;
    const int d_c6 = (6<nc) ? args.c_diags[cb+6].c_offset : d_c0;
    const int d_c7 = (7<nc) ? args.c_diags[cb+7].c_offset : d_c0;

    /* c_val_start0..7: offset into C_vals[] where this diagonal's output begins.
     * Used only at the end to write results. */
    const int c_val_start0 = (0<nc) ? args.c_diags[cb+0].values_start : 0;
    const int c_val_start1 = (1<nc) ? args.c_diags[cb+1].values_start : 0;
    const int c_val_start2 = (2<nc) ? args.c_diags[cb+2].values_start : 0;
    const int c_val_start3 = (3<nc) ? args.c_diags[cb+3].values_start : 0;
    const int c_val_start4 = (4<nc) ? args.c_diags[cb+4].values_start : 0;
    const int c_val_start5 = (5<nc) ? args.c_diags[cb+5].values_start : 0;
    const int c_val_start6 = (6<nc) ? args.c_diags[cb+6].values_start : 0;
    const int c_val_start7 = (7<nc) ? args.c_diags[cb+7].values_start : 0;

    /* c_length0..7: number of elements in this C diagonal.
     * Used only at the end to bounds-check the output write. */
    const int c_length0 = (0<nc) ? args.c_diags[cb+0].length : 0;
    const int c_length1 = (1<nc) ? args.c_diags[cb+1].length : 0;
    const int c_length2 = (2<nc) ? args.c_diags[cb+2].length : 0;
    const int c_length3 = (3<nc) ? args.c_diags[cb+3].length : 0;
    const int c_length4 = (4<nc) ? args.c_diags[cb+4].length : 0;
    const int c_length5 = (5<nc) ? args.c_diags[cb+5].length : 0;
    const int c_length6 = (6<nc) ? args.c_diags[cb+6].length : 0;
    const int c_length7 = (7<nc) ? args.c_diags[cb+7].length : 0;

    /* acc0..acc7: running dot-product sums for each C diagonal.
     * Persist in registers across all A-partitions. Written to C_vals[] once at the end. */
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    /* ════════════════════════════════════════════════════════════
     * Partition loop: stream A in chunks of 53 diagonals.
     * Each iteration loads one A-partition + its matching B set.
     * ════════════════════════════════════════════════════════════ */
    for (int a_off = 0; a_off < task.a_count; a_off += HYBRID_PARTITION_SIZE) {

        PartBMeta pmeta = args.part_b_meta[task.part_b_base + a_off / HYBRID_PARTITION_SIZE];
        /* pmeta.b_count:   number of B diags matching this A-partition */
        /* pmeta.b_begin:   offset into b_contrib[] for this partition's B set */
        /* pmeta.b_d_min:   smallest d_b among this partition's B diags */
        /* pmeta.b_d_range: span of d_b values (lookup table width) */

        /* ── Load: lookup init (must finish before fill) ── */
        for (int i = tid; i < pmeta.b_d_range; i += HYBRID_BLOCK)
            smem_B_lookup[i] = null_sb;  /* default: miss → null slot */
        __syncthreads();

        /* ── Load: A, B, and lookup fill (write to disjoint smem regions, one sync) ── */

        /* Fill lookup: map each B diagonal's offset to its smem_B row */
        for (int sb = tid; sb < pmeta.b_count; sb += HYBRID_BLOCK) {
            int bi = args.b_contrib[task.b_begin + pmeta.b_begin + sb];  /* global B diagonal index */
            smem_B_lookup[args.B_offsets[bi] - pmeta.b_d_min] = sb;      /* d_b → slot */
        }

        /* Zero the null slot (float4 vectorized) */
        for (int j = tid; j < cb4; j += HYBRID_BLOCK) {
            int i0 = j << 2;  /* float index = float4 index × 4 */
            *reinterpret_cast<float4*>(&smem_B[null_sb * chunk_b + i0]) = make_float4(0,0,0,0);  /* single 128-bit store: zeros 4 floats in one instruction */
        }

        /* Load B values (float4 vectorized): 128 threads cooperatively fill each B row */
        for (int sb = 0; sb < pmeta.b_count; sb++) {
            int bi    = args.b_contrib[task.b_begin + pmeta.b_begin + sb];  /* global B diagonal index */
            int b_len = args.B_lengths[bi];   /* number of elements in this B diagonal */
            int b_st  = args.B_starts[bi];    /* offset into B_vals[] flat array */
            int bp    = task.min_c_sc + task.tile_p_begin - max(0, args.B_offsets[bi]);  /* base position for this B diag at tile start */
            float* dst = smem_B + sb * chunk_b;  /* destination row in smem */
            for (int j = tid; j < cb4; j += HYBRID_BLOCK) {
                int p = bp + (j << 2);  /* position in the B diagonal's flat data */
                float4 v;
                v.x = (p   >= 0 && p   < b_len) ? args.B_vals[b_st + p]   : 0.f;
                v.y = (p+1 >= 0 && p+1 < b_len) ? args.B_vals[b_st + p+1] : 0.f;
                v.z = (p+2 >= 0 && p+2 < b_len) ? args.B_vals[b_st + p+2] : 0.f;
                v.w = (p+3 >= 0 && p+3 < b_len) ? args.B_vals[b_st + p+3] : 0.f;
                *reinterpret_cast<float4*>(dst + (j << 2)) = v;  /* single 128-bit store: writes 4 consecutive floats in one instruction */
            }
        }

        /* Load A values (float4 vectorized): 128 threads cooperatively fill each A row */
        int a_batch = min(HYBRID_PARTITION_SIZE, task.a_count - a_off);  /* A diags in this partition (≤53) */
        for (int s = 0; s < a_batch; s++) {
            int ai    = args.a_contrib[task.a_begin + a_off + s];  /* global A diagonal index */
            int a_len = args.A_lengths[ai];   /* number of elements in this A diagonal */
            int a_st  = args.A_starts[ai];    /* offset into A_vals[] flat array */
            int ap    = task.min_c_sr - max(0, -args.A_offsets[ai]) + task.tile_p_begin;  /* base position for this A diag at tile start */
            float* dst = smem_A + s * chunk;  /* destination row in smem */
            for (int j = tid; j < chunk4; j += HYBRID_BLOCK) {
                int p = ap + (j << 2);  /* position in the A diagonal's flat data */
                float4 v;
                v.x = (p   >= 0 && p   < a_len) ? args.A_vals[a_st + p]   : 0.f;
                v.y = (p+1 >= 0 && p+1 < a_len) ? args.A_vals[a_st + p+1] : 0.f;
                v.z = (p+2 >= 0 && p+2 < a_len) ? args.A_vals[a_st + p+2] : 0.f;
                v.w = (p+3 >= 0 && p+3 < a_len) ? args.A_vals[a_st + p+3] : 0.f;
                *reinterpret_cast<float4*>(dst + (j << 2)) = v;  /* single 128-bit store: writes 4 consecutive floats in one instruction */
            }
        }
        __syncthreads();

        /* ── Accumulate: read A and B directly from smem ── */
        for (int s = 0; s < a_batch; s++) {
            int d_a = args.A_offsets[args.a_contrib[task.a_begin + a_off + s]];  /* this A diagonal's offset */
            int bdm = pmeta.b_d_min;      /* B lookup table base offset */
            int bdr = pmeta.b_d_range;    /* B lookup table width */

            ACCUM(acc0, d_c0)  /* acc0 += smem_A[row for d_c0] * smem_B[matched B slot] */
            ACCUM(acc1, d_c1)
            ACCUM(acc2, d_c2)
            ACCUM(acc3, d_c3)
            ACCUM(acc4, d_c4)
            ACCUM(acc5, d_c5)
            ACCUM(acc6, d_c6)
            ACCUM(acc7, d_c7)
        }

        __syncthreads();  /* free smem for next partition's lookup init */
    }

    /* ── Write: one coalesced store per C diagonal ── */
    const int pb = task.tile_p_begin;  /* first position in this tile */
    if (0 < nc && pb+tid < c_length0) args.C_vals[c_val_start0+pb+tid] = acc0;
    if (1 < nc && pb+tid < c_length1) args.C_vals[c_val_start1+pb+tid] = acc1;
    if (2 < nc && pb+tid < c_length2) args.C_vals[c_val_start2+pb+tid] = acc2;
    if (3 < nc && pb+tid < c_length3) args.C_vals[c_val_start3+pb+tid] = acc3;
    if (4 < nc && pb+tid < c_length4) args.C_vals[c_val_start4+pb+tid] = acc4;
    if (5 < nc && pb+tid < c_length5) args.C_vals[c_val_start5+pb+tid] = acc5;
    if (6 < nc && pb+tid < c_length6) args.C_vals[c_val_start6+pb+tid] = acc6;
    if (7 < nc && pb+tid < c_length7) args.C_vals[c_val_start7+pb+tid] = acc7;
}
