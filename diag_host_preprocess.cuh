/* ============================================================
 * diag_host_preprocess.cuh
 *
 * Host-side preprocessing for DiagSpMM.
 *
 * ALL complex mapping, tiling, grouping, contributor
 * enumeration, and B packing is done here so the kernel
 * never needs to do runtime pair lookup or index gymnastics.
 *
 * Pipeline:
 *   1. build_output_diagonals()   – enumerate all d_c = d_a + d_b
 *   2. build_contributors()       – find all (a,b) pairs per d_c
 *   3. group_contributors_by_a_diag() – sort pairs for A reuse
 *   4. estimate_tile_work()       – sum of pair overlaps in tile
 *   5. build_all()                – tile, bucket, emit tables,
 *                                   pack B in warp-major layout
 *
 * Header-only (inline) for single-compilation-unit convenience.
 * ============================================================ */
#pragma once

#include "diag_types.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

/* ============================================================
 * Internal scratch types (not exposed to kernel)
 * ============================================================ */

/* One contributor pair before tiling. */
struct ContributorPair {
    int a_diag_idx;
    int a_offset;        // d_a
    int b_diag_idx;
    int b_offset;        // d_b
    int valid_begin;     // global valid position range along c diagonal
    int valid_end;
};

/* Per-output-diagonal working data. */
struct OutputDiagInfo {
    int c_offset;        // d_c
    int c_start_row;
    int c_start_col;
    int c_length;
    std::vector<ContributorPair> contributors;
};

/* ============================================================
 * Result bundle returned by build_all()
 * ============================================================ */
struct PreprocessResult {
    std::vector<Task>       tasks;
    std::vector<Group>      groups;
    std::vector<PairMeta>   pairs;
    std::vector<float>      packedB;
    std::vector<OutputDiag> output_diags;
    int                     total_c_values;  // total output elements

    /* Bucket-sorted task id lists (indices into tasks[]). */
    std::vector<int> light_task_ids;
    std::vector<int> medium_task_ids;
    std::vector<int> heavy_task_ids;
    std::vector<int> wide_task_ids;   // bucket 3: WIDE (15.5)
};

/* ============================================================
 * Step 1:  Enumerate output diagonals
 *
 * For every pair (d_a, d_b) compute d_c = d_a + d_b.
 * Only keep diagonals with positive length in C (M x N).
 * ============================================================ */
inline std::map<int, OutputDiagInfo>
build_output_diagonals(const DiagMatrix& A, const DiagMatrix& B,
                       int M, int N)
{
    std::map<int, OutputDiagInfo> out;
    for (int ai = 0; ai < A.num_diags; ++ai) {
        for (int bi = 0; bi < B.num_diags; ++bi) {
            int d_c = A.offsets[ai] + B.offsets[bi];
            if (out.count(d_c)) continue;           // already registered
            int len = DiagMatrix::diag_length(M, N, d_c);
            if (len <= 0) continue;
            OutputDiagInfo info;
            info.c_offset    = d_c;
            info.c_start_row = DiagMatrix::diag_start_row(d_c);
            info.c_start_col = DiagMatrix::diag_start_col(d_c);
            info.c_length    = len;
            out[d_c] = info;
        }
    }
    return out;
}

/* ============================================================
 * Step 2:  Build contributor pairs for every output diagonal
 *
 * For each (a_diag, b_diag) with d_a + d_b = d_c, compute
 * the valid position range along d_c where both A and B
 * have data.
 *
 * Position arithmetic:
 *   For output diagonal d_c at position p (0-indexed):
 *     row = c_start_row + p
 *     The A element on diagonal d_a is at position
 *         p_a = row - a_start_row             (a_base + p)
 *     The B element on diagonal d_b is at position
 *         p_b = row + d_a - b_start_row       (b_base + p)
 *     Valid iff  0 <= p_a < a_len  AND  0 <= p_b < b_len
 *
 *   Solving for p gives a contiguous valid interval
 *   [valid_begin, valid_end).
 * ============================================================ */
inline void
build_contributors(std::map<int, OutputDiagInfo>& out_diags,
                   const DiagMatrix& A, const DiagMatrix& B)
{
    for (int ai = 0; ai < A.num_diags; ++ai) {
        int d_a  = A.offsets[ai];
        int a_sr = DiagMatrix::diag_start_row(d_a);
        int a_len = A.diag_lengths[ai];

        for (int bi = 0; bi < B.num_diags; ++bi) {
            int d_b  = B.offsets[bi];
            int b_sr = DiagMatrix::diag_start_row(d_b);
            int b_len = B.diag_lengths[bi];
            int d_c  = d_a + d_b;

            auto it = out_diags.find(d_c);
            if (it == out_diags.end()) continue;
            OutputDiagInfo& info = it->second;
            int c_sr = info.c_start_row;

            /* a_base: tile-local offset → A diagonal index.
             *   p_a = a_base + p  where a_base = c_sr - a_sr         */
            int a_base = c_sr - a_sr;
            /* b_base: tile-local offset → B diagonal index.
             *   p_b = b_base + p  where b_base = c_sr + d_a - b_sr   */
            int b_base = c_sr + d_a - b_sr;

            /* Solve  0 <= a_base + p < a_len
             *    AND 0 <= b_base + p < b_len
             *    AND 0 <= p < c_length                                */
            int lo = 0;
            lo = std::max(lo, -a_base);           // p_a >= 0
            lo = std::max(lo, -b_base);           // p_b >= 0
            int hi = info.c_length;
            hi = std::min(hi, a_len - a_base);    // p_a < a_len
            hi = std::min(hi, b_len - b_base);    // p_b < b_len

            if (lo >= hi) continue;               // no overlap

            ContributorPair cp;
            cp.a_diag_idx  = ai;
            cp.a_offset    = d_a;
            cp.b_diag_idx  = bi;
            cp.b_offset    = d_b;
            cp.valid_begin = lo;
            cp.valid_end   = hi;
            info.contributors.push_back(cp);
        }
    }
}

/* ============================================================
 * Step 3:  Group contributors by a_diag
 *
 * Sorting by a_diag_idx ensures that when the kernel iterates
 * groups, each A slice is loaded into shared memory exactly
 * once and reused by all pairs in the group.
 * ============================================================ */
inline void
group_contributors_by_a_diag(std::vector<ContributorPair>& contributors)
{
    std::sort(contributors.begin(), contributors.end(),
              [](const ContributorPair& x, const ContributorPair& y) {
                  return x.a_diag_idx < y.a_diag_idx;
              });
}

/* ============================================================
 * Step 4:  Estimate work for a tile
 *
 * work(tile) = Σ  overlap_len(pair, tile)
 *            = Σ  max(0, min(pair.valid_end, tile_end)
 *                      - max(pair.valid_begin, tile_begin))
 *
 * This accounts for both the number of pairs AND the per-pair
 * useful overlap (not just a fixed tile length).
 * ============================================================ */
inline int
estimate_tile_work(const std::vector<ContributorPair>& contributors,
                   int p_begin, int p_end)
{
    int work = 0;
    for (const auto& cp : contributors) {
        int ov = std::min(cp.valid_end, p_end)
               - std::max(cp.valid_begin, p_begin);
        if (ov > 0) work += ov;
    }
    return work;
}

/* ============================================================
 * Step 5:  pack_B_warp_major
 *
 * For a single pair within a tile, pack B values into a
 * contiguous buffer of `packed_count` floats (padded to a
 * multiple of WARP_SIZE).
 *
 * Layout:
 *   packedB[ offset + q ]  for q = 0 .. packed_count-1
 *
 * Within a warp, lane l reads packedB[offset + warp_id*32 + l],
 * which is address-consecutive across the 32 lanes → perfectly
 * coalesced 128-byte transaction.
 *
 * Invalid positions (outside B's valid range) are zero-padded
 * so the kernel can unconditionally multiply without branching.
 * ============================================================ */
inline void
pack_B_for_pair(std::vector<float>& packedB,
                const DiagMatrix& B,
                int b_diag_idx,
                int b_map_offset,    // tile-local q → B index: p_b = b_map_offset + q
                int tile_begin,      // global p_begin of the tile
                int tile_len)
{
    int b_start = B.diag_starts[b_diag_idx];
    int b_len   = B.diag_lengths[b_diag_idx];

    /* Pad to a multiple of WARP_SIZE for clean warp loads. */
    int packed_count = ((tile_len + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    for (int q = 0; q < packed_count; ++q) {
        int p_b = b_map_offset + tile_begin + q;
        float val = 0.0f;
        if (q < tile_len && p_b >= 0 && p_b < b_len) {
            val = B.values[b_start + p_b];
        }
        packedB.push_back(val);
    }
}

/* ============================================================
 * Step 5b:  Adaptive Tile Size Selection (15.6)
 *
 * Chooses an appropriate tile size based on matrix statistics.
 * Strategy:
 *   - If max diagonal length >> 128, use larger tiles to
 *     reduce task count and kernel launch overhead.
 *   - If contributor count per diagonal is very high (HEAVY),
 *     use smaller tiles to reduce smemA pressure and allow
 *     more active warps per SM.
 *   - Returns a tile size that is always a multiple of
 *     WARP_SIZE and at least WARP_SIZE.
 *
 * The adaptive tile size is used by build_all_adaptive() for
 * per-diagonal tile size selection.
 * ============================================================ */
struct TileConfig {
    int tile_size;     // chosen tile size for this diagonal
    int block_size;    // matching block size for the kernel
};

inline TileConfig
choose_tile_config(int diag_length, int num_contributors, int avg_pair_count)
{
    TileConfig cfg;

    if (num_contributors == 0) {
        cfg.tile_size  = TILE_SIZE;
        cfg.block_size = BLOCK_SIZE_MED;
        return cfg;
    }

    /* Estimate work density: average pairs times average overlap. */
    int est_work_per_tile_128 = avg_pair_count * std::min(diag_length, 128);

    if (est_work_per_tile_128 <= LIGHT_WORK_MAX) {
        /* Light work: use warp-sized tiles (32). */
        cfg.tile_size  = TILE_SIZE_LIGHT;
        cfg.block_size = BLOCK_SIZE_LIGHT;
    } else if (est_work_per_tile_128 > MEDIUM_WORK_MAX) {
        /* Heavy work: use larger tiles (256) with heavy kernel. */
        cfg.tile_size  = TILE_SIZE_HEAVY;
        cfg.block_size = BLOCK_SIZE_HEAVY;
    } else {
        /* Medium: default 128. But if the diagonal is very long,
         * consider larger tiles to reduce task count. */
        if (diag_length > ADAPTIVE_HUGE_DIAG_THRESH && avg_pair_count <= 4) {
            /* Very long diagonal, few pairs: WIDE tile (512) with multi-output. */
            cfg.tile_size  = WIDE_TILE_SIZE;
            cfg.block_size = WIDE_BLOCK_SIZE;
        } else if (diag_length > ADAPTIVE_LARGE_DIAG_THRESH && avg_pair_count <= 8) {
            /* Long diagonal, moderate pair count: use 256 tile. */
            cfg.tile_size  = TILE_SIZE_HEAVY;
            cfg.block_size = BLOCK_SIZE_HEAVY;
        } else {
            cfg.tile_size  = TILE_SIZE;
            cfg.block_size = BLOCK_SIZE_MED;
        }
    }

    return cfg;
}

/* ============================================================
 * sort_buckets_by_work
 *
 * Sorts each bucket's task_ids in DESCENDING order of work_est.
 * Heavy tasks launch first → fill SMs early; small tasks fill
 * the gaps at the tail.  Reduces warp-level imbalance.
 * ============================================================ */
inline void sort_buckets_by_work(PreprocessResult& res);

/* ============================================================
 * Step 6 + 7:  build_all  (master driver)
 *
 * Runs the full host preprocessing pipeline and returns
 * a PreprocessResult containing all tables ready for
 * device upload.
 *
 * The tile_size parameter enables Multi-precision Tiling
 * (15.5): callers can pass any tile size that is a multiple
 * of WARP_SIZE.  Default = TILE_SIZE (128) for backward
 * compatibility.
 * ============================================================ */
inline PreprocessResult
build_all(const DiagMatrix& A, const DiagMatrix& B,
          int M, int K, int N, int tile_size = TILE_SIZE)
{
    PreprocessResult res;

    /* ---- 1. enumerate output diagonals ---- */
    auto out_map = build_output_diagonals(A, B, M, N);

    /* ---- 2. find contributors ---- */
    build_contributors(out_map, A, B);

    /* ---- assign output diagonal storage ---- */
    int c_val_offset = 0;
    std::map<int, int> c_idx_map;   // d_c → index in output_diags

    for (auto& [d_c, info] : out_map) {
        int idx = static_cast<int>(res.output_diags.size());
        c_idx_map[d_c] = idx;
        OutputDiag od;
        od.offset       = d_c;
        od.start_row    = info.c_start_row;
        od.start_col    = info.c_start_col;
        od.length       = info.c_length;
        od.values_start = c_val_offset;
        res.output_diags.push_back(od);
        c_val_offset += info.c_length;
    }
    res.total_c_values = c_val_offset;

    /* ---- 3-7. per-output-diagonal: group, tile, emit ---- */
    for (auto& [d_c, info] : out_map) {
        if (info.contributors.empty()) continue;

        /* 3. sort contributors by a_diag */
        group_contributors_by_a_diag(info.contributors);

        int c_idx = c_idx_map[d_c];
        int c_sr  = info.c_start_row;

        /* 4-5. tile the output diagonal using the given tile_size */
        for (int p = 0; p < info.c_length; p += tile_size) {
            int p_begin = p;
            int p_len   = std::min(tile_size, info.c_length - p);
            int p_end   = p_begin + p_len;

            int work = estimate_tile_work(info.contributors, p_begin, p_end);
            if (work == 0) continue;

            int bucket;
            if      (work <= LIGHT_WORK_MAX)  bucket = 0;
            else if (work <= MEDIUM_WORK_MAX) bucket = 1;
            else                              bucket = 2;

            /* ---- build Group and PairMeta entries for this tile ---- */
            int task_group_begin = static_cast<int>(res.groups.size());
            int task_group_count = 0;

            int ci = 0;
            while (ci < static_cast<int>(info.contributors.size())) {
                int cur_a = info.contributors[ci].a_diag_idx;

                int grp_pair_begin = static_cast<int>(res.pairs.size());
                int grp_pair_count = 0;

                int j = ci;
                while (j < static_cast<int>(info.contributors.size()) &&
                       info.contributors[j].a_diag_idx == cur_a) {
                    const auto& cp = info.contributors[j];

                    /* tile-local valid range */
                    int ov_begin = std::max(cp.valid_begin, p_begin) - p_begin;
                    int ov_end   = std::min(cp.valid_end,   p_end)  - p_begin;

                    if (ov_begin < ov_end) {
                        int b_map = c_sr + cp.a_offset
                                  - DiagMatrix::diag_start_row(cp.b_offset);

                        int pair_pb_off = static_cast<int>(res.packedB.size());

                        /* Pack B for this pair (warp-major layout). */
                        pack_B_for_pair(res.packedB, B,
                                        cp.b_diag_idx, b_map,
                                        p_begin, p_len);

                        PairMeta pm;
                        pm.b_diag_idx      = cp.b_diag_idx;
                        pm.b_offset        = cp.b_offset;
                        pm.out_valid_begin = ov_begin;
                        pm.out_valid_end   = ov_end;
                        pm.a_base          = c_sr
                                           - DiagMatrix::diag_start_row(cp.a_offset);
                        pm.b_base          = b_map;
                        pm.packedB_offset  = pair_pb_off;
                        res.pairs.push_back(pm);
                        grp_pair_count++;
                    }
                    ++j;
                }

                if (grp_pair_count > 0) {
                    int a_sr = DiagMatrix::diag_start_row(
                                   info.contributors[ci].a_offset);
                    Group g;
                    g.a_diag_idx    = cur_a;
                    g.a_offset      = info.contributors[ci].a_offset;
                    g.a_global_start = A.diag_starts[cur_a];
                    g.a_diag_len    = A.diag_lengths[cur_a];
                    g.a_map_offset  = c_sr + p_begin - a_sr;
                    g.pair_begin    = grp_pair_begin;
                    g.pair_count    = grp_pair_count;
                    res.groups.push_back(g);
                    task_group_count++;
                }

                ci = j;
            }

            if (task_group_count > 0) {
                Task t;
                t.c_diag_idx  = c_idx;
                t.c_offset    = d_c;
                t.p_begin     = p_begin;
                t.p_len       = p_len;
                t.group_begin = task_group_begin;
                t.group_count = task_group_count;
                t.work_est    = work;
                t.bucket      = bucket;

                int tid = static_cast<int>(res.tasks.size());
                res.tasks.push_back(t);

                if      (bucket == 0) res.light_task_ids.push_back(tid);
                else if (bucket == 1) res.medium_task_ids.push_back(tid);
                else                  res.heavy_task_ids.push_back(tid);
            }
        }
    }

    sort_buckets_by_work(res);
    return res;
}

/* ---- sort_buckets_by_work (definition) ---- */
inline void
sort_buckets_by_work(PreprocessResult& res)
{
    auto cmp = [&](int a, int b) {
        return res.tasks[a].work_est > res.tasks[b].work_est;
    };
    std::sort(res.light_task_ids.begin(),  res.light_task_ids.end(),  cmp);
    std::sort(res.medium_task_ids.begin(), res.medium_task_ids.end(), cmp);
    std::sort(res.heavy_task_ids.begin(),  res.heavy_task_ids.end(),  cmp);
    std::sort(res.wide_task_ids.begin(),   res.wide_task_ids.end(),   cmp);
}

/* ============================================================
 * build_all_adaptive (15.5 + 15.6 combined)
 *
 * Uses per-output-diagonal adaptive tile sizing.
 * For each output diagonal, computes statistics (length,
 * contributor count, average pair count per group) and
 * selects an appropriate tile size via choose_tile_config().
 *
 * This allows:
 *   - Long, sparse diagonals to use larger tiles (256/512)
 *     reducing task count and launch overhead
 *   - Short, dense diagonals to use smaller tiles (32/64)
 *     reducing smemA pressure and allowing more occupancy
 *   - Default behavior identical to build_all() when all
 *     diagonals happen to choose TILE_SIZE=128
 * ============================================================ */
inline PreprocessResult
build_all_adaptive(const DiagMatrix& A, const DiagMatrix& B,
                   int M, int K, int N)
{
    PreprocessResult res;

    /* ---- 1. enumerate output diagonals ---- */
    auto out_map = build_output_diagonals(A, B, M, N);

    /* ---- 2. find contributors ---- */
    build_contributors(out_map, A, B);

    /* ---- assign output diagonal storage ---- */
    int c_val_offset = 0;
    std::map<int, int> c_idx_map;

    for (auto& [d_c, info] : out_map) {
        int idx = static_cast<int>(res.output_diags.size());
        c_idx_map[d_c] = idx;
        OutputDiag od;
        od.offset       = d_c;
        od.start_row    = info.c_start_row;
        od.start_col    = info.c_start_col;
        od.length       = info.c_length;
        od.values_start = c_val_offset;
        res.output_diags.push_back(od);
        c_val_offset += info.c_length;
    }
    res.total_c_values = c_val_offset;

    /* ---- 3-7. per-output-diagonal: adaptive tile, group, emit ---- */
    for (auto& [d_c, info] : out_map) {
        if (info.contributors.empty()) continue;

        /* 3. sort contributors by a_diag */
        group_contributors_by_a_diag(info.contributors);

        int c_idx = c_idx_map[d_c];
        int c_sr  = info.c_start_row;

        /* Compute per-diagonal statistics for adaptive tile selection. */
        int num_contrib = static_cast<int>(info.contributors.size());

        /* Count unique a_diags to estimate average pairs per group. */
        int num_groups_est = 1;
        for (int ci = 1; ci < num_contrib; ++ci) {
            if (info.contributors[ci].a_diag_idx !=
                info.contributors[ci - 1].a_diag_idx) {
                num_groups_est++;
            }
        }
        int avg_pairs = (num_groups_est > 0)
                        ? (num_contrib + num_groups_est - 1) / num_groups_est
                        : 0;

        /* Choose tile size adaptively. */
        TileConfig tc = choose_tile_config(info.c_length, num_contrib, avg_pairs);
        int chosen_tile = tc.tile_size;

        /* 4-5. tile the output diagonal with adaptive tile size */
        for (int p = 0; p < info.c_length; p += chosen_tile) {
            int p_begin = p;
            int p_len   = std::min(chosen_tile, info.c_length - p);
            int p_end   = p_begin + p_len;

            int work = estimate_tile_work(info.contributors, p_begin, p_end);
            if (work == 0) continue;

            int bucket;
            if (chosen_tile == WIDE_TILE_SIZE) {
                /* Wide tile selected by adaptive config: bucket 3. */
                bucket = 3;
            } else if (work <= LIGHT_WORK_MAX) {
                bucket = 0;
            } else if (work <= MEDIUM_WORK_MAX) {
                bucket = 1;
            } else {
                bucket = 2;
            }

            /* ---- build Group and PairMeta entries for this tile ---- */
            int task_group_begin = static_cast<int>(res.groups.size());
            int task_group_count = 0;

            int ci = 0;
            while (ci < static_cast<int>(info.contributors.size())) {
                int cur_a = info.contributors[ci].a_diag_idx;

                int grp_pair_begin = static_cast<int>(res.pairs.size());
                int grp_pair_count = 0;

                int j = ci;
                while (j < static_cast<int>(info.contributors.size()) &&
                       info.contributors[j].a_diag_idx == cur_a) {
                    const auto& cp = info.contributors[j];

                    int ov_begin = std::max(cp.valid_begin, p_begin) - p_begin;
                    int ov_end   = std::min(cp.valid_end,   p_end)  - p_begin;

                    if (ov_begin < ov_end) {
                        int b_map = c_sr + cp.a_offset
                                  - DiagMatrix::diag_start_row(cp.b_offset);

                        int pair_pb_off = static_cast<int>(res.packedB.size());

                        pack_B_for_pair(res.packedB, B,
                                        cp.b_diag_idx, b_map,
                                        p_begin, p_len);

                        PairMeta pm;
                        pm.b_diag_idx      = cp.b_diag_idx;
                        pm.b_offset        = cp.b_offset;
                        pm.out_valid_begin = ov_begin;
                        pm.out_valid_end   = ov_end;
                        pm.a_base          = c_sr
                                           - DiagMatrix::diag_start_row(cp.a_offset);
                        pm.b_base          = b_map;
                        pm.packedB_offset  = pair_pb_off;
                        res.pairs.push_back(pm);
                        grp_pair_count++;
                    }
                    ++j;
                }

                if (grp_pair_count > 0) {
                    int a_sr = DiagMatrix::diag_start_row(
                                   info.contributors[ci].a_offset);
                    Group g;
                    g.a_diag_idx    = cur_a;
                    g.a_offset      = info.contributors[ci].a_offset;
                    g.a_global_start = A.diag_starts[cur_a];
                    g.a_diag_len    = A.diag_lengths[cur_a];
                    g.a_map_offset  = c_sr + p_begin - a_sr;
                    g.pair_begin    = grp_pair_begin;
                    g.pair_count    = grp_pair_count;
                    res.groups.push_back(g);
                    task_group_count++;
                }

                ci = j;
            }

            if (task_group_count > 0) {
                Task t;
                t.c_diag_idx  = c_idx;
                t.c_offset    = d_c;
                t.p_begin     = p_begin;
                t.p_len       = p_len;
                t.group_begin = task_group_begin;
                t.group_count = task_group_count;
                t.work_est    = work;
                t.bucket      = bucket;

                int tid = static_cast<int>(res.tasks.size());
                res.tasks.push_back(t);

                if      (bucket == 0) res.light_task_ids.push_back(tid);
                else if (bucket == 1) res.medium_task_ids.push_back(tid);
                else if (bucket == 2) res.heavy_task_ids.push_back(tid);
                else                  res.wide_task_ids.push_back(tid);
            }
        }
    }

    sort_buckets_by_work(res);
    return res;
}

/* ============================================================
 * build_unified_task_ids
 *
 * Creates a single flat task list (ALL tasks, no bucket
 * separation) sorted by work_est DESCENDING.
 *
 * For use with launch_unified_kernel():
 *   PreprocessResult pr = build_all(A, B, M, K, N, WARP_SIZE);
 *   auto ids = build_unified_task_ids(pr);
 *   int* d_ids = upload(ids);
 *   launch_unified_kernel(..., d_ids, ..., ids.size());
 *
 * The descending sort ensures heavy tasks are assigned to
 * low-numbered warps (processed first in grid-stride),
 * minimizing tail effect.
 * ============================================================ */
inline std::vector<int>
build_unified_task_ids(const PreprocessResult& pr)
{
    std::vector<int> ids(pr.tasks.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
        return pr.tasks[a].work_est > pr.tasks[b].work_est;
    });
    return ids;
}
