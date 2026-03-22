/* ============================================================
 * diag_host_preprocess.cuh
 *
 * Host-side preprocessing for DiagSpMM.
 *
 * Pipeline:
 *   1. build_output_diagonals()   – enumerate all d_c = d_a + d_b
 *   2. build_contributors()       – find valid ranges per (a,b) pair
 *   3. estimate_tile_work()       – sum of pair overlaps in tile
 *   4. build_all()                – tile, bucket, emit Task table
 *   5. build_b_diag_lookup()      – O(1) B diagonal lookup for kernel
 *
 * The kernel uses NO Group or PairMeta arrays — it computes
 * B indices on the fly from the A/B diagonal metadata.
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
 * Result bundle returned by build_all()
 * ============================================================ */
struct PreprocessResult {
    std::vector<Task>       tasks;
    std::vector<OutputDiag> output_diags;
    int                     total_c_values;  // total output elements

    /* Bucket-sorted task id lists (indices into tasks[]). */
    std::vector<int> light_task_ids;
    std::vector<int> medium_task_ids;
    std::vector<int> heavy_task_ids;
    std::vector<int> wide_task_ids;
};

/* ============================================================
 * Step 1:  Enumerate output diagonals
 * ============================================================ */
inline std::map<int, OutputDiagInfo>
build_output_diagonals(const DiagMatrix& A, const DiagMatrix& B,
                       int M, int N)
{
    std::map<int, OutputDiagInfo> out;
    for (int ai = 0; ai < A.num_diags; ++ai) {
        for (int bi = 0; bi < B.num_diags; ++bi) {
            int d_c = A.offsets[ai] + B.offsets[bi];
            if (out.count(d_c)) continue;
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

            int a_base = c_sr - a_sr;
            int b_base = c_sr + d_a - b_sr;

            int lo = 0;
            lo = std::max(lo, -a_base);
            lo = std::max(lo, -b_base);
            int hi = info.c_length;
            hi = std::min(hi, a_len - a_base);
            hi = std::min(hi, b_len - b_base);

            if (lo >= hi) continue;

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
 * Step 3:  Estimate work for a tile
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
 * Adaptive tile-size selection
 * ============================================================ */
struct TileConfig {
    int tile_size;
    int block_size;
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
    int est_work_per_tile_128 = avg_pair_count * std::min(diag_length, 128);
    if (est_work_per_tile_128 <= LIGHT_WORK_MAX) {
        cfg.tile_size  = TILE_SIZE_LIGHT;
        cfg.block_size = BLOCK_SIZE_LIGHT;
    } else if (est_work_per_tile_128 > MEDIUM_WORK_MAX) {
        cfg.tile_size  = TILE_SIZE_HEAVY;
        cfg.block_size = BLOCK_SIZE_HEAVY;
    } else {
        if (diag_length > ADAPTIVE_HUGE_DIAG_THRESH && avg_pair_count <= 4) {
            cfg.tile_size  = WIDE_TILE_SIZE;
            cfg.block_size = WIDE_BLOCK_SIZE;
        } else if (diag_length > ADAPTIVE_LARGE_DIAG_THRESH && avg_pair_count <= 8) {
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
 * ============================================================ */
inline void sort_buckets_by_work(PreprocessResult& res)
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
 * build_all  (master driver)
 *
 * Simplified: only builds Tasks and OutputDiags.
 * No Group or PairMeta construction — the kernel computes
 * B indices on the fly.
 * ============================================================ */
inline PreprocessResult
build_all(const DiagMatrix& A, const DiagMatrix& B,
          int M, int K, int N, int tile_size = TILE_SIZE)
{
    PreprocessResult res;

    auto out_map = build_output_diagonals(A, B, M, N);
    build_contributors(out_map, A, B);

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

    /* Tile each output diagonal, compute work, assign bucket. */
    for (auto& [d_c, info] : out_map) {
        if (info.contributors.empty()) continue;
        int c_idx = c_idx_map[d_c];

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

            /* Clamp: bucket must have enough threads for tile_size. */
            if (tile_size > TILE_SIZE       && bucket < 2) bucket = 2;
            if (tile_size > TILE_SIZE_HEAVY && bucket < 3) bucket = 3;

            Task t;
            t.c_diag_idx  = c_idx;
            t.c_offset    = d_c;
            t.p_begin     = p_begin;
            t.p_len       = p_len;
            t.work_est    = work;
            t.bucket      = bucket;

            int tid = static_cast<int>(res.tasks.size());
            res.tasks.push_back(t);

            if      (bucket == 0) res.light_task_ids.push_back(tid);
            else if (bucket == 1) res.medium_task_ids.push_back(tid);
            else                  res.heavy_task_ids.push_back(tid);
        }
    }

    sort_buckets_by_work(res);
    return res;
}

/* ============================================================
 * build_all_adaptive (per-diagonal adaptive tile sizing)
 * ============================================================ */
inline PreprocessResult
build_all_adaptive(const DiagMatrix& A, const DiagMatrix& B,
                   int M, int K, int N)
{
    PreprocessResult res;

    auto out_map = build_output_diagonals(A, B, M, N);
    build_contributors(out_map, A, B);

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

    for (auto& [d_c, info] : out_map) {
        if (info.contributors.empty()) continue;
        int c_idx = c_idx_map[d_c];

        int num_contrib = static_cast<int>(info.contributors.size());
        TileConfig tc = choose_tile_config(info.c_length, num_contrib,
                                           num_contrib);
        int chosen_tile = tc.tile_size;

        for (int p = 0; p < info.c_length; p += chosen_tile) {
            int p_begin = p;
            int p_len   = std::min(chosen_tile, info.c_length - p);
            int p_end   = p_begin + p_len;

            int work = estimate_tile_work(info.contributors, p_begin, p_end);
            if (work == 0) continue;

            int bucket;
            if (chosen_tile == WIDE_TILE_SIZE) {
                bucket = 3;
            } else if (work <= LIGHT_WORK_MAX) {
                bucket = 0;
            } else if (work <= MEDIUM_WORK_MAX) {
                bucket = 1;
            } else {
                bucket = 2;
            }

            /* Clamp: bucket must have enough threads for tile_size.
             *   LIGHT  (bucket 0): up to TILE_SIZE_LIGHT  (32)
             *   MEDIUM (bucket 1): up to TILE_SIZE         (128)
             *   HEAVY  (bucket 2): up to TILE_SIZE_HEAVY   (256)
             *   WIDE   (bucket 3): up to WIDE_TILE_SIZE    (512)  */
            if (chosen_tile > TILE_SIZE       && bucket < 2) bucket = 2;
            if (chosen_tile > TILE_SIZE_HEAVY && bucket < 3) bucket = 3;

            Task t;
            t.c_diag_idx  = c_idx;
            t.c_offset    = d_c;
            t.p_begin     = p_begin;
            t.p_len       = p_len;
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

    sort_buckets_by_work(res);
    return res;
}

/* ============================================================
 * build_unified_task_ids
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

/* ============================================================
 * build_b_diag_lookup
 *
 * Build a dense lookup array of size 2n-1:
 *   lookup[d + (n-1)] = index into B's diagonal list, or -1.
 * This enables O(1) diagonal matching in the kernel.
 * ============================================================ */
inline std::vector<int>
build_b_diag_lookup(const DiagMatrix& B, int n)
{
    std::vector<int> lookup(2 * n - 1, -1);
    for (int i = 0; i < B.num_diags; ++i) {
        int d = B.offsets[i];
        if (d >= -(n - 1) && d <= (n - 1))
            lookup[d + (n - 1)] = i;
    }
    return lookup;
}
