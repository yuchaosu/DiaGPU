/* ============================================================
 * gemm_diag.cu
 *
 * GEMM-style tiled diagonal SpMM kernel.
 *
 * C = A × B where A, B are banded matrices stored in
 * row-packed diagonal format.
 *
 * Key formula:
 *   C[k][t] = sum over (i,j) where i+j=k of DA[i][t] * DB[j][t+i]
 *
 * Tile: Ti A-diags × Tj B-diags × Tc columns.
 * SA loaded once into shared memory, reused across B-tile iterations.
 * Within-CTA reduction: sum Ti×Tj products → min(Ti+Tj-1) unique k
 * before atomicAdd, reducing atomic pressure by ~Ti×Tj.
 *
 * H100 optimization:
 *   Ti=4, Tj=4, Tc=32 → 512 threads, ~3 KB smem, 4 CTAs/SM.
 * ============================================================ */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t e = (call);                                      \
    if (e != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(e));       \
        exit(1);                                                 \
    }                                                            \
} while(0)

/* ============================================================
 * Tile sizes — tuned for H100 (sm_90)
 *
 * Threads per block: Ti * Tj * Tc = 4 * 4 * 32 = 512
 * Shared memory per CTA:
 *   SA: Ti × Tc × 4 = 512 B
 *   SB: Tj × SB_WIDTH × 4 = 576 B
 *   scratch: Ti × Tj × Tc × 4 = 2048 B
 *   Total: ~3.1 KB → 4 CTAs/SM → 2048 threads/SM (max)
 * ============================================================ */
constexpr int TI = 4;        /* A diagonals per tile */
constexpr int TJ = 4;        /* B diagonals per tile */
constexpr int TC = 32;       /* columns per tile (warp-aligned) */
constexpr int SB_PAD = TI;   /* extra SB columns for shift (worst case Ti-1, padded to Ti) */
constexpr int SB_WIDTH = TC + SB_PAD;

constexpr int BLOCK_SIZE = TI * TJ * TC;  /* 512 */

/* ============================================================
 * KERNEL
 *
 * Grid: (num_a_tiles, num_col_tiles)
 * Block: TI × TJ × TC = 512 threads
 *
 * Each CTA owns one A-tile and one column-tile.
 * Inner loop over B-tiles.
 *
 * Shared memory layout:
 *   SA[TI][TC]                    — A tile (loaded once)
 *   SB[TJ][SB_WIDTH]             — B tile (reloaded per B-tile iter)
 *   scratch[TI][TJ][TC]          — element-wise products
 *   reduce_buf[MAX_K_PER_TILE][TC] — reduction accumulators
 *
 * MAX_K_PER_TILE = TI + TJ - 1 = 7
 * ============================================================ */
constexpr int MAX_K_PER_TILE = TI + TJ - 1;

__global__ void
__launch_bounds__(BLOCK_SIZE, 4)
gemm_diag_kernel(
    /* Row-packed diagonal matrices (P×n and Q×n) */
    const float* __restrict__ DA,     /* [P][n] */
    const float* __restrict__ DB,     /* [Q][n] */
    const int*   __restrict__ off_A,  /* [P] diagonal offsets */
    const int*   __restrict__ off_B,  /* [Q] diagonal offsets */
    int P, int Q, int n,
    /* Output C in flat diagonal storage */
    float*       __restrict__ C_data,
    const int*   __restrict__ C_offsets, /* [num_k] start in C_data */
    const int*   __restrict__ C_lens,   /* [num_k] diagonal length */
    const int*   __restrict__ k_to_idx, /* [2n-1] k+n-1 → index, or -1 */
    int num_a_tiles, int num_col_tiles)
{
    /* --- Shared memory --- */
    __shared__ float SA[TI][TC];
    __shared__ float SB[TJ][SB_WIDTH];
    __shared__ float scratch[TI * TJ][TC]; /* flattened [TI*TJ][TC] */

    /* Per-tile info for k mapping (populated by thread 0). */
    __shared__ int s_off_a[TI];           /* A offsets in this tile */
    __shared__ int s_min_off_a, s_max_off_a;

    const int tid = threadIdx.x;
    const int t_local = tid % TC;                 /* 0..TC-1 */
    const int pair_id = tid / TC;                 /* 0..TI*TJ-1 */
    const int ri = pair_id / TJ;                  /* 0..TI-1 */
    const int rj = pair_id % TJ;                  /* 0..TJ-1 */

    const int ia = blockIdx.x;                    /* A tile index */
    const int tc = blockIdx.y;                    /* column tile index */
    const int t0 = tc * TC;                       /* global column start */

    /* --- Phase 0: Load A-tile metadata --- */
    if (tid < TI) {
        int ai = ia * TI + tid;
        s_off_a[tid] = (ai < P) ? off_A[ai] : 0;
    }
    __syncthreads();
    if (tid == 0) {
        s_min_off_a = s_off_a[0];
        s_max_off_a = s_off_a[0];
        for (int i = 1; i < TI; ++i) {
            if (ia * TI + i >= P) break;
            s_min_off_a = min(s_min_off_a, s_off_a[i]);
            s_max_off_a = max(s_max_off_a, s_off_a[i]);
        }
    }
    __syncthreads();

    const int min_off_a = s_min_off_a;

    /* --- Phase 1: Load SA[TI][TC] from DA --- */
    if (tid < TI * TC) {
        int r = tid / TC, c = tid % TC;
        int ai = ia * TI + r;
        int col = t0 + c;
        SA[r][c] = (ai < P && col < n) ? DA[ai * n + col] : 0.0f;
    }

    /* --- Main loop over B tiles --- */
    const int num_b_tiles = (Q + TJ - 1) / TJ;

    for (int jb = 0; jb < num_b_tiles; ++jb) {

        /* --- Phase 2: Load SB[TJ][SB_WIDTH] from DB --- */
        /* SB covers global columns [sb_base, sb_base + SB_WIDTH)
         * where sb_base = t0 + min_off_a to handle negative shifts. */
        const int sb_base = t0 + min_off_a;

        /* Collaborative load: TJ * SB_WIDTH elements. */
        for (int idx = tid; idx < TJ * SB_WIDTH; idx += BLOCK_SIZE) {
            int r = idx / SB_WIDTH;
            int c = idx % SB_WIDTH;
            int bj = jb * TJ + r;
            int col = sb_base + c;
            SB[r][c] = (bj < Q && col >= 0 && col < n)
                      ? DB[bj * n + col] : 0.0f;
        }
        __syncthreads();

        /* --- Phase 3: Compute element-wise products --- */
        {
            int ai = ia * TI + ri;
            int bj = jb * TJ + rj;
            float product = 0.0f;

            if (ai < P && bj < Q && (t0 + t_local) < n) {
                float a_val = SA[ri][t_local];
                /* Shifted B access: SB[rj][(t0 + t_local + off_a) - sb_base]
                 * = SB[rj][t_local + off_a - min_off_a] */
                int sb_col = t_local + s_off_a[ri] - min_off_a;
                float b_val = (sb_col >= 0 && sb_col < SB_WIDTH)
                            ? SB[rj][sb_col] : 0.0f;
                product = a_val * b_val;
            }
            scratch[ri * TJ + rj][t_local] = product;
        }
        __syncthreads();

        /* --- Phase 4: Sequential reduction per output diagonal k ---
         *
         * Zero shared-memory atomics.  One thread per (k, t) reads
         * all contributing scratch[ri][rj][t] entries directly.
         *
         * For Ti=4, Tj=4: up to TI+TJ-1=7 unique k values.
         * Each k has 1..min(Ti,Tj) contributing (ri,rj) pairs.
         *
         * Precompute contributor list per k in shared memory:
         *   s_k_contribs[ki][ci] = ri*TJ + rj (scratch row index)
         *   s_k_num_contribs[ki] = count
         *
         * Thread assignment: tid = ki * TC + t  (ki < num_k, t < TC)
         * Each thread sequentially sums s_k_num_contribs[ki] entries.
         * Max contributors = min(Ti,Tj) = 4 → trivial loop.
         * ---------------------------------------------------------- */
        {
            __shared__ int s_off_b[TJ];
            __shared__ int s_k_vals[MAX_K_PER_TILE];
            __shared__ int s_k_contribs[MAX_K_PER_TILE][TI < TJ ? TJ : TI];
            __shared__ int s_k_num_contribs[MAX_K_PER_TILE];
            __shared__ int s_num_k;

            /* Load B offsets for this B tile. */
            if (tid < TJ) {
                int bj = jb * TJ + tid;
                s_off_b[tid] = (bj < Q) ? off_B[bj] : 0;
            }

            /* Thread 0 builds k-value list and contributor map. */
            if (tid == 0) {
                s_num_k = 0;
                for (int i = 0; i < TI && (ia * TI + i) < P; ++i) {
                    for (int j = 0; j < TJ && (jb * TJ + j) < Q; ++j) {
                        int k = s_off_a[i] + s_off_b[j];
                        /* Find or create k entry. */
                        int ki = -1;
                        for (int x = 0; x < s_num_k; ++x) {
                            if (s_k_vals[x] == k) { ki = x; break; }
                        }
                        if (ki < 0 && s_num_k < MAX_K_PER_TILE) {
                            ki = s_num_k++;
                            s_k_vals[ki] = k;
                            s_k_num_contribs[ki] = 0;
                        }
                        if (ki >= 0) {
                            int ci = s_k_num_contribs[ki]++;
                            s_k_contribs[ki][ci] = i * TJ + j;
                        }
                    }
                }
            }
            __syncthreads();

            const int num_k = s_num_k;

            /* Each thread (ki, t) sequentially sums contributors.
             * No shared-memory atomics — pure register accumulation. */
            if (tid < num_k * TC) {
                int ki = tid / TC;
                int t  = tid % TC;
                int k  = s_k_vals[ki];
                int nc = s_k_num_contribs[ki];

                float acc = 0.0f;
                for (int ci = 0; ci < nc; ++ci) {
                    acc += scratch[s_k_contribs[ki][ci]][t];
                }

                /* Write to global C (atomicAdd: different A-tile CTAs
                 * may contribute to the same (k,t) position). */
                if (acc != 0.0f) {
                    int glob_t = t0 + t;
                    int idx = k + (n - 1);
                    if (idx >= 0 && idx < 2 * n - 1) {
                        int k_idx = k_to_idx[idx];
                        if (k_idx >= 0) {
                            int c_sr = (k >= 0) ? 0 : -k;
                            int p_c = glob_t - c_sr;
                            if (p_c >= 0 && p_c < C_lens[k_idx]) {
                                atomicAdd(&C_data[C_offsets[k_idx] + p_c], acc);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    } /* end B-tile loop */
}

/* ============================================================
 * CPU Reference
 * ============================================================ */
static void
cpu_diag_multiply(const float* DA, const int* off_A, int P,
                  const float* DB, const int* off_B, int Q,
                  int n, std::vector<float>& C_dense)
{
    C_dense.assign(n * n, 0.0f);
    for (int ri = 0; ri < P; ++ri) {
        int i = off_A[ri];
        for (int rj = 0; rj < Q; ++rj) {
            int j = off_B[rj];
            for (int t = 0; t < n; ++t) {
                float a = DA[ri * n + t];
                int bt = t + i;
                if (bt < 0 || bt >= n) continue;
                float b = DB[rj * n + bt];
                if (a == 0.0f || b == 0.0f) continue;
                int row = t, col = t + i + j;
                if (col < 0 || col >= n) continue;
                C_dense[row * n + col] += a * b;
            }
        }
    }
}

/* ============================================================
 * Build row-packed diagonal matrix
 * ============================================================ */
static void
pack_diag_matrix(int n, const std::vector<int>& offsets,
                 unsigned seed,
                 std::vector<float>& DA_out,
                 std::vector<int>& off_out)
{
    int P = (int)offsets.size();
    DA_out.assign(P * n, 0.0f);
    off_out = offsets;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);

    for (int ri = 0; ri < P; ++ri) {
        int d = offsets[ri];
        int sr = (d >= 0) ? 0 : -d;
        int len = n - std::abs(d);
        for (int p = 0; p < len; ++p) {
            DA_out[ri * n + sr + p] = dist(rng);
        }
    }
}

/* ============================================================
 * Build C diagonal layout
 * ============================================================ */
static void
build_c_layout(int n, const std::vector<int>& off_A,
               const std::vector<int>& off_B,
               std::vector<int>& C_offsets,
               std::vector<int>& C_lens,
               std::vector<int>& k_to_idx,
               int& total_c, int& num_k,
               std::vector<int>& k_list)
{
    /* Find all unique k = i + j */
    std::vector<bool> present(2 * n - 1, false);
    for (int i : off_A)
        for (int j : off_B) {
            int k = i + j;
            if (k >= -(n-1) && k <= (n-1))
                present[k + (n-1)] = true;
        }

    k_to_idx.assign(2 * n - 1, -1);
    k_list.clear();
    C_offsets.clear();
    C_lens.clear();
    total_c = 0;
    num_k = 0;

    for (int k = -(n-1); k <= (n-1); ++k) {
        if (!present[k + (n-1)]) continue;
        int len = n - std::abs(k);
        if (len <= 0) continue;
        k_to_idx[k + (n-1)] = num_k;
        k_list.push_back(k);
        C_offsets.push_back(total_c);
        C_lens.push_back(len);
        total_c += len;
        num_k++;
    }
}

/* ============================================================
 * Verify
 * ============================================================ */
static bool
verify(const float* gpu_C, const std::vector<int>& C_offsets,
       const std::vector<int>& C_lens, const std::vector<int>& k_list,
       const std::vector<float>& C_dense, int n, float tol)
{
    int mismatches = 0;
    for (int ki = 0; ki < (int)k_list.size(); ++ki) {
        int k = k_list[ki];
        int c_sr = (k >= 0) ? 0 : -k;
        int c_sc = (k >= 0) ? k : 0;
        for (int p = 0; p < C_lens[ki]; ++p) {
            float gpu_val = gpu_C[C_offsets[ki] + p];
            float cpu_val = C_dense[(c_sr + p) * n + (c_sc + p)];
            if (std::fabs(gpu_val - cpu_val) > tol) {
                if (mismatches < 5)
                    fprintf(stderr, "  MISMATCH k=%d p=%d: gpu=%.4f cpu=%.4f\n",
                            k, p, gpu_val, cpu_val);
                mismatches++;
            }
        }
    }
    return mismatches == 0;
}

/* ============================================================
 * Upload helper
 * ============================================================ */
template<typename T>
static T* upload(const std::vector<T>& v) {
    if (v.empty()) return nullptr;
    T* d; CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return d;
}

/* ============================================================
 * MAIN
 * ============================================================ */
int main(int argc, char** argv)
{
    int n  = (argc > 1) ? atoi(argv[1]) : 4096;
    int hb = (argc > 2) ? atoi(argv[2]) : 10;
    int iters = (argc > 3) ? atoi(argv[3]) : 100;

    printf("GEMM-Diag SpMM  n=%d  bw=%d  diags=%d  iters=%d\n",
           n, hb, 2*hb+1, iters);

    /* Build symmetric offsets {-hb, ..., +hb} */
    std::vector<int> offsets;
    for (int d = -hb; d <= hb; ++d) offsets.push_back(d);
    int P = (int)offsets.size();
    int Q = P;

    /* Pack DA, DB */
    std::vector<float> h_DA, h_DB;
    std::vector<int> h_offA, h_offB;
    pack_diag_matrix(n, offsets, 42, h_DA, h_offA);
    pack_diag_matrix(n, offsets, 137, h_DB, h_offB);

    /* Build C layout */
    std::vector<int> h_C_offsets, h_C_lens, h_k_to_idx, h_k_list;
    int total_c, num_k;
    build_c_layout(n, h_offA, h_offB, h_C_offsets, h_C_lens,
                   h_k_to_idx, total_c, num_k, h_k_list);

    printf("  P=%d  Q=%d  output_diags=%d  total_C=%d\n", P, Q, num_k, total_c);

    /* CPU reference */
    std::vector<float> C_dense;
    if (n <= 4096) {
        cpu_diag_multiply(h_DA.data(), h_offA.data(), P,
                          h_DB.data(), h_offB.data(), Q,
                          n, C_dense);
    }

    /* Upload */
    float* d_DA = upload(h_DA);
    float* d_DB = upload(h_DB);
    int*   d_offA = upload(h_offA);
    int*   d_offB = upload(h_offB);
    int*   d_C_off = upload(h_C_offsets);
    int*   d_C_len = upload(h_C_lens);
    int*   d_k2idx = upload(h_k_to_idx);

    float* d_C = nullptr;
    size_t c_bytes = total_c * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));

    /* Grid configuration */
    int num_a_tiles  = (P + TI - 1) / TI;
    int num_col_tiles = (n + TC - 1) / TC;

    dim3 grid(num_a_tiles, num_col_tiles);
    dim3 block(BLOCK_SIZE);

    printf("  grid=(%d, %d)  block=%d  SA:%dB  SB:%dB  scratch:%dB\n",
           num_a_tiles, num_col_tiles, BLOCK_SIZE,
           (int)(TI * TC * sizeof(float)),
           (int)(TJ * SB_WIDTH * sizeof(float)),
           (int)(TI * TJ * TC * sizeof(float)));

    /* Warmup */
    CUDA_CHECK(cudaMemset(d_C, 0, c_bytes));
    gemm_diag_kernel<<<grid, block>>>(
        d_DA, d_DB, d_offA, d_offB, P, Q, n,
        d_C, d_C_off, d_C_len, d_k2idx,
        num_a_tiles, num_col_tiles);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Verify */
    if (n <= 4096) {
        std::vector<float> h_C(total_c);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_bytes,
                              cudaMemcpyDeviceToHost));
        bool pass = verify(h_C.data(), h_C_offsets, h_C_lens, h_k_list,
                           C_dense, n, 1e-1f);
        printf("  Verification: %s\n", pass ? "PASS" : "FAIL");
    }

    /* Timed runs */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iters; ++iter) {
        CUDA_CHECK(cudaMemset(d_C, 0, c_bytes));
        gemm_diag_kernel<<<grid, block>>>(
            d_DA, d_DB, d_offA, d_offB, P, Q, n,
            d_C, d_C_off, d_C_len, d_k2idx,
            num_a_tiles, num_col_tiles);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    printf("  Kernel time: %.4f ms (avg over %d iters)\n", avg_ms, iters);

    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_DA); cudaFree(d_DB);
    cudaFree(d_offA); cudaFree(d_offB);
    cudaFree(d_C_off); cudaFree(d_C_len);
    cudaFree(d_k2idx); cudaFree(d_C);

    return 0;
}
