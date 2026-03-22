/* ============================================================
 * paper_algorithms.cu
 *
 * Implementation of the two GPU algorithms from:
 *   "GPU Algorithms for Structured Sparse Matrix Multiplication
 *    with Diagonal Storage Schemes"
 *   Haque, Parvez, Hossain — Algorithms 2024, 17, 31.
 *
 * Algorithm 1: Banded Matrix–Matrix Multiplication using CDM
 *   - Compact Diagonal Method storage
 *   - Shared-memory tiled A, global-memory B
 *   - 2D thread block (T×T), each thread computes one C entry
 *
 * Algorithm 2: Structured Sparse Matrix–Matrix Multiplication using HM
 *   - HM diagonal storage (Hossain–Mahmud)
 *   - One thread per nonzero of A
 *   - Atomic updates to C
 *   - 13 cases for diagonal type combinations
 *
 * Both algorithms operate on square matrices of dimension n×n
 * stored in their respective diagonal formats.
 *
 * Compile:
 *   nvcc paper_algorithms.cu -o paper_alg -std=c++17 -O2 -arch=sm_86
 * ============================================================ */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

#include <cuda_runtime.h>

/* ============================================================
 * CUDA error checking
 * ============================================================ */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/* ============================================================
 * Tile size T for Algorithm 1 (banded CDM multiplication).
 * Paper uses T = 2^d with d = 4, so T = 16.
 * T×T = 256 threads per block (multiple of warp size 32).
 * ============================================================ */
constexpr int T_TILE = 16;

/* ============================================================
 *                     STORAGE FORMATS
 * ============================================================ */

/* ------------------------------------------------------------
 * CDM: Compact Diagonal Method (Section 3.2 of the paper)
 *
 * For a banded n×n matrix with lower bandwidth kl, upper
 * bandwidth ku:
 *   - dgX  = max(kl, ku) + 1  compact diagonals (indices 0..dgX-1)
 *   - Compact diagonal k (for k >= 1) packs the kth super-
 *     diagonal (length n-k) on top of the (n-k)th subdiagonal
 *     (length k), forming a vector of length n.
 *   - Compact diagonal 0 is just the main diagonal (length n),
 *     padded to n if needed (already length n for square).
 *
 * Storage:
 *   DsCDM[i]  = diagonal index i (0, 1, 2, ..., dgX-1)
 *   Xlist[]   = flat array of size dgX × n
 *               Xlist[i*n .. i*n + n - 1] = compact diagonal i
 * ------------------------------------------------------------ */
struct CDMMatrix {
    int n;              /* matrix dimension (n×n) */
    int kl;             /* lower bandwidth */
    int ku;             /* upper bandwidth */
    int num_cdiags;     /* number of compact diagonals = max(kl,ku)+1 */
    std::vector<float> values;  /* flat: num_cdiags × n */

    /* Access compact diagonal d, position p (0-indexed within
     * the length-n compact vector). */
    float& at(int d, int p) { return values[d * n + p]; }
    float  at(int d, int p) const { return values[d * n + p]; }
};

/* Build a CDM representation from a dense n×n matrix.
 * The dense matrix is row-major: dense[i*n + j].             */
static CDMMatrix
dense_to_cdm(const float* dense, int n, int kl, int ku)
{
    CDMMatrix cdm;
    cdm.n  = n;
    cdm.kl = kl;
    cdm.ku = ku;
    cdm.num_cdiags = std::max(kl, ku) + 1;
    cdm.values.assign(cdm.num_cdiags * n, 0.0f);

    /* Compact diagonal 0: main diagonal A0.
     * A0[p] = dense[p][p] for p = 0..n-1. */
    for (int p = 0; p < n; ++p) {
        cdm.at(0, p) = dense[p * n + p];
    }

    /* Compact diagonal k (k >= 1):
     * Top part = kth superdiagonal Ak (length n-k):
     *   Ak[p] = dense[p][p+k]  for p = 0..n-k-1
     *   stored at positions 0..n-k-1 of compact diagonal k.
     *
     * Bottom part = (n-k)th subdiagonal A_{n-k} (length k):
     *   A_{n-k}[p] = dense[n-k+p][p]  for p = 0..k-1
     *   stored at positions n-k..n-1 of compact diagonal k.   */
    for (int k = 1; k < cdm.num_cdiags; ++k) {
        /* Superdiagonal k (if k <= ku) */
        if (k <= ku) {
            for (int p = 0; p < n - k; ++p) {
                cdm.at(k, p) = dense[p * n + (p + k)];
            }
        }

        /* Subdiagonal (n-k) (if n-k <= kl) */
        int sub_idx = n - k;
        if (sub_idx > 0 && sub_idx <= kl) {
            for (int p = 0; p < k; ++p) {
                cdm.at(k, n - k + p) = dense[(sub_idx + p) * n + p];
            }
        }
    }

    return cdm;
}

/* Convert CDM back to dense for verification. */
static void
cdm_to_dense(const CDMMatrix& cdm, float* dense)
{
    int n = cdm.n;
    memset(dense, 0, n * n * sizeof(float));

    /* Main diagonal from compact diagonal 0 */
    for (int p = 0; p < n; ++p) {
        dense[p * n + p] = cdm.at(0, p);
    }

    /* Compact diagonal k (k >= 1) */
    for (int k = 1; k < cdm.num_cdiags; ++k) {
        /* Top part: superdiagonal k */
        if (k <= cdm.ku) {
            for (int p = 0; p < n - k; ++p) {
                dense[p * n + (p + k)] = cdm.at(k, p);
            }
        }

        /* Bottom part: subdiagonal (n-k) */
        int sub_idx = n - k;
        if (sub_idx > 0 && sub_idx <= cdm.kl) {
            for (int p = 0; p < k; ++p) {
                dense[(sub_idx + p) * n + p] = cdm.at(k, n - k + p);
            }
        }
    }
}

/* HM types and helpers are shared with bench_compare.cu */
#include "paper_hm.cuh"

/* dense_to_hm() and hm_to_dense() are now provided by paper_hm.cuh */

/* ============================================================
 *       ALGORITHM 1: Banded Matrix Multiplication (CDM)
 *
 * Paper Section 6.1, Algorithm 1.
 *
 * C = A × B, where A and B are banded matrices stored in CDM.
 * C is also banded; its compact diagonal indices are computed
 * as (i + k) mod n for all pairs of A and B diagonals.
 *
 * Kernel: 2D thread block of T×T.
 *   - blockIdx.y selects the compact diagonal of C
 *   - blockIdx.x × T selects the column tile
 *   - Each thread computes one element of C
 *   - Shared memory: T×T tile of A values
 *   - B accessed from global memory
 *
 * Key formulas (Section 4):
 *   C_k = C_k + A_k * B_0^{rk}
 *       + Σ_{i=1}^{n-1} A_{(k+i) mod n} * B̃_i^{rk}
 *
 * where B̃_i is the "reverse" compact diagonal (sub/super swapped)
 * and ^{rk} denotes upward circular shift by k positions.
 *
 * In the tiled implementation, rather than explicitly constructing
 * the shifted/reversed B, we compute the required B element
 * indices on the fly from the diagonal indices of A and C.
 * ============================================================ */

/* BdiagDense: a lookup table of size n.
 * BdiagDense[d] = starting index of compact diagonal d in B's
 * flat value array, or -1 if diagonal d doesn't exist in B.
 * This enables O(1) lookup of B diagonal positions.           */

__global__ void
__launch_bounds__(T_TILE * T_TILE)
cdm_banded_matmul_kernel(
    const float* __restrict__ A_vals,    /* A.values: dgA × n */
    const int*   __restrict__ A_diags,   /* DsA: compact diagonal indices of A */
    int                       dgA,       /* number of compact diagonals in A */
    const float* __restrict__ B_vals,    /* B.values: dgB × n */
    const int*   __restrict__ BdiagDense,/* BdiagDense[0..n-1]: start offset or -1 */
    int                       dgB,       /* number of compact diagonals in B */
    float*       __restrict__ C_vals,    /* C.values: dgC × n */
    const int*   __restrict__ C_diags,   /* DsC: compact diagonal indices of C */
    int                       dgC,       /* number of compact diagonals in C */
    int                       n)         /* matrix dimension */
{
    /* Shared memory tile for a T×T block of A. */
    __shared__ float daS[T_TILE][T_TILE];

    /* This thread's position in the output C CDM array:
     *   myX = column within the compact diagonal (position along diagonal)
     *   myY = which compact diagonal of C                              */
    int myX = blockIdx.x * T_TILE + threadIdx.x;
    int myY = blockIdx.y * T_TILE + threadIdx.y;

    /* Out-of-bounds check: myY must index a valid C diagonal,
     * myX must be within [0, n). */
    if (myY >= dgC || myX >= n) return;

    int c_diag = C_diags[myY];  /* compact diagonal index of C row */
    float result = 0.0f;

    /* Number of tile iterations along A's diagonal dimension. */
    int maxLoop = (dgA + T_TILE - 1) / T_TILE;

    for (int j = 0; j < maxLoop; ++j) {
        /* Load A tile into shared memory.
         * Each thread loads one element:
         *   A diagonal index = T_TILE * j + threadIdx.x  (the A row in CDM)
         *   Position along diagonal = myX (same column as the C output)
         *
         * threadIdx.y selects the position, threadIdx.x selects
         * the A diagonal within this tile.                         */
        int a_diag_local = T_TILE * j + threadIdx.x;
        if (a_diag_local < dgA) {
            int a_diag = A_diags[a_diag_local];
            daS[threadIdx.x][threadIdx.y] = A_vals[a_diag * n + myX];
        } else {
            daS[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        /* For each A diagonal in this tile, find the corresponding
         * B element and accumulate.
         *
         * From the paper's formula:
         *   Compact diagonal c_diag of C gets contributions from
         *   A diagonal a_diag and B diagonal b_diag where:
         *     b_diag = (c_diag - a_diag + n) mod n
         *
         * The position in B depends on the shift operation.
         * For compact diagonal storage:
         *   - Position in A's compact diagonal a_diag at pos myX
         *     corresponds to matrix element (row, col) where:
         *     If myX < n - a_diag: from superdiagonal a_diag
         *       row = myX, col = myX + a_diag
         *     Else: from subdiagonal (n - a_diag)
         *       Let p = myX - (n - a_diag)
         *       row = (n - a_diag) + p, col = p
         *
         *   - The B element needed is determined by the shared
         *     index (column of A = row of B).
         *
         * Rather than fully deriving the shift formula, we use
         * the direct index computation:
         *   A at CDM position (a_diag, myX) hits matrix (row_a, k)
         *   B at CDM must provide element (k, col_c)
         *   where (row_c, col_c) is determined by (c_diag, myX). */
        for (int t = 0; t < T_TILE; ++t) {
            int a_diag_idx = T_TILE * j + t;
            if (a_diag_idx >= dgA) break;

            int a_diag = A_diags[a_diag_idx];

            /* Determine the matrix (row, col) for this C entry. */
            int row_c, col_c;
            if (c_diag == 0) {
                row_c = myX; col_c = myX;
            } else if (myX < n - c_diag) {
                /* From superdiagonal c_diag */
                row_c = myX; col_c = myX + c_diag;
            } else {
                /* From subdiagonal (n - c_diag) */
                int p = myX - (n - c_diag);
                row_c = (n - c_diag) + p; col_c = p;
            }

            /* Determine (row_a, k) from A's compact diagonal. */
            int row_a, k;
            if (a_diag == 0) {
                row_a = myX; k = myX;
            } else if (myX < n - a_diag) {
                row_a = myX; k = myX + a_diag;
            } else {
                int p = myX - (n - a_diag);
                row_a = (n - a_diag) + p; k = p;
            }

            /* A contributes to C[row_c][col_c] only if row_a == row_c. */
            if (row_a != row_c) continue;

            /* Find B element (k, col_c) in B's CDM storage. */
            int b_offset = (col_c >= k) ? (col_c - k) : (k - col_c);
            if (b_offset == 0) {
                /* Main diagonal of B */
                if (BdiagDense[0] >= 0) {
                    int b_pos = k; /* position along main diagonal */
                    result += daS[t][threadIdx.y] * B_vals[BdiagDense[0] + b_pos];
                }
            } else {
                /* b_offset is the diagonal index in B.
                 * Need to find which compact diagonal it belongs to
                 * and the position within it. */
                int b_diag_cdm = b_offset; /* compact diagonal index */
                if (b_diag_cdm < n && BdiagDense[b_diag_cdm] >= 0) {
                    /* Determine position within compact diagonal b_diag_cdm.
                     * If col_c > k: superdiagonal part, pos = k
                     * If col_c < k: subdiagonal part, pos = n - b_diag_cdm + col_c */
                    int b_pos;
                    if (col_c > k) {
                        /* Superdiagonal b_offset: pos = min(row_b, col_b) = k */
                        b_pos = k;
                    } else {
                        /* Subdiagonal (n - b_offset):
                         * pos in compact diag = (n - b_offset) + col_c
                         * = n - b_diag_cdm + col_c */
                        b_pos = n - b_diag_cdm + col_c;
                    }

                    if (b_pos >= 0 && b_pos < n) {
                        result += daS[t][threadIdx.y]
                                * B_vals[BdiagDense[b_diag_cdm] + b_pos];
                    }
                }
            }
        }

        __syncthreads();
    }

    /* Write result to C. */
    C_vals[c_diag * n + myX] = result;
}

/* ============================================================
 *   ALGORITHM 2: Structured Sparse Matrix Multiplication (HM)
 *
 * Paper Section 6.2, Algorithm 2.
 *
 * C = A × B, where A and B are structured sparse matrices
 * stored in HM format.
 *
 * Approach: one thread per nonzero of A.
 *   - Thread i handles A_values[i]
 *   - Finds which diagonal of A this element belongs to
 *     (via binary search on StartDgA)
 *   - For each diagonal of B, computes the index of the B
 *     element to multiply with and the C element to update
 *   - Updates C via atomicAdd
 *
 * The 13 cases (Section 6.2) are handled by two general rules:
 *   Rule 1: A[i][j] * B[j][k] -> C[i][k]  (coordinate form)
 *   Rule 2: coordinate to HM: element at (i,j) with d=j-i
 *           is stored at HM diagonal d, position min(i,j)
 * ============================================================ */

/* hm_structured_sparse_matmul_kernel is declared in paper_hm.cuh
 * and defined in paper_hm_kernel.cu. When compiling paper_algorithms.cu
 * as a standalone binary, link with paper_hm_kernel.cu. */

/* ============================================================
 *              CPU REFERENCE IMPLEMENTATIONS
 * ============================================================ */

/* Dense matrix multiply: C = A × B (row-major, n×n). */
static void
cpu_dense_matmul(const float* A, const float* B, float* C, int n)
{
    memset(C, 0, n * n * sizeof(float));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            float a = A[i * n + k];
            if (a == 0.0f) continue;
            for (int j = 0; j < n; ++j)
                C[i * n + j] += a * B[k * n + j];
        }
}

/* ============================================================
 *              HOST PREPROCESSING FOR ALGORITHM 1
 * ============================================================ */

/* Compute the set of compact diagonal indices for C = A × B.
 *
 * From the paper (Section 6.1):
 *   Compact diagonal i of C is computed by multiplying
 *   compact diagonal k of A and compact diagonal (i-k) mod n of B,
 *   for all k.
 *
 * The set of C's compact diags = { (a + b) mod n : a in A_diags, b in B_diags }
 */
static std::vector<int>
compute_c_cdm_diags(const CDMMatrix& A, const CDMMatrix& B)
{
    int n = A.n;
    std::vector<bool> present(n, false);

    /* For CDM, diagonal indices range from 0 to num_cdiags-1.
     * We enumerate all (a_diag + b_diag) mod n combinations. */
    for (int ai = 0; ai < A.num_cdiags; ++ai) {
        for (int bi = 0; bi < B.num_cdiags; ++bi) {
            int c_diag = (ai + bi) % n;
            present[c_diag] = true;
        }
    }

    std::vector<int> result;
    for (int d = 0; d < n; ++d) {
        if (present[d]) result.push_back(d);
    }
    return result;
}

/* Build BdiagDense: BdiagDense[d] = offset into B.values for
 * compact diagonal d, or -1 if B doesn't have diagonal d. */
static std::vector<int>
build_b_diag_dense(const CDMMatrix& B)
{
    std::vector<int> dense(B.n, -1);
    for (int i = 0; i < B.num_cdiags; ++i) {
        dense[i] = i * B.n;  /* compact diagonal i starts at i*n */
    }
    return dense;
}

/* compute_c_hm_structure() and build_c_diag_lookup() are now
 * provided by paper_hm.cuh. */

/* ============================================================
 *                    TEST GENERATORS
 * ============================================================ */

/* Generate a random banded matrix of size n×n with bandwidth bw
 * (kl = ku = bw). Values in [0.5, 1.5]. */
static void
generate_banded(float* dense, int n, int bw, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    memset(dense, 0, n * n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        for (int j = std::max(0, i - bw); j <= std::min(n - 1, i + bw); ++j) {
            dense[i * n + j] = dist(rng);
        }
    }
}

/* Generate a structured sparse matrix with the given diagonal offsets. */
static void
generate_structured_sparse(float* dense, int n,
                           const std::vector<int>& offsets, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    memset(dense, 0, n * n * sizeof(float));
    for (int d : offsets) {
        int sr = (d >= 0) ? 0 : -d;
        int sc = (d >= 0) ? d :  0;
        int len = n - std::abs(d);
        for (int p = 0; p < len; ++p) {
            dense[(sr + p) * n + (sc + p)] = dist(rng);
        }
    }
}

/* ============================================================
 *                  VERIFICATION HELPERS
 * ============================================================ */
static bool
verify(const float* gpu, const float* cpu, int n, float tol,
       const char* label)
{
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < n * n; ++i) {
        float diff = std::fabs(gpu[i] - cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > tol) {
            if (mismatches < 5) {
                int r = i / n, c = i % n;
                printf("  %s MISMATCH at (%d,%d): gpu=%.6f cpu=%.6f diff=%.6f\n",
                       label, r, c, gpu[i], cpu[i], diff);
            }
            ++mismatches;
        }
    }
    if (mismatches == 0) {
        printf("  %s PASS (max_diff=%.2e, tol=%.0e)\n", label, max_diff, tol);
        return true;
    } else {
        printf("  %s FAIL: %d mismatches (max_diff=%.2e, tol=%.0e)\n",
               label, mismatches, max_diff, tol);
        return false;
    }
}

/* ============================================================
 *                         MAIN
 * ============================================================ */
int main()
{
    printf("================================================================\n");
    printf("  Paper Algorithms: GPU Structured Sparse Matrix Multiplication\n");
    printf("  Haque, Parvez, Hossain — Algorithms 2024, 17, 31\n");
    printf("================================================================\n\n");

    /* ============================================================
     * TEST 1: Algorithm 1 — Banded Matrix Multiplication (CDM)
     * ============================================================ */
    {
        const int n  = 512;
        const int bw = 10;  /* bandwidth (kl = ku = 10) */

        printf("=== Algorithm 1: Banded CDM Multiplication ===\n");
        printf("  Matrix size: %d x %d, bandwidth: %d\n\n", n, n, bw);

        /* Generate test matrices */
        std::vector<float> denseA(n * n), denseB(n * n), denseC_cpu(n * n);
        generate_banded(denseA.data(), n, bw, 42);
        generate_banded(denseB.data(), n, bw, 137);

        /* CPU reference */
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_dense_matmul(denseA.data(), denseB.data(), denseC_cpu.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  CPU reference: %.2f ms\n", cpu_ms);

        /* Convert to CDM */
        CDMMatrix A_cdm = dense_to_cdm(denseA.data(), n, bw, bw);
        CDMMatrix B_cdm = dense_to_cdm(denseB.data(), n, bw, bw);

        /* Compute C's compact diagonal set */
        std::vector<int> c_diags = compute_c_cdm_diags(A_cdm, B_cdm);
        int dgC = static_cast<int>(c_diags.size());
        printf("  CDM: A has %d compact diags, B has %d, C has %d\n",
               A_cdm.num_cdiags, B_cdm.num_cdiags, dgC);

        /* BdiagDense lookup */
        std::vector<int> bdiag_dense = build_b_diag_dense(B_cdm);

        /* Allocate C CDM (dgC × n) */
        std::vector<float> C_cdm_vals(dgC * n, 0.0f);

        /* A diagonal indices: just 0, 1, ..., dgA-1 for CDM */
        std::vector<int> a_diag_indices(A_cdm.num_cdiags);
        std::iota(a_diag_indices.begin(), a_diag_indices.end(), 0);

        /* Upload to device */
        float *d_A, *d_B, *d_C;
        int *d_A_diags, *d_BdiagDense, *d_C_diags;

        CUDA_CHECK(cudaMalloc(&d_A, A_cdm.values.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, A_cdm.values.data(),
                   A_cdm.values.size() * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_B, B_cdm.values.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_B, B_cdm.values.data(),
                   B_cdm.values.size() * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_C, dgC * n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_C, 0, dgC * n * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_A_diags, a_diag_indices.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_A_diags, a_diag_indices.data(),
                   a_diag_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_BdiagDense, bdiag_dense.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_BdiagDense, bdiag_dense.data(),
                   bdiag_dense.size() * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_C_diags, c_diags.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_C_diags, c_diags.data(),
                   c_diags.size() * sizeof(int), cudaMemcpyHostToDevice));

        /* Launch kernel */
        dim3 block(T_TILE, T_TILE);
        dim3 grid((n + T_TILE - 1) / T_TILE,
                  (dgC + T_TILE - 1) / T_TILE);

        /* Warmup */
        cdm_banded_matmul_kernel<<<grid, block>>>(
            d_A, d_A_diags, A_cdm.num_cdiags,
            d_B, d_BdiagDense, B_cdm.num_cdiags,
            d_C, d_C_diags, dgC, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Timed run */
        CUDA_CHECK(cudaMemset(d_C, 0, dgC * n * sizeof(float)));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        cdm_banded_matmul_kernel<<<grid, block>>>(
            d_A, d_A_diags, A_cdm.num_cdiags,
            d_B, d_BdiagDense, B_cdm.num_cdiags,
            d_C, d_C_diags, dgC, n);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
        printf("  GPU kernel: %.3f ms\n", gpu_ms);
        printf("  Speedup: %.1fx\n", cpu_ms / gpu_ms);

        /* Download and convert to dense for verification */
        CUDA_CHECK(cudaMemcpy(C_cdm_vals.data(), d_C,
                   dgC * n * sizeof(float), cudaMemcpyDeviceToHost));

        /* Reconstruct dense C from CDM output.
         * Each C compact diagonal c_diags[i] maps to C_cdm_vals[i*n..]. */
        std::vector<float> denseC_gpu(n * n, 0.0f);
        for (int ci = 0; ci < dgC; ++ci) {
            int cd = c_diags[ci];
            if (cd == 0) {
                /* Main diagonal */
                for (int p = 0; p < n; ++p)
                    denseC_gpu[p * n + p] = C_cdm_vals[ci * n + p];
            } else {
                /* Superdiagonal cd (top part, positions 0..n-cd-1) */
                if (cd <= 2 * bw) {  /* C bandwidth <= 2*bw */
                    for (int p = 0; p < n - cd; ++p)
                        denseC_gpu[p * n + (p + cd)] += C_cdm_vals[ci * n + p];
                }
                /* Subdiagonal (n-cd) (bottom part, positions n-cd..n-1) */
                int sub = n - cd;
                if (sub > 0 && sub <= 2 * bw) {
                    for (int p = 0; p < cd; ++p)
                        denseC_gpu[(sub + p) * n + p] += C_cdm_vals[ci * n + (n - cd + p)];
                }
            }
        }

        verify(denseC_gpu.data(), denseC_cpu.data(), n, 1e-1f, "Algorithm1-CDM");

        /* Cleanup */
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaFree(d_A_diags); cudaFree(d_BdiagDense); cudaFree(d_C_diags);
        printf("\n");
    }

    /* ============================================================
     * TEST 2: Algorithm 2 — Structured Sparse Multiplication (HM)
     * ============================================================ */
    {
        const int n = 1024;

        /* Random diagonal offsets (structured sparse pattern) */
        std::vector<int> offsets_a = {0, 1, -1, 3, -3, 7, -7, 15, -15,
                                      50, -50, 100, -100, 200, -200};
        std::vector<int> offsets_b = {0, 2, -2, 5, -5, 10, -10, 20, -20,
                                      80, -80, 150, -150};

        /* Filter out-of-range offsets */
        offsets_a.erase(
            std::remove_if(offsets_a.begin(), offsets_a.end(),
                           [n](int d) { return std::abs(d) >= n; }),
            offsets_a.end());
        offsets_b.erase(
            std::remove_if(offsets_b.begin(), offsets_b.end(),
                           [n](int d) { return std::abs(d) >= n; }),
            offsets_b.end());

        printf("=== Algorithm 2: Structured Sparse HM Multiplication ===\n");
        printf("  Matrix size: %d x %d\n", n, n);
        printf("  A diagonals: %zu, B diagonals: %zu\n",
               offsets_a.size(), offsets_b.size());

        /* Generate dense matrices */
        std::vector<float> denseA(n * n), denseB(n * n), denseC_cpu(n * n);
        generate_structured_sparse(denseA.data(), n, offsets_a, 42);
        generate_structured_sparse(denseB.data(), n, offsets_b, 137);

        /* Count nonzeros */
        int nzA = 0, nzB = 0;
        for (int i = 0; i < n * n; ++i) {
            if (denseA[i] != 0.0f) nzA++;
            if (denseB[i] != 0.0f) nzB++;
        }
        printf("  nzA = %d, nzB = %d\n", nzA, nzB);

        /* CPU reference */
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_dense_matmul(denseA.data(), denseB.data(), denseC_cpu.data(), n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  CPU reference: %.2f ms\n", cpu_ms);

        /* Convert to HM */
        HMMatrix A_hm = dense_to_hm(denseA.data(), n, offsets_a);
        HMMatrix B_hm = dense_to_hm(denseB.data(), n, offsets_b);

        /* Compute C structure */
        HMMatrix C_hm = compute_c_hm_structure(A_hm, B_hm, n);
        printf("  C diagonals: %d, total C values: %d\n",
               C_hm.num_diags, C_hm.total_nz);

        /* C diagonal lookup */
        std::vector<int> c_lookup = build_c_diag_lookup(C_hm, n);

        /* Upload to device */
        float *d_A_vals, *d_B_vals, *d_C_vals;
        int *d_A_offsets, *d_A_starts, *d_A_lengths;
        int *d_B_offsets, *d_B_starts, *d_B_lengths;
        int *d_C_offsets, *d_C_starts, *d_C_lengths;
        int *d_C_lookup;

        auto upload_vec_f = [](const std::vector<float>& v) -> float* {
            float* d; CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            return d;
        };
        auto upload_vec_i = [](const std::vector<int>& v) -> int* {
            int* d; CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            return d;
        };

        d_A_vals    = upload_vec_f(A_hm.values);
        d_A_offsets = upload_vec_i(A_hm.diag_offsets);
        d_A_starts  = upload_vec_i(A_hm.diag_starts);
        d_A_lengths = upload_vec_i(A_hm.diag_lengths);

        d_B_vals    = upload_vec_f(B_hm.values);
        d_B_offsets = upload_vec_i(B_hm.diag_offsets);
        d_B_starts  = upload_vec_i(B_hm.diag_starts);
        d_B_lengths = upload_vec_i(B_hm.diag_lengths);

        /* C values: zero-initialized on device */
        CUDA_CHECK(cudaMalloc(&d_C_vals, C_hm.total_nz * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_C_vals, 0, C_hm.total_nz * sizeof(float)));
        d_C_offsets = upload_vec_i(C_hm.diag_offsets);
        d_C_starts  = upload_vec_i(C_hm.diag_starts);
        d_C_lengths = upload_vec_i(C_hm.diag_lengths);
        d_C_lookup  = upload_vec_i(c_lookup);

        /* Launch kernel */
        const int block_size = 256;
        int grid_size = (A_hm.total_nz + block_size - 1) / block_size;

        /* Warmup */
        hm_structured_sparse_matmul_kernel<<<grid_size, block_size>>>(
            d_A_vals, d_A_offsets, d_A_starts, d_A_lengths, A_hm.num_diags,
            d_B_vals, d_B_offsets, d_B_starts, d_B_lengths, B_hm.num_diags,
            d_C_vals, d_C_offsets, d_C_starts, d_C_lengths, C_hm.num_diags,
            d_C_lookup, A_hm.total_nz, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Timed run */
        CUDA_CHECK(cudaMemset(d_C_vals, 0, C_hm.total_nz * sizeof(float)));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        hm_structured_sparse_matmul_kernel<<<grid_size, block_size>>>(
            d_A_vals, d_A_offsets, d_A_starts, d_A_lengths, A_hm.num_diags,
            d_B_vals, d_B_offsets, d_B_starts, d_B_lengths, B_hm.num_diags,
            d_C_vals, d_C_offsets, d_C_starts, d_C_lengths, C_hm.num_diags,
            d_C_lookup, A_hm.total_nz, n);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
        printf("  GPU kernel: %.3f ms\n", gpu_ms);
        printf("  Speedup: %.1fx\n", cpu_ms / gpu_ms);

        /* Download and convert to dense for verification */
        std::vector<float> h_C_vals(C_hm.total_nz);
        CUDA_CHECK(cudaMemcpy(h_C_vals.data(), d_C_vals,
                   C_hm.total_nz * sizeof(float), cudaMemcpyDeviceToHost));

        /* Copy into a local HMMatrix and convert to dense */
        HMMatrix C_result = C_hm;
        C_result.values = h_C_vals;
        std::vector<float> denseC_gpu(n * n, 0.0f);
        hm_to_dense(C_result, denseC_gpu.data());

        verify(denseC_gpu.data(), denseC_cpu.data(), n, 1e-2f, "Algorithm2-HM");

        /* Cleanup */
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cudaFree(d_A_vals); cudaFree(d_A_offsets);
        cudaFree(d_A_starts); cudaFree(d_A_lengths);
        cudaFree(d_B_vals); cudaFree(d_B_offsets);
        cudaFree(d_B_starts); cudaFree(d_B_lengths);
        cudaFree(d_C_vals); cudaFree(d_C_offsets);
        cudaFree(d_C_starts); cudaFree(d_C_lengths);
        cudaFree(d_C_lookup);
    }

    printf("\n================================================================\n");
    printf("  Done.\n");
    printf("================================================================\n");

    return 0;
}
