# DiagSpMM — Diagonal Sparse Matrix Multiplication on GPU

```bash
nvcc test_hybrid.cu diag_hybrid_kernel.cu paper_hm_kernel.cu \
     -o test_hybrid -std=c++17 -O3 -arch=sm_90
```

High-performance CUDA kernel for diagonal sparse matrix multiplication (C = A × B in DIA format). Tuned for H100.

---

## Table of Contents

1. [Background](#1-background)
2. [Key Property](#2-key-property)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Structures](#4-data-structures)
5. [Host Preprocessing](#5-host-preprocessing)
6. [Kernel Design](#6-kernel-design)
7. [Shared Memory Layout](#7-shared-memory-layout)
8. [Memory Access Analysis](#8-memory-access-analysis)
9. [File Structure](#9-file-structure)
10. [Build & Run](#10-build--run)

---

## 1. Background

### Diagonal Format (DIA)

A sparse matrix whose nonzeros lie on diagonals. Each diagonal is described by:

| Field | Meaning |
|-------|---------|
| `offset` (int) | Diagonal offset: 0 = main, positive = super, negative = sub |
| `values[]` | Element values (contiguous storage) |

For an M × N matrix, diagonal with offset `d`:

```
start_row = max(0, -d)
start_col = max(0,  d)
length    = min(M - start_row, N - start_col)
```

---

## 2. Key Property

**Theorem**: A diagonal `d_a` of A (M×K) multiplied with diagonal `d_b` of B (K×N) contributes exclusively to diagonal `d_c = d_a + d_b` of C (M×N).

```
A[i][i + d_a] * B[i + d_a][(i + d_a) + d_b] = contribution to C[i][i + d_a + d_b]
=> d_c = d_a + d_b
```

This is an exact structural property — the entire kernel is built on it.

---

## 3. Architecture Overview

### Unified C-Centric Design

Each CTA owns a **group** of K=8 consecutive output C diagonals and one exclusive position **tile** of 128 elements. It iterates over A-diagonal **partitions**, loading each partition's A and B diagonals into shared memory, accumulating into register-resident `acc[K]`.

```
CTA
 |
 +-- Group: K=8 consecutive C diagonals (c_begin..c_begin+c_count)
      |
      +-- Tile: position range [tile_p_begin, tile_p_begin + TILE)
           |
           +-- Partition loop (split-K):
                for each partition of HYBRID_PARTITION_SIZE=53 A diagonals:
                    load B partition into smem  (one sync per phase)
                    load A partition into smem  (one sync)
                    accumulate into acc[K]      (register-resident, no sync)
                    __syncthreads()             (smem free for next partition)
           |
           +-- Write acc[K] -> C_vals[]
```

### Design Invariants

- **Zero atomics** in the kernel — each CTA owns an exclusive output region
- **O(1) shared memory** — fixed 57 KB per block regardless of matrix size
- **Accumulator in registers** — `acc[K]` persists across all partition iterations, no global partial buffer
- **Single kernel** — no corner/heavy split, no reduction pass

---

## 4. Data Structures

### PartBMeta — Per-partition B metadata

One entry per A-diagonal partition per group. Precomputed by the host.

```cpp
struct PartBMeta {
    int b_begin;    // offset into b_contrib[] relative to task.b_begin
    int b_count;    // B diagonals in this partition
    int b_d_min;    // min d_b among this partition's B contributors
    int b_d_range;  // b_d_max - b_d_min + 1  (lookup table width)
};
```

### HybridCDiag — Output diagonal metadata

```cpp
struct HybridCDiag {
    int c_offset;       // d_c
    int c_sr;           // start row: max(0, -d_c)
    int length;         // number of elements
    int values_start;   // offset into C_vals[]
};
```

### HybridTask — One task per (group, tile)

```cpp
struct HybridTask {
    int c_begin;       // first index into c_diags[]
    int c_count;       // C diagonals in this group (<= K=8)
    int min_c_sr;      // min start-row across group
    int spread;        // max_c_sr - min_c_sr
    int max_c_len;     // longest C diagonal in group
    int a_begin;       // start index into a_contrib[]
    int a_count;       // total A diagonals in this group
    int min_c_sc;      // min start-col across group
    int spread_sc;     // max_c_sc - min_c_sc
    int b_begin;       // start of this group's b_contrib entries
    int part_b_base;   // index into part_b_meta[] for partition 0
    int n_parts;       // ceil(a_count / HYBRID_PARTITION_SIZE)
    int tile_p_begin;  // position offset for this tile
};
```

### HybridKernelArgs — Kernel arguments

```cpp
struct HybridKernelArgs {
    const HybridTask*   tasks;
    int                 n_tasks;
    int                 max_smem;
    const HybridCDiag*  c_diags;
    int                 n_c_diags;
    const int*          a_contrib;    // flat array of A diagonal indices
    const int*          b_contrib;    // flat array of B diagonal indices
    const PartBMeta*    part_b_meta;  // per-partition B metadata
    const float*        A_vals;
    const int*          A_offsets;
    const int*          A_starts;
    const int*          A_lengths;
    int                 A_num_diags;
    const float*        B_vals;
    const int*          B_offsets;
    const int*          B_starts;
    const int*          B_lengths;
    float*              C_vals;
};
```

---

## 5. Host Preprocessing

`build_hybrid_plan()` runs entirely on the CPU before any GPU work:

```
1. Enumerate output diagonals
   For every (ai, bi) pair with d_a + d_b = d_c:
     record the contributing pair and its valid position range

2. Build c_diags[] table
   Assign values_start offsets into C_vals[]

3. Group consecutive C diagonals into groups of K=8

4. For each group:
   a. Compute geometry: min_c_sr, spread, min_c_sc, spread_sc, max_c_len
   b. Collect A contributors; sort by d_a ascending
   c. For each partition of HYBRID_PARTITION_SIZE A diagonals:
        - Compute d_b range: [min_d_c - d_a_max_p, max_d_c - d_a_min_p]
        - Collect B contributors in that d_b range
        - Emit PartBMeta entry
   d. Emit one HybridTask per tile (TILE=128 positions)

5. Compute max_smem across all groups
```

**Why sort A by d_a:** With A diagonals sorted ascending, each partition `[p*53, (p+1)*53)` maps to a contiguous d_b range (`d_b = d_c - d_a`). This makes per-partition B collection exact and contiguous — no set operations needed.

**Smem bound:**

```
max_b_per_part = HYBRID_PARTITION_SIZE + HYBRID_DIAGS_PER_CTA - 1 = 53 + 8 - 1 = 60
chunk    = TILE + spread       ≈ 128..136  (padded to multiple of 4)
chunk_b  = TILE + spread_sc    ≈ 128..136

smem = sizeof(float) × (53×chunk + 60×chunk_b + 60)
     ≈ 29 KB + 33 KB + 0.25 KB ≈ 57 KB  ✓ fits 4 blocks/SM on H100
```

---

## 6. Kernel Design

### 6.1 Thread-to-Output Mapping

```
Block:  HYBRID_BLOCK = 128 threads
Thread tid owns position (tile_p_begin + tid) within each C diagonal.
Each thread holds K=8 independent accumulators in registers.
```

### 6.2 Partition Loop

```cpp
for (int a_off = 0; a_off < total_a; a_off += HYBRID_PARTITION_SIZE) {

    // --- Per-partition B load ---
    const PartBMeta pmeta = args.part_b_meta[task.part_b_base + p_idx];

    // Init B lookup table
    for (i = tid; i < part_b_d_range_pad; i += BLOCK)  smem_B_lookup[i] = -1;
    __syncthreads();

    // Fill B lookup: d_b offset -> slot index
    for (sb = tid; sb < part_b_count; sb += BLOCK)
        smem_B_lookup[B_offsets[b_contrib[b_begin + b_begin_p + sb]] - b_d_min] = sb;
    __syncthreads();

    // Load B values into smem (vectorised float4)
    for each B diagonal sb in partition:
        for j = tid; j < chunk_b/4; j += BLOCK:
            smem_B[sb*chunk_b + j*4 .. +3] = B_vals[...];
    __syncthreads();

    // --- A load + accumulate ---
    for (a_batch of HYBRID_A_UNROLL=4 A diagonals):
        load A batch into smem_A (vectorised float4)
        __syncthreads()
        for each C diagonal ki:
            if !active[ki]: continue
            for each A diagonal in batch:
                sb = smem_B_lookup[d_b - b_d_min]
                if sb < 0: continue           // B has no such diagonal
                acc[ki] += smem_A[...] * smem_B[sb*chunk_b + ...]

    __syncthreads();  // smem free for next partition's B-lookup init
}

// --- Write output ---
for ki in [0, c_count):
    if active[ki]:  C_vals[values_start[ki] + tile_p_begin + tid] = acc[ki]
```

### 6.3 Sync Sequence per Partition

| Step | Barrier | Purpose |
|------|---------|---------|
| Lookup init | `__syncthreads()` | All `-1` writes visible before fill |
| Lookup fill | `__syncthreads()` | All slot indices visible before B load |
| B value load | `__syncthreads()` | All B values visible before accumulate |
| A load (each batch) | `__syncthreads()` | A values visible before accumulate |
| End of partition | `__syncthreads()` | smem free before next iteration's init |

### 6.4 Register Usage

| Data | Registers |
|------|-----------|
| `c_offset[8]`, `c_sr[8]`, `c_len[8]`, `c_sc[8]`, `c_start[8]` | 40 |
| `acc[8]` | 8 |
| `active[8]` | 8 |
| Loop vars, addresses, temporaries | ~20 |
| **Total estimate** | **~76** |

Fits within 128 registers at 4 blocks/SM on H100.

---

## 7. Shared Memory Layout

```
smem (57 KB per block):

[  smem_A  ][      smem_B      ][ smem_B_lookup ]
  53×chunk     60×chunk_b           60×4 bytes

smem_A:       HYBRID_PARTITION_SIZE × chunk floats
smem_B:       max_b_per_part × chunk_b floats
smem_B_lookup: max_b_per_part ints  (padded to 4-alignment)

chunk   = (TILE + spread   + 3) & ~3   ≈ 128–136
chunk_b = (TILE + spread_sc + 3) & ~3  ≈ 128–136
```

The same fixed smem region is **reused every partition** — `acc[K]` in registers carries the partial sums across iterations. smem does not grow with N.

---

## 8. Memory Access Analysis

### Global Memory

| Operation | Pattern | Notes |
|-----------|---------|-------|
| A load | coalesced float4 | 128 threads × 4 floats = 512 B/transaction |
| B load | coalesced float4 | per B diagonal per partition |
| B_lookup | broadcast | `d_b - b_d_min` same for all threads given (A diag, C diag) |
| C write | coalesced | 128 threads × 1 float per C diagonal |

### Shared Memory Bank Conflicts

```
smem_A read: smem_A[slot * chunk + (c_sr[ki] - min_c_sr) + tid]
  => consecutive threads read consecutive addresses → zero bank conflicts

smem_B read: smem_B[sb * chunk_b + (c_sc[ki] - min_c_sc) + tid]
  => same structure → zero bank conflicts
```

---

## 9. File Structure

```
DiaGPU/
+-- diag_types.cuh              Core type: DiagMatrix
+-- diag_host_preprocess.cuh    Host preprocessing utilities
|                                 build_output_diagonals()
|                                 sort_diag_matrix_by_offset()
|
+-- diag_hybrid_kernel.cuh      Kernel header
|                                 Constants (HYBRID_TILE, HYBRID_BLOCK,
|                                   HYBRID_DIAGS_PER_CTA, HYBRID_PARTITION_SIZE)
|                                 PartBMeta, HybridCDiag, HybridTask
|                                 HybridKernelArgs, HybridPlan
|                                 Declarations: hybrid_kernel, launch_hybrid
|
+-- diag_hybrid_kernel.cu       Implementation
|                                 build_hybrid_plan()
|                                 hybrid_kernel()
|                                 launch_hybrid()
|
+-- test_hybrid.cu              Test + benchmark driver
|                                 CPU reference
|                                 Correctness validation
|                                 Timing (CUDA events)
|                                 Comparison vs paper_hm_kernel
|
+-- paper_hm_kernel.cu/.cuh     Baseline: atomicAdd-based diagonal SpMM
+-- dia_spmv.cu / test_spmv.cu  Diagonal SpMV kernel
+-- gemm_diag/                  Dense GEMM baseline
```

---

## 10. Build & Run

```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Compile (adjust -arch as needed)
#   sm_80 = A100    sm_86 = RTX 3090
#   sm_89 = RTX 4090  sm_90 = H100
nvcc test_hybrid.cu diag_hybrid_kernel.cu paper_hm_kernel.cu \
     -o test_hybrid -std=c++17 -O3 -arch=sm_90

# Run (results written to results.md)
./test_hybrid

# Profile with Nsight Compute
ncu --profile-from-start off ./test_hybrid
```

### Key Tuning Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `HYBRID_TILE` | 128 | Positions per tile = threads per block |
| `HYBRID_BLOCK` | 128 | Threads per block |
| `HYBRID_DIAGS_PER_CTA` | 8 | C diagonals per group (K) |
| `HYBRID_BLOCKS_PER_SM` | 4 | Target occupancy |
| `HYBRID_PARTITION_SIZE` | 53 | A diagonals per smem partition |
| `HYBRID_A_UNROLL` | 4 | A diagonals unrolled per accumulate step |
