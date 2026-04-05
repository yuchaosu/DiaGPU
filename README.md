# DiagSpMM — Diagonal Sparse Matrix Multiplication on GPU

```
nvcc test_hybrid.cu diag_hybrid_kernel.cu paper_hm_kernel.cu -o test_hybrid -std=c++17 -O3 -arch=sm_90
```

High-performance CUDA kernel for diagonal sparse matrix multiplication (C = A x B in DIA format). Tuned for H100.

---

## Table of Contents

1. [Background](#1-background)
2. [Key Property](#2-key-property)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Structures](#4-data-structures)
5. [Host Preprocessing](#5-host-preprocessing)
6. [Kernel Design](#6-kernel-design)
7. [Optimisations](#7-optimisations)
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

For an M x N matrix, diagonal offset `d`:

```
start_row = max(0, -d)
start_col = max(0,  d)
length    = min(M - start_row, N - start_col)
```

---

## 2. Key Property

**Theorem**: A diagonal `d_a` of A(M x K) multiplied with diagonal `d_b` of B(K x N) contributes exclusively to diagonal `d_c = d_a + d_b` of C(M x N).

```
A[i][i + d_a] * B[i + d_a][(i + d_a) + d_b] = contribution to C[i][i + d_a + d_b]
=> d_c = d_a + d_b
```

This is an exact structural property. The entire framework builds on it.

---

## 3. Architecture Overview

### Unified C-Centric Design

Each CTA owns a **group** of K=8 consecutive output C diagonals. It tiles by position, preloads all contributing A diagonal chunks into shared memory, then accumulates with B values loaded directly into registers.

```
CTA (persistent, grid-stride)
 |
 +-- Group: K=8 consecutive C diagonals
      |
      +-- Tile loop: p_begin = 0, TILE, 2*TILE, ...
           |
           +-- Phase 1: preload A chunks into smem (one sync)
           +-- Phase 2: accumulate A*B for all K diags (sync-free)
           +-- Phase 3: write results
```

### Corner vs Heavy Split

Based on shared memory capacity, not pair count:

- **Corner group**: all contributing A diagonals fit in smem in one pass. CTA writes directly to `C_vals`. One task per group.
- **Heavy group**: too many A diagonals for smem. A contributors are partitioned. Each partition is a separate task writing to `partial_buf`. A reduction kernel sums partials into `C_vals`.

Both paths use the **same computation kernel** (template-specialized at compile time). The corner kernel gets a smaller smem allocation, leaving more L1 cache for B loads.

### Design Invariants

- Zero atomic operations in computation kernel
- One CTA = exclusive output region (no write conflicts)
- A staged in shared memory, B in registers
- K local accumulators per thread, single final writeback

---

## 4. Data Structures

### HybridCDiag — Output diagonal metadata

```cpp
struct HybridCDiag {
    int c_offset;       // d_c
    int c_sr;           // start row: max(0, -d_c)
    int length;         // number of elements
    int values_start;   // offset into C_vals[]
};
```

### HybridTask — Computation task (corner or heavy partition)

```cpp
struct HybridTask {
    int c_begin;       // first index into c_diags[]
    int c_count;       // C diags in this group (<=K)
    int min_c_sr;      // min c_sr across group
    int spread;        // max_c_sr - min_c_sr
    int max_c_len;     // longest C diagonal in group
    int a_begin;       // index into a_contrib[]
    int a_count;       // number of A diags in this partition
    int is_direct;     // 1=corner (C_vals), 0=heavy (partial_buf)
    int out_offset;    // partial_buf offset (heavy only)
};
```

### HybridReduceTask — Reduction task (heavy groups only)

```cpp
struct HybridReduceTask {
    int c_begin, c_count;
    int min_c_sr, spread, max_c_len;
    int partial_base;   // start in partial_buf
    int num_partials;   // partitions to sum
};
```

### HybridKernelArgs — Kernel arguments

Contains pointers to corner/heavy task arrays, reduce tasks, C diagonal metadata, A/B matrix data, extended B_lookup, and output buffers.

---

## 5. Host Preprocessing

`build_hybrid_plan()` performs all preprocessing on the CPU:

```
1. Enumerate output diagonals and contributor pairs
2. Build c_diags[] table
3. Group consecutive c_diags into groups of K=8
4. For each group:
   a. Compute geometry (min_c_sr, spread, max_c_len)
   b. Find contributing A diagonals (union across K C diags)
   c. If all A diags fit in smem -> corner (1 task, direct write)
   d. Else -> heavy (partition A, P tasks + 1 reduce task)
5. Build a_contrib[] flat array of A diagonal indices per task
```

The corner/heavy threshold is purely a shared memory capacity check:

```
smem_needed = 2 * a_count * chunk * sizeof(float)    // 2x for double buffering
fits = (smem_needed <= HYBRID_SMEM_BUDGET)            // 57 KB per block on H100
```

---

## 6. Kernel Design

### 6.1 Thread-to-Output Mapping

```
Block:  128 threads (HYBRID_BLOCK)
Thread tid owns position p_begin + tid within each C diagonal.
Each thread holds K=8 independent accumulators in registers.
```

### 6.2 Shared Memory: A Chunk Layout

For K consecutive C diagonals with different `c_sr` values, the A data at the same row is at different offsets. The smem chunk is wider than TILE to accommodate this:

```
chunk = TILE + spread       (spread = max_c_sr - min_c_sr, <= K-1)
chunk is padded to multiple of 4 for float4 loads

smem layout (double-buffered):
  buf[0]: smem[0 .. a_count*chunk - 1]
  buf[1]: smem[a_count*chunk .. 2*a_count*chunk - 1]
```

Thread tid for C diagonal ki reads:

```
smem_A[slot * chunk + (c_sr[ki] - min_c_sr) + tid]
```

Different C diagonals read at different offsets within the same chunk. Thread 0 for ki=1 may read data loaded by thread 1 -- genuine cross-thread shared memory reuse.

### 6.3 Execution Flow (per tile)

**Phase 1 -- Preload A (double-buffered, float4 vectorised):**

```
for each contributing A diagonal:
    for j = tid; j < chunk/4; j += BLOCK:
        load float4 from A_vals -> smem buf[cur]
cp.async commit + wait
__syncthreads()
```

Next tile's A is loaded asynchronously into `buf[1-cur]` while current tile computes.

**Phase 2 -- Accumulate (sync-free, B in registers):**

```
Precompute: active[ki] = (ki < c_count) && (p_begin + tid < c_len[ki])

for each A diagonal (groups of 4 for ILP):
    Read 4 A values from smem into registers (av_g[4][K])
    for each C diagonal ki:
        if (!active[ki]) continue
        for each of 4 A diags:
            bi = B_lookup[c_offset[ki] - d_a + b_lookup_base]  // no bounds check
            if (bi < 0) continue                                // sparsity
            b_val = B_vals[clamp(b_pos)] * mask                 // no bounds branch
            acc[ki] += a_val * b_val
```

**Phase 3 -- Write (template-specialised):**

```
if constexpr (DIRECT):   // corner -- compiled out for heavy kernel
    C_vals[c_start[ki] + p_begin + tid] = acc[ki]
else:                     // heavy -- compiled out for corner kernel
    partial_buf[out_offset + ki*padded + p_begin + tid] = acc[ki]
```

### 6.4 Reduction Kernel (heavy groups only)

Persistent grid-stride, no shared memory. Each thread independently sums its position across all partitions:

```
for each reduce task:
    for each tile position:
        for each C diagonal ki:
            sum = 0
            for each partition:
                sum += partial_buf[...]
            C_vals[...] = sum
```

### 6.5 Launch Structure

```
launch_hybrid():
    1. hybrid_compute_kernel<true>   -- corner tasks (small smem, more L1)
    2. hybrid_compute_kernel<false>  -- heavy tasks  (large smem)
    3. hybrid_reduce_kernel          -- sum partials -> C_vals
```

All on the same stream. FIFO ordering guarantees heavy finishes before reduce.

---

## 7. Optimisations

### Implemented

| Optimisation | Effect |
|---|---|
| **Double-buffered A preload (cp.async)** | Next tile's A loads overlap with current tile's computation. Hides A load latency entirely. |
| **B load ILP (A_UNROLL=4)** | 4 A diagonals processed simultaneously. Multiple B loads in flight, saturating memory pipeline. |
| **Vectorised A loads (float4)** | 128-bit loads reduce Phase 1 instruction count 4x. Chunk padded to 4-alignment. |
| **Extended B_lookup (4N-3)** | Covers all possible d_b values. Eliminates b_idx bounds check branch. |
| **Active mask** | Precomputed per tile. Merges c_count and c_len checks into one bool. |
| **Clamp + predicate for b_pos** | Clamps b_pos to valid range, multiplies by 0/1 mask. Eliminates bounds branch. |
| **Template-split kernel** | `<true>` for corner, `<false>` for heavy. Phase 3 branch resolved at compile time. Corner launch gets smaller smem = more L1. |

### Branch count in inner loop

Only **one data-dependent branch**: `if (bi < 0) continue` (fundamental sparsity check). Everything else is either precomputed, predicated, or eliminated by design.

### Future (recorded in kernel source)

| Optimisation | Description |
|---|---|
| **L2 cache residency** | Pin B_vals in L2 with `cudaAccessPropertyPersisting`. Beneficial when B fits in L2 (50 MB on H100). |
| **Thread block clusters** | Hopper distributed shared memory (dsmem). CTAs in a cluster share A data across groups, eliminating redundant loads at group boundaries. |
| **Warp shuffle fast path** | When spread=0 (all positive C diags), bypass smem entirely -- A lives in a register. |

---

## 8. Memory Access Analysis

### Global Memory

| Operation | Pattern | Notes |
|---|---|---|
| A preload (Phase 1) | **coalesced float4** | 128 threads load chunk/4 float4s per A diagonal |
| B load (Phase 2) | **coalesced** per C diagonal | One load per (A diag, C diag) pair. 4 outstanding via ILP unroll. |
| C write (Phase 3) | **coalesced** | 128 threads write 128 consecutive floats per C diagonal |
| B_lookup | broadcast | Same value for all threads in warp (d_b depends only on ki, d_a) |

### Shared Memory Bank Conflicts

```
Thread tid reads smem_A[slot * chunk + offset + tid]
  where offset = c_sr[ki] - min_c_sr (0..7)

For a given (slot, ki): consecutive threads read consecutive addresses.
  bank(tid) = (slot*chunk + offset + tid) % 32
  adjacent threads -> adjacent banks -> zero bank conflicts
```

### Register Budget (per thread, H100 @ 4 blocks/SM = 128 regs available)

| Usage | Registers |
|---|---|
| c_offset[8], c_sr[8], c_len[8], c_start[8] | 32 |
| acc[8] | 8 |
| active[8] | 8 (bools, may be predicate regs) |
| av_g[4][8] (A unroll temps) | 32 |
| da_g[4], loop vars, addresses | ~20 |
| **Total estimate** | **~100** |

Fits within the 128-register limit with headroom. K=8 is the practical maximum.

### Shared Memory Budget (H100: 228 KB/SM, 4 blocks/SM = 57 KB/block)

```
Double-buffered: 2 * a_count * chunk * 4 bytes
chunk = TILE + spread (padded to 4) = 128..136

Corner: a_count <= 52 A diags per partition (28.5 KB per buffer)
Heavy:  partitioned to fit the same budget
```

---

## 9. File Structure

```
DiaGPU/
+-- diag_types.cuh              Core types: DiagMatrix, OutputDiagInfo
+-- diag_host_preprocess.cuh    Host preprocessing utilities
|                                 build_output_diagonals()
|                                 build_contributors()
|                                 build_b_diag_lookup()
|                                 sort_diag_matrix_by_offset()
|
+-- diag_hybrid_kernel.cuh      Unified kernel header
|                                 Constants (H100 tuned)
|                                 HybridCDiag, HybridTask, HybridReduceTask
|                                 HybridKernelArgs, HybridPlan
|                                 Kernel declarations (template<bool DIRECT>)
|                                 launch_hybrid() declaration
|
+-- diag_hybrid_kernel.cu       Kernel implementation
|                                 hybrid_compute_kernel<true>   (corner)
|                                 hybrid_compute_kernel<false>  (heavy)
|                                 hybrid_reduce_kernel
|                                 build_hybrid_plan()
|                                 launch_hybrid()
|
+-- test_hybrid.cu              Test + benchmark driver
|                                 CPU reference
|                                 Correctness validation
|                                 Timing (CUDA events)
|                                 Comparison vs paper_hm_kernel
|
+-- paper_hm_kernel.cu/.cuh     Baseline: atomicAdd-based kernel
```

### Compile

```
nvcc test_hybrid.cu diag_hybrid_kernel.cu paper_hm_kernel.cu \
     -o test_hybrid -std=c++17 -O3 -arch=sm_90
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

# Run
./test_hybrid
# Results written to results.md

# Profile with Nsight Compute
ncu --profile-from-start off ./test_hybrid
```
