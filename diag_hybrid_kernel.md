# diag_hybrid_kernel — Design & Implementation Notes

Diagonal sparse matrix multiplication: A × B = C, where every matrix is stored as a flat array of diagonals. Implemented in `diag_hybrid_kernel.cuh` (types + declarations) and `diag_hybrid_kernel.cu` (implementation).

---

## Core Math

For output diagonal `dC`, every element at position `k` is:

```
C[dC][k] = Σ  A[dA][k_a] * B[dB][k_b]
           dA+dB=dC
```

The set of valid (dA, dB) pairs for a given dC is called the **contributor list**. Its size drives the kernel path selection.

Index arithmetic mapping position `k` in the output tile to flat array offsets:

```
c_sr  = max(0, -dC)           // row where diagonal dC starts
a_sr  = max(0, -dA)
a_pos = c_sr - a_sr + k       // index into A's flat value array

b_sr  = max(0, -dB)
b_pos = c_sr + dA - b_sr + k  // index into B's flat value array
```

Both `a_pos` and `b_pos` are linear in `k`, so thread `tid` (= offset within the tile, i.e. `k - p_begin`) maps to exactly one element of A and one element of B for each pair. This one-to-one mapping is the foundation of the tiling strategy.

---

## Two-Level Structure: Host Plan + GPU Kernels

### 1. Host Side — `build_hybrid_plan()`

Runs entirely on the CPU before any GPU work. The output is a `HybridPlan` struct containing all task tables and the global pair list. The GPU kernels are pure consumers of this plan; they perform no dynamic scheduling or metadata construction.

**Steps:**

1. **Enumerate output diagonals** — for every (ai, bi) pair, compute `dC = dA + dB`. Collect unique dC values and compute their lengths.
2. **Build contributor lists** — for each dC, record every (ai, bi) with `dA + dB = dC` and compute the valid global position range `[valid_begin, valid_end)` within the dC diagonal where that pair actually contributes (respects A and B diagonal lengths).
3. **Route corner vs heavy** — if `num_contributors <= HYBRID_CORNER_THRESH (16)`, the diagonal is a corner diagonal; otherwise heavy.
4. **Tile and emit tasks** — slice each diagonal into segments of length `HYBRID_TILE_CORNER (128)` or `HYBRID_TILE_HEAVY (256)`, emit the corresponding task structs, and assign partial buffer offsets.

For corner diagonals, the valid A-diagonal range `[ai_begin, ai_end)` is precomputed on the host via binary search on the sorted `A.offsets` array using:
```
d_a_lo = dC - B_offset_max
d_a_hi = dC - B_offset_min
```
This range is baked directly into each `HybridCornerTask`, eliminating on-device binary search entirely.

For heavy diagonals, the full contributor list is appended to a **global flat pairs array** (`HybridPair[] pairs`). Each s1 task records a `pair_begin` index and `pair_count` into this array. The s2 task for the same segment is emitted first and its index is recorded in each s1 task as `s2_task_idx` — needed by the fused pipelined kernel to know which pending counter to decrement.

---

### 2. GPU Kernels

Four kernels total. The first three are launched together as `launch_hybrid` (3 sequential persistent kernels). The fourth (`hybrid_heavy_fused_kernel`) replaces the last two as `launch_hybrid_pipelined` and enables per-segment pipelining of s1 and s2 within a single kernel launch.

---

#### `hybrid_corner_kernel` — persistent, 128 threads/block

**Ownership:** one block = one or more `HybridCornerTask`s (grid-stride loop).
**Tile:** 128 output elements per task, one thread per element.

**Shared memory: 1 KB per block**

```
__shared__ float smem_A[128];   // 512 bytes — staged A diagonal tile
__shared__ float smem_B[128];   // 512 bytes — staged B diagonal tile
```

**Inner loop (one iteration per contributor pair):**

```
for ai in [task.ai_begin, task.ai_end):
    dB = dC - dA
    bi = B_lookup[dB + n-1]           // O(1) hash table lookup, no divergence
    if bi < 0: continue               // B doesn't have this diagonal

    // --- load phase ---
    smem_A[tid] = A_vals[A_starts[ai] + a_pos]   // guarded by bounds check
    smem_B[tid] = B_vals[B_starts[bi] + b_pos]
    __syncthreads()                               // barrier: all loads complete

    // --- compute phase ---
    if active: acc += smem_A[tid] * smem_B[tid]  // fused multiply-add
    __syncthreads()                               // barrier: smem safe to overwrite

C_vals[values_start + p_begin + tid] = acc        // single direct write
```

**Why shared memory here:**
The global loads for each pair are coalesced (128 threads loading 128 consecutive floats = one 512-byte transaction per warp). Routing them through shared memory gives a well-defined synchronization point that also serves as a write barrier for the next iteration, and keeps all threads in lockstep for the `__syncthreads` calls — which would otherwise be unsafe if any thread exited early. The `active` flag (not an early `return`) enforces this.

**Latency hiding in the corner kernel:**
The L2 cache typically services 128-float loads in ~30 cycles on modern GPUs. The fused MAC that follows (`smem_A[tid] * smem_B[tid]`) is only ~4 cycles. There is no explicit software pipelining here: the kernel pays the full global memory latency on every pair iteration. This is acceptable for corner diagonals because the contributor count is small (≤ 16), so the total number of load-barrier-compute cycles is bounded. For high-pair-count diagonals the heavy path is used instead.

---

#### `hybrid_heavy_s1_kernel` — persistent, 256 threads/block

**Ownership:** one block = one or more `HybridS1Task`s (grid-stride loop).
**Tile:** 256 output elements per task, one thread per element.
**Pair batch:** up to 8 pairs per task (`HYBRID_PAIRS_PER_PART`).

**Shared memory: 4 KB per block**

```
__shared__ float smem_A[256];   // 1024 bytes — current A tile
__shared__ float smem_B[256];   // 1024 bytes — current B tile
```

With a 256-thread block and 4 KB of shared memory, and targeting 4 blocks per SM, this occupies 16 KB of the 48 KB shared memory bank per SM — leaving the remaining 32 KB for L1 cache of global memory reads (set via `cudaSharedmemCarveoutMaxShared`).

**Inner loop (one iteration per pair in the partition):**

```
for pi in [0, task.pair_count):          // at most HYBRID_PAIRS_PER_PART = 8
    ai, bi = pairs[task.pair_begin + pi]
    dA = A_offsets[ai],  dB = B_offsets[bi]

    // --- load phase ---
    smem_A[tid] = A_vals[A_starts[ai] + a_pos]   // 256-wide coalesced load
    smem_B[tid] = B_vals[B_starts[bi] + b_pos]
    __syncthreads()

    // --- compute phase ---
    if active: acc += smem_A[tid] * smem_B[tid]
    __syncthreads()

partial_buf[task.partial_offset + tid] = acc       // exclusive slot, no atomics
```

**Memory traffic per task:**
Each pair loads 256 A floats + 256 B floats = 2 KB from global memory, processes them into 256 MACs, then the next pair's loads start. With 8 pairs, the task performs 16 KB of global reads and 256 × 8 = 2048 MACs. The arithmetic intensity is 2048 MACs / 16 KB ≈ 0.125 FLOP/byte — firmly memory-bound. The shared memory staging ensures all loads are aligned 512-byte transactions regardless of the a_pos/b_pos base offset.

**Latency hiding in the s1 kernel:**
The two `__syncthreads()` per pair iteration (one after loads, one after compute) serialize the pipeline: no pair's loads are issued until the previous pair's MACs are done. This gives:

```
timeline per pair:
  [load A+B: ~30 cy][sync][mac: ~4 cy][sync]
  total ~34 cycles per pair, heavily load-bound
```

The future path to hide this latency is `cp.async` (Ampere+): replace the shared memory stores with async copies, advance to the next pair's loads while the current pair's MACs execute. This would give:

```
timeline with cp.async double-buffering:
  [load A+B pair i+1 (async)]
                              [mac pair i]
  overlap → ~30 cy saved per pair
```

The current code has a comment marking this upgrade point. The smem is sized for single-buffering; double-buffering would require `smem_A[2][256]` and `smem_B[2][256]` (8 KB total), which is still within budget at 4 blocks per SM (32 KB total vs 48 KB available).

---

#### `hybrid_heavy_s2_reduce_kernel` — persistent, 256 threads/block

**Ownership:** one block = one or more `HybridS2Task`s (grid-stride loop).
**Tile:** up to 256 output elements per task.
**Shared memory: 0 bytes** (no `__syncthreads` needed).

**Inner loop:**

```
sum = 0.0
for p in [0, task.num_partials):
    sum += partial_buf[task.partial_offset + p * 256 + tid]

C_vals[values_start + p_begin + tid] = sum
```

Each thread reduces its own scalar position `tid` across `num_partials` slots. Threads are completely independent — no data sharing between them. The reads from `partial_buf` are coalesced: thread `tid` reads element `tid` from each partial row, so 256 threads collectively issue one 1 KB transaction per partial row.

**Why no shared memory:**
The reduction is scalar per thread. There is no cross-thread communication. Using shared memory here would add unnecessary synchronization and reduce occupancy for no gain.

**Latency hiding in s2:**
The loop over `num_partials` is a sequential scalar reduction per thread. The compiler typically unrolls it (`#pragma unroll 4`) and pipelines the loads from `partial_buf`, since successive iterations access independent memory locations. On a 256-element partial_buf row, consecutive threads access consecutive addresses, so the hardware prefetcher can pipeline the loads across the partial rows automatically.

---

#### `hybrid_heavy_fused_kernel` — pipelined s1 + s2, single launch

**Block size:** 256 threads (same as s1).
**Shared memory:** same 4 KB as s1 kernel — `smem_A[256]` + `smem_B[256]`.
**Ownership:** dynamic, via device-side atomic counters in `ctrl[]`.

This kernel replaces both `hybrid_heavy_s1_kernel` and `hybrid_heavy_s2_reduce_kernel` with a single persistent kernel that pipelines s1 and s2 at per-segment granularity.

**Control array layout (`ctrl[]`, device memory):**

```
ctrl[0]           : s1_next     — atomic counter, next s1 task index to claim
ctrl[1]           : s2_claim    — atomic counter, next s2 task index to claim
ctrl[2 + i]       : pending[i]  — countdown for s2 task i (init = num_partitions[i])
ctrl[2 + n_s2 + i]: num_partitions[i]  — source for re-initialisation
```

Total size: `(2 + 2 * n_s2) * sizeof(int)`.

**CTA life-cycle:**

```
while (s2_processed < n_s2):

    // Phase A: claim and execute one s1 task
    if not s1_done:
        s1_idx = atomicAdd(ctrl[s1_next], 1)   // thread 0 only
        if s1_idx < n_s1:
            ... compute partial sum (same inner loop as s1 kernel) ...
            __threadfence()                     // ensure partial_buf writes visible globally
            atomicSub(pending[s2_task_idx], 1)  // signal one partition complete
        else:
            s1_done = true

    // Phase B: claim and execute one s2 task
    s2_idx = atomicAdd(ctrl[s2_claim], 1)       // thread 0 only
    if s2_idx < n_s2:
        // spin until all partitions for this segment are committed
        while atomicAdd(pending[s2_idx], 0) != 0:
            __nanosleep(64)                     // yield SM briefly (Volta+)
        ... reduce partial_buf → C_vals ...
        s2_processed++
    else:
        atomicSub(ctrl[s2_claim], 1)            // undo overflow, exit
        break
```

**Why `__threadfence()` before `atomicSub`:**
CUDA's memory model does not guarantee that writes to `partial_buf` by one CTA are visible to another CTA without an explicit fence. `__threadfence()` flushes all pending writes to L2/DRAM, ensuring the s2 spin-wait sees the completed partial sums — not stale cache lines. Without this fence, the s2 CTA might read zeroes from `partial_buf` even though `pending[i]` reached zero.

**Pipelining timeline:**

```
Without fused kernel (sequential launches):
  [====== ALL s1 tasks ======][=== ALL s2 tasks ===]
                               ^^^
                               entire s2 phase delayed until all s1 done

With fused kernel (per-segment overlap):
  CTA 0: [s1 seg0,p0][spin seg0 ][s2 seg0][s1 seg3,p0]...
  CTA 1: [s1 seg0,p1][s1 seg1,p0][spin seg1][s2 seg1]...
  CTA 2: [s1 seg1,p1][s1 seg2,p0][s1 seg2,p1][spin seg2][s2 seg2]...
                                   ^^^^^^^^^
                                   seg0 pending hits 0 → CTA 0 unblocks
```

For a workload with many heavy segments, the s2 reduction for early segments overlaps with s1 computation for later segments. The latency of s2 is hidden inside the much longer s1 phase.

**Quantifying the benefit:**
Let `T_s1` = total s1 time, `T_s2` = total s2 time. Without the fused kernel the total heavy kernel time is `T_s1 + T_s2`. With the fused kernel, s2 for segment 0 can start as soon as its partitions finish — in the best case overlapping all of `T_s2` with the tail of `T_s1`. The theoretical saved time is `min(T_s2, T_s1)`, which in practice is `T_s2` since s2 is much lighter than s1. For workloads where s2 is 10–20% of total time, this directly translates to 10–20% speedup on the heavy path.

**Deadlock proof:**
A CTA only enters Phase B's spin-wait after completing Phase A — which either processed one s1 task (reducing the unclaimed s1 pool by one) or discovered that s1 is exhausted. In the first case, at least one CTA made forward progress on s1. In the second case, all s1 tasks have already been claimed; since every claimed s1 task eventually finishes (it's compute-bound with no blocking), every `pending[i]` counter eventually reaches zero. Therefore every spinning s2 CTA unblocks in finite time.

The grid size `min(SMs × 4, n_s1 + n_s2)` ensures no CTA is permanently starved: the pool of tasks (n_s1 + n_s2) is always ≥ the number of CTAs, so every CTA finds work.

---

### 3. Persistent Grid Sizing

All kernels use fixed grids, not one-block-per-task. Launch overhead is O(1) regardless of task count.

| Kernel | Grid size | Block | Smem/block | Target occupancy |
|---|---|---|---|---|
| corner | `min(SMs×8, n_corner)` | 128 | 1 KB | 8 blocks/SM |
| s1 | `min(SMs×4, n_s1)` | 256 | 4 KB | 4 blocks/SM |
| s2 | `min(SMs×4, n_s2)` | 256 | 0 KB | 4 blocks/SM |
| fused | `min(SMs×4, n_s1+n_s2)` | 256 | 4 KB | 4 blocks/SM |

**Why grid-stride instead of one-block-per-task:**
A naive one-block-per-task launch where `n_tasks = 10 000` means 10 000 blocks must be scheduled by the hardware. On a 108-SM GPU with 4 resident blocks per SM, only 432 blocks run concurrently; the other 9 568 sit in the scheduler queue. Each scheduler step has overhead (~few microseconds on the driver side). With a persistent grid of 432 blocks each processing `ceil(10000 / 432) ≈ 24` tasks via the grid-stride loop, the scheduler overhead is incurred once and the 24-task inner loop runs at full throughput with no re-scheduling pauses between tasks.

---

### 4. Shared Memory Budgets

Each SM on a modern NVIDIA GPU has 48–228 KB of configurable L1/shared memory (architecture-dependent). The `cudaFuncSetAttribute` call with `cudaFuncAttributePreferredSharedMemoryCarveout` controls the split.

| Kernel | Smem request | Carveout setting | Effective L1 |
|---|---|---|---|
| corner | 1 KB | `MaxL1` (prefer L1) | ~32–48 KB L1 |
| s1 / fused | 4 KB | `MaxShared` (prefer smem) | ~16–32 KB L1 |
| s2 | 0 KB | `MaxL1` | ~32–48 KB L1 |

**Corner kernel** sets `MaxL1` because its shared memory (1 KB) is tiny and the L1 cache is more valuable for the `B_lookup[]` array (size `2n-1` ints, ~8 KB for n=1024) and the task metadata that all threads broadcast from constant memory.

**s1/fused kernel** sets `MaxShared` because the 4 KB smem is actively used every iteration, and the global loads from `A_vals`/`B_vals` are sequential and rarely reused across tasks (so L1 cache hit rate is low anyway — more L1 doesn't help here).

**s2 kernel** uses no smem and sets `MaxL1` to help cache `partial_buf` reads, which are reused when multiple s2 tasks access the same L2 cache lines.

---

### 5. Memory Layout

#### Output buffer `C_vals[]`

Flat array of floats. Each output diagonal `cd` occupies a contiguous slice:

```
C_vals[cd.values_start + k]   for k in [0, cd.length)
```

The host preprocessor assigns `values_start` in diagonal-offset order. There are no holes; total size is `plan.total_c_values` floats.

#### Partial buffer `partial_buf[]`

Used only by heavy path. For each heavy (c_diag, segment) there are `num_partitions` slots of `HYBRID_TILE_HEAVY` floats each:

```
For segment s of heavy diagonal dC:
  partial_buf[seg_base + p * HYBRID_TILE_HEAVY + tid]
    = partial sum contributed by partition p, thread tid

  seg_base is assigned sequentially by build_hybrid_plan()
```

Reading pattern for s2:

```
thread tid reads:
  partial_buf[partial_offset + 0 * 256 + tid]
  partial_buf[partial_offset + 1 * 256 + tid]
  ...
  partial_buf[partial_offset + (num_partials-1) * 256 + tid]
```

Consecutive threads access consecutive addresses within each row → fully coalesced 1 KB reads per partial row.

#### Control array `ctrl[]` (fused kernel)

```
Offset  Content
──────  ───────────────────────────────────────────────
0       s1_next          (atomicAdd counter, init 0)
1       s2_claim         (atomicAdd counter, init 0)
2..     pending[i]       (countdown, init = num_partitions[i])
2+n_s2  num_partitions[i] (source for re-init, written once by host)
```

`init_fused_ctrl()` copies `num_partitions[]` → `pending[]` and zeros `s1_next` and `s2_claim` before each launch, allowing the ctrl buffer to be reused across repeated calls without a new `cudaMalloc`.

---

## Key Design Constants

| Constant | Value | Meaning |
|---|---|---|
| `HYBRID_TILE_CORNER` | 128 | Segment length (= block size) for corner kernel |
| `HYBRID_BLOCK_CORNER` | 128 | 4 warps, 1 thread per output element |
| `HYBRID_TILE_HEAVY` | 256 | Segment length (= block size) for heavy kernels |
| `HYBRID_BLOCK_HEAVY_S1` | 256 | 8 warps, 1 thread per output element |
| `HYBRID_BLOCK_HEAVY_S2` | 256 | 8 warps, 1 thread per output element |
| `HYBRID_PAIRS_PER_PART` | 8 | Pairs per s1 partition; controls partial buffer size |
| `HYBRID_CORNER_THRESH` | 16 | Contributor count above which heavy path is used |

**Choosing `HYBRID_PAIRS_PER_PART`:**
Larger values mean fewer s1 tasks (less scheduling overhead) but each task runs longer, reducing the granularity at which s2 can start in the fused kernel. Smaller values increase s1 task count and give finer-grained pipelining at the cost of more `atomicSub` calls and more `partial_buf` slots. Value 8 is a balance: at 8 pairs × 256 elements × 4 bytes = 8 KB of A reads + 8 KB of B reads per task, each task performs ~32–50 µs of work — enough to amortize scheduling overhead but not so much that the fused pipeline stalls waiting for it.

---

## Data Structures

### Host-side plan (`HybridPlan`)

Produced by `build_hybrid_plan()` before any GPU work.

| Field | Type | Description |
|---|---|---|
| `c_diags` | `vector<HybridCDiag>` | Metadata for each output diagonal |
| `corner_tasks` | `vector<HybridCornerTask>` | One entry per (corner diagonal, segment) |
| `s1_tasks` | `vector<HybridS1Task>` | One entry per (heavy segment, pair partition) |
| `s2_tasks` | `vector<HybridS2Task>` | One entry per (heavy segment) |
| `pairs` | `vector<HybridPair>` | Global flat list of (ai, bi) pairs for heavy diagonals |
| `total_c_values` | `int` | Total floats needed for `C_vals` |
| `partial_buf_size` | `int` | Total floats needed for `partial_buf` |

### Device task structs

**`HybridCornerTask`**
```cpp
int c_idx;      // index into c_diags[]
int c_offset;   // dC
int p_begin;    // segment start position along the diagonal
int p_len;      // segment length (<= HYBRID_TILE_CORNER = 128)
int ai_begin;   // valid A-diagonal index range, precomputed on host
int ai_end;     //   eliminates on-device binary search
```

**`HybridS1Task`**
```cpp
int c_idx;
int c_offset;
int p_begin;
int p_len;          // <= HYBRID_TILE_HEAVY = 256
int pair_begin;     // start index into global pairs[]
int pair_count;     // <= HYBRID_PAIRS_PER_PART = 8
int partial_offset; // element offset into partial_buf[] for this slot
int s2_task_idx;    // index of the corresponding s2 task (for fused kernel)
```

**`HybridS2Task`**
```cpp
int c_idx;
int p_begin;
int p_len;
int partial_offset; // start of partial slots for this (c_diag, segment)
int num_partials;   // number of slots to reduce (= ceil(num_pairs / PAIRS_PER_PART))
```

---

## Launch Paths

### `launch_hybrid(args, stream)` — 3 sequential persistent kernels

```
stream: [corner_kernel][─── s1_kernel ───][── s2_kernel ──]
                                           ^^^
                                           full global barrier:
                                           every s1 block must retire
                                           before any s2 block starts
```

Correct and simple. s2 has zero overlap with s1. The stream FIFO enforces the ordering without any explicit event.

### `launch_hybrid_pipelined(args, ctrl, stream)` — 2 kernel launches

```
stream: [corner_kernel][──────── fused_kernel ────────]
                         CTA 0: [s1][s1][s2][s1][s2]...
                         CTA 1: [s1][s2][s1][s1][s2]...
                         CTA 2: [s1][s1][s1][s2][s2]...
                                      ^^^
                                      s2 for segment S starts as soon as
                                      pending[S] hits 0, while other CTAs
                                      continue s1 work for later segments
```

s2 for segment S begins as soon as all s1 partitions for S write `__threadfence()` and decrement `pending[S]` to zero — without waiting for s1 work on any other segment. Requires a `ctrl` device allocation of `(2 + 2*n_s2) * sizeof(int)`, initialised by `init_fused_ctrl()` before each call.

---

## Future Optimisations

### cp.async double-buffering in s1 (Ampere+)

Replace the blocking `smem_A[tid] = A_vals[...]` stores with asynchronous copies using CUDA's `cp.async` instruction. This allows the next pair's A and B loads to issue while the current pair's MACs execute:

```
// Current (blocking):
smem_A[tid] = A_vals[...];
smem_B[tid] = B_vals[...];
__syncthreads();
acc += smem_A[tid] * smem_B[tid];
__syncthreads();

// With cp.async (non-blocking):
// Issue async load for pair i+1 into alternate smem buffer
cuda::memcpy_async(smem_A[nxt], A_vals[...], cuda::aligned_size_t<16>{});
cuda::memcpy_async(smem_B[nxt], B_vals[...], cuda::aligned_size_t<16>{});
// Compute with current buffer (no sync needed until we read nxt)
acc += smem_A[cur][tid] * smem_B[cur][tid];
// Commit async loads before next iteration reads them
cuda::pipeline::arrive_and_wait();
swap(cur, nxt);
```

This would require doubling the smem to `smem_A[2][256]` + `smem_B[2][256]` = 8 KB per block. At 4 blocks per SM that is 32 KB — still within the 48 KB budget. Expected benefit: hide ~30-cycle global memory latency behind ~4-cycle MAC computation, recovering most of the latency cost per pair.

### Warp-level reduction in s2

When `num_partials` is small (≤ 32), the reduction in s2 can be parallelised within a warp using `__shfl_down_sync` instead of a scalar per-thread loop. For large `num_partials` the current scalar loop is already efficient (sequential loads, hardware prefetcher friendly); this optimisation targets low-`num_partials` cases.
