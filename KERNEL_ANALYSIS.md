# Diagonal Sparse Matrix Multiplication: Kernel Design Analysis

## Platform
- **GPU**: NVIDIA H100 NVL (132 SMs, 1785 MHz, 50 MB L2, 3.9 TB/s HBM3)
- **Problem**: C = A × B where A, B are n×n matrices with nonzeros confined to specific diagonals
- **Diagonal storage**: symmetric offsets {-hb, ..., 0, ..., +hb}, same for A and B
- **Baseline**: cuSPARSE SpGEMM (CSR format)

---

## Kernels Tested

### 1. Paper HM (Haque et al. 2024, Algorithm 2)

**Design**: One thread per nonzero of A. Each thread holds its A value in a register, iterates all B diagonals, and uses `atomicAdd` to write partial products to C.

```
Thread i:
    a_val = A_values[i]                  // register, loaded ONCE
    (row, k) = coordinates of A element i
    for each B diagonal d_b:
        p_b = k - b_sr(d_b)
        if valid:
            b_val = B_values[B_starts[bi] + p_b]   // 1 global read
            atomicAdd(&C[row][k+d_b], a_val * b_val) // 1 atomic
```

**Per-FMA cost**: 1 global read (B) + 1 atomic write (C) + 0 A reads (register)

**Strengths**:
- Optimal A amortization: each A value loaded once, reused across all B_num_diags iterations
- Uniform work per thread (always B_num_diags iterations)
- Simple kernel, no preprocessing needed
- Memory traffic: nzA × 4B (A) + nzA × B_num_diags × 4B (B) = minimal

**Weaknesses**:
- Uses atomicAdd (H100 L2 hardware makes this fast, but still ~4-8 cycles per atomic)
- At high diagonal counts, each C element receives ~(2hb+1) atomic updates (contention)

**Benchmark results** (kernel-only, 100-iteration average):

| n | diags | HM (ms) |
|---|---|---|
| 1024 | 41 | 0.0256 |
| 4096 | 41 | 0.0271 |
| 8192 | 101 | 0.2083 |
| 16384 | 201 | 1.4562 |

---

### 2. DiagSpMM v1 — C-Stationary Unified (Zero Metadata)

**Design**: Each warp owns a 32-element tile on one output diagonal. Iterates matching A diagonals, loading both A and B per iteration. Binary search for A range. Grid-stride persistent kernel.

```
Warp (d_c, p_begin):
    compute_a_range(d_c) via binary search
    for each A diagonal ai in [ai_begin, ai_end):
        a_val = A_values[A_starts[ai] + p_a]     // 1 global read (L1/L2)
        b_val = B_values[B_starts[bi] + p_b]     // 1 global read
        acc += a_val * b_val                      // register FMA
    C_values[...] = acc                           // 1 direct write
```

**Per-FMA cost**: 2 global reads (A from L1/L2 + B) + 0 atomics

**Strengths**:
- Zero atomics (exclusive tile ownership)
- Zero metadata arrays (no Group, no PairMeta — eliminated during optimization)
- Register accumulation (1 cycle per FMA)
- Coalesced A, B reads and C writes
- Binary search narrows A range to only matching diagonals (eliminates ~50% wasted iterations)

**Weaknesses**:
- **2x memory traffic vs HM**: Each A value is a fresh global/L1 read every iteration because a different A diagonal is needed each time. HM keeps A in a register.
- Binary search overhead per warp (~64 instructions for 2 searches)
- Task setup overhead in grid-stride loop (load Task, compute d_c, c_sr)
- Edge output diagonals have few matching A diags but still occupy a full warp

**Root cause of 1.5-1.8x gap vs HM**:

Total 128-byte memory transactions are nearly identical:
- DiagSpMM: sum over d_c of 2 × matches(d_c) × tiles = **80,802** per row-group
- Paper HM: A_num_diags × (1 + 2 × B_num_diags) = **81,003** per row-group

But DiagSpMM's inner loop does **2 global reads per FMA** (A + B) while HM does **1 read + 1 atomic** (B + C). On H100, one global L1/L2 read costs ~30-200 cycles; one atomic costs ~4-8 cycles. So DiagSpMM pays ~230-400 cycles/FMA vs HM's ~204-208 cycles/FMA.

**Benchmark results**:

| n | diags | DiagSpMM (ms) | vs HM |
|---|---|---|---|
| 1024 | 41 | 0.0245 | 0.96x (faster) |
| 4096 | 41 | 0.0441 | 1.63x slower |
| 8192 | 101 | 0.3483 | 1.67x slower |
| 16384 | 201 | 2.5675 | 1.76x slower |

At small scale (n≤1024), launch overhead dominates and DiagSpMM matches HM.
At large scale, the 2x A-load overhead dominates.

---

### 3. Row-Tiled v1 — Shared Memory Accumulation

**Design**: Each CTA owns R consecutive rows. Pre-load A values into shared memory. Iterate output diagonals in chunks. For each chunk, iterate A/B pairs and accumulate into shared memory: `smem[d_c_local][tid] += a_val * b_val`.

```
Phase 1: Load A into smem (once)
Phase 2: For each output diagonal chunk:
    Zero smem accumulators
    for each A diagonal:
        for each B diagonal in chunk:
            smem[d_c_local][tid] += a_val * b_val   // read-modify-write smem
    Write back smem to C
```

**Per-FMA cost**: 1 smem read (A, ~5 cyc) + 1 global read (B) + 1 smem read-modify-write (accumulator, ~20-30 cyc)

**Strengths**:
- Zero atomics
- A loaded once into shared memory (optimal amortization)

**Weaknesses**:
- **Shared memory read-modify-write for accumulation is 20-30x slower than register FMA**. This is the killer. Every FMA goes through smem instead of registers.
- 3 `__syncthreads()` per chunk (zero + compute + writeback)
- Low occupancy for large diagonal counts (smem = num_out_diags × R × 4 bytes)
- Much worse than both HM and DiagSpMM v1

**Benchmark results**:

| n | diags | RowTiled (ms) | vs HM |
|---|---|---|---|
| 1024 | 41 | 0.4937 | 19.3x slower |
| 4096 | 41 | 0.5113 | 18.9x slower |
| 8192 | 101 | 4.0279 | 19.4x slower |
| 16384 | 201 | 16.8131 | 11.5x slower |

**Conclusion**: Using shared memory for accumulation is fundamentally wrong for this problem. Shared memory should only hold read-only data.

---

### 4. Row-Tiled v2 — Shared Memory A Cache + Register Accumulation

**Design**: Fix the v1 mistake. Shared memory holds A values (read-only, loaded once). Accumulation stays in registers. Iterate output diagonals with a sliding window (zero binary search).

```
Phase 1: Load ALL A values into smem (once)
Phase 2: Sliding window over output diags:
    for each d_c:
        acc = 0  (register)
        for each matching A diagonal:
            a_val = smemA[ai][tid]           // ~5 cyc smem read (read-only)
            b_val = B_values[...]            // 1 global read
            acc += a_val * b_val             // register FMA
        C_values[...] = acc                  // direct write
```

**Per-FMA cost**: 1 smem read (A, ~5 cyc) + 1 global read (B) + 1 register FMA

**Strengths**:
- Zero atomics
- A in shared memory (loaded once, ~5 cycle reads vs ~30-200 for global/L1)
- Register accumulation (fast)
- Sliding window: zero binary search, O(1) amortized per d_c

**Weaknesses**:
- **Still iterates all output diagonals per row**: for hb=100, 401 output diags × ~100 A diags each = 40,401 iterations per row position. The outer d_c loop creates massive iteration count.
- Smem read (5 cycles) is better than global (30-200) but worse than register (0)
- Low occupancy for large A_num_diags (smem = A_num_diags × R × 4 bytes)
- The outer d_c loop has per-d_c overhead (sliding window advance, C write-back)

**Why still slower**: The loop structure — outer over d_c, inner over matching A diags — means each A smem value is read once per d_c that uses it. For the center output diagonal (d_c=0), ALL A diags are read. For edge diags, few. Total A reads from smem = sum of matches = 40,401 per row. Same total work as DiagSpMM v1, just from smem instead of L1. The 5-cycle smem read vs ~30-cycle L1 read helps, but the iteration structure overhead (d_c loop, sliding window bookkeeping, C write-back per d_c) overwhelms the gain.

**Benchmark results**:

| n | diags | RowTiled v2 (ms) | vs HM |
|---|---|---|---|
| 1024 | 41 | 0.4937 | 19.3x slower |
| 8192 | 101 | 3.9675 | 19.1x slower |
| 16384 | 201 | 16.8131 | 11.5x slower |

**Conclusion**: The outer d_c loop structure kills performance regardless of where A is cached. The total iteration count is the same; only the per-iteration latency changes.

---

### 5. Batched DiagSpMM — Output Diagonal Batching (K=32)

**Design**: Process K=32 consecutive output diagonals per warp simultaneously. Each thread has K register accumulators. For each A diagonal in the UNION range, load A once and compute contributions to all K output diags via an unrolled inner loop. `#pragma unroll` ensures `acc[k]` stays in registers.

```
Warp (d_c_base, p_begin):
    float acc[32] = {0}                            // 32 register accumulators
    ai_range = union_range(d_c_base..d_c_base+31)  // binary search once

    for each A diagonal ai in union range:
        a_val = A_values[...]                       // 1 global read, REUSED 32x
        k_lo, k_hi = valid output diag range for this d_a
        #pragma unroll
        for k = 0..31:                              // constant-trip, register-indexed
            if (k >= k_lo && k < k_hi):
                b_val = B_values[...]               // 1 global read
                acc[k] += a_val * b_val             // register FMA

    // Write back 32 accumulators
    for k = 0..31: C[d_c_base+k][p_c] = acc[k]
```

**Per-FMA cost**: ~1/32 global read (A, amortized) + 1 global read (B) + 1 register FMA

**Strengths**:
- Zero atomics
- A amortization: each A value loaded once, reused K=32 times
- Register-only accumulation (no shared memory)
- Unrolled K loop with predicated execution (no divergence)
- Binary search done once per warp (amortized across K×32 = 1024 output elements)
- A loads reduced ~32x vs DiagSpMM v1

**Weaknesses**:
- K=32 gives ~32x A amortization, but HM has ~201x (for 201 diags). Still 6-7x more A loads than HM.
- The unrolled K loop iterates all 32 slots even when fewer are valid (predicated NOPs)
- B_diag_lookup check inside unrolled loop adds instruction overhead
- Increased register pressure: 32 accumulators + ~30 other = ~62 regs/thread

**Theoretical analysis** (n=16384, 201 diags):
- A loads per row: 201 × 7 batches = 1,407 (vs HM's 201, vs v1's 40,401)
- Reads per FMA: 1,407/40,401 + 1.0 = **1.035** (vs HM's 1.005 + atomic, vs v1's 2.0)
- Estimated time: 2.57 × 1.035/2.0 = **1.33 ms** (vs HM's 1.46 ms)
- Expected to be ~9% faster than HM

**Actual benchmark results**: [pending - kernel still slower than predicted, likely due to unrolled loop instruction overhead and B_diag_lookup latency in the hot inner loop]

---

## Summary Table

| Kernel | A Load Strategy | Accumulation | Atomics | Per-FMA Reads | Relative Speed |
|--------|----------------|--------------|---------|---------------|----------------|
| Paper HM | Register (1x) | Register + Atomic | Yes | 1.005 + atomic | **1.00x** (fastest) |
| DiagSpMM v1 | Global/L1 per iter | Register | No | 2.000 | ~1.76x slower |
| Row-Tiled v1 | Smem (1x) | **Smem RMW** | No | 1 + smem_rmw | ~19x slower |
| Row-Tiled v2 | Smem (1x) | Register | No | 1 + smem_read | ~11-19x slower |
| Batched K=32 | Global/L1 (1/32x) | Register | No | 1.035 | TBD (~1.0-1.3x) |

## Key Insights

### 1. Memory traffic is NOT the bottleneck — instruction throughput is
Total 128-byte memory transactions are nearly identical between DiagSpMM and HM (~80K per row-group of 32). The problem is per-FMA instruction mix.

### 2. The A-load amortization gap is the dominant factor
HM loads each A value into a register once and reuses it across all B iterations (201x amortization). DiagSpMM v1 loads A fresh each iteration (1x). Batched K=32 achieves 32x. Closing this gap is the key to matching HM.

### 3. Shared memory is wrong for accumulation, correct for caching
Using smem for read-write accumulators (20-30 cyc/access) is catastrophic. Using smem for read-only A cache (5 cyc) is better but doesn't overcome the iteration structure overhead.

### 4. Atomics are nearly free on H100
H100's L2 hardware atomic unit handles `atomicAdd(float)` at ~4-8 cycles. For structured sparse patterns with contention ~(2hb+1) per C element, the atomic throughput is high. The "zero atomic" advantage provides minimal benefit on modern hardware.

### 5. The fundamental tension
- **C-stationary** (no atomics): must reload A each iteration → 2x memory traffic
- **A-stationary** (no atomics): must accumulate across output diags → shared memory or too many registers
- **A-stationary + atomics** (HM): optimal A reuse + simple accumulation, atomics nearly free

Output diagonal batching (K=32) partially breaks this tension by amortizing A across K output diags in K registers. Full resolution would require K = B_num_diags registers (~201), which exceeds practical limits for large diagonal counts.

### 6. All custom kernels vastly outperform cuSPARSE
Even the slowest custom kernel (Row-Tiled v1) matches cuSPARSE at n=16384/201 diags. The fastest (HM) is **11.7x faster** than cuSPARSE. This demonstrates the value of diagonal-specific storage and algorithms vs generic CSR-based SpGEMM.

## Possible Future Directions

1. **Larger K via register tiling**: K=64 or K=128 with careful register management. Diminishing returns above K=32 (from 1.035 to 1.02 reads/FMA).

2. **Warp-shuffle A broadcast**: Distribute A values across 32 lanes, broadcast via `__shfl_sync`. Problem: each lane needs A value for its OWN row, not a shared value. Shuffle distributes one value to all lanes — doesn't help when each lane needs a different value.

3. **L1 cache optimization**: Use `__ldg()` or `__ldcs()` intrinsics to hint the compiler to use read-only/streaming cache for A values. May reduce L1 miss penalty.

4. **Hybrid approach**: Use HM's structure (A-stationary, iterate B) but replace atomics with warp-cooperative reduction. Each warp's 32 threads could collectively own a small number of output positions and use `__shfl_sync` for cross-lane accumulation.

5. **Accept atomics on modern hardware**: On H100/H200/B100 where atomics are hardware-accelerated, the HM approach may simply be optimal. Focus the paper on the diagonal storage format advantage over CSR (11.7x over cuSPARSE) rather than eliminating atomics.

6. **Thread block clusters (H100 DSMEM)**: The C-centric design causes the same A diagonal chunk to be loaded independently by every CTA processing a different C diagonal group at the same tile position. With Hopper thread block clusters, rank-0 CTA loads A into smem once; the other 7 CTAs in the cluster read it via distributed shared memory (DSMEM) at L1-cache speed. B is still loaded independently per CTA (each group needs different B diagonals). Net effect: **8× reduction in A global memory traffic** with CLUSTER_SIZE=8. Cost: smem per CTA grows (A smem allocated on all CTAs even though only rank-0 fills it), requiring 3 blocks/SM instead of 4. Requires CUDA 12 + `__cluster_dims__` + `cooperative_groups::this_cluster()`. Design in progress: `diag_cluster_kernel.cuh/.cu`.

7. **Smem hash table for B diagonal lookup**: The current `smem_B_lookup` is a dense array of size `b_d_range` (max d_b − min d_b + 1). For sparse B matrices with large diagonal offset gaps, `b_d_range` can exceed `max_b_per_part = 60`, causing an out-of-bounds smem write (latent bug). Suggested fix: replace with a fixed-size open-addressing hash table in smem, size = next power of 2 ≥ max_b_per_part (e.g., 64 slots). Key = d_b value, value = slot index in smem_B. Hash: `d_b & (HT_SIZE - 1)`, linear probe on collision. Benefits: smem footprint bounded regardless of d_b spread; fixes the latent OOB bug. Cost: 1–2 extra comparisons per lookup in the inner accumulation loop.
