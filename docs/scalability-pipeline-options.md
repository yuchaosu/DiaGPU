# Scalability: Split-K Pipeline Options for Large N

## The Problem

For a group of K consecutive C diagonals, all contributing (A, B) diagonal pairs
must be processed. The number of pairs grows **O(N)** with matrix size:

```
smem needed = (a_count + b_count) × chunk × 4 bytes  ∝  N × TILE × 4
```

For N = 16384, fully dense diagonal matrices:
- a_count ≈ 32767, b_count ≈ 32767
- chunk ≈ 136 floats
- smem needed ≈ 35.7 MB  vs.  budget = 57 KB

**No tuning of TILE, BLOCK, or DIAGS_PER_CTA solves this** — the pair count
is independent of all those parameters. The only correct solution is to
partition the (A, B) pairs across CTAs (split-K along the A diagonal axis).

### Why Corner-First Ordering

A diagonals with large |d_a| (far from main diagonal) are shorter and
contribute to a smaller region of the output. Processing them first means
their partial results are the smallest, complete soonest, and are cheapest
to reduce. Processing inward (|d_a| decreasing) means the heaviest work
(main diagonal contribution) comes last, by which time all corner partials
are already reduced.

```
partition 0:  d_a = ±(N-1)   shortest, corner
partition 1:  d_a = ±(N-2)
...
partition P:  d_a = 0         longest, center
```

---

## Task Structure (Chosen Design)

One CTA owns one `(C diagonal group, position tile, A-partition)` triple.

Each CTA loops through A-partitions serially, keeping smem small:

```
CTA (g, t):
  for p = 0 .. P-1:
    load A-partition[p] + B-partition[p] into smem   ← fits: O(partition_size)
    accumulate into slot[p % 2]
    if p > 0: add slot[(p-1) % 2] → running_sum
  add slot[(P-1) % 2] → running_sum
  write running_sum → C_vals
```

**Smem per CTA (constant, independent of N):**
```
slot[0] + slot[1]  =  2 × partition_size × chunk × 4
B smem             =  partition_size × chunk_b × 4
B lookup           =  b_d_range_pad × 4
```

**Chosen because:** Zero inter-CTA synchronization. Self-contained.
No kernel launch overhead. GPU parallelizes across different (g, t) CTAs
naturally. Smem stays bounded regardless of N.

**Limitation recorded:** Compute and reduce are serial within each CTA.
No true hardware-level overlap between them. The corner-first ordering
provides correctness and clean memory reuse but no latency hiding in
this variant.

---

## Option B — Two Streams, Separate Kernel Launches (True Overlap)

```
for p = 0 .. P-1:
    stream0: compute_kernel(wave=p)  →  slot[p % 2]
    record event_p on stream0

    stream1: wait(event_{p-1})
    stream1: reduce_kernel(wave=p-1)  →  running_sum += slot[(p-1)%2]
```

**Advantage:** True hardware-level compute/reduce overlap. Stream 0 and
stream 1 run on different SMs simultaneously. Corner-first ordering
genuinely hides reduction latency — early (short) compute waves finish
fast so the reduce stream never stalls.

**Drawback:** One kernel launch per wave per stream. For P = N (one A
diagonal per partition), N = 16384 → ~32K launches. Launch overhead
dominates. Practical only when P is small, meaning each partition batches
many A diagonals (reducing overlap granularity).

**When to try:** If profiling shows the serial compute→reduce loop in
Option A is a bottleneck (e.g., reduction latency >> next partition's
compute time), batch partitions into groups of ~16–32 A diagonals,
reduce P to ~1000, and use this two-stream model.

---

## Option C — Cooperative Groups, Single Persistent Kernel

Single kernel launch using `cudaLaunchCooperativeKernel`. Split the CTA
pool into a compute half and a reduce half. Both halves run concurrently
within each wave, separated by `grid.sync()` at wave boundaries:

```
kernel (cooperative):
  for p = 0 .. P-1:
    if CTA in compute_group:
        compute partition p → slot[p % 2]
    if CTA in reduce_group:
        running_sum += slot[(p-1) % 2]
    grid.sync()
```

**Advantage:** Single launch. No launch overhead. True within-kernel
overlap. Architecturally the cleanest expression of the pipeline.

**Drawback:** `cudaLaunchCooperativeKernel` requires the entire grid to
be resident on the GPU simultaneously. For large N:

```
CTAs needed = n_groups × n_tiles = (2N/K) × (N/TILE)
            ≈ (2×16384/8) × (16384/128) = 4096 × 128 = 524288
```

H100 supports ~2048 concurrent CTAs per SM × 132 SMs = ~270K CTAs max.
This exceeds the limit for N ≥ ~11K, forcing a smaller grid that
serializes work and kills occupancy.

**When to try:** Only if matrix sizes are known to be small (N ≤ 8K)
and the single-launch overhead savings matter more than occupancy.

---

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-08 | Option A chosen | Zero sync complexity, self-contained, no launch overhead, scales to any N |
| 2026-04-08 | Option B recorded | Try if reduction latency becomes a measured bottleneck |
| 2026-04-08 | Option C recorded | Only viable for small N (≤ 8K) due to cooperative grid size limit |
