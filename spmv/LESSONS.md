# Tensor-Core Diagonal SpMV — Design Lessons

_H100 NVL (sm_90a), CUDA 13.2, TF32. Work from 2026-06-07._

This note records what we learned converting diagonal/banded SpMV into a dense
tensor-core MMA, and benchmarking it against cuSPARSE and Drawloom.

---

## 0. The idea under test

A banded matrix `H` in DIA form is reconstructed into `Recon[k, c] = H[c-d_k, c]`,
and SpMV is realised as the **diagonal of a dense MMA**:

```
A[r,k] = Recon[k, tile_row+r+d_k]      B[k,c] = x[tile_row+c+d_k]
C = A*B   ->   C[r,r] = y[tile_row+r]    (off-diagonal entries discarded)
```

Verified correct end-to-end (`test_reconstruct.cpp`, `test_tc_spmv.cu`).

---

## 1. The first version didn't even compile — and why that mattered

The initial kernel used `nvcuda::wmma` fragments of shape **m16n8k8**. That shape
**does not exist in the WMMA C++ API** — the only TF32 WMMA shape is `m16n16k8`.
`m16n8k8` is a *PTX `mma.sync`* shape, reachable only via inline PTX / CUTLASS.

**Lesson:** the WMMA API and the PTX `mma.sync` ISA expose *different* shape sets.
If you want the small/asymmetric shapes (m16n8k8, m8n8k4, m16n8k32 sparse …), you
must drop to raw PTX. WMMA is the convenient-but-restrictive subset.

Two valid fixes:
- `m16n16k8` via WMMA (what we did first — simple, compiles, correct).
- `m16n8k8` via raw PTX (what the fast version uses — full control).

---

## 2. Don't optimize the thing that isn't the bottleneck

The "obvious" worry was **wasted MMA compute**: only the 16 diagonal entries of a
16×16 output are used (~3.5% MMA utilization). It feels like the problem. It is not.

`ncu` on the WMMA kernel at N=4M:

| metric | value |
|---|---|
| DRAM throughput | **15.75%** |
| Compute (SM) throughput | 49.2% |

The kernel saturated **neither** memory nor compute → it was **overhead/latency
bound**. SpMV is memory-bound, and an H100 has ~1000 TF32 TFLOP/s, so the discarded
off-diagonal FLOPs are essentially free and add **zero** DRAM traffic (the MMA reads
from registers, not memory).

**Lesson:** profile before optimizing. "Wasted FLOPs" on a memory-bound op is a red
herring. Count **bytes moved** and **instructions issued**, not MMA-lattice
utilization.

---

## 3. For single-vector banded SpMV, the diagonal mapping is intrinsic

We considered switching to a "M = rows, K = nonzeros, no diagonal" mapping (how
DASP/Drawloom present it). Working through the algebra:

```
y[m] = sum_k A[m,k] * x[m + off_k]      <- the x index depends on row m
```

For `C[m,n] = sum_k A[m,k] B[k,n]` with a **shared** B, the only way to land
`y[m]` is on `C[m,m]` — the diagonal is **algebraically forced** whenever each row
reads a row-shifted window of `x`. DASP/Drawloom only escape the diagonal by
**column-tiling** general sparsity (all rows in a tile share one `x` segment); for a
band that means ~5 column-tile MMAs instead of 2 diagonal-batch MMAs — *more* work.

**Lesson:** the diagonal trick is a legitimate, near-optimal mapping for a *band*.
The win is not the mapping — it's the mechanics. Don't change what's already right.

---

## 4. The mechanics are where the speed was hiding

Same math, two changes (`tc_spmv_regdirect_kernel.cu`): raw
`mma.sync.m16n8k8.f32.tf32` PTX with operands streamed **global → registers**, and
the result read **straight out of the accumulator registers** (the m16n8k8 C-lane
map is fixed: the 16 diagonal entries live in lanes `g==2t` → c0/c2, `g==2t+1` →
c1/c3, so 8 lanes write to global directly).

| | WMMA + smem | regdirect (PTX) |
|---|---|---|
| speedup vs WMMA | — | **1.6–1.8×** |
| DRAM @4M | 16% | **28%** |
| duration @4M | 296 µs | 168 µs |
| vs cuSPARSE @1M | 0.70× (lost) | **1.23× (won)** |

### Why it got faster — trace the data's journey per 16-row tile

The two versions issue the **same global loads** and the **same MMAs**. The only
difference is the path the data takes *between* global memory and the MMA registers.

**WMMA + smem path** (each operand makes a detour through shared memory):

```
global ─► smem (store) ─► [__syncwarp] ─► ldmatrix ─► registers ─► MMA
                                                                     │
global ◄─ regs ◄─ smem (read 16) ◄─ [implicit sync] ◄─ store_matrix_sync (write 256)
```

Per tile that is, just for the shared-memory layer:
- A: 128 floats written + 128 read
- B (top+bot): 128 written + 128 read
- C: **256 floats written by `store_matrix_sync` to recover only 16** → 16× write
  amplification on the output
- plus the `__syncwarp` barriers gating each phase

≈ **~780 shared-memory float transactions + 2 barriers to produce 16 results.**

**Register-direct path:**

```
global ─► registers ─► MMA ─► registers ─► global
```

≈ **0 shared-memory transactions, 0 barriers.** Operands land in the exact registers
the MMA reads; the 16 outputs are written from the 8 lanes that already hold them.

### The four mechanisms, ranked

1. **Barrier removal is the big one.** `load_matrix_sync` from smem can't fire until
   the smem is fully written, so it needs a `__syncwarp`. On a *latency-bound* kernel
   (which §2 proved this was), barriers are pure stall: they stop the scheduler from
   hiding global-load latency behind other warps' compute. No smem ⇒ no barrier ⇒ the
   warp scheduler overlaps freely. (Raising the block to 4 warps amplifies this.)
2. **No output write-amplification.** `store_matrix_sync` materializes the whole
   accumulator (256 floats) to extract 16 — a 16× waste in smem write traffic and
   instructions. Register extraction touches only the 16 needed values.
3. **Fewer instructions per tile.** Each `*_matrix_sync` expands to address math +
   `ldmatrix`/`st.shared` + sync. Deleting them shrinks the per-tile instruction
   count — the direct lever when you are instruction/latency-bound, not FLOP-bound.
4. **PTX exposes the layout, which is what makes 1–3 possible.** Knowing the
   register↔element map lets you place operands and harvest results with zero data
   movement. WMMA *hides* the layout, so the smem round-trip is the only
   API-sanctioned way to get known-layout data in and out of a fragment. The smem
   tax is not incidental to WMMA — it is the price of the abstraction.

Net effect shows up exactly where predicted: DRAM utilization climbed 16% → 28% (the
kernel started actually moving data instead of stalling), duration nearly halved, and
the structured-format bandwidth edge (§5) finally became reachable.

> ### ⭐ KEY OPTIMIZATION RULE — minimize the data's journey to the MMA registers
>
> **A tensor-core MMA consumes and produces *registers*. Every shared-memory hop and
> every `*_matrix_sync` between global memory and those registers is overhead you pay
> per tile. Pay it only to *reuse* data across the warp; never pay it merely to
> *reshape* data into a fragment.**
>
> Decision rule:
> - **Use WMMA + shared memory** when an smem tile is read by *many* MMAs (true GEMM
>   tiling: operand reuse amortizes the staging cost), or when you don't need the
>   small/asymmetric PTX shapes.
> - **Drop to `mma.sync` PTX + register-direct** when the kernel is
>   **latency/overhead-bound**, operands are **used once** (streaming kernels — SpMV,
>   SpMM with low reuse, fused epilogues), and you can **compute operand addresses and
>   know the C-fragment layout**. Then smem staging and `*_matrix_sync` are pure tax —
>   delete them and stream `global → reg → MMA → reg → global`.
>
> Litmus test before adding shared memory to a TC kernel: *"How many MMAs read each
> value I stage?"* If the answer is **one**, shared memory is costing you, not helping.

---

## 5. The structured format's real edge: implicit indices

CSR stores a 4-byte **column index per nonzero** — roughly doubling the matrix bytes
it streams. A band encodes column position arithmetically (`col = row + offset`), so
it reads **zero** index data. On a memory-bound op that is up to a **~2× bandwidth
advantage** over any CSR-based method — but you only collect it once the kernel is
bandwidth-bound (i.e. after fixing §4). Before that, overhead masks it entirely.

**Lesson:** "structured beats general when zeros dominate" is true, but it's a
*bandwidth* claim — it only shows up after the kernel stops being overhead-bound.

---

## 6. What we learned from the baselines (DASP → Drawloom)

- **DASP (SC'23):** tensor core as a **batched reducer**. Bin rows long/medium/short,
  pack nonzeros into small MMAs (`m8n8k4`), finish with warp shuffles. Static; loads
  operands directly; no pipeline, no reorder, no sparse-MMA.
- **Drawloom (2025, successor):** adds (1) **row reordering by column similarity**
  (MinHash-LSH) to densify TC tiles, (2) a **`cp.async` multi-stage pipeline**, (3)
  `ldmatrix` staging, (4) **2:4 sparse tensor cores** (`mma.sp`), plus a TF32 path.

What transfers to a uniform band, and what doesn't:

| Drawloom feature | Use for a band? |
|---|---|
| TF32 `m16n8k8` PTX | **Yes — copied.** |
| register-direct / `ldmatrix` | **Yes (register-direct half) — copied.** |
| `cp.async` pipeline | **No (marginal)** — only `ceil(num_diags/8)` batches to overlap. |
| LSH row reorder / ZCF format | **No** — a band already has perfect column locality. |
| 2:4 sparse MMA | **No** — band rarely fits the 2:4 pattern. |

**Lesson:** a newer/fancier baseline is not strictly better *for your problem*.
Drawloom's headline contribution (reorder-to-densify) targets **irregular** sparsity;
on a uniform band it's dead weight. Copy the mechanics, skip the generality.

---

## 7. Benchmark result vs Drawloom (same band, fair yardstick)

The two harnesses time differently (Drawloom reports cuSPARSE ~1.25–1.3× faster than
this repo's bench), so compare via **speedup over each harness's own cuSPARSE**:

| N | regdirect (ours) | Drawloom TF32 |
|---|---|---|
| 65 K | **3.16×** | 2.57× |
| 256 K | **2.34×** | 1.50× |
| 1 M | **1.23×** | 0.86× |

The band-specialized kernel wins at all sizes; at 1M Drawloom is slower than its own
cuSPARSE on this matrix. (Reproduce: `DRAWLOOM_BIN=… ./run_comparison.sh`.)

**Lesson:** always cross-check timing methodology. Raw ms across harnesses is not
comparable; speedup-over-own-baseline is the robust metric. State the caveat.

---

## 7.5 Full size sweep — where the advantage peaks (the sweet point)

Swept N from 16 K to 4.2 M rows (same band, `bw=9`, `run_comparison.sh`). Each row:
regdirect's raw multiplier over cuSPARSE, plus Drawloom's speedup over **its own**
cuSPARSE, plus the **normalized margin** = (our speedup) / (Drawloom's speedup).

| N | nnz | cuSPARSE GF/s | regdirect GF/s | **ours vs cuSPARSE** | Drawloom (norm) | **ours/Drawloom (norm)** |
|---|---|---|---|---|---|---|
| 16 K | 147 K | 21.2 | 67.0 | **3.16×** | 3.23× | 0.98× |
| 32 K | 295 K | 40.4 | 128.2 | **3.17×** | 3.03× | 1.05× |
| 65 K | 590 K | 71.5 | 222.6 | **3.11×** | 2.62× | 1.19× |
| 131 K | 1.18 M | 108.7 | 310.4 | **2.86×** | 2.06× | 1.39× |
| 262 K | 2.36 M | 188.0 | 432.9 | **2.30×** | 1.54× | 1.49× |
| **524 K** | **4.72 M** | 267.3 | **496.7** | **1.86×** | 1.00× | **1.86×** |
| 1.05 M | 9.44 M | 345.1 | 424.1 | **1.23×** | 0.86× | 1.43× |
| 2.10 M | 18.9 M | 404.2 | 433.4 | **1.07×** | 0.76× | 1.41× |
| 4.19 M | 37.7 M | 403.5 | 446.2 | **1.11×** | *crash* | — |

(H100 NVL, sm_90a, TF32. Drawloom **fails to produce a result on the 4.19 M band** —
no timing output / exit 139 — so it's not just slower, it doesn't run that size here.)

There are **three different "sweet points"** depending on the question:

1. **Biggest multiplier over cuSPARSE → small N (16 K–65 K, ~3.1–3.2×).** This is an
   *overhead* regime, not a bandwidth one: cuSPARSE is launch/setup-dominated (only
   21–72 GF/s) while our lightweight kernel isn't. The multiplier is large but the
   absolute work is tiny — flattering, not the real win.
2. **Widest margin over Drawloom → N ≈ 524 K (1.86× normalized).** Drawloom has
   collapsed to *parity with its own cuSPARSE* (1.00×) here while we're still at 1.86×.
   The relative margin grows monotonically from 16 K up to 524 K, then narrows.
3. **Our kernel's own peak throughput → N ≈ 524 K (496.7 GF/s),** its global max
   across the whole sweep.

**The honest single answer is N ≈ 524 K (~0.5 M rows, ~4.7 M nnz).** It's the
"Goldilocks" point: enough work to amortize our launch overhead and reach peak
occupancy (our max GF/s), while cuSPARSE is **not yet bandwidth-saturated** (267 GF/s)
— so we still get 1.86×, *and* it's our widest gap over Drawloom.

**Why the cuSPARSE multiplier decays toward large N — and why it *floors* near 1.1×
instead of falling below 1×.** As N grows, cuSPARSE amortizes overhead and climbs to a
**bandwidth ceiling (~404 GF/s)**. Ours rises to 496 then settles ~430–446. At ≥1 M
both are bandwidth-bound — so this is exactly where the §5 implicit-index advantage
*should* hand us ~2× (a band streams no column array, ~half CSR's bytes). We only get
**~1.1×**. The 2× is on the table but unrealized: per §8 the kernel sits at ~28% DRAM,
so it isn't actually saturating bandwidth, so the structural byte-savings don't convert
to speed. **The large-N gap between the observed ~1.1× and the structural ~2× is the
single most actionable headroom number in this whole study.**

**Lesson:** "find the sweet point" has no scalar answer — name the regime. Peak
*multiplier* (small N, overhead-bound) and peak *absolute advantage* (mid N,
~524 K, near our roofline) live in different places, and the large-N convergence is a
**diagnosis** (bandwidth left unclaimed), not a law of nature.

---

## 8. Where the ceiling is, and the real opportunity

The regdirect kernel is now **60% compute / 28% DRAM** — better balanced but **not
bandwidth-saturated**. Remaining costs: the wasted-N MMA work and bounds-checked
scalar gathers. Single-vector SpMV fundamentally underuses a tensor core by ~1/N
(you get M outputs from an M×N×K MMA).

The clean escape is **SpMM / block methods** (multiple right-hand sides): then `x`
is a block, the MMA's N dimension carries real columns, there is **no diagonal trick
and no wasted dimension**, and the tensor core runs at full utilization. For
diagonally-dominant iterative solvers (block CG/Krylov) this is often natural.

**Lesson:** single-vector SpMV is the *worst* case for tensor cores. If the
application admits multiple RHS, that's where the TC approach wins unambiguously.

---

## TL;DR

1. WMMA ≠ PTX `mma.sync` shapes — know which API gates which shape.
2. Profile first; "wasted FLOPs" on a memory-bound op is a red herring.
3. For banded single-vector SpMV the diagonal mapping is intrinsic — fix mechanics,
   not math.
4. **⭐ KEY RULE: minimize the data's journey to the MMA registers.** An MMA eats and
   emits registers; every smem hop and `*_matrix_sync` is per-tile tax. Pay it only to
   *reuse* data across the warp, never just to *reshape* into a fragment. For
   single-use/streaming operands, drop to `mma.sync` PTX and go
   `global → reg → MMA → reg → global` (here: −~780 smem float-ops and −2 barriers per
   tile → 1.6–1.8× and a win over cuSPARSE). Litmus test: *if each staged value feeds
   only one MMA, shared memory is costing you.*
5. Structured formats win on **bandwidth** (implicit indices), but only after you're
   bandwidth-bound.
6. Copy a baseline's mechanics, not its generality — Drawloom's reorder/format is
   for irregular sparsity, useless on a band.
7. Compare across harnesses via speedup-over-own-baseline, not raw ms.
8. **Sweet point ≈ N=524 K (~0.5 M rows):** our kernel's own throughput peaks
   (496 GF/s) *and* the margin over Drawloom is widest (1.86× normalized) while
   cuSPARSE isn't yet bandwidth-saturated. Peak *multiplier* over cuSPARSE (~3.2×) is
   at small N but that's an overhead regime. At large N (≥1 M) both go bandwidth-bound
   and we fall to ~1.1× — vs the ~2× the implicit-index byte savings *should* give:
   that gap (kernel only ~28% DRAM) is the actionable headroom. Drawloom outright
   **crashes on the 4.19 M band**; ours runs it.
9. The real headroom is SpMM / multiple RHS, where the TC is fully utilized.
