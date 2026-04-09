# Split-K Partition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current design (all B loaded once per task, A batched) with a true split-K partition design where each loop iteration loads exactly `HYBRID_PARTITION_SIZE` A diagonals and their corresponding B diagonals into smem, keeping smem footprint O(1) in N.

**Architecture:** A contributors per group are sorted by d_a (ascending). Each partition of `HYBRID_PARTITION_SIZE` consecutive A contributors has a deterministic B range (d_b = d_c − d_a is contiguous). The host precomputes per-partition B metadata (`PartBMeta[]`). The kernel's A-batch loop is extended to also load B for each partition before accumulating. The accumulator `acc[ki]` lives in registers across all partition iterations — no inter-partition synchronisation or global partial buffer.

**Tech Stack:** CUDA C++17, H100 (228 KB smem / SM, 4 blocks/SM target, 57 KB budget per block). Compile: `nvcc test_hybrid.cu diag_hybrid_kernel.cu paper_hm_kernel.cu -o test_hybrid -std=c++17`

---

## File Map

| File | Change |
|------|--------|
| `diag_hybrid_kernel.cuh` | Add `HYBRID_PARTITION_SIZE`; add `PartBMeta` struct; update `HybridTask`, `HybridKernelArgs`, `HybridPlan` |
| `diag_hybrid_kernel.cu` | `build_hybrid_plan`: sort A by d_a, compute `PartBMeta[]` per partition; `hybrid_kernel`: move B load inside A-batch loop |
| `test_hybrid.cu` | Allocate `d_part_b_meta`, pass to kargs |

---

## Task 1: Update constants and data structures in diag_hybrid_kernel.cuh

**Files:**
- Modify: `diag_hybrid_kernel.cuh`

Background: `HybridTask` currently carries a single group-level `b_begin/b_count/b_d_min/b_d_range` covering all B diagonals. In the new design these become per-partition. We add a `PartBMeta` struct (one per partition) and reference it from the task.

- [ ] **Step 1: Add `HYBRID_PARTITION_SIZE` constant**

  In `diag_hybrid_kernel.cuh`, after the existing `HYBRID_BLOCKS_PER_SM` constant line, add:

  ```cpp
  constexpr int HYBRID_PARTITION_SIZE = 53;  // A (and ~B) diags per smem partition
  ```

- [ ] **Step 2: Add `PartBMeta` struct**

  After the `HYBRID_PARTITION_SIZE` line, add:

  ```cpp
  /* Per-partition B metadata.  Stored in a flat array on device;
   * task.part_b_base indexes into it. */
  struct PartBMeta {
      int b_begin;    // offset into b_contrib[] relative to task.b_begin
      int b_count;    // B diagonals in this partition
      int b_d_min;    // min d_b among this partition's B contributors
      int b_d_range;  // b_d_max - b_d_min + 1  (lookup table width)
  };
  ```

- [ ] **Step 3: Update `HybridTask`**

  Remove `b_count`, `b_d_min`, `b_d_range`, and `a_smem_cap` from `HybridTask`.
  **Keep `b_begin`** — it is still needed by the kernel to find the group's b_contrib base.
  Add:

  ```cpp
  int part_b_base;   // first index into part_b_meta[] for this task's group
  int n_parts;       // ceil(a_count / HYBRID_PARTITION_SIZE)
  ```

  Remove `a_smem_cap` — it is now always `HYBRID_PARTITION_SIZE`.

  The updated struct should be:

  ```cpp
  struct HybridTask {
      int c_begin;
      int c_count;
      int min_c_sr;
      int spread;
      int max_c_len;
      int a_begin;
      int a_count;
      /* B smem — column side */
      int min_c_sc;
      int spread_sc;
      int b_begin;       // start of this group's b_contrib entries
      /* Partition metadata */
      int part_b_base;   // index into part_b_meta[]
      int n_parts;       // number of A partitions
      /* Output tile */
      int tile_p_begin;
  };
  ```

- [ ] **Step 4: Update `HybridKernelArgs`**

  Remove `max_smem` (moved to launch, not kernel). Add:

  ```cpp
  const PartBMeta* part_b_meta;   // flat array, indexed by task.part_b_base + p
  ```

  The updated struct:

  ```cpp
  struct HybridKernelArgs {
      const HybridTask*   tasks;
      int                 n_tasks;
      int                 max_smem;
      const HybridCDiag*  c_diags;
      int                 n_c_diags;
      const int*          a_contrib;
      const int*          b_contrib;
      const PartBMeta*    part_b_meta;
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

- [ ] **Step 5: Update `HybridPlan`**

  Add `part_b_meta` vector:

  ```cpp
  struct HybridPlan {
      std::vector<HybridCDiag>  c_diags;
      std::vector<HybridTask>   tasks;
      std::vector<int>          a_contrib;
      std::vector<int>          b_contrib;
      std::vector<PartBMeta>    part_b_meta;
      int total_c_values;
      int max_smem;
  };
  ```

- [ ] **Step 6: Verify the file compiles (headers only)**

  ```bash
  nvcc -std=c++17 -x cu -c /dev/null --include diag_hybrid_kernel.cuh -o /dev/null 2>&1 | head -20
  ```

  Expected: no errors (warnings about unused variables are fine at this stage).

---

## Task 2: Update `build_hybrid_plan` in diag_hybrid_kernel.cu

**Files:**
- Modify: `diag_hybrid_kernel.cu` (the `build_hybrid_plan` function)

Background: The current plan sorts A contributors in whatever order `A.offsets` happens to be. We must sort them by d_a ascending so that each partition of `HYBRID_PARTITION_SIZE` A contributors maps to a contiguous d_b range (B partition). We also precompute `PartBMeta` for each partition.

- [ ] **Step 1: Sort A contributors by d_a ascending**

  Find the block in `build_hybrid_plan` where `group_a_indices` is populated (lines ~284–296 of current file). After `group_a_indices` is filled, add a sort:

  ```cpp
  std::sort(group_a_indices.begin(), group_a_indices.end(),
            [&](int a, int b){ return A.offsets[a] < A.offsets[b]; });
  ```

  This replaces whatever implicit ordering existed. (Currently A contributors are added in A.offsets iteration order which is already sorted, so this is a no-op for well-formed input — but makes the guarantee explicit.)

- [ ] **Step 2: Remove the global group-level B collection**

  Delete the `group_b_indices` computation block (the one building a `std::set<int> b_set` over all A indices). We will replace it with per-partition B collection in the next step.

  Also delete `b_count_g`, `b_d_min_g`, `b_d_max_g`, `b_d_range_g`, `b_d_range_pad_g`, `b_smem_bytes_g` — they are replaced by `PartBMeta`.

- [ ] **Step 3: Compute per-partition B metadata and populate b_contrib**

  Replace the deleted code with a partition loop. Insert after `a_base_g` is set:

  ```cpp
  const int a_base_g   = static_cast<int>(plan.a_contrib.size());
  for (int ai : group_a_indices)
      plan.a_contrib.push_back(ai);

  const int total_a    = static_cast<int>(group_a_indices.size());
  const int n_parts    = (total_a + HYBRID_PARTITION_SIZE - 1) / HYBRID_PARTITION_SIZE;
  const int part_b_base_g = static_cast<int>(plan.part_b_meta.size());
  const int b_base_g   = static_cast<int>(plan.b_contrib.size());

  /* Build PartBMeta for each A partition. */
  for (int p = 0; p < n_parts; ++p) {
      const int a_p_begin = p * HYBRID_PARTITION_SIZE;
      const int a_p_end   = std::min(a_p_begin + HYBRID_PARTITION_SIZE, total_a);

      /* d_a range for this partition (A sorted by d_a ascending). */
      int d_a_min_p = A.offsets[group_a_indices[a_p_begin]];
      int d_a_max_p = A.offsets[group_a_indices[a_p_end - 1]];

      /* d_b range: for each C diagonal d_c and each d_a in partition,
       * d_b = d_c - d_a.  Range: [min_d_c - d_a_max, max_d_c - d_a_min]. */
      int min_d_c = INT_MAX, max_d_c = INT_MIN;
      for (int ki = 0; ki < g_count; ++ki) {
          const int d_c = plan.c_diags[g_base + ki].c_offset;
          min_d_c = std::min(min_d_c, d_c);
          max_d_c = std::max(max_d_c, d_c);
      }
      const int d_b_lo = min_d_c - d_a_max_p;
      const int d_b_hi = max_d_c - d_a_min_p;

      /* Collect B contributors with d_b in [d_b_lo, d_b_hi]. */
      std::vector<int> part_b;
      for (int bi = 0; bi < static_cast<int>(B.offsets.size()); ++bi) {
          const int d_b = B.offsets[bi];
          if (d_b >= d_b_lo && d_b <= d_b_hi)
              part_b.push_back(bi);
      }
      std::sort(part_b.begin(), part_b.end(),
                [&](int a, int b){ return B.offsets[a] < B.offsets[b]; });

      PartBMeta meta;
      meta.b_begin  = static_cast<int>(plan.b_contrib.size()) - b_base_g;
      meta.b_count  = static_cast<int>(part_b.size());
      meta.b_d_min  = part_b.empty() ? 0 : B.offsets[part_b.front()];
      meta.b_d_range = part_b.empty() ? 0
                      : (B.offsets[part_b.back()] - meta.b_d_min + 1);
      plan.part_b_meta.push_back(meta);

      for (int bi : part_b)
          plan.b_contrib.push_back(bi);
  }
  ```

- [ ] **Step 4: Update the task struct population**

  Find the loop `for (int tile = 0; tile < n_tiles; ++tile)` and update the task fields:

  - Remove: `t.a_smem_cap`, `t.b_begin`, `t.b_count`, `t.b_d_min`, `t.b_d_range`
  - Add:

  ```cpp
  t.b_begin      = b_base_g;
  t.part_b_base  = part_b_base_g;
  t.n_parts      = n_parts;
  ```

  The full task init becomes:

  ```cpp
  HybridTask t;
  t.c_begin      = g_base;
  t.c_count      = g_count;
  t.min_c_sr     = min_c_sr;
  t.spread       = spread;
  t.max_c_len    = max_c_len;
  t.a_begin      = a_base_g;
  t.a_count      = total_a;
  t.min_c_sc     = min_c_sc_g;
  t.spread_sc    = spread_sc_g;
  t.b_begin      = b_base_g;
  t.part_b_base  = part_b_base_g;
  t.n_parts      = n_parts;
  t.tile_p_begin = tile * HYBRID_TILE;
  plan.tasks.push_back(t);
  ```

- [ ] **Step 5: Update `plan.max_smem` calculation**

  Replace the old `smem_needed` formula with:

  ```cpp
  /* Worst-case smem: N_part A + (N_part + K - 1) B + lookup.
   * chunk and chunk_b_g are already in scope (computed earlier in the group loop). */
  const int max_b_per_part    = HYBRID_PARTITION_SIZE + HYBRID_DIAGS_PER_CTA - 1;
  const int lookup_pad        = (max_b_per_part + 3) & ~3;
  const int smem_needed       = static_cast<int>(sizeof(float))
                              * (HYBRID_PARTITION_SIZE * chunk
                                 + max_b_per_part * chunk_b_g
                                 + lookup_pad);
  plan.max_smem = std::max(plan.max_smem, smem_needed);
  ```

- [ ] **Step 6: Compile (host side only, will fail on kernel until Task 3)**

  ```bash
  cd /Volumes/STORAGE/Yuchao/DiaGPU && nvcc -std=c++17 -c diag_hybrid_kernel.cu -o /dev/null 2>&1 | head -30
  ```

  Expected: errors only in `hybrid_kernel` referencing removed fields, not in `build_hybrid_plan`.

---

## Task 3: Update `hybrid_kernel` in diag_hybrid_kernel.cu

**Files:**
- Modify: `diag_hybrid_kernel.cu` (the `hybrid_kernel` function)

Background: Currently B is loaded once before the A-batch loop. In the new design, B is loaded inside the loop, one partition at a time. The smem layout now holds exactly `HYBRID_PARTITION_SIZE` A diagonals + up to `HYBRID_PARTITION_SIZE + K - 1` B diagonals simultaneously.

- [ ] **Step 1: Update smem layout constants at top of kernel**

  In `hybrid_kernel`, replace:

  ```cpp
  const int a_smem_cap = task.a_smem_cap;
  ```

  with:

  ```cpp
  constexpr int a_smem_cap = HYBRID_PARTITION_SIZE;
  ```

  Keep `chunk` and `chunk_b` as-is (they depend on spread which is still in task).

  Update smem pointers — smem layout is now:
  - `smem_A`: `a_smem_cap × chunk` floats
  - `smem_B`: `(a_smem_cap + HYBRID_DIAGS_PER_CTA - 1) × chunk_b` floats (max B per partition)
  - `smem_B_lookup`: immediately after smem_B

  ```cpp
  constexpr int max_b_per_part = a_smem_cap + HYBRID_DIAGS_PER_CTA - 1;
  float* smem_A        = smem;
  float* smem_B        = smem + a_smem_cap * chunk;
  int*   smem_B_lookup = reinterpret_cast<int*>(smem_B + max_b_per_part * chunk_b);
  ```

- [ ] **Step 2: Remove the upfront full-B loading block**

  Delete the entire block that initialises `smem_B_lookup` and loads all B diagonals into `smem_B` (currently lines ~91–130, before the `const int p_begin = task.tile_p_begin;` line).

  Also delete the upfront lookup init `__syncthreads()` pair.

- [ ] **Step 3: Move B loading inside the A-batch loop**

  The A-batch loop currently starts at:
  ```cpp
  for (int a_off = 0; a_off < total_a; a_off += a_smem_cap) {
  ```

  At the top of this loop body (before the A load), add the per-partition B load:

  ```cpp
  /* Load B partition for this A partition. */
  const int p_idx  = a_off / HYBRID_PARTITION_SIZE;
  const PartBMeta pmeta = args.part_b_meta[task.part_b_base + p_idx];
  const int part_b_count   = pmeta.b_count;
  const int part_b_d_min   = pmeta.b_d_min;
  const int part_b_d_range = pmeta.b_d_range;
  const int part_b_d_range_pad = (part_b_d_range + 3) & ~3;

  /* Init B lookup for this partition. */
  for (int i = tid; i < part_b_d_range_pad; i += HYBRID_BLOCK)
      smem_B_lookup[i] = -1;
  __syncthreads();

  for (int sb = tid; sb < part_b_count; sb += HYBRID_BLOCK) {
      const int bi = args.b_contrib[task.b_begin + pmeta.b_begin + sb];
      smem_B_lookup[args.B_offsets[bi] - part_b_d_min] = sb;
  }
  __syncthreads();

  for (int sb = 0; sb < part_b_count; ++sb) {
      const int bi    = args.b_contrib[task.b_begin + pmeta.b_begin + sb];
      const int d_b   = args.B_offsets[bi];
      const int b_len = args.B_lengths[bi];
      const int b_st  = args.B_starts[bi];
      const int bp_min = task.min_c_sc + p_begin - max(0, d_b);
      float* dst = smem_B + sb * chunk_b;
      const int cb4 = chunk_b >> 2;
      for (int j = tid; j < cb4; j += HYBRID_BLOCK) {
          const int i0 = j << 2;
          float4 v;
          int p0 = bp_min + i0;
          v.x = (p0   >= 0 && p0   < b_len) ? args.B_vals[b_st + p0]   : 0.f;
          v.y = (p0+1 >= 0 && p0+1 < b_len) ? args.B_vals[b_st + p0+1] : 0.f;
          v.z = (p0+2 >= 0 && p0+2 < b_len) ? args.B_vals[b_st + p0+2] : 0.f;
          v.w = (p0+3 >= 0 && p0+3 < b_len) ? args.B_vals[b_st + p0+3] : 0.f;
          *reinterpret_cast<float4*>(dst + i0) = v;
      }
  }
  __syncthreads();
  ```

- [ ] **Step 4: Update the accumulation inner loop to use per-partition B metadata**

  In the accumulation loops (both the unrolled `s < a_main` loop and the scalar tail), replace:
  - `task.b_d_min` → `part_b_d_min`
  - `task.b_d_range` → `part_b_d_range`

  The lookup array index `rel = d_b - part_b_d_min` and the bounds check `(unsigned)rel < (unsigned)part_b_d_range` are otherwise identical.

- [ ] **Step 5: Replace the inter-batch `__syncthreads()` guard**

  The current code has:
  ```cpp
  if (a_off + a_smem_cap < total_a) __syncthreads();
  ```

  Replace with an unconditional sync at the very end of the loop body (after accumulation):

  ```cpp
  __syncthreads();  // ensure smem_A is free before next iteration's B-lookup init
  ```

  Note: on the final partition iteration this sync is still correct (harmless extra sync
  before the write-out). There is no double-sync: the syncs at the top of the loop
  are between the lookup-init phase and the B-value load phase — a different smem
  hazard. Do NOT remove either of those inner syncs.

- [ ] **Step 6: Full compile**

  ```bash
  cd /Volumes/STORAGE/Yuchao/DiaGPU && nvcc -std=c++17 -arch=sm_90 -c diag_hybrid_kernel.cu -o diag_hybrid_kernel.o 2>&1 | head -30
  ```

  Expected: clean compile.

---

## Task 4: Update test_hybrid.cu

**Files:**
- Modify: `test_hybrid.cu`

Background: We need to allocate `d_part_b_meta` on the device and pass it in `kargs`. The test also prints plan stats — update the print to show partition count.

- [ ] **Step 1: Find where device buffers are allocated for the plan**

  Search for the block that allocates `d_a_contrib`, `d_b_contrib`, `d_tasks` in `test_hybrid.cu`. It should be a set of `cudaMalloc` + `cudaMemcpy` calls.

- [ ] **Step 2: Add allocation and copy for `d_part_b_meta`**

  ```cpp
  PartBMeta* d_part_b_meta = nullptr;
  if (!plan.part_b_meta.empty()) {
      CUDA_CHECK(cudaMalloc(&d_part_b_meta,
                            plan.part_b_meta.size() * sizeof(PartBMeta)));
      CUDA_CHECK(cudaMemcpy(d_part_b_meta, plan.part_b_meta.data(),
                            plan.part_b_meta.size() * sizeof(PartBMeta),
                            cudaMemcpyHostToDevice));
  }
  ```

- [ ] **Step 3: Add `d_part_b_meta` to the `kargs` struct**

  Find where `HybridKernelArgs kargs` is populated. Add:

  ```cpp
  kargs.part_b_meta = d_part_b_meta;
  ```

- [ ] **Step 4: Update the plan summary print**

  Find line ~304 in `test_hybrid.cu`:
  ```cpp
  fprintf(g_out, "**Plan:** tasks=%zu  a\\_contrib=%zu  max\\_smem=%d bytes\n\n",
          plan.tasks.size(), plan.a_contrib.size(), plan.max_smem);
  ```

  Update it to include b_contrib and part_b_meta counts (keep `g_out` and escaped underscores):
  ```cpp
  fprintf(g_out, "**Plan:** tasks=%zu  a\\_contrib=%zu  b\\_contrib=%zu"
                 "  part\\_b\\_meta=%zu  max\\_smem=%d bytes\n\n",
          plan.tasks.size(), plan.a_contrib.size(),
          plan.b_contrib.size(), plan.part_b_meta.size(),
          plan.max_smem);
  ```

- [ ] **Step 5: Add `cudaFree` for `d_part_b_meta`**

  In the cleanup block, add:

  ```cpp
  cudaFree(d_part_b_meta);
  ```

- [ ] **Step 6: Full compile**

  ```bash
  cd /Volumes/STORAGE/Yuchao/DiaGPU && nvcc -std=c++17 -arch=sm_90 test_hybrid.cu diag_hybrid_kernel.cu paper_hm_kernel.cu -o test_hybrid 2>&1 | head -30
  ```

  Expected: clean compile.

---

## Task 5: Verify correctness and commit

**Files:**
- Run: `test_hybrid` binary

- [ ] **Step 1: Run the correctness test**

  ```bash
  cd /Volumes/STORAGE/Yuchao/DiaGPU && ./test_hybrid 2>&1 | grep -E "(PASS|FAIL|error|max abs)"
  ```

  Expected: all test cases report max absolute error < 1e-4 (same threshold as before refactor).

- [ ] **Step 2: Verify smem stays bounded for large N**

  Add a temporary print in `build_hybrid_plan` (remove after check):

  ```cpp
  printf("group g_base=%d: n_parts=%d, part_b_meta entries=[%d..%d]\n",
         g_base, n_parts, part_b_base_g,
         (int)plan.part_b_meta.size() - 1);
  ```

  Run with the largest test case and confirm `n_parts` grows linearly with N while the printed `max_smem` stays ≤ 57 KB.

- [ ] **Step 3: Remove the temporary print**

- [ ] **Step 4: Commit**

  ```bash
  git add diag_hybrid_kernel.cuh diag_hybrid_kernel.cu test_hybrid.cu
  git commit -m "feat: split-K partition — O(1) smem per CTA regardless of N

  Replace all-B-upfront + A-batched design with true partition loop:
  each iteration loads HYBRID_PARTITION_SIZE A and their corresponding
  B diagonals, keeping smem bounded at ~57 KB for any matrix size.

  - Add HYBRID_PARTITION_SIZE=53, PartBMeta struct
  - build_hybrid_plan: sort A by d_a, precompute per-partition B metadata
  - hybrid_kernel: B load moved inside A-batch loop, uses PartBMeta

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```
