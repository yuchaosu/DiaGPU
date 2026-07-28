---
title: "One Storage, Plan-Selected Kernels: Hybrid Diagonal SpMV and SpMSpM for Fixed-Operator Iteration on GPUs"
venue: PPoPP (due 2026-08-01 AoE)
tags: [paper, ppopp, spmv, spmspm, hybrid, dia]
status: body-drafted, numbers measured, folding pass pending
date: 2026-07-27
---

# Paper structure — PPoPP (updated 2026-07-27)

Supersedes the 07-26 outline. The frame moved from "instruction-bound + plan-based kernels" to its concrete result: the **hybrid** — one packed-diagonal storage, multiple plan encodings, selection at build time by measured crossovers. Draft state: Intro + Background frozen (user-owned); Design/Evaluation/Related/Conclusion rewritten in ppopp2027/*.tex; abstract deliberately last.

## Thesis (as written in the draft)

> In diagonal form both primitives are sums of shifted vector–vector products, and the operator is fixed for the length of a run, so every irregular decision is made once, on the host, in a plan. The open question — how to encode interior zeros — has no single answer but a computable one: per-nonzero work lists and coarse-segmented streaming are complementary Pareto encodings separated by a byte-crossover law (fill vs 0.8ρ) in quantities known at plan time. The hybrid selects with three comparisons; plan once, stream always.

## The hybrid (all measured)

| plan | mapping | selected when | evidence |
|---|---|---|---|
| work list (didx) | thr→row, 1–2B dictionary index | interior-sparse, moderate D | fastest on every such operator, 6.1–13.5× vs cuSPARSE (H100), 5.5× fewer instructions than dense row kernel (ncu, BH_20) |
| streaming (+coarse zero-preproc, tile lists, vec4) | blk→tile, 4 consecutive lanes | D≤4 / fill≥0.8ρ / D>3072 / chain outputs | wins D=1 (launch-bound), vib_18 1.27×, O2_20 1.68×, within 10% at bh_24 where all public baselines fail |
| pair plan (gather_flat) | blk→slice of output diagonal | SpMSpM (single plan; host build per product, device rebuild for chains) | 1.25–2.3× vs HM scatter, 2.3–162× vs libdiaq fp32, 0 vs 1.8–6.2M RED atomics |

D-threshold 3072: bracketed by four measurements on two GPUs (work list wins D=1243, 2859; streaming wins D=3359, 5367); stated in the paper as an empirically bracketed constant with the offset-table-occupancy mechanism. Known borderline: qmaxcut fill=0.50, gopt +11–20%, selector stays with work list — reported, not hidden.

## Evidence status (ledger: sim/paper_bench/results/, commit 86631a3)

| claim | value | source | status |
|---|---|---|---|
| SpMV vs cuSPARSE | 6.1–13.5× (wl), 2.1–6.7× (str) | h100_spmv_sweep + a100_all_266847 | HAVE |
| SpMV vs libdiaq fp32/fp64 | 1.6–9.4× / up to 18× | both sweeps | HAVE |
| SpMV vs Drawloom | they win ~1.1–1.35× at q≤14; we win 2–7× at q≥16; they fail >q20 | a100_all_266847 (50 rows) | HAVE (A100; H100 \ph) |
| hybrid auto = best plan | ~90% of rows; O2_20 fixed by 3072 | a100_fix_267616 | HAVE |
| SpMSpM vs HM / libdiaq / cuSPARSE | 1.25–2.3× / 2.3–162× / 27–1524× (context only) | a100_all_266847 | HAVE (A100; H100 \ph) |
| atomics | 0 vs 1,847,674 (heis_16), 6,200,758 (BeH_12) RED/product; libdiaq also 0 | ncu_atomics_267616 | HAVE |
| mechanism | didx: 12.1M vs 66.8M inst, L1 72.7%, DRAM 44.8%; diaq 10.8× inst, 82.8% DRAM; vec4 −3.2× inst | ncu_*_266847 | HAVE |
| e2e evolution 1000×6 | 3.3–9.3× vs identical cuSPARSE pipeline, 54 matrices, ≤8e-7 agreement, q→24 | a100_e2e_267635 | HAVE |
| plan amortization | plan build ≈ cuSPARSE CSR setup; break-even ≤ ~70/1000 steps | a100_e2e_267635 | HAVE |
| losing encodings | bitmap 2×↓, pattern 1.4–3.6×↓ (100% coverage on spin!), fine segments 200×↓ | archive/spmv/spmv_gather benches (4060 Ti) | HAVE (dev card; flag platform or rerun) |
| budget curves | molecular 26%→0.99 vs spin cliff | prior evolve_budget assets | CARRY-OVER (verify provenance before final) |
| H100 v2 (auto rows, SpMSpM, e2e, Drawloom-H100, ncu) | — | run_all.sh on gpu16 | PENDING (one command) |

## Baseline policy (locked 2026-07-27)

Baselines = public artifacts only: cuSPARSE, Drawloom, libdiaq/HamSim (fp32 same-precision + fp64 published config, device-resident path only), HM/Haque scatter. Our classical-DIA reimplementation (cite Bell–Garland) and CSR-encoded work list appear ONLY as ablations (layout-vs-plan split: layout alone = 1.9–4×; dictionary vs CSR encoding = up to 1.7× at D≤256). SpMV table normalized to cuSPARSE with absolute ms column.

## Section state (ppopp2027/)

| file | state |
|---|---|
| Introduction.tex, Background.tex | FROZEN (user). Pending user touches: C1 Hermitian "halve operand traffic" → footprint only; \TODO speedups → "up to 13× cuSPARSE / 9× libdiaq / e2e 3.3–9.3×"; "hybrid SpMV kernel" not "hybrid kernels" |
| Design.tex | DONE: storage → diagonal form → 3 plans (streaming w/ coarse-split rank-map argument; work-list dictionary + Alg.1; hybrid + Eq. crossover + fig hybrid_plan.pdf) → SpMSpM (pair plan, Alg.2, chain) → e2e (fused NV=2 at measured 9–45%, amortization, chain crosses fill boundary). TC-free (verified 0 mentions) |
| Evaluation.tex | DRAFTED, folding pass pending: SpMV table renormalized to cuSPARSE (done); remaining \ph = SpMSpM table, mechanism table, e2e numbers, coverage grid, Drawloom-H100, budget |
| related.tex | DONE: HamSim measured head-to-head + pair-list credit; format family via bytes-not-bottleneck; no TC-engine claims |
| Conclusion.tex | DONE (hybrid framing, 13×/9× headline) |
| abstract.tex | LAST (user) |
| figures | spmv_design/spmspm_design (existing, math+CUDA panels) + hybrid_plan.pdf (matplotlib, crossover + plan diagram); results figures TBD from CSVs |

## Hard don'ts (unchanged + additions)

- No TC content anywhere in Design (verified); Drawloom = measured baseline row only, no mechanism discussion.
- Never headline cuSPARSE SpGEMM ratios (context only); never time libdiaq's per-call-malloc API; fp64 columns labeled "published config".
- Atomic claim worded vs the SCATTER baseline (libdiaq is also atomic-free).
- Evolution loop = "Krylov-type polynomial evolution (S=1000, K=6, truncated-Taylor coefficients)"; operator chain = "Taylor"; keep Background Table 1 mapping consistent.
- Budget adopted from HamSim, cited, never claimed.
- Raw data immutable in sim/paper_bench/results/; every paper number traces to a file+commit there.

## Remaining work, ordered (due 8/1 AoE)

1. H100 v2 run (user: git pull; DATA=… bash run_all.sh) — fills last \ph family.
2. Folding pass over Evaluation.tex from results/ CSVs (SpMSpM + mechanism + e2e tables, coverage grid, suite table fills).
3. Results figures (per-family bars, e2e bars, gap-vs-size) from CSVs via matplotlib.
4. User: intro touches, abstract, first Overleaf compile of new body (check: 7-col SpMV table width; hybrid_plan.pdf path; \ph→final recolor).
5. Optional: O2_20/BH_24 e2e rows with --no-cusparse; negative-encoding rerun on A100 for platform consistency; ledger sync in ppopp2027/RESULTS_LEDGER.md.
