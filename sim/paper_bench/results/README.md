# Paper benchmark results (raw, immutable)

Raw measured data for the PPoPP evaluation. Do not edit in place; derived
tables/plots go elsewhere.

## A100 (NCSU cluster, c33, A100 80GB, sm_80, CUDA 13.2)

- `a100_all_266847.csv` — full kernel sweep, job 266847 (2026-07-27, code
  cfbae1b): SPMVCSV + SPMSPMCSV rows for all kernels/baselines incl.
  Drawloom rows; matrices = beegfs dia_e2e set (heis/tfim/fermi/BH/B2/BeH/
  O2/maxcut/qmaxcut/tsp, q8-24). Auto-plan picks in `a100_plan_266847.log`.
  KNOWN ISSUE superseded by 267616: auto picked the work list on O2_20
  (D-threshold was 4096).
- `a100_fix_267616.csv` — targeted SpMV rerun, job 267616 (code e6208e6,
  D-threshold 3072): O2_16/O2_20/qmaxcut rows; picks in
  `a100_fix_plan_267616.log`. O2_20 auto now selects streaming (gopt),
  2.850 ms vs work list 4.776 ms.
- `a100_e2e_267635.csv` — end-to-end, job 267635 (code e6208e6): E2EEVCSV
  (Taylor evolution, 1000 steps x K=6, hybrid fused NV=2 vs cuSPARSE
  identical pipeline, final states cross-checked) + E2EOPCSV (H^2/H^3
  pair-plan chain). O2_20 / BH_24 blocks absent: cuSPARSE-side host CSR
  setup died at that scale (per-matrix containment).
- `ncu/ncu_{spmv,spmspm}_<mat>_266847.csv` — mechanism counters (DRAM%,
  L1/L2 hit, SM%, occupancy, long-scoreboard, inst, op_atom) per kernel.
- `ncu/ncu_atomics_*_267616.csv` — op_red + op_atom counts: atomicAdd with
  unused return compiles to RED ops, so op_atom alone reads 0; HM scatter =
  1,847,674 (heis_16) / 6,200,758 (BeH_12) RED requests per product; ours
  and libdiaq = 0.

## H100 (external platform, gpu16, H100 PCIe, sm_90)

- `h100_spmv_sweep.csv` — SpMV kernel sweep (2026-07-27, code be80b2d,
  D-threshold then 4096): DIAMOND-dataset matrices (bh/fermi/heis/vib/
  maxcut/max3sat/maxkcut/qmaxcut/reg3/tfim/tsp, q8-24). Predates ours_auto
  rows and the SM-based gather dispatch: gather/gopt rows at n=65k-262k are
  starved (64-256 tiles vs 114 SMs); rerun planned with e6208e6+.

All timings: device-side CUDA-event medians, kernel-only (e2e rows time the
full evolution loop; plan/CSR builds reported separately). Every kernel
verified against an fp64 CPU reference within the producing driver.
