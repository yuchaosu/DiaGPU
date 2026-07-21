# Need-to-Run List — sequenced: kernels first, then end-to-end

Two phases by user decision: **(1) kernel comparison vs Drawloom + cuSPARSE
(+ Haque for SpMSpM), then (2) end-to-end.** Each item names the gap it closes
(`RESULTS_LEDGER.md`), the goal, a command grounded in the real harnesses, the
output artifact, and effort. Where a harness does not exist yet it says "BUILD".

**Verified entry points**
- SpMV: `spmv/bench/run_comparison.sh` → builds `bench_cusparse.cu` +
  `src/tc_spmv_regdirect_kernel.cu`; env `ARCH` (default `sm_90a`),
  `SIZES` (default `65536 262144 1048576`), `DRAWLOOM_BIN`, `MKLROOT`. Compares
  cuSPARSE vs **TC regdirect (ours)** vs **Drawloom** vs MKL on the same matrix.
- SpMV matrix gen: `gen_banded_mtx N out.mtx` (offset set hardcoded in the `.cpp`).
- SpMSpM: `bench_spmspm <n> <w> <iters>` — **`w` = #diagonals** (budget knob);
  baselines compiled in: **cuSPARSE + Haque (`paper_hm`)**. (No Drawloom: it has no SpMSpM.)

---

## Run 0 — reproducibility preamble (log with every result)
`nvidia-smi` (H100 + driver), `nvcc --version`, CUDA runtime; `DiaGPU` commit hash;
`ARCH=sm_90a`; generator params/seeds; exact command line. Keep raw CSVs immutable.

---

# PHASE 1 — Kernel comparison (do first)

### P1.1 — SpMV: DiaGPU vs cuSPARSE + Drawloom  ·  closes **G-DRAW-HEAD** · low effort
The harness already supports all three; just supply Drawloom's binary.
```bash
DRAWLOOM_BIN=/path/to/Drawloom/myfloat ARCH=sm_90a \
SIZES="65536 262144 1048576" bash spmv/bench/run_comparison.sh | tee results/spmv_kernel.csv
```
- Run on the **eval matrices** (c2h-10/16, b2-10/14) **and** spin (TFIM/Heisenberg)
  so the case study has SpMV evidence on its own Hamiltonians (see P1.4).
- If no Drawloom binary: run without it (cuSPARSE-only) and keep Drawloom as a
  cited baseline — but getting the binary is the point of this phase.
- **Output:** `results/spmv_kernel.csv` → SpMV speedup figure (replaces the
  reconstructed `kernel_speedups.pdf` flagged in the ledger).

### P1.2 — SpMSpM: DiaGPU vs cuSPARSE + Haque  ·  confirms **R-SPMSPM** · low effort
```bash
for N in 65536 262144 1048576; do bench_spmspm $N 16 50 | tee -a results/spmspm_kernel.csv; done
```
- Sweep `w` to the diagonal counts of the eval/spin operators.
- **Output:** `results/spmspm_kernel.csv` → SpMSpM speedup figure (cuSPARSE + Haque bars).

### P1.3 — diagonal-budget sweep (the "money figure")  ·  supports IDEA §6.1(B) · low effort
Speedup **grows as the budget tightens** — the plot only this paper can produce.
```bash
N=262144; for w in 2 4 8 16 32 64 128 256; do
  bench_spmspm $N $w 50 | tee -a results/budget_sweep_spmspm.csv
done
```
- SpMV version needs the generator's offset set varied (edit `gen_banded_mtx.cpp`).
- Strong form: drive `w` from **HamSim's `D_max`** on one real Hamiltonian so the
  x-axis is literally HamSim's budget; the pure-`w` loop is a synthetic proxy.
- **Output:** `results/budget_sweep_*.csv` → speedup-vs-#diagonals figure.

### P1.4 — spin-model kernel runs  ·  aligns speedup evidence with case study · med effort
Current speedup bars are on **molecular** matrices; the primary case study is
**spin chains**. Generate TFIM/Heisenberg operands and re-run P1.1/P1.2 on them.
- **Output:** `results/spin_spmv.csv`, `results/spin_spmspm.csv`.

### P1.5 — kernel ablations (optimization attribution)  ·  closes **G-ABLATION** · low effort · **CODE BUILT**
Base → all-optimizations for BOTH kernels, so the paper can attribute the speedup
to each lever (the reviewer asks "which optimization does the work?"). All levels
cross-checked **bit-exact** vs the full/reference kernel. Harnesses built + smoke-tested
this session (`sim/suite/hpc/smoke_hamlib.sh`, q8 = ALL PASS).

**TC SpMV ablation** — T0 full-TC → +symmetric(half recon) → +zero-tile-skip → +both(best TC):
```bash
: > results/tc_ablation.csv
for m in heis_16_5 heis_18_5 heis_20_6 tfim_18_5 qmaxcut_18_4 fermi_16_5 bh_18_5 O2_16_5; do
  $BIN/spmv_tc_ablation /mnt/beegfs/ysu34/hamlib/dia_e2e/$m.txt 200 --csv 2>/dev/null | grep '^TCABL,' >> results/tc_ablation.csv
done   # cols: file,n,D,Dsym,kept_ztile,kept_symztile,t0,t1_sym,t2_ztile,t3_both,s1,s2,s3,relerr
```

**SpMSpM ablation** — L0 HM(Haque) → +atomic-free(meta) → +flat/pairs → +adaptive-ILP → +symmetric:
```bash
: > results/spmspm_ablation.csv
for m in heis_14_5 heis_16_5 heis_18_5 tfim_16_5 tfim_18_5 qmaxcut_16_4 qmaxcut_18_4 fermi_14_5 fermi_16_5; do
  $BIN/spmspm_ablation /mnt/beegfs/ysu34/hamlib/dia_e2e/$m.txt 30 --csv 2>/dev/null | grep '^SPABL,' >> results/spmspm_ablation.csv
done   # cols: file,n,Hd,nnzC,L0_hm,L1_meta,L2_flat1,L3_flat4,L4_sym,s1,s2,s3,s4,relerr
```

One-shot build+run: **`bash sim/suite/hpc/run_ablation.sh`** (builds both harnesses sm_80, runs both sweeps into `results/`).
- Harness src: `sim/spmv_tc_ablation.cu` (+ `spmv/src/tc_spmv_sym.cuh`, `tc_spmv_ztile.cuh`), `sim/spmspm_ablation.cu` (+ `spmspm/gather_flat_sym.cuh`).
- **Measured (A100, `hamlib_bench/{tc,spmspm}_ablation.csv`):** TC best (sym+ztile) **2.5–6.1×** vs full-TC (O2 6.1×, bh_18 3.4×, heis_18 3.35×); SpMSpM +symmetric **3.3–4.8×** vs HM. All bit-exact.
- **Output:** `results/tc_ablation.csv`, `results/spmspm_ablation.csv` → two ablation-bar figures (per-optimization contribution).

> **Note — real-HamLib harnesses now exist (built this session), superseding the synthetic P1.1/P1.2 benches:**
> `sim/spmv_dia_vs_tc.cu` → ours(DIA zero-skip) vs TC / dense-CUDA / **cuSPARSE** / **Drawloom** on a real DIA `.txt`;
> `sim/spmspm_hamlib.cu` (`spmspm_pure`) → ours(gather_flat) vs **HM(Haque)** / **cuSPARSE SpGEMM**.
> The full sweep + both ablations + ncu-all is `sim/suite/hpc/bench_hamlib.slurm` → `/mnt/beegfs/ysu34/hamlib_bench/`.

---

# PHASE 2 — End-to-end (after kernels are locked)

### P2.1 — two-kernel Taylor pipeline  ·  closes **G-TAYLOR-E2E** · high effort
Build `U` once via SpMSpM (truncated Taylor / SpME), then apply via SpMV over many
steps. **BUILD** a driver chaining build→apply; time DiaGPU vs a cuSPARSE-backed
pipeline (same operator, same step count).
- **Output:** `results/taylor_e2e.csv` (build, per-step apply, total).

### P2.2 — QAOA end-to-end  ·  closes **G-QAOA** · low effort if data exists
Locate the QAOA CSV you ran, commit under `results/`, record baseline tool+version.
If no harness, BUILD a minimal QAOA driver on the SpMV path. Scope honestly: **SpMV path only**.
- **Output:** `results/qaoa_e2e.csv`.

### P2.3 — standard-tool baseline  ·  closes **G-BASELINE-TOOL** · med effort
e2e credibility needs a recognized simulator: cuQuantum/cuStateVec (closest, GPU),
QuTiP, or QuEST. Same sizes as P2.1/P2.2; add as baseline columns.

---

# DEFERRED — still required before submission (do not forget)

### D1 — DiaGPU vs HamSim-GPU  ·  closes **G-HAMSIM-GPU**  ·  🔴 top novelty risk
You deferred this to run kernels-vs-Drawloom/cuSPARSE first — fine as sequencing,
**but it cannot be dropped.** HamSim (same group, ICS'26) already has diagonal GPU
SpGEMM + SpMV + operator construction; beating cuSPARSE/Haque/Drawloom does not
answer "did you beat HamSim's own GPU kernels?" Wire a HamSim row into
`run_comparison.sh` (like `DRAWLOOM_BIN`). Schedule it right after Phase 1 while
the kernel harness is warm.

### D2 — breadth & scaling (reviewer pre-empts)
- **G-SCALE:** large-N SpMSpM to trigger split-K / Hopper DSMEM
  (`for N in 1048576 4194304 16777216; do bench_spmspm $N 64 20; done`).
- **G-FAMILIES:** broader HamLib family sweep + diagonal-prevalence histogram.
- **G-A100:** optional A100 repeat of P1.1/P1.2 for Drawloom's A100/H100 split.

---

## Ordering
Phase 1 (P1.1 → P1.2 → P1.3 → P1.4) → **D1 while harness is warm** → Phase 2
(P2.1 + P2.3, then P2.2) → D2.
