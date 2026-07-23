#!/bin/bash
# ===========================================================================
# run_all.sh — MASTER driver: build + run the ENTIRE paper evaluation (6.2-6.8)
# into results/, plus a meta.txt provenance manifest. One command reproduces
# every experiment. Kernels live in spmv/src + spmspm (clean); harnesses in sim/;
# this only orchestrates.
#
#   6.2 SpMV compare     (run_plan.sh)   -> spmv_kernel.csv
#   6.3 SpMSpM compare   (run_plan.sh)   -> spmspm_kernel.csv
#   6.5 ablations        (run_plan.sh)   -> tc_ablation.csv, spmspm_ablation.csv
#   6.6 scalability      (run_plan.sh)   -> budget_sweep.csv (derived)
#   6.7 e2e + anchor     (run_e2e.sh)    -> e2e_speedup.csv, qutip_anchor.csv
#   6.8 sparsification   (run_sparsity.sh) -> spmv_trim, evolve_trim, evolve_budget, spmspm_trim
#   6.4 ncu mechanism    (run_ncu.sh)    -> ncu_profile.csv
#
#   usage: bash run_all.sh
#          ARCH=sm_90a bash run_all.sh                    # H100
#          OUT=/path DIA=/path STEPS=1000 bash run_all.sh
#          FAST=1 bash run_all.sh                         # STEPS=100, subset ncu (quick check)
#          NCU=0 bash run_all.sh                          # skip 6.4 ncu (no profiler available)
#
#   NCU:  1 (default) run ncu profiling if `ncu` is on PATH (auto-skips if not);
#         0           skip 6.4 entirely (use where the profiler is unavailable).
# ===========================================================================
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
ARCH=${ARCH:-sm_80}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
OUT=${OUT:-$REPO/results}
NCU=${NCU:-1}
export REPO ARCH DIA OUT
if [ "${FAST:-0}" = 1 ]; then export STEPS=${STEPS:-100}; fi
mkdir -p "$OUT"
HPC=$REPO/sim/suite/hpc

echo "############################################################"
echo "#  DiaGPU — full evaluation (6.2-6.8)"
echo "#  node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "#  arch=$ARCH  dia=$DIA  out=$OUT"
echo "############################################################"

# ---- provenance manifest ----
{ echo "date_utc: $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "host: $(hostname)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1)"
  echo "git_commit: $(cd $REPO && git rev-parse HEAD 2>/dev/null || echo NA)"
  echo "git_branch: $(cd $REPO && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo NA)"
  echo "arch: $ARCH"
  echo "dia_dir: $DIA"
  echo "steps: ${STEPS:-1000}"
  echo "cuda: $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1)"
} > $OUT/meta.txt
echo "== wrote $OUT/meta.txt =="

# ---- 6.2/6.3/6.5/6.6/6.7/6.8 (run_plan chains these) ----
echo; echo "===== PHASE A: kernels + ablations + e2e + sparsification (run_plan.sh) ====="
bash $HPC/run_plan.sh || echo "  (run_plan.sh reported a non-zero exit; continuing)"

# ---- kernels under the fidelity-preserving diagonal budget (SpMV + SpMSpM) ----
echo; echo "===== PHASE A2: kernels under fidelity->=0.99 diagonal budget (run_budget.sh) ====="
bash $HPC/run_budget.sh || echo "  (run_budget.sh non-zero exit; continuing)"

# ---- 6.4 ncu mechanism profiling (gated by NCU=1; run_ncu also self-skips if absent) ----
echo; echo "===== PHASE B: ncu mechanism profiling (run_ncu.sh) ====="
if [ "$NCU" = 1 ]; then
  bash $HPC/run_ncu.sh || echo "  (run_ncu.sh reported a non-zero exit; ncu optional)"
else
  echo "  NCU=0 -> skipping 6.4 ncu profiling"
fi

# ---- manifest ----
echo; echo "############################################################"
echo "#  DONE — all evaluation CSVs in $OUT:"
for f in spmv_kernel spmspm_kernel budget_sweep tc_ablation spmspm_ablation \
         e2e_speedup qutip_anchor spmv_trim evolve_trim evolve_budget spmspm_trim \
         budget_summary spmv_budget spmspm_budget ncu_profile; do
  [ -f $OUT/$f.csv ] && printf "#    %-22s %s rows\n" "$f.csv" "$(( $(wc -l < $OUT/$f.csv) - 1 ))"
done
echo "#  provenance: $OUT/meta.txt"
echo "############################################################"
