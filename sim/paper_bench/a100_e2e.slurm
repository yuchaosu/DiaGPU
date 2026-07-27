#!/bin/bash
# ===========================================================================
# a100_e2e.slurm — end-to-end evaluation on the A100 (c33):
# Taylor evolution (hybrid SpMV, fused NV=2) vs cuSPARSE identical pipeline,
# 1000 steps x K=6 (the paper workload, ~6000 applies), plus the H^2/H^3
# pair-plan operator chain.  One E2EEVCSV + E2EOPCSV block per matrix;
# plan/CSR build times included (the amortization table).
# submit: cd sim/paper_bench && sbatch a100_e2e.slurm
# ===========================================================================
#SBATCH -J a100_e2e
#SBATCH -p a100
#SBATCH -w c33
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH -o logs/a100_e2e_%j.out
set -u
REPO=/home/ysu34/DiaGPU
cd "$REPO/sim/paper_bench" || exit 2

rm -f e2e_driver
make e2e_driver ARCH=sm_80 || exit 3

DATA=/mnt/beegfs/ysu34/hamlib/dia_e2e
OUTDIR=/mnt/beegfs/ysu34/paper_bench
OUT="$OUTDIR/a100_e2e_${SLURM_JOB_ID}.csv"
mkdir -p "$OUTDIR"

for f in "$DATA"/*.txt; do
  case "$f" in *imag*) continue;; esac
  ./e2e_driver "$f" 1000 6 | tee -a "$OUT"
done

echo "done -> $OUT"
