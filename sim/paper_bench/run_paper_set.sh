#!/bin/bash
# ===========================================================================
# run_paper_set.sh — THE standard paper run: matrices_paper.txt ONLY
# (q16-20 HamLib + SuiteSparse ecology2/atmosmodd/Lin; user decision
# 2026-08-01: nothing else gets run).  Protocol: SpMV = didx single scheme,
# diaq fp32 only, stream/csr_zskip off (defaults of the trimmed drivers),
# honest true-nonzero cuSPARSE, HamLib e2e --sym / SS full.
# Sequential (exclusive GPU).  Appends to the standard CSV corpus; use
# REPS=3 for the median policy once adopted.
#   bash run_paper_set.sh            (or sbatch: works unmodified on h100)
# ===========================================================================
#SBATCH -J paper_set
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/paper_set_%j.out
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
REPO=/home/ysu34/DiaGPU
PB=$REPO/sim/paper_bench
B=/mnt/beegfs/ysu34/paper_bench
LIST=$PB/matrices_paper.txt
ITERS=${ITERS:-50}
SPITERS=${SPITERS:-100}
STEPS=${STEPS:-1000}
REPS=${REPS:-1}
cd "$PB" || exit 2
echo "host=$(hostname) $(date -Is) reps=$REPS"

for rep in $(seq 1 $REPS); do
for f in $(cat "$LIST"); do
  m=$(basename "$f" .txt)
  case "$f" in */dia_e2e/*) SYM="--sym";; *) SYM="";; esac
  echo "== [$rep] $m =="
  timeout 2400 stdbuf -oL ./prep_driver "$f" "$ITERS"        | tee -a $B/prep_h100_c29patch.csv
  timeout 2400 stdbuf -oL ./spmspm_driver "$f" "$SPITERS"    | tee -a $B/spmspm_cusparse_fix_h100.csv
  timeout 1200 stdbuf -oL ./cusparse_fix_bench "$f" "$ITERS" | tee -a $B/cusparse_fix_h100.csv
  timeout 1200 stdbuf -oL ./cfill_bench "$f"                 | tee -a $B/cfill_h100.csv
  case "$f" in */dia_e2e/*)   # e2e is HamLib-only (user decision: SS skips e2e)
    timeout 3600 stdbuf -oL ./e2e_driver_sm_90a "$f" "$STEPS" $SYM | tee -a $B/e2e_diaq_h100_c29patch.csv ;;
  esac
done
done
python3 make_paper_tables.py
echo "== PAPER SET DONE $(date -Is)"
