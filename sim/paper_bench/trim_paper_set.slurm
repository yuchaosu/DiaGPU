#!/bin/bash
# ===========================================================================
# trim_paper_set.slurm — ABLATION TIER A0: the full paper protocol rerun with
# the uniform load-time trim DIA_TRIM=1e-8 (2026-08-01 decision: threshold
# from the clean-file sweep; matrices with nothing below tau load identically,
# so no family dispatch is needed and untrimmed-clean rows are reproduced).
# Baselines see the SAME trimmed operator (dia_to_csr keeps true nonzeros).
# Appends to a PARALLEL *_trim CSV corpus — never mixes with the untrimmed
# medians. Chain after fixslice (needs the O2c_20 safety net to have run).
#   sbatch -d afterany:<fixslice> trim_paper_set.slurm
# ===========================================================================
#SBATCH -J trim_paper_set
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/trim_paper_set_%j.out
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export DIA_TRIM=1e-8
export OURS_ONLY=1   # baselines are NOT run on trimmed data (user decision);
                     # their official numbers live in the untrimmed corpus
REPO=/home/ysu34/DiaGPU
PB=$REPO/sim/paper_bench
B=/mnt/beegfs/ysu34/paper_bench
LIST=$PB/matrices_paper.txt
ITERS=${ITERS:-50}; SPITERS=${SPITERS:-100}; STEPS=${STEPS:-1000}; REPS=${REPS:-3}
DRAW=/mnt/beegfs/ysu34/drawloom_ae_h100/Drawloom_code/myfloat
D2M=$REPO/sim/suite/bin/dia_to_mtx
cd "$PB" || exit 2
echo "host=$(hostname) $(date -Is) DIA_TRIM=$DIA_TRIM reps=$REPS"

echo "== rebuild (dia_io trim hook) =="
NV="nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr"
$NV prep_driver.cu   "$REPO/spmspm/paper_hm_kernel.cu" -o prep_driver   -lcusparse -lnvidia-ml && \
$NV spmspm_driver.cu "$REPO/spmspm/paper_hm_kernel.cu" -o spmspm_driver -lcusparse -lnvidia-ml -Xcompiler -fopenmp && \
$NV e2e_driver.cu  -o e2e_driver_sm_90a -lcusparse -lnvidia-ml && \
$NV cfill_bench.cu -o cfill_bench -lnvidia-ml && \
$NV amort_bench.cu -o amort_bench_v2 -lcusparse -lnvidia-ml && \
$NV amort_spmspm_bench.cu -o amort_spmspm_bench_v2 -lcusparse -lnvidia-ml || { echo build_fail; exit 3; }

for rep in $(seq 1 $REPS); do
for f in $(cat "$LIST"); do
  m=$(basename "$f" .txt)
  case "$f" in */dia_e2e/*) SYM="--sym";; *) SYM="";; esac
  echo "== [trim $rep] $m =="
  timeout 2400 stdbuf -oL ./prep_driver "$f" "$ITERS"        | tee -a $B/prep_h100_trim.csv
  timeout 2400 stdbuf -oL ./spmspm_driver "$f" "$SPITERS"    | tee -a $B/spmspm_h100_trim.csv
  timeout 1200 stdbuf -oL ./cfill_bench "$f"                 | tee -a $B/cfill_h100_trim.csv
  case "$f" in */dia_e2e/*)
    timeout 3600 stdbuf -oL ./e2e_driver_sm_90a "$f" "$STEPS" $SYM --no-cusparse --no-diaq | tee -a $B/e2e_h100_trim.csv ;;
  esac
done
done

echo "== amortization (trim) =="
for f in $(cat "$LIST"); do
  timeout 2400 stdbuf -oL ./amort_bench_v2 "$f"        | tee -a $B/amort_h100_trim.csv        || true
  timeout 2400 stdbuf -oL ./amort_spmspm_bench_v2 "$f" | tee -a $B/amort_spmspm_h100_trim.csv || true
done
echo "== energy (trim) =="
for f in $(cat "$LIST"); do
  m=$(basename "$f" .txt); echo "== energy $m ==" | tee -a $B/energy_h100_trim.csv
  ENERGY=6 timeout 3600 stdbuf -oL ./prep_driver   "$f" 20 | tee -a $B/energy_h100_trim.csv
  ENERGY=6 timeout 3600 stdbuf -oL ./spmspm_driver "$f" 20 | tee -a $B/energy_h100_trim.csv
done
echo "== TRIM PAPER SET DONE $(date -Is)"
