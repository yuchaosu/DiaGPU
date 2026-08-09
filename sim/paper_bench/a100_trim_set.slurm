#!/bin/bash
# a100_trim_set.slurm — ABLATION TIER A0 on A100: DIA_TRIM=1e-8, OUR kernels
# only (OURS_ONLY=1; baselines never touch trimmed data — their official
# numbers are the untrimmed a100_paperset corpus from job 269665).
# Chain: sbatch -d afterany:<a100_paper_set> a100_trim_set.slurm
# Rebuilds bin_sm80_trim from current sources so the env gates are guaranteed
# present (stale gate-less binaries would silently run baselines).
#SBATCH -J a100_trim_set
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 30:00:00
#SBATCH -o logs/a100_trim_set_%j.out
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export DIA_TRIM=1e-8
export OURS_ONLY=1
REPO=/home/ysu34/DiaGPU
PB=$REPO/sim/paper_bench
BIN=$PB/bin_sm80_trim
B=/mnt/beegfs/ysu34/paper_bench
LIST=$PB/matrices_paper.txt
ITERS=${ITERS:-50}; SPITERS=${SPITERS:-100}; STEPS=${STEPS:-1000}; REPS=${REPS:-3}
mkdir -p "$BIN"
cd "$PB" || exit 2
echo "host=$(hostname) $(date -Is) DIA_TRIM=$DIA_TRIM OURS_ONLY=$OURS_ONLY reps=$REPS"

echo "== build (sm_80 -> $BIN) =="
NV="nvcc -O3 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr"
$NV prep_driver.cu   "$REPO/spmspm/paper_hm_kernel.cu" -o "$BIN/prep_driver"   -lcusparse -lnvidia-ml && \
$NV spmspm_driver.cu "$REPO/spmspm/paper_hm_kernel.cu" -o "$BIN/spmspm_driver" -lcusparse -lnvidia-ml -Xcompiler -fopenmp && \
$NV e2e_driver.cu  -o "$BIN/e2e_driver"  -lcusparse -lnvidia-ml && \
$NV cfill_bench.cu -o "$BIN/cfill_bench" -lnvidia-ml && \
$NV amort_bench.cu -o "$BIN/amort_bench" -lcusparse -lnvidia-ml && \
$NV amort_spmspm_bench.cu -o "$BIN/amort_spmspm_bench" -lcusparse -lnvidia-ml || { echo build_fail; exit 3; }

for rep in $(seq 1 $REPS); do
for f in $(cat "$LIST"); do
  m=$(basename "$f" .txt)
  case "$f" in */dia_e2e/*) SYM="--sym";; *) SYM="";; esac
  echo "== [trim $rep] $m =="
  timeout 3600 stdbuf -oL "$BIN/prep_driver" "$f" "$ITERS"     | tee -a $B/prep_a100_trim.csv
  timeout 3600 stdbuf -oL "$BIN/spmspm_driver" "$f" "$SPITERS" | tee -a $B/spmspm_a100_trim.csv
  timeout 1200 stdbuf -oL "$BIN/cfill_bench" "$f"              | tee -a $B/cfill_a100_trim.csv
  case "$f" in */dia_e2e/*)
    timeout 5400 stdbuf -oL "$BIN/e2e_driver" "$f" "$STEPS" $SYM --no-cusparse --no-diaq | tee -a $B/e2e_a100_trim.csv ;;
  esac
done
done

echo "== amortization (trim, ours-only) =="
for f in $(cat "$LIST"); do
  timeout 2400 stdbuf -oL "$BIN/amort_bench" "$f"        | tee -a $B/amort_a100_trim.csv        || true
  timeout 2400 stdbuf -oL "$BIN/amort_spmspm_bench" "$f" | tee -a $B/amort_spmspm_a100_trim.csv || true
done
echo "== energy (trim, ours-only) =="
for f in $(cat "$LIST"); do
  m=$(basename "$f" .txt); echo "== energy $m ==" | tee -a $B/energy_a100_trim.csv
  ENERGY=6 timeout 3600 stdbuf -oL "$BIN/prep_driver"   "$f" 20 | tee -a $B/energy_a100_trim.csv
  ENERGY=6 timeout 3600 stdbuf -oL "$BIN/spmspm_driver" "$f" 20 | tee -a $B/energy_a100_trim.csv
done
echo "== A100 TRIM SET DONE $(date -Is)"
