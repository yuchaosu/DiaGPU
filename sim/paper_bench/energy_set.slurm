#!/bin/bash
#SBATCH -J energy_set
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH -o logs/energy_set_%j.out
# ENERGY=1: every tms() call loops its kernel ~6s with NVML sampling and
# emits ENERGYRAW before the variant's CSV row (pair in post-processing).
set -u
export PATH=/usr/local/cuda/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
REPO=/home/ysu34/DiaGPU; PB=$REPO/sim/paper_bench; BIN=$PB/bin_energy; B=/mnt/beegfs/ysu34/paper_bench
mkdir -p "$BIN"; cd "$PB"
NV="nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr"
$NV prep_driver.cu   "$REPO/spmspm/paper_hm_kernel.cu" -o "$BIN/prep_driver"   -lcusparse -lnvidia-ml || exit 3
$NV spmspm_driver.cu "$REPO/spmspm/paper_hm_kernel.cu" -o "$BIN/spmspm_driver" -lcusparse -lnvidia-ml -Xcompiler -fopenmp || exit 3
E=$B/energy_h100.csv
[ -f $E ] || echo "# $(date -Is) ENERGY=1 J/apply (net of idle): ENERGYRAW,J_net,avg_W,idle_W,iters then the variant row it belongs to" > $E
for f in $(cat $PB/matrices_paper.txt); do
  m=$(basename "$f" .txt); echo "== energy $m ==" | tee -a $E
  ENERGY=6 timeout 3600 stdbuf -oL "$BIN/prep_driver"   "$f" 20 | tee -a $E
  ENERGY=6 timeout 3600 stdbuf -oL "$BIN/spmspm_driver" "$f" 20 | tee -a $E
done
echo "== ENERGY DONE $(date -Is)"
