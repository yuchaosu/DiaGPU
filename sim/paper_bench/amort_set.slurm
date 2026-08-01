#!/bin/bash
#SBATCH -J amort_set
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 6:00:00
#SBATCH -o logs/amort_set_%j.out
# Completes the H100 amortization corpus with the mainline-sym decision:
#  - SpMSpM amort: finish flat/cusparse for matrices the killed local sweep
#    missed, and add ours_sym rows everywhere symmetric (SYM_ONLY passes)
#  - SpMV amort: add didx_sym rows (AMORT_SYM_ONLY; flat didx/cusparse/diaq
#    rows are complete from the local 375-row sweep)
set -u
export PATH=/usr/local/cuda/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
PB=/home/ysu34/DiaGPU/sim/paper_bench; B=/mnt/beegfs/ysu34/paper_bench
cd $PB
AS=$B/amort_spmspm_h100.csv; AV=$B/amort_h100.csv
for f in $(cat matrices_paper.txt); do
  m=$(basename "$f" .txt)
  if ! grep -q "/$m.txt,cusparse,100," $AS; then
    timeout 2400 stdbuf -oL ./amort_spmspm_bench_v2 "$f" | tee -a $AS || echo "AMORTFAIL,$m" | tee -a $AS
  elif ! grep -q "/$m.txt,ours_sym," $AS; then
    SYM_ONLY=1 timeout 2400 stdbuf -oL ./amort_spmspm_bench_v2 "$f" | tee -a $AS
  fi
  grep -q "/$m.txt,didx_sym," $AV || \
    AMORT_SYM_ONLY=1 timeout 2400 stdbuf -oL ./amort_bench_v2 "$f" | tee -a $AV
done
echo "== AMORT SET DONE: spmspm=$(grep -c AMORTSPMSPM $AS) spmv=$(grep -c AMORTCSV $AV)"
