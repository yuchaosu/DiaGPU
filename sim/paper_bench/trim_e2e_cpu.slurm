#!/bin/bash
# trim_e2e_cpu.slurm — f64 end-to-end trim accuracy for the clean molecular
# rebuilds (tau=1e-6 vs 1e-8 vs untrimmed; CPU-only, no GPU needed).
# The three matrices run in parallel (independent processes).
#SBATCH -J trim_e2e_cpu
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 20:00:00
#SBATCH -o logs/trim_e2e_cpu_%j.out
set -u
PB=/home/ysu34/DiaGPU/sim/paper_bench
D=/mnt/beegfs/ysu34/hamlib/dia_e2e
B=/mnt/beegfs/ysu34/paper_bench
cd "$PB" || exit 2
python3 molecular_trim_e2e.py $D/O2c_16_5.txt > $B/trim_e2e_O2c.log 2>&1 &
python3 molecular_trim_e2e.py $D/H8c_16_4.txt > $B/trim_e2e_H8c.log 2>&1 &
python3 molecular_trim_e2e.py $D/C2c_18_4.txt > $B/trim_e2e_C2c.log 2>&1 &
wait
cat $B/trim_e2e_O2c.log $B/trim_e2e_H8c.log $B/trim_e2e_C2c.log > $B/trim_e2e_molecular_f64.log
echo "== TRIM E2E CPU DONE $(date -Is)"
