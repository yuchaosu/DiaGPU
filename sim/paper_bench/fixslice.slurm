#!/bin/bash
# fixslice.slurm — full-protocol rerun for the CLEAN molecular rebuilds
# (O2c/C2c/H8c: e-notation corruption fix 2026-08-01). List built at runtime
# from whatever clean files exist; missing ones are skipped, and any name not
# yet in matrices_paper.txt is appended so the tables and the pending A100
# job (269665) pick them up.
#SBATCH -J fixslice
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o logs/fixslice_%j.out
set -u
PB=/home/ysu34/DiaGPU/sim/paper_bench
cd "$PB" || exit 2
# safety net: regenerate O2c_20 if the interactive session died before it landed
if ! ls /mnt/beegfs/ysu34/hamlib/dia_e2e/O2c_20_*.txt >/dev/null 2>&1; then
  ( cd /mnt/beegfs/ysu34/hamlib && \
    HAMLIB_TAG=O2c timeout 14400 python3.11 fetchham_sparse_dia.py O2/O2.hdf5 ham_JW20 1.2 1000 dia_e2e )
fi
ls /mnt/beegfs/ysu34/hamlib/dia_e2e/O2c_*.txt \
   /mnt/beegfs/ysu34/hamlib/dia_e2e/C2c_*.txt \
   /mnt/beegfs/ysu34/hamlib/dia_e2e/H8c_*.txt \
   /mnt/beegfs/ysu34/hamlib/dia_e2e/bhc_*.txt 2>/dev/null > matrices_fixslice.txt
[ -s matrices_fixslice.txt ] || { echo "no clean files found"; exit 3; }
while read -r f; do
  grep -qxF "$f" matrices_paper.txt || echo "$f" >> matrices_paper.txt
done < matrices_fixslice.txt
echo "fixslice list:"; cat matrices_fixslice.txt
ADD=$PB/matrices_fixslice.txt bash run_additions.sh
