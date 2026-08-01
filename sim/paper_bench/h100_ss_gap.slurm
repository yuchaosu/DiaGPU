#!/bin/bash
#SBATCH -J h100_ss_gap
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -o logs/h100_ss_gap_%j.out
# Drawloom + diaq SpMV timing for the NEW SS matrices on H100 (old sweep
# predates them), REPS=3 -> spmv_sweep_extra_h100.csv
set -u
export PATH=/usr/local/cuda/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
PB=/home/ysu34/DiaGPU/sim/paper_bench; B=/mnt/beegfs/ysu34/paper_bench
DRAW=/mnt/beegfs/ysu34/drawloom_ae_h100/Drawloom_code/myfloat
D2M=/home/ysu34/DiaGPU/sim/suite/bin/dia_to_mtx
cd $PB
for m in ecology2 atmosmodd Lin; do
  f=/mnt/beegfs/ysu34/hamlib/dia_suitesparse/$m.txt
  for rep in 1 2 3; do
    timeout 1800 stdbuf -oL ./spmv_driver "$f" 50 | tee -a $B/spmv_sweep_extra_h100.csv
  done
  mtx=$B/_${m}_dl.mtx
  "$D2M" "$f" "$mtx" >/dev/null 2>&1 || continue
  for rep in 1 2 3; do
    ms=$(OMP_NUM_THREADS=16 timeout 1800 "$DRAW" -filename "$mtx" 2>/dev/null | sed -n 's/.*drawloom time:\s*\([0-9.]*\)\s*ms.*/\1/p' | head -1)
    [ -n "$ms" ] && echo "SPMVCSV,$f,na,na,na,na,drawloom,$ms,na" | tee -a $B/spmv_sweep_extra_h100.csv
  done
  rm -f "$mtx"
done
echo DONE
