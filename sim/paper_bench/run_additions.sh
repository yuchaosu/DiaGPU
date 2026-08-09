#!/bin/bash
# ===========================================================================
# run_additions.sh — full-protocol run for the 14 q16-20 HamLib additions
# (matrices_additions.txt; user decision 2026-08-01: D=1 candidates excluded,
# single-diagonal structure is already covered by maxcut/tsp in the main set).
# Stages mirror the main corpus: core pipeline (REPS=3, via run_paper_set.sh)
# -> spmv sweep + Drawloom -> amortization -> energy -> tables.
#   bash run_additions.sh          (H100 c29; appends to the standard CSVs)
# ===========================================================================
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
PB=/home/ysu34/DiaGPU/sim/paper_bench
B=/mnt/beegfs/ysu34/paper_bench
ADD=${ADD:-$PB/matrices_additions.txt}
DRAW=/mnt/beegfs/ysu34/drawloom_ae_h100/Drawloom_code/myfloat
D2M=/home/ysu34/DiaGPU/sim/suite/bin/dia_to_mtx
cd "$PB" || exit 2
echo "== ADDITIONS start $(date -Is) host=$(hostname)"

echo "== stage 1: core pipeline (REPS=3)"
LIST=$ADD REPS=3 bash run_paper_set.sh

echo "== stage 2: spmv sweep + drawloom"
for rep in 1 2 3; do
for f in $(cat "$ADD"); do
  timeout 1800 stdbuf -oL ./spmv_driver "$f" 50 | tee -a $B/spmv_sweep_extra_h100.csv
done
done
if [ -x "$DRAW" ] && [ -x "$D2M" ]; then
  for f in $(cat "$ADD"); do
    m=$(basename "$f" .txt); mtx=$B/_${m}_dl.mtx
    "$D2M" "$f" "$mtx" >/dev/null 2>&1 || { echo "SKIP drawloom $m"; continue; }
    for rep in 1 2 3; do
      ms=$(OMP_NUM_THREADS=16 timeout 1800 "$DRAW" -filename "$mtx" 2>/dev/null | sed -n 's/.*drawloom time:\s*\([0-9.]*\)\s*ms.*/\1/p' | head -1)
      [ -n "$ms" ] && echo "SPMVCSV,$f,na,na,na,na,drawloom,$ms,na" | tee -a $B/spmv_sweep_extra_h100.csv
    done
    rm -f "$mtx"
  done
fi

echo "== stage 3: amortization"
for f in $(cat "$ADD"); do
  timeout 2400 stdbuf -oL ./amort_bench_v2 "$f"        | tee -a $B/amort_h100.csv        || echo "AMORTFAIL,$(basename $f)" | tee -a $B/amort_h100.csv
  timeout 2400 stdbuf -oL ./amort_spmspm_bench_v2 "$f" | tee -a $B/amort_spmspm_h100.csv || echo "AMORTFAIL,$(basename $f)" | tee -a $B/amort_spmspm_h100.csv
done

echo "== stage 4: energy"
for f in $(cat "$ADD"); do
  m=$(basename "$f" .txt); echo "== energy $m ==" | tee -a $B/energy_h100.csv
  ENERGY=6 timeout 3600 stdbuf -oL ./prep_driver   "$f" 20 | tee -a $B/energy_h100.csv
  ENERGY=6 timeout 3600 stdbuf -oL ./spmspm_driver "$f" 20 | tee -a $B/energy_h100.csv
done

python3 make_paper_tables.py
python3 pair_energy.py
echo "== ADDITIONS DONE $(date -Is)"
