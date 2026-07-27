#!/usr/bin/env bash
# run_all.sh — run both drivers (+ external Drawloom if provided) over a
# matrix list; append every CSV row to one results file.
#
#   DATA=/mnt/beegfs/ysu34/hamlib/dia_e2e \
#   MATS="heis_14_5 heis_16_5 tfim_14_4 maxcut_14_5 BeH_12_6 B2_14_4" \
#   OUT=results_$(hostname).csv \
#   DRAWLOOM_BIN=/path/to/drawloom D2MTX=../suite/bin/dia_to_mtx \
#   bash run_all.sh
#
# Drawloom is an external binary consuming .mtx (prints "drawloom time: X ms");
# it is appended as a SPMVCSV row with kernel=drawloom, relerr=na.
set -u
DATA=${DATA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
MATS=${MATS:-"heis_14_5 heis_16_5 tfim_14_4 maxcut_14_5 BeH_12_6 B2_14_4 BH_20_7"}
OUT=${OUT:-results.csv}
ITERS=${ITERS:-200}
SPITERS=${SPITERS:-100}
DRAWLOOM_BIN=${DRAWLOOM_BIN:-}
D2MTX=${D2MTX:-$(dirname "$0")/../suite/bin/dia_to_mtx}
TMP=${TMPDIR:-/tmp}

echo "# $(date -Is) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" >> "$OUT"

for m in $MATS; do
  f="$DATA/$m.txt"
  if [ ! -f "$f" ]; then echo "SKIP,$m,missing" | tee -a "$OUT"; continue; fi
  echo "== $m =="
  ./spmv_driver   "$f" "$ITERS"   | tee -a "$OUT"
  ./spmspm_driver "$f" "$SPITERS" | tee -a "$OUT"
  # e2e: Taylor evolution (hybrid SpMV, fused NV=2, vs cuSPARSE same
  # pipeline) + operator chain H^2/H^3 (pair-plan SpMSpM).  E2E=0 skips.
  if [ "${E2E:-1}" = "1" ] && [ -x ./e2e_driver ]; then
    ./e2e_driver "$f" "${ESTEPS:-1000}" "${EK:-6}" | tee -a "$OUT"
  fi
  # Drawloom consumes a TEXT .mtx: skip when the source .txt is huge (the
  # conversion alone would dominate the job; matches the old 1.5e8-nnz cap).
  if [ -n "$DRAWLOOM_BIN" ] && [ -x "$DRAWLOOM_BIN" ] && [ -x "$D2MTX" ] \
     && [ "$(stat -c%s "$f")" -le 800000000 ]; then
    mtx="$TMP/cur_$m.mtx"
    if "$D2MTX" "$f" "$mtx" >/dev/null 2>&1 && [ -f "$mtx" ]; then
      out=$(OMP_NUM_THREADS=16 "$DRAWLOOM_BIN" -filename "$mtx" 2>&1)
      ms=$(echo "$out" | sed -n 's/.*drawloom time:\s*\([0-9.]*\)\s*ms.*/\1/p' | head -1)
      rm -f "$mtx"
      if [ -n "$ms" ]; then
        echo "SPMVCSV,$f,na,na,na,na,drawloom,$ms,na" | tee -a "$OUT"
      else
        echo "SKIP,$m,drawloom_fail" | tee -a "$OUT"
      fi
    else
      echo "SKIP,$m,mtx_fail" | tee -a "$OUT"
    fi
  fi
done
echo "done -> $OUT"
