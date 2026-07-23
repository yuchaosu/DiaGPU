#!/bin/bash
# ===========================================================================
# run_sparsity.sh — Eval 6.8: structure-aware sparsification studies.
# Driver only; harnesses in sim/ (kept clean). All CSVs -> results/.
#
#   6.8a  SpMV entry-magnitude trim (nnz/speed/error vs tau)     -> spmv_trim.csv
#   6.8b  1000-step compounding (fidelity vs tau over evolution) -> evolve_trim.csv
#   6.8c  diagonal budget (fidelity vs #diagonals kept)          -> evolve_budget.csv
#   6.8d  SpMSpM fill-in budget (H^k diagonals/nnz/error vs tau) -> spmspm_trim.csv
#
# Runs a representative BOTH-structure-class set (spin + molecular) so the
# family dependence (molecular compressible, spin irreducible) is visible.
#
#   usage: bash run_sparsity.sh
#          ARCH=sm_89 STEPS=100 MATS="heis_16_5 O2_16_5" bash run_sparsity.sh  # smoke
# ===========================================================================
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
BIN=${SPARSE_BIN:-/tmp/sparse_bin}
ARCH=${ARCH:-sm_80}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
OUT=${OUT:-$REPO/results}
STEPS=${STEPS:-1000}
K=${K:-12}
KMAX=${KMAX:-4}
THREADS=${THREADS:-16}
mkdir -p "$BIN" "$OUT"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) arch=$ARCH out=$OUT steps=$STEPS"

echo "== build sparsification harnesses =="
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/trim_sweep.cu    -o $BIN/trim_sweep    || { echo BUILD_FAIL trim_sweep;    exit 1; }
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/evolve_trim.cu   -o $BIN/evolve_trim   || { echo BUILD_FAIL evolve_trim;   exit 1; }
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/evolve_budget.cu -o $BIN/evolve_budget || { echo BUILD_FAIL evolve_budget; exit 1; }
g++  -O3 -std=c++17 -fopenmp    $REPO/sim/spmspm_trim.cpp  -o $BIN/spmspm_trim   || { echo BUILD_FAIL spmspm_trim;   exit 1; }
echo "  builds OK"

# representative set: spin (heis/tfim/qmaxcut/maxcut) + molecular (fermi/O2)
MATS=${MATS:-"heis_16_5 tfim_16_5 qmaxcut_16_4 maxcut_16_5 fermi_16_5 O2_16_5"}
# SpMSpM fill-in feasible only on narrow bands (O2^2 overflows offset cap)
MATS_SP=${MATS_SP:-"heis_16_5 tfim_16_5 qmaxcut_16_4 fermi_16_5"}
paths(){ for m in $1; do p=$DIA/$m.txt; [ -f "$p" ] && printf "%s " "$p"; done; }

echo "== 6.8a SpMV entry trim -> $OUT/spmv_trim.csv =="
$BIN/trim_sweep $(paths "$MATS") 2>/dev/null > $OUT/spmv_trim.csv \
  && echo "  ok ($(( $(wc -l < $OUT/spmv_trim.csv) - 1 )) rows)" || echo "  FAIL"

echo "== 6.8b 1000-step compounding -> $OUT/evolve_trim.csv =="
first=1
for m in $MATS; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m"; continue; }
  if [ $first -eq 1 ]; then $BIN/evolve_trim "$p" $STEPS $K 2>/dev/null > $OUT/evolve_trim.csv; first=0
  else $BIN/evolve_trim "$p" $STEPS $K 2>/dev/null | tail -n +2 >> $OUT/evolve_trim.csv; fi
  echo "  ok $m"
done

echo "== 6.8c diagonal budget -> $OUT/evolve_budget.csv =="
first=1
for m in $MATS; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m"; continue; }
  if [ $first -eq 1 ]; then $BIN/evolve_budget "$p" $STEPS $K 2>/dev/null > $OUT/evolve_budget.csv; first=0
  else $BIN/evolve_budget "$p" $STEPS $K 2>/dev/null | tail -n +2 >> $OUT/evolve_budget.csv; fi
  echo "  ok $m"
done

echo "== 6.8d SpMSpM fill-in budget -> $OUT/spmspm_trim.csv =="
KMAX=$KMAX OMP_NUM_THREADS=$THREADS $BIN/spmspm_trim $(paths "$MATS_SP") 2>/dev/null > $OUT/spmspm_trim.csv \
  && echo "  ok ($(( $(wc -l < $OUT/spmspm_trim.csv) - 1 )) rows)" || echo "  FAIL"

echo "== done. 6.8 CSVs in $OUT: spmv_trim, evolve_trim, evolve_budget, spmspm_trim =="
