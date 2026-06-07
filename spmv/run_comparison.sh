#!/bin/bash
# ============================================================
# run_comparison.sh
#
# Apples-to-apples-ish comparison of the diagonal tensor-core
# SpMV kernels against cuSPARSE and the Drawloom TF32 baseline,
# on the SAME banded matrix.
#
# Because the two harnesses (this repo's bench vs Drawloom's
# binary) time differently, we ALSO report speedup-over-cuSPARSE
# *within each harness* — a methodology-robust yardstick.
#
# env:
#   ARCH          (default sm_90a)
#   DRAWLOOM_BIN  path to Drawloom's `myfloat` (TF32) binary.
#                 If unset/missing, Drawloom rows are skipped.
# ============================================================
set -e
cd "$(dirname "$0")"
ARCH="${ARCH:-sm_90a}"
SIZES="${SIZES:-65536 262144 1048576}"
DRAWLOOM_BIN="${DRAWLOOM_BIN:-/tmp/drawloom_ae/drawloom_ae/Drawloom_code/myfloat}"

echo ">>> building bench + kernels (arch=$ARCH)"
nvcc -std=c++17 -O3 -arch=$ARCH bench_cusparse.cu tc_spmv_dense_kernel.cu \
     tc_spmv_regdirect_kernel.cu -o /tmp/bench3 -lcusparse
g++ -O2 -std=c++17 gen_banded_mtx.cpp -o /tmp/gen_banded_mtx

gflops() { awk -v nnz="$1" -v ms="$2" 'BEGIN{printf "%.1f", (ms>0)?(2*nnz/(ms*1e6)):0}'; }

for N in $SIZES; do
  echo
  echo "================= N = $N ================="
  # ---- this repo: cuSPARSE + TC dense + TC regdirect ----
  out=$(/tmp/bench3 "$N")
  nnz=$(echo "$out" | awk '/nnz =/{print $3}')
  cs=$(echo  "$out" | awk '/^  cuSPARSE/{print $2}')
  td=$(echo  "$out" | awk '/^  TC dense/{print $3}')
  rd=$(echo  "$out" | awk '/^  TC regdirect/{print $3}')
  printf "  nnz = %s\n" "$nnz"
  printf "  %-22s %9s ms  %8s GFLOP/s   %s\n" "cuSPARSE (this harness)" "$cs" "$(gflops $nnz $cs)" ""
  printf "  %-22s %9s ms  %8s GFLOP/s   %5.2fx vs cusp\n" "TC dense (WMMA+smem)"   "$td" "$(gflops $nnz $td)" "$(awk -v a=$cs -v b=$td 'BEGIN{print a/b}')"
  printf "  %-22s %9s ms  %8s GFLOP/s   %5.2fx vs cusp\n" "TC regdirect (ours)"     "$rd" "$(gflops $nnz $rd)" "$(awk -v a=$cs -v b=$rd 'BEGIN{print a/b}')"

  # ---- Drawloom TF32 (separate harness) ----
  if [ -x "$DRAWLOOM_BIN" ]; then
    /tmp/gen_banded_mtx "$N" "/tmp/banded_$N.mtx" 2>/dev/null
    dout=$(cd "$(dirname "$DRAWLOOM_BIN")" && OMP_NUM_THREADS=32 "./$(basename "$DRAWLOOM_BIN")" -filename "/tmp/banded_$N.mtx" 2>/dev/null)
    dcs=$(echo "$dout" | awk '/best-cusp time/{print $4}')
    dl=$(echo  "$dout" | awk '/drawloom time/{print $3}')
    printf "  %-22s %9s ms  %8s GFLOP/s   (Drawloom harness)\n" "cuSPARSE (Drawloom)" "$dcs" "$(gflops $nnz $dcs)"
    printf "  %-22s %9s ms  %8s GFLOP/s   %5.2fx vs cusp\n" "Drawloom TF32" "$dl" "$(gflops $nnz $dl)" "$(awk -v a=$dcs -v b=$dl 'BEGIN{print a/b}')"
  else
    echo "  (Drawloom binary not found at $DRAWLOOM_BIN — skipping)"
  fi
done
