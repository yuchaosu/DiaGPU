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
#   MKLROOT       Intel oneMKL root (default /opt/intel/oneapi/mkl/latest).
#                 If mkl.h is found, an MKL CPU baseline row is added
#                 (hardware-axis context: GPU kernel vs CPU library).
# ============================================================
set -e
cd "$(dirname "$0")"
ARCH="${ARCH:-sm_90a}"
SIZES="${SIZES:-65536 262144 1048576}"
DRAWLOOM_BIN="${DRAWLOOM_BIN:-/tmp/drawloom_ae/drawloom_ae/Drawloom_code/myfloat}"
MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/latest}"

echo ">>> building bench + kernels (arch=$ARCH)"
MKL_INC=""; MKL_LIB=""
if [ -f "$MKLROOT/include/mkl.h" ]; then
  echo ">>> MKL found at $MKLROOT — enabling CPU baseline (-DUSE_MKL)"
  MKL_INC="-DUSE_MKL -I$MKLROOT/include"
  MKL_LIB="-L$MKLROOT/lib/intel64 -lmkl_rt"
  # mkl_rt dlopens the OpenMP runtime (libiomp5) at run time
  IOMP_DIR="$(dirname "$(dirname "$MKLROOT")")/compiler/latest/lib"
  export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:${IOMP_DIR}:${LD_LIBRARY_PATH}"
else
  echo ">>> MKL not found (set MKLROOT to enable CPU baseline) — skipping"
fi
nvcc -std=c++17 -O3 -arch=$ARCH $MKL_INC bench_cusparse.cu \
     ../src/tc_spmv_regdirect_kernel.cu $MKL_LIB -lcusparse -o /tmp/bench3
g++ -O2 -std=c++17 ../tools/gen_banded_mtx.cpp -o /tmp/gen_banded_mtx

gflops() { awk -v nnz="$1" -v ms="$2" 'BEGIN{printf "%.1f", (ms>0)?(2*nnz/(ms*1e6)):0}'; }

for N in $SIZES; do
  echo
  echo "================= N = $N ================="
  # ---- this repo: cuSPARSE + TC regdirect ----
  out=$(/tmp/bench3 "$N")
  nnz=$(echo "$out" | awk '/nnz =/{print $3}')
  cs=$(echo  "$out" | awk '/^  cuSPARSE/{print $2}')
  rd=$(echo  "$out" | awk '/^  TC regdirect/{print $3}')
  printf "  nnz = %s\n" "$nnz"
  printf "  %-22s %9s ms  %8s GFLOP/s   %s\n" "cuSPARSE (this harness)" "$cs" "$(gflops $nnz $cs)" ""
  printf "  %-22s %9s ms  %8s GFLOP/s   %5.2fx vs cusp\n" "TC regdirect (ours)"     "$rd" "$(gflops $nnz $rd)" "$(awk -v a=$cs -v b=$rd 'BEGIN{print a/b}')"

  # ---- MKL CSR on CPU (hardware-axis context, not algorithmic) ----
  mk=$(echo "$out" | awk '/^  MKL CSR/{print $3}')
  if [ -n "$mk" ]; then
    printf "  %-22s %9s ms  %8s GFLOP/s   (CPU MKL, hardware-axis)\n" "MKL CSR (CPU)" "$mk" "$(gflops $nnz $mk)"
  fi

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
