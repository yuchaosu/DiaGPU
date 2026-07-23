#!/bin/bash
# ===========================================================================
# run_e2e.sh — Eval 6.7: end-to-end solver comparison + correctness anchor.
# Driver only; kernels in spmv/src, harnesses in sim/ (kept clean).
#
#   6.7a  e2e speedup : ours(CUDA zero-skip) vs cuSPARSE, IDENTICAL Taylor loop
#                       (also TF32 tensor-core column, which loses) -> e2e_speedup.csv
#   6.7b  correctness : our fixed-K Taylor vs QuTiP sesolve + scipy expm_multiply,
#                       small q (recognized-tool anchor) -> qutip_anchor.csv
#
#   usage: bash run_e2e.sh
#          ARCH=sm_89 STEPS=200 MATS="heis_16_5" bash run_e2e.sh   # smoke
# ===========================================================================
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
BIN=${E2E_BIN:-/tmp/e2e_bin}
ARCH=${ARCH:-sm_80}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
OUT=${OUT:-$REPO/results}
STEPS=${STEPS:-1000}
K=${K:-6}
PY=${PY:-python3.11}
mkdir -p "$BIN" "$OUT"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) arch=$ARCH out=$OUT steps=$STEPS K=$K"

echo "== build e2e harness =="
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/e2e_speedup.cu $REPO/spmv/src/tc_spmv_regdirect_kernel.cu \
  -o $BIN/e2e_speedup -lcusparse || { echo BUILD_FAIL e2e; exit 1; }
echo "  build OK"

# full family/q set for the speedup sweep; small-q set for the correctness anchor
MATS=${MATS:-"heis_16_5 heis_18_5 heis_20_6 tfim_16_5 tfim_18_5 qmaxcut_16_4 qmaxcut_18_4 fermi_16_5 fermi_18_5 maxcut_16_5 O2_16_5"}
MATS_ANCHOR=${MATS_ANCHOR:-"heis_8_4 tfim_8_4 fermi_8_4 qmaxcut_8_4"}

echo "== 6.7a e2e speedup -> $OUT/e2e_speedup.csv =="
echo "file,n,D,nnz,steps,K,zs_full_ms,tf_full_ms,cusp_full_ms,zs_vs_cusp,tf_vs_cusp,zs_spmv_ms,cusp_spmv_ms" > $OUT/e2e_speedup.csv
for m in $MATS; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m (missing)"; continue; }
  line=$($BIN/e2e_speedup "$p" 1.2 $STEPS $K 2>/dev/null | grep '^E2E,')
  [ -z "$line" ] && { echo "  $m: OOM/none"; continue; }
  echo "$line" | sed 's#^E2E,[^,]*/#E2E,#; s/^E2E,//' >> $OUT/e2e_speedup.csv && echo "  ok $m"
done

echo "== 6.7b correctness anchor -> $OUT/qutip_anchor.csv =="
echo "file,N,K,steps,fid_qutip_taylor64,fid_qutip_expm,fid_expm_taylor64,fid_expm_taylor32" > $OUT/qutip_anchor.csv
for m in $MATS_ANCHOR; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m (missing)"; continue; }
  line=$($PY $REPO/sim/qutip_anchor.py "$p" 1.2 $STEPS $K 2>/dev/null | grep '^QANCHOR,')
  [ -z "$line" ] && { echo "  $m: anchor failed"; continue; }
  echo "$line" | sed 's#^QANCHOR,[^,]*/#QANCHOR,#; s/^QANCHOR,//' >> $OUT/qutip_anchor.csv && echo "  ok $m"
done

echo "== done: $(( $(wc -l < $OUT/e2e_speedup.csv) - 1 )) e2e rows, $(( $(wc -l < $OUT/qutip_anchor.csv) - 1 )) anchor rows =="
