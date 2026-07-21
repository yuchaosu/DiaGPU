#!/bin/bash
# ===========================================================================
# run_ablation.sh — build the two ablation harnesses and run both sweeps into
# results/. Driver only: kernels live in spmv/src + spmspm (kept clean),
# harnesses in sim/. Override MATS_TC / MATS_SP for a quick (e.g. q8) test.
#   usage: bash run_ablation.sh
#          ARCH=sm_89 MATS_TC="heis_8_4 tfim_8_4" bash run_ablation.sh   # smoke
# ===========================================================================
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
BIN=${ABL_BIN:-/tmp/abl_bin}
ARCH=${ARCH:-sm_80}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
OUT=${OUT:-$REPO/results}
ITERS_TC=${ITERS_TC:-200}
ITERS_SP=${ITERS_SP:-30}
mkdir -p "$BIN" "$OUT"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) arch=$ARCH out=$OUT"

echo "== build ablation harnesses =="
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmv_tc_ablation.cu $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $BIN/spmv_tc_ablation                    || { echo BUILD_FAIL tc;   exit 1; }
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmspm_ablation.cu  $REPO/spmspm/paper_hm_kernel.cu            -o $BIN/spmspm_ablation -Xcompiler -fopenmp || { echo BUILD_FAIL sp;   exit 1; }
echo "  builds OK"

# default matrix sets (narrow+wide for TC; SpMSpM-feasible narrow bands for SpMSpM)
MATS_TC=${MATS_TC:-"heis_16_5 heis_18_5 heis_20_6 tfim_18_5 qmaxcut_18_4 fermi_16_5 bh_18_5 O2_16_5"}
MATS_SP=${MATS_SP:-"heis_14_5 heis_16_5 heis_18_5 tfim_16_5 tfim_18_5 qmaxcut_16_4 qmaxcut_18_4 fermi_14_5 fermi_16_5"}

echo "== TC ablation -> $OUT/tc_ablation.csv =="
echo "file,n,D,Dsym,kept_ztile,kept_symztile,t0_full,t1_sym,t2_ztile,t3_both,s1_sym,s2_ztile,s3_both,relerr" > $OUT/tc_ablation.csv
for m in $MATS_TC; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m (missing)"; continue; }
  $BIN/spmv_tc_ablation "$p" $ITERS_TC --csv 2>/dev/null | grep '^TCABL,' | sed 's#^TCABL,[^,]*/#TCABL,#; s/^TCABL,//' >> $OUT/tc_ablation.csv \
    && echo "  ok $m" || echo "  FAIL $m"
done

echo "== SpMSpM ablation -> $OUT/spmspm_ablation.csv =="
echo "file,n,Hd,nnzC,L0_hm,L1_meta,L2_flat1,L3_flat4,L4_sym,s1_meta,s2_flat1,s3_flat4,s4_sym,relerr" > $OUT/spmspm_ablation.csv
for m in $MATS_SP; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m (missing)"; continue; }
  $BIN/spmspm_ablation "$p" $ITERS_SP --csv 2>/dev/null | grep '^SPABL,' | sed 's#^SPABL,[^,]*/#SPABL,#; s/^SPABL,//' >> $OUT/spmspm_ablation.csv \
    && echo "  ok $m" || echo "  FAIL $m"
done

echo "== done: $(wc -l < $OUT/tc_ablation.csv) TC rows, $(wc -l < $OUT/spmspm_ablation.csv) SpMSpM rows (incl headers) =="
