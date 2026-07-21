#!/bin/bash
# ===========================================================================
# run_plan.sh — build EVERY harness and run NEEDTORUN Phase 1 end-to-end into
# results/, on real HamLib DIA matrices. Driver only; kernels live in
# spmv/src + spmspm (kept clean), harnesses in sim/.
#
#   P1.1 SpMV     : ours(zero-skip) vs TC / dense-CUDA / cuSPARSE / Drawloom
#   P1.2 SpMSpM   : ours(gather_flat) vs Haque(HM) / cuSPARSE SpGEMM
#   P1.3 budget   : speedup vs #diagonals (D), derived from P1.1 across families
#   P1.5 ablations: TC (T0->T3) + SpMSpM (L0->L4), base -> all optimizations
#
#   usage: bash run_plan.sh
#          ARCH=sm_89 MATS="heis_8_4 tfim_8_4" MATS_SP="heis_8_4" bash run_plan.sh  # smoke
# ===========================================================================
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
BIN=${PLAN_BIN:-/tmp/plan_bin}
ARCH=${ARCH:-sm_80}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
OUT=${OUT:-$REPO/results}
DRAW=${DRAWLOOM_BIN:-/mnt/beegfs/ysu34/drawloom_ae/drawloom_ae/Drawloom_code/myfloat}
mkdir -p "$BIN" "$OUT"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) arch=$ARCH out=$OUT"

echo "== build all harnesses ($ARCH) =="
B(){ nvcc -O3 -std=c++17 -arch=$ARCH "$@"; }
B $REPO/sim/spmv_dia_vs_tc.cu   $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $BIN/spmv_dia_vs_tc   -lcusparse            || { echo FAIL; exit 1; }
B $REPO/sim/spmspm_hamlib.cu    $REPO/spmspm/paper_hm_kernel.cu            -o $BIN/spmspm_pure       -lcusparse -Xcompiler -fopenmp || { echo FAIL; exit 1; }
B $REPO/sim/spmv_tc_ablation.cu $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $BIN/spmv_tc_ablation                        || { echo FAIL; exit 1; }
B $REPO/sim/spmspm_ablation.cu  $REPO/spmspm/paper_hm_kernel.cu            -o $BIN/spmspm_ablation   -Xcompiler -fopenmp    || { echo FAIL; exit 1; }
g++ -O2 -std=c++17 $REPO/sim/dia_to_mtx.cpp -o $BIN/dia_to_mtx || { echo FAIL; exit 1; }
echo "  builds OK"

# real-HamLib matrix sets (override via env for smoke)
MATS=${MATS:-"heis_16_5 heis_18_5 tfim_18_5 fermi_16_5 qmaxcut_18_4 bh_18_5 O2_16_5 B2_14_4 BeH_12_4 maxcut_18_5 tsp_18_4"}
MATS_SP=${MATS_SP:-"heis_14_5 heis_16_5 heis_18_5 tfim_16_5 tfim_18_5 qmaxcut_16_4 qmaxcut_18_4 fermi_14_5 fermi_16_5"}
bn(){ basename "$1" .txt; }

# ---------- P1.1 SpMV kernel comparison ----------
echo "== P1.1 SpMV -> $OUT/spmv_kernel.csv =="
echo "file,q,D,fill,zs_ms,tc_ms,dense_ms,cusparse_ms,zs_vs_tc,zs_vs_dense,zs_vs_cusparse,drawloom_ms" > $OUT/spmv_kernel.csv
for m in $MATS; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m"; continue; }
  line=$($BIN/spmv_dia_vs_tc "$p" 200 --csv 2>/dev/null | grep '^CSV,')
  [ -z "$line" ] && { echo "  $m: OOM/none"; continue; }
  q=$(echo $m | sed -E 's/.*_([0-9]+)_[0-9]+$/\1/')
  # CSV: ,file,n,D,fill,nnz,zsv,tc,dense,zs,csp,zs_vs_tc,zs_vs_dense,zs_vs_csp,relerr,bw,skip
  dl=na; if [ -x "$DRAW" ]; then $BIN/dia_to_mtx "$p" $BIN/_p.mtx >/dev/null 2>&1
    dl=$(cd "$(dirname $DRAW)" && OMP_NUM_THREADS=16 ./myfloat -filename $BIN/_p.mtx 2>/dev/null | grep -oE 'drawloom time:[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+$'); dl=${dl:-na}; fi
  echo "$line" | awk -F, -v q=$q -v dl=$dl '{gsub(/.*\//,"",$2); printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",$2,q,$4,$5,$10,$8,$9,$11,$12,$13,$14,dl}' >> $OUT/spmv_kernel.csv
  echo "  ok $m"
done

# ---------- P1.2 SpMSpM kernel comparison ----------
echo "== P1.2 SpMSpM -> $OUT/spmspm_kernel.csv =="
echo "file,n,Hdiags,nnzC,ours_ms,hm_ms,cusparse_ms,ours_vs_hm,ours_vs_cusparse" > $OUT/spmspm_kernel.csv
for m in $MATS_SP; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m"; continue; }
  line=$($BIN/spmspm_pure "$p" 30 --csv 2>/dev/null | grep '^SPCSV,')
  [ -z "$line" ] && { echo "  $m: OOM/none"; continue; }
  # SPCSV,file,n,Hd,nnzH,Cd,nnzC,ours,hm,cusp,o_hm,o_csp
  echo "$line" | awk -F, '{gsub(/.*\//,"",$2); printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n",$2,$3,$4,$7,$8,$9,$10,$11,$12}' >> $OUT/spmspm_kernel.csv
  echo "  ok $m"
done

# ---------- P1.3 budget sweep (speedup vs #diagonals, derived from P1.1) ----------
echo "== P1.3 budget sweep -> $OUT/budget_sweep.csv =="
echo "D,file,q,fill,zs_vs_tc,zs_vs_cusparse" > $OUT/budget_sweep.csv
awk -F, 'NR>1 && $9!="na"{printf "%s,%s,%s,%s,%s,%s\n",$3,$1,$2,$4,$9,$11}' $OUT/spmv_kernel.csv | sort -t, -k1,1n >> $OUT/budget_sweep.csv

# ---------- P1.5 ablations ----------
echo "== P1.5 ablations -> $OUT/{tc,spmspm}_ablation.csv =="
ABL_BIN=$BIN OUT=$OUT ARCH=$ARCH DIA=$DIA MATS_TC="$MATS" MATS_SP="$MATS_SP" bash $REPO/sim/suite/hpc/run_ablation.sh >/dev/null 2>&1 \
  && echo "  ablations OK" || echo "  ablations FAIL"

echo "== DONE. results in $OUT: =="
for f in spmv_kernel spmspm_kernel budget_sweep tc_ablation spmspm_ablation; do
  [ -f $OUT/$f.csv ] && echo "  $f.csv ($(( $(wc -l < $OUT/$f.csv) - 1 )) rows)"
done
