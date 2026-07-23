#!/bin/bash
# ===========================================================================
# run_budget.sh — Eval: kernels UNDER the fidelity-preserving diagonal budget.
# Adopts HamSim's diagonal budget as a uniform methodology: for each matrix pick
# the MOST aggressive budget (smallest D) whose evolution fidelity >= FIDMIN
# (default 0.99), then evaluate BOTH kernels on that budgeted operator:
#   SpMV   : ours(zero-skip) vs cuSPARSE  (sim/spmv_budget.cu, trusted tight-loop)
#   SpMSpM : ours(gather_flat) vs HM / cuSPARSE SpGEMM (on the budgeted H)
# No-op for lattice/spin families (irreducible -> keeps all diagonals); D-narrowing
# for molecular. Full-density numbers stay in 6.2/6.3 for transparency.
#
#   usage: bash run_budget.sh
#          FIDMIN=0.99 MATS="heis_16_5 O2_16_5" bash run_budget.sh
# ===========================================================================
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
BIN=${BUDGET_BIN:-/tmp/budget_bin}
ARCH=${ARCH:-sm_80}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
OUT=${OUT:-$REPO/results}
FIDMIN=${FIDMIN:-0.99}
STEPS=${STEPS:-1000}
K=${K:-6}
ITERS=${ITERS:-200}
PY=${PY:-python3.11}
mkdir -p "$BIN" "$OUT"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) arch=$ARCH out=$OUT fidmin=$FIDMIN"

echo "== build harnesses =="
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmv_budget.cu   $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $BIN/spmv_budget  -lcusparse            || { echo BUILD_FAIL spmv_budget; exit 1; }
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmspm_hamlib.cu $REPO/spmspm/paper_hm_kernel.cu            -o $BIN/spmspm_pure  -lcusparse -Xcompiler -fopenmp || { echo BUILD_FAIL spmspm; exit 1; }
echo "  builds OK"

MATS=${MATS:-"heis_16_5 tfim_16_5 qmaxcut_16_4 fermi_16_5 O2_16_5"}

# full per-budget sweep (SpMV, all levels + fidelity)
echo "file,n,Ng,budget,Dkept,nnz,fidelity,zs_ms,cusp_ms,zs_vs_cusp" > $OUT/spmv_budget.csv
# SpMSpM on the budgeted operator
echo "file,n,Hdiags,nnzC,ours_ms,hm_ms,cusparse_ms,ours_vs_hm,ours_vs_cusparse" > $OUT/spmspm_budget.csv
# unified summary: the fidelity>=FIDMIN operating point for BOTH kernels
echo "file,D_full,groups_star,D_budget,fidelity,spmv_zs_vs_cusp,spmspm_ours_vs_hm,spmspm_ours_vs_cusp" > $OUT/budget_summary.csv

for m in $MATS; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m (missing)"; continue; }
  # 1) SpMV budget sweep -> all rows + capture
  raw=$($BIN/spmv_budget "$p" 1.2 $STEPS $K $ITERS 2>/dev/null | grep '^SPBUD' | grep -v ',Ng,budget,')
  [ -z "$raw" ] && { echo "  $m: spmv_budget none"; continue; }
  echo "$raw" | sed 's#^SPBUD,[^,]*/#SPBUD,#; s/^SPBUD,//' >> $OUT/spmv_budget.csv
  # 2) pick smallest-D budget with fidelity>=FIDMIN (SPBUD cols after strip:
  #    file=1,n=2,Ng=3,budget=4,Dkept=5,nnz=6,fid=7,zs_ms=8,cusp_ms=9,zs_vs_cusp=10;
  #    but here $raw still has the SPBUD tag -> shift by 1: budget=5,Dkept=6,fid=8,ratio=11)
  read gstar dstar fid zscmp dfull < <(echo "$raw" | awk -F, -v f=$FIDMIN '
    NR==1{dfull=$6}
    { if($8+0>=f){ if(best==""||$6+0<bestD){best=$5;bestD=$6;bestF=$8;bestR=$11} } }
    END{ print best, bestD, bestF, bestR, dfull }')
  # 3) budgeted DIA at gstar groups (== full when lattice/irreducible)
  $PY $REPO/sim/budget_dia.py "$p" $BIN/_bud.txt $gstar >/dev/null 2>&1
  # 4) SpMSpM on the budgeted operator
  sp=$($BIN/spmspm_pure $BIN/_bud.txt 30 --csv 2>/dev/null | grep '^SPCSV')
  if [ -n "$sp" ]; then
    echo "$sp" | awk -F, -v m=$m '{printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n",m,$3,$4,$7,$8,$9,$10,$11,$12}' >> $OUT/spmspm_budget.csv
    ohm=$(echo "$sp" | awk -F, '{print $11}'); ocs=$(echo "$sp" | awk -F, '{print $12}')
  else ohm=NA; ocs=NA; fi
  # 5) unified summary row
  echo "$m,$dfull,$gstar,$dstar,$fid,$zscmp,$ohm,$ocs" >> $OUT/budget_summary.csv
  echo "  ok $m: fid>=$FIDMIN at $gstar groups (D=$dstar, fid=$fid) -> SpMV ${zscmp}x, SpMSpM ours/hm=$ohm"
done
rm -f $BIN/_bud.txt
echo "== done: budget_summary.csv, spmv_budget.csv, spmspm_budget.csv in $OUT =="