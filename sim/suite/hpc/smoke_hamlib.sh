#!/bin/bash
# ===========================================================================
# smoke_hamlib.sh — build every bench/ablation harness and smoke-test them on
# tiny (q<=10) matrices, checking each harness's self-verify. Fast end-to-end
# validation before a full run. Runs on whatever GPU node it is invoked on
# (sm_80 binaries JIT/run on A100 / Ampere / Ada).
#   usage: bash smoke_hamlib.sh            (uses /tmp/smoke_hamlib_bin, sm_80)
#          ARCH=sm_89 SMOKE_BIN=/path bash smoke_hamlib.sh
# ===========================================================================
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
SP=${SMOKE_BIN:-/tmp/smoke_hamlib_bin}
ARCH=${ARCH:-sm_80}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
DRAW=${DRAWLOOM_BIN:-/mnt/beegfs/ysu34/drawloom_ae/drawloom_ae/Drawloom_code/myfloat}
mkdir -p "$SP"
echo "node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)  arch=$ARCH"

echo "== build =="
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmv_dia_vs_tc.cu    $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $SP/spmv_dia_vs_tc   -lcusparse            || { echo BUILD_FAIL spmv; exit 1; }
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmspm_hamlib.cu     $REPO/spmspm/paper_hm_kernel.cu            -o $SP/spmspm_pure     -lcusparse -Xcompiler -fopenmp || { echo BUILD_FAIL spmspm; exit 1; }
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmv_tc_ablation.cu  $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $SP/spmv_tc_ablation                       || { echo BUILD_FAIL tc_abl; exit 1; }
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmspm_ablation.cu   $REPO/spmspm/paper_hm_kernel.cu            -o $SP/spmspm_ablation -Xcompiler -fopenmp     || { echo BUILD_FAIL sp_abl; exit 1; }
g++  -O2 -std=c++17 $REPO/sim/dia_to_mtx.cpp -o $SP/dia_to_mtx || { echo BUILD_FAIL dia_to_mtx; exit 1; }
echo "  builds OK"

MATS="heis_8_4 tfim_8_4 fermi_8_4 B2_8_4 BeH_8_3 tsp_8_4"
pass=0; fail=0
chk(){ # $1=label $2=logfile $3=awk-extract-max-err
  local err; err=$(eval "$3" < "$2" 2>/dev/null)
  if [ -z "$err" ]; then echo "    $1: FAIL (no verify line)"; fail=$((fail+1)); return; fi
  # pass if |err| < 1e-3 (TF32 tolerance)
  if awk "BEGIN{exit !($err < 1e-3)}"; then echo "    $1: PASS (max_err=$err)"; pass=$((pass+1));
  else echo "    $1: FAIL (max_err=$err)"; fail=$((fail+1)); fi
}

echo "== smoke run (q8) =="
for m in $MATS; do
  p=$DIA/$m.txt; [ -f "$p" ] || { echo "  $m: missing"; continue; }
  echo "  --- $m ($(head -1 $p)) ---"
  L=$SP/_log
  # 1) SpMV: check zeroskip + cuSPARSE errors
  $SP/spmv_dia_vs_tc "$p" 20 > $L 2>&1
  chk "SpMV(zeroskip)"  $L "awk -F'zeroskip_err=' '/verify/{split(\$2,a,\" \"); print a[1]+0}'"
  chk "SpMV(cuSPARSE)"  $L "awk -F'cusparse_err=' '/verify/{split(\$2,a,\" \"); print a[1]+0}'"
  # 2) SpMSpM: check flat-vs-cpu (small n has CPU ref)
  $SP/spmspm_pure "$p" 10 > $L 2>&1
  chk "SpMSpM(flat)"    $L "awk -F'max\\\\|flat - cpu\\\\| = ' '/flat - cpu/{print \$2+0}'"
  # 3) TC ablation: max of T0..T3 rel err
  $SP/spmv_tc_ablation "$p" 20 > $L 2>&1
  chk "TCabl(T0-T3)"    $L "awk -F'verify vs dense] ' '/verify vs dense/{n=split(\$2,a,\" \"); m=0; for(i=1;i<=n;i++){if(a[i]~/^T[0-9]=/){split(a[i],b,\"=\"); v=b[2]+0; if(v>m)m=v}} print m}'"
  # 4) SpMSpM ablation: max of the three verify diffs
  $SP/spmspm_ablation "$p" 10 > $L 2>&1
  chk "SPabl(L0-L4)"    $L "awk '/\\[verify\\]/{m=0; for(i=1;i<=NF;i++){split(\$i,b,\"=\"); v=b[2]+0; if(v>m)m=v} print m}'"
done

# Drawloom smoke (one matrix) if binary present
if [ -x "$DRAW" ]; then
  echo "  --- Drawloom (heis_8_4) ---"
  $SP/dia_to_mtx $DIA/heis_8_4.txt $SP/_s.mtx >/dev/null 2>&1
  dl=$(cd "$(dirname $DRAW)" && OMP_NUM_THREADS=4 ./myfloat -filename $SP/_s.mtx 2>/dev/null | grep -oE 'drawloom time:\s*[0-9.]+')
  [ -n "$dl" ] && { echo "    Drawloom: PASS ($dl ms)"; pass=$((pass+1)); } || { echo "    Drawloom: FAIL"; fail=$((fail+1)); }
fi

echo "== SMOKE RESULT: $pass passed, $fail failed =="
[ "$fail" -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit $fail
