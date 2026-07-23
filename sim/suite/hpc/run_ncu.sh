#!/bin/bash
# ===========================================================================
# run_ncu.sh — Eval 6.4: ncu mechanism profiling of the SpMV kernels.
# One ncu pass per OUR named kernel (tc_spmv_regdirect / cuda_spmv_dia /
# spmv_dia_zeroskip). Long-format ncu_profile.csv (matrix,kernel,metric,value).
# This is the "why we win / why TC loses" data: DRAM%, L1/L2 hit, TensorPipe%,
# occupancy. cuSPARSE kernels are library-internal names (timing lives in 6.2),
# so only our named kernels are profiled here.
#
#   usage: bash run_ncu.sh
#          MATS="heis_16_5 O2_16_5" bash run_ncu.sh          # subset
#          MATS_ALL=1 bash run_ncu.sh                        # every q>=thr matrix
# ===========================================================================
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
BIN=${NCU_BIN:-/tmp/ncu_bin}
ARCH=${ARCH:-sm_80}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
OUT=${OUT:-$REPO/results}
PY=${PY:-python3.11}
TIMEOUT=${NCU_TIMEOUT:-600}
mkdir -p "$BIN" "$OUT"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) arch=$ARCH out=$OUT"

command -v ncu >/dev/null 2>&1 || { echo "  ncu not found -> skipping 6.4 (profiling optional)"; exit 0; }

echo "== build SpMV harness =="
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmv_dia_vs_tc.cu $REPO/spmv/src/tc_spmv_regdirect_kernel.cu \
  -o $BIN/spmv_dia_vs_tc -lcusparse || { echo BUILD_FAIL; exit 1; }

NCU_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
PROF=$OUT/ncu_profile.csv
echo "matrix,kernel,metric,value" > $PROF

emit_ncu(){ # $1=matrix-tag $2=raw-csv -> append (matrix,kernel,metric,value)
  $PY - "$1" "$2" "$PROF" <<'PY'
import sys,csv
mat,raw,prof=sys.argv[1],sys.argv[2],sys.argv[3]
try: rows=list(csv.reader(open(raw)))
except: sys.exit(0)
hi=[i for i,r in enumerate(rows) if "Metric Name" in r]
if not hi: sys.exit(0)
h=rows[hi[0]]
try: kn=h.index("Kernel Name"); mn=h.index("Metric Name"); mv=h.index("Metric Value")
except: sys.exit(0)
with open(prof,"a") as f:
    for r in rows[hi[0]+1:]:
        if len(r)>max(kn,mn,mv) and r[mn]:
            k=r[kn].split('(')[0].split('<')[0][:44]
            f.write(f"{mat},{k},{r[mn]},{r[mv]}\n")
PY
}

# default: representative set covering the mechanism story (spin + molecular);
# MATS_ALL=1 profiles every q>=threshold matrix (slow, matches the full A100 run)
if [ "${MATS_ALL:-0}" = 1 ]; then
  MATS=""; for p in $DIA/*.txt; do case "$p" in *.imag.txt) continue;; esac
    b=$(basename "$p" .txt); fam=$(echo "$b"|sed -E 's/_[0-9]+_[0-9]+$//'); q=$(echo "$b"|sed -E 's/.*_([0-9]+)_[0-9]+$/\1/')
    case "$fam" in B2|O2|BeH|Li2|c2h|hnc) thr=12;; *) thr=18;; esac
    [ "$q" -ge "$thr" ] 2>/dev/null && MATS="$MATS $b"; done
else
  MATS=${MATS:-"heis_16_5 tfim_16_5 fermi_16_5 qmaxcut_16_4 O2_16_5"}
fi

echo "== ncu profile -> $PROF =="
for m in $MATS; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m (missing)"; continue; }
  any=0
  for kre in tc_spmv_regdirect cuda_spmv_dia spmv_dia_zeroskip; do
    timeout $TIMEOUT ncu --csv --metrics "$NCU_METRICS" --kernel-name "regex:$kre" -c 1 \
      $BIN/spmv_dia_vs_tc "$p" 3 > $OUT/_ncu_raw.csv 2>/dev/null && { emit_ncu "$m" "$OUT/_ncu_raw.csv"; any=1; }
  done
  [ "$any" = 0 ] && echo "$m,ncu_skipped,timeout_or_error,-" >> $PROF
  echo "  profiled $m"
done
rm -f $OUT/_ncu_raw.csv
echo "== done: ncu_profile.csv ($(( $(wc -l < $PROF) - 1 )) rows) =="
