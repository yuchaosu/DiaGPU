#!/bin/bash
# ===========================================================================
# finish_c29.slurm — safety net for the interactive c29 session: figure out
# from the DATA FILES what is already finished and run only what is missing.
# Idempotent; safe to submit while the interactive batch still runs (it waits
# for the log heartbeat to go stale before touching anything).
#
# Completion criteria (per matrix, across original+resume+patch CSVs):
#   prep : any spmspm row present, OR matrix in PREP_WALL (real OOM: spmv-only
#          is its terminal state)
#   e2e  : an E2EOPBUILDCSV row present (patched driver emits one for EVERY
#          matrix, wall rows included)
#   ncu  : base kernel rows and spmv_csr rows present in ncu_shared_h100.csv
#
# submit:  cd sim/paper_bench && sbatch finish_c29.slurm
# ===========================================================================
#SBATCH -J finish_c29
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o logs/finish_c29_%j.out
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
REPO=/home/ysu34/DiaGPU
PB=$REPO/sim/paper_bench
HAM=/mnt/beegfs/ysu34/hamlib/dia_e2e
SSD=/mnt/beegfs/ysu34/hamlib/dia_suitesparse
OUTDIR=/mnt/beegfs/ysu34/paper_bench
ITERS=${ITERS:-50}
STEPS=${STEPS:-1000}
ARCH=sm_90a
cd "$PB" || exit 2
echo "host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name --format=csv,noheader || exit 2

# ---- wait out the interactive session (log heartbeat < 30 min = still alive)
# NOTE: when running this script interactively with output appended to the
# heartbeat log itself, override ILOG (e.g. ILOG=/nonexistent) or it waits on
# its own output forever.
ILOG=${ILOG:-$PB/logs/resume_c29.out}
while [ -f "$ILOG" ] && [ $(( $(date +%s) - $(stat -c %Y "$ILOG") )) -lt 1800 ]; do
  echo "interactive run still active ($(date -Is)); waiting 10 min"; sleep 600
done

echo "== build (patched drivers, $ARCH) =="
nvcc -O3 -std=c++17 -arch=$ARCH --expt-relaxed-constexpr prep_driver.cu \
  "$REPO/spmspm/paper_hm_kernel.cu" -o prep_driver -lcusparse || exit 3
nvcc -O3 -std=c++17 -arch=$ARCH --expt-relaxed-constexpr e2e_driver.cu \
  -o e2e_driver_${ARCH} -lcusparse || exit 3

PCSVS="$OUTDIR/prep_h100_268657.csv $OUTDIR/prep_h100_c29resume.csv $OUTDIR/prep_h100_c29patch.csv"
ECSVS="$OUTDIR/e2e_diaq_h100_268662.csv $OUTDIR/e2e_diaq_h100_c29resume.csv $OUTDIR/e2e_diaq_h100_c29patch.csv"
PPATCH=$OUTDIR/prep_h100_c29patch.csv
EPATCH=$OUTDIR/e2e_diaq_h100_c29patch.csv
[ -f "$PPATCH" ] || echo "# $(date -Is) finish_c29 prep patch  PREPCSV,file,n,D,stage,variant,prep_host_ms,upload_ms,kernel_ms,amort_applies" > "$PPATCH"
[ -f "$EPATCH" ] || echo "# $(date -Is) finish_c29 e2e patch (HamLib --sym, SuiteSparse full)" > "$EPATCH"
PREP_WALL="BH_24_8 O2_20_5 O2_20_8"   # ours_flat C > device memory: spmv-only is terminal

prep_done(){ # $1=basename
  for w in $PREP_WALL; do [ "$1" = "$w" ] && { awk -F, -v m="/$1.txt" '$1=="PREPCSV" && index($2,m){f=1} END{exit !f}' $PCSVS 2>/dev/null && return 0; }; done
  awk -F, -v m="/$1.txt" '$1=="PREPCSV" && index($2,m) && $5=="spmspm"{f=1} END{exit !f}' $PCSVS 2>/dev/null
}
e2e_done(){ # $1=basename
  awk -F, -v m="/$1.txt" '$1=="E2EOPBUILDCSV" && index($2,m){f=1} END{exit !f}' $ECSVS 2>/dev/null
}

echo "== stage A: prep completeness =="
for f in $(ls "$HAM"/*.txt "$SSD"/*.txt 2>/dev/null | grep -v imag | sort); do
  m=$(basename "$f" .txt)
  if prep_done "$m"; then continue; fi
  echo "== prep $m =="
  timeout 2400 ./prep_driver "$f" "$ITERS" | tee -a "$PPATCH"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "SKIP(rc=$rc),$m.txt"
done

echo "== stage B: e2e completeness =="
for f in $(ls "$HAM"/*.txt 2>/dev/null | grep -v imag | sort); do
  m=$(basename "$f" .txt); e2e_done "$m" && continue
  echo "== e2e $m [L4-sym] =="
  timeout 3600 ./e2e_driver_${ARCH} "$f" "$STEPS" --sym | tee -a "$EPATCH"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "SKIP(rc=$rc),$m.txt"
done
for f in $(ls "$SSD"/*.txt 2>/dev/null | grep -v imag | sort); do
  m=$(basename "$f" .txt); e2e_done "$m" && continue
  echo "== e2e $m [full] =="
  timeout 3600 ./e2e_driver_${ARCH} "$f" "$STEPS" | tee -a "$EPATCH"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "SKIP(rc=$rc),$m.txt"
done

echo "== stage C: ncu shared-mem completeness =="
NCSV=$OUTDIR/ncu_shared_h100.csv
if [ ! -f "$NCSV" ] || ! grep -q ",gather_meta_kernel\|,void gather_meta_kernel" "$NCSV"; then
  bash "$REPO/sim/suite/hpc/run_ncu_shared.sh"          # base pass missing -> full run
elif ! grep -q ",spmv_csr" "$NCSV"; then
  PROF_APPEND=1 SPMSPM_KERNELS="" SPMV_KERNELS="spmv_csr" bash "$REPO/sim/suite/hpc/run_ncu_shared.sh"
else
  echo "  ncu complete"
fi

echo "== FINISH_C29 DONE $(date -Is) =="
echo "  prep rows: $(cat $PCSVS 2>/dev/null | grep -c '^PREPCSV')  e2e opbuild: $(cat $ECSVS 2>/dev/null | grep -c '^E2EOPBUILDCSV')  ncu rows: $(( $(wc -l < "$NCSV") - 1 ))"
