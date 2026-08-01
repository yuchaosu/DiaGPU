#!/bin/bash
# ===========================================================================
# resume_c29.sh — finish the root-killed H100 jobs 268657 (prep) and 268662
# (e2e_diaq) interactively on c29, then run the shared-mem ncu profile.
# Stages run SEQUENTIALLY (exclusive GPU -> timings stay clean).
#
# Idempotent: a matrix is skipped iff it is listed in the done-file; it is
# appended there only when its driver exits 0. Done-files are seeded from the
# killed jobs' CSVs (BeH_12_6 deliberately NOT seeded for e2e: its first run
# hit an illegal memory access and must be redone). Killed-job CSVs are never
# modified; resumed rows go to *_c29resume.csv.
#
#   usage: bash resume_c29.sh            (or nohup/background)
# ===========================================================================
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

echo "== build ($ARCH) =="
nvcc -O3 -std=c++17 -arch=$ARCH --expt-relaxed-constexpr prep_driver.cu \
  "$REPO/spmspm/paper_hm_kernel.cu" -o prep_driver -lcusparse || { echo build_fail_prep; exit 3; }
nvcc -O3 -std=c++17 -arch=$ARCH --expt-relaxed-constexpr e2e_driver.cu \
  -o e2e_driver_${ARCH} -lcusparse || { echo build_fail_e2e; exit 3; }

# ---- stage 1: prep resume (same cmdline as prep_h100.slurm) ---------------
PDONE=$OUTDIR/prep_h100_done.txt
PCSV=$OUTDIR/prep_h100_c29resume.csv
if [ ! -f "$PDONE" ]; then  # seed: matrices 268657 finished (ours_sym row present)
  awk -F, '$1=="PREPCSV" && $5=="spmspm" && $6=="ours_sym"{print $2}' \
    "$OUTDIR"/prep_h100_268657.csv | sort -u > "$PDONE"
  echo "seeded $PDONE with $(wc -l < "$PDONE") matrices from job 268657"
fi
[ -f "$PCSV" ] || echo "# $(date -Is) host=$(hostname) ITERS=$ITERS resume-of-268657  PREPCSV,file,n,D,stage,variant,prep_host_ms,upload_ms,kernel_ms,amort_applies (take LAST row per file,variant)" > "$PCSV"
echo "== stage 1: prep resume -> $PCSV =="
for f in $(ls "$HAM"/*.txt "$SSD"/*.txt 2>/dev/null | grep -v imag | sort); do
  grep -qxF "$f" "$PDONE" && continue
  echo "== prep $(basename "$f" .txt) =="
  timeout 1800 ./prep_driver "$f" "$ITERS" | tee -a "$PCSV"
  rc=${PIPESTATUS[0]}
  # done = clean exit OR terminal ours_sym row emitted (wide-band matrices can
  # crash in the sym block AFTER all useful rows are out — known, tolerated)
  if [ "$rc" -eq 0 ] || awk -F, -v f="$f" '$1=="PREPCSV" && $2==f && $5=="spmspm" && $6=="ours_sym"{ok=1} END{exit !ok}' "$PCSV"; then
    echo "$f" >> "$PDONE"
  else echo "SKIP(rc=$rc),$(basename "$f")"; fi
done

# ---- stage 2: e2e_diaq resume (same cmdline as e2e_diaq_h100.slurm) -------
EDONE=$OUTDIR/e2e_diaq_h100_done.txt
ECSV=$OUTDIR/e2e_diaq_h100_c29resume.csv
if [ ! -f "$EDONE" ]; then  # seed: 268662's OPBUILD matrices MINUS BeH_12_6 (illegal access -> redo)
  grep '^E2EOPBUILDCSV' "$OUTDIR"/e2e_diaq_h100_268662.csv | awk -F, '{print $2}' \
    | sort -u | grep -v 'BeH_12_6' > "$EDONE"
  echo "seeded $EDONE with $(wc -l < "$EDONE") matrices from job 268662 (BeH_12_6 excluded)"
fi
[ -f "$ECSV" ] || echo "# $(date -Is) host=$(hostname) steps=$STEPS resume-of-268662 (HamLib=--sym, SuiteSparse=full; +diaq ev arm, +full opbuild under --sym)" > "$ECSV"
echo "== stage 2: e2e resume -> $ECSV =="
e2e_one(){ # $1=file $2=--sym|""
  echo "== e2e $(basename "$1" .txt) [${2:-full}] =="
  timeout 3600 ./e2e_driver_${ARCH} "$1" "$STEPS" $2 | tee -a "$ECSV"
  rc=${PIPESTATUS[0]}
  # done = clean exit OR OPBUILD row emitted (known tolerated sym-block crash
  # on wide bands happens after all useful rows are printed)
  if [ "$rc" -eq 0 ] || grep -q "^E2EOPBUILDCSV,$1," "$ECSV"; then
    echo "$1" >> "$EDONE"
  else echo "SKIP(rc=$rc),$(basename "$1")"; fi
}
for f in $(ls "$HAM"/*.txt 2>/dev/null | grep -v imag | sort); do
  grep -qxF "$f" "$EDONE" && continue; e2e_one "$f" --sym; done
for f in $(ls "$SSD"/*.txt 2>/dev/null | grep -v imag | sort); do
  grep -qxF "$f" "$EDONE" && continue; e2e_one "$f" ""; done

# ---- stage 3: shared-mem ncu profile --------------------------------------
echo "== stage 3: ncu shared-mem profile =="
bash "$REPO/sim/suite/hpc/run_ncu_shared.sh"

echo "== ALL DONE $(date -Is) =="
echo "  prep rows:  268657=$(grep -c '^PREPCSV' "$OUTDIR"/prep_h100_268657.csv)  resume=$(grep -c '^PREPCSV' "$PCSV")"
echo "  e2e opbuild: 268662=$(grep -c '^E2EOPBUILDCSV' "$OUTDIR"/e2e_diaq_h100_268662.csv)  resume=$(grep -c '^E2EOPBUILDCSV' "$ECSV")"
