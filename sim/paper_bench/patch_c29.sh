#!/bin/bash
# ===========================================================================
# patch_c29.sh — make-up runs after the int32 guards (see resume_c29.sh for
# stages 1-3).  Rows go to *_c29patch.csv; at merge time, for a matrix present
# in both resume and patch files the PATCH rows win (they carry the recovered
# variants; duplicated spmv rows are fresh remeasurements, same config).
#   stage 4: prep  — 15 matrices that died on baseline int32 limits
#   stage 5: e2e   — 10 matrices (auto cusparse-skip + opbuild int32 wall rows)
#   stage 6: ncu   — csr_zskip kernels (spmv_csr_*) missed by the old regex
# ===========================================================================
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
REPO=/home/ysu34/DiaGPU
PB=$REPO/sim/paper_bench
HAM=/mnt/beegfs/ysu34/hamlib/dia_e2e
OUTDIR=/mnt/beegfs/ysu34/paper_bench
ITERS=${ITERS:-50}
STEPS=${STEPS:-1000}
cd "$PB" || exit 2
echo "patch host=$(hostname) $(date -Is)"

PREP_MATS="BH_20_7 BH_24_8 O2_16_5 O2_16_7 O2_20_5 O2_20_8 heis_20_5 heis_20_6 heis_22_5 heis_22_6 heis_24_5 heis_24_6 tfim_20_5 tfim_22_5 tfim_24_5"
E2E_MATS="BH_24_8 O2_16_5 O2_16_7 O2_20_5 O2_20_8 heis_18_5 qmaxcut_18_4 qmaxcut_20_4 tfim_18_4 tfim_18_5"

PCSV=$OUTDIR/prep_h100_c29patch.csv
[ -f "$PCSV" ] || echo "# $(date -Is) host=$(hostname) ITERS=$ITERS int32-guard patch of 268657-resume  PREPCSV,file,n,D,stage,variant,prep_host_ms,upload_ms,kernel_ms,amort_applies" > "$PCSV"
echo "== stage 4: prep patch -> $PCSV =="
for m in $PREP_MATS; do f=$HAM/$m.txt; [ -f "$f" ] || { echo "SKIP(missing),$m"; continue; }
  echo "== prep $m =="
  timeout 2400 ./prep_driver "$f" "$ITERS" | tee -a "$PCSV"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "SKIP(rc=$rc),$m.txt"
done

ECSV=$OUTDIR/e2e_diaq_h100_c29patch.csv
[ -f "$ECSV" ] || echo "# $(date -Is) host=$(hostname) steps=$STEPS int32-guard patch of 268662-resume (HamLib --sym)" > "$ECSV"
echo "== stage 5: e2e patch -> $ECSV =="
for m in $E2E_MATS; do f=$HAM/$m.txt; [ -f "$f" ] || { echo "SKIP(missing),$m"; continue; }
  echo "== e2e $m [L4-sym] =="
  timeout 3600 ./e2e_driver_sm_90a "$f" "$STEPS" --sym | tee -a "$ECSV"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "SKIP(rc=$rc),$m.txt"
done

echo "== stage 6: ncu csr_zskip make-up =="
PROF_APPEND=1 SPMSPM_KERNELS="" SPMV_KERNELS="spmv_csr" bash "$REPO/sim/suite/hpc/run_ncu_shared.sh"

echo "== PATCH DONE $(date -Is) =="
echo "  prep patch rows=$(grep -c '^PREPCSV' "$PCSV")  e2e patch: ev=$(grep -c '^E2EEVCSV' "$ECSV") opbuild=$(grep -c '^E2EOPBUILDCSV' "$ECSV")"
