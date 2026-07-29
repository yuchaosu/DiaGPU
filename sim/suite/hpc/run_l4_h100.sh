#!/bin/bash
# ===========================================================================
# run_l4_h100.sh — kernel sweep (+L4 ours_sym) + SpMSpM ablation + e2e, on gpu17.
#   bash sim/suite/hpc/run_l4_h100.sh
#
# L4 / upper-half (gather_flat_sym) is VALID ONLY for symmetric H.
#   HamLib matrices  -> e2e run WITH --sym (L4 upper-half operator-build)
#   SuiteSparse      -> e2e run WITHOUT --sym (full-C); NEVER use upper-half
# In the kernel sweep, spmspm_driver always reports BOTH ours_flat (full) and
# ours_sym (L4); for SuiteSparse use ours_flat (ours_sym's check flags non-sym).
# ===========================================================================
set -u
PROJ=/rsstu/users/d/dlee48/HRI-NCSU_LearningEnhancedMotionPlanning/xinlei/projects
REPO=$PROJ/DiaGPU; DIA=$PROJ/DIAMOND/data; PB=$REPO/sim/paper_bench; BIN=$REPO/_work/bin
DRAW=$REPO/_work/drawloom_ae/drawloom_ae/Drawloom_code/myfloat; D2MTX=$BIN/dia_to_mtx
ITERS=${ITERS:-200}; SPITERS=${SPITERS:-100}; STEPS=${STEPS:-1000}
SS="tols4000 rdb5000 fv1 flowmeter0"          # SuiteSparse: never upper-half

set +u; source ~/.bashrc 2>/dev/null; conda activate cudadev 2>/dev/null; set -u
echo "host=$(hostname)"; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || { echo "no GPU"; exit 2; }
mkdir -p "$BIN" "$REPO/results"
echo "== build (sm_90a) =="
( cd "$PB" && make spmv_driver spmspm_driver e2e_driver ARCH=sm_90a ) || { echo build_fail; exit 1; }
nvcc -O3 -std=c++17 -arch=sm_90a $REPO/sim/spmspm_ablation.cu $REPO/spmspm/paper_hm_kernel.cu \
  -o $BIN/spmspm_ablation -Xcompiler -fopenmp || { echo abl_build_fail; exit 1; }

is_ss(){ for s in $SS; do [ "$1" = "$s" ] && return 0; done; return 1; }
ALL=$(cd "$DIA" && ls *.txt 2>/dev/null | grep -v imag | sed 's/\.txt$//' | sort)

# ---- 1) kernel sweep: SpMV + SpMSpM (ours_flat + ours_sym/L4 + all baselines) ----
CSV=$REPO/results/paper_sweep_l4.csv
echo "# $(date -Is) host=$(hostname) L4-enabled" > "$CSV"
echo "== kernel sweep -> $CSV =="
for m in $ALL; do f="$DIA/$m.txt"; echo "== $m =="
  "$PB/spmv_driver"   "$f" "$ITERS"   | tee -a "$CSV"
  "$PB/spmspm_driver" "$f" "$SPITERS" | tee -a "$CSV"
  if [ -x "$DRAW" ] && [ -x "$D2MTX" ]; then mtx=/tmp/${m}_$$.mtx
    "$D2MTX" "$f" "$mtx" >/dev/null 2>&1 && { ms=$(OMP_NUM_THREADS=16 "$DRAW" -filename "$mtx" 2>/dev/null | sed -n 's/.*drawloom time:\s*\([0-9.]*\)\s*ms.*/\1/p'|head -1); [ -n "$ms" ] && echo "SPMVCSV,$f,na,na,na,na,drawloom,$ms,na" | tee -a "$CSV"; rm -f "$mtx"; }
  fi
done

# ---- 2) SpMSpM ablation (L0 HM .. L4 sym) ----
ACSV=$REPO/results/spmspm_ablation_l4.csv
echo "file,n,Hd,nnzC,L0_hm,L1_meta,L2_flat1,L3_flat4,L4_sym,s1_meta,s2_flat1,s3_flat4,s4_sym,relerr" > "$ACSV"
echo "== SpMSpM ablation -> $ACSV =="
for m in $ALL; do "$BIN/spmspm_ablation" "$DIA/$m.txt" "$SPITERS" --csv 2>/dev/null \
  | grep '^SPABL,' | sed 's#^SPABL,[^,]*/#SPABL,#; s/^SPABL,//' >> "$ACSV" && echo "  ok $m" || echo "  skip $m"; done

# ---- 3) e2e sweep: HamLib --sym (L4 upper), SuiteSparse full ----
ECSV=$REPO/results/e2e_sweep_l4.csv
echo "# $(date -Is) steps=$STEPS K=from-filename  (HamLib=--sym L4, SuiteSparse=full)" > "$ECSV"
echo "== e2e sweep -> $ECSV =="
for m in $ALL; do
  if is_ss "$m"; then flag=""; tag="full"; else flag="--sym"; tag="L4-sym"; fi
  echo "== $m [$tag] =="
  "$PB/e2e_driver" "$DIA/$m.txt" "$STEPS" $flag | tee -a "$ECSV" || echo "SKIP,$m"
done

echo "== DONE =="
echo "  kernel: SpMV=$(grep -c '^SPMVCSV' "$CSV") SpMSpM=$(grep -c '^SPMSPMCSV' "$CSV") ours_sym=$(grep -c 'ours_sym' "$CSV")"
echo "  ablation=$(($(wc -l < "$ACSV")-1))  e2e opbuild=$(grep -c '^E2EOPBUILDCSV' "$ECSV")"
