#!/bin/bash
#SBATCH -J eval_1420
#SBATCH -p a100
#SBATCH -w c33
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -o logs/eval_1420_%j.out
# ===========================================================================
# eval_1420.slurm — whole evaluation sweep over ALL existing q14-20 matrices.
# Stages every family variant into one dir, then runs run_all.sh (kernels,
# ablations, e2e+anchor, sparsification, fidelity>=0.99 diagonal budget for
# SpMV+SpMSpM, ncu mechanism). Results -> /mnt/beegfs/ysu34/eval_1420.
# SpMSpM scoped to diagonal+banded+bh (molecular-electronic fill-in auto-N/A).
# ===========================================================================
set -u
REPO=/home/ysu34/DiaGPU
SRC=/mnt/beegfs/ysu34/hamlib
DIA=/mnt/beegfs/ysu34/eval_1420_dia
OUT=/mnt/beegfs/ysu34/eval_1420
mkdir -p "$DIA" "$OUT" "$REPO/logs"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) start=$(date)"

# ALL existing q14-20 matrices (34), every parameter variant
MATS="B2_14_4 BH_20_7 Li2_14_6 O2_16_5 O2_16_7 O2_20_5 O2_20_8 bh_18_5 bh_20_5 \
fermi_14_5 fermi_16_5 fermi_18_5 fermi_20_5 heis_14_5 heis_16_5 heis_18_5 heis_20_5 heis_20_6 \
maxcut_14_5 maxcut_16_5 maxcut_18_5 maxcut_20_5 qmaxcut_14_4 qmaxcut_16_4 qmaxcut_18_4 qmaxcut_20_4 \
tfim_14_4 tfim_16_4 tfim_16_5 tfim_18_4 tfim_18_5 tfim_20_5 tsp_15_4 tsp_18_4"
# SpMSpM-feasible (diagonal + banded + bh); molecular-electronic fill-in excluded
MATS_SP="bh_18_5 bh_20_5 fermi_14_5 fermi_16_5 fermi_18_5 fermi_20_5 \
heis_14_5 heis_16_5 heis_18_5 heis_20_5 heis_20_6 maxcut_14_5 maxcut_16_5 maxcut_18_5 maxcut_20_5 \
qmaxcut_14_4 qmaxcut_16_4 qmaxcut_18_4 qmaxcut_20_4 tfim_14_4 tfim_16_4 tfim_16_5 tfim_18_4 tfim_18_5 tfim_20_5 \
tsp_15_4 tsp_18_4"

# small-q matrices for the QuTiP correctness anchor (run_e2e's default MATS_ANCHOR)
MATS_ANCHOR="heis_8_4 tfim_8_4 fermi_8_4 qmaxcut_8_4"
# stage every matrix + the anchor matrices into one dir (dia_e2e or dia_oom)
for m in $MATS $MATS_ANCHOR; do s=$SRC/dia_e2e/$m.txt; [ -f "$s" ] || s=$SRC/dia_oom/$m.txt
  [ -f "$s" ] && ln -sf "$(readlink -f "$s")" "$DIA/$m.txt" || echo "  WARN missing $m"; done
echo "staged $(ls "$DIA"/*.txt 2>/dev/null | wc -l) matrices (34 sweep + anchor) into $DIA"

export REPO
DIA="$DIA" OUT="$OUT" ARCH=sm_80 STEPS=200 K=6 KMAX=4 ITERS=100 FIDMIN=0.99 NCU=1 \
  MATS="$MATS" MATS_SP="$MATS_SP" \
  bash "$REPO/sim/suite/hpc/run_all.sh"
echo "end=$(date)"
