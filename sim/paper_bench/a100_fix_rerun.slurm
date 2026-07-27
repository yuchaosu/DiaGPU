#!/bin/bash
# ===========================================================================
# a100_fix_rerun.slurm — targeted SpMV rerun after the auto-plan D-threshold
# fix (4096 -> 2048): the O2_20 rows should now pick gopt (was didx, 1.68x
# loss); O2_16 must stay didx; qmaxcut rows re-measured as the honest
# borderline (fill = 0.5) — the fix does not change their pick.
# submit: cd sim/paper_bench && sbatch a100_fix_rerun.slurm
# ===========================================================================
#SBATCH -J a100_fix
#SBATCH -p a100
#SBATCH -w c33
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -o logs/a100_fix_%j.out
set -u
REPO=/home/ysu34/DiaGPU
cd "$REPO/sim/paper_bench" || exit 2

make clean && make ARCH=sm_80 spmv_driver || exit 3

DATA=/mnt/beegfs/ysu34/hamlib/dia_e2e
OUTDIR=/mnt/beegfs/ysu34/paper_bench
OUT="$OUTDIR/a100_fix_${SLURM_JOB_ID}.csv"

for m in O2_16_5 O2_16_7 O2_20_5 O2_20_8 qmaxcut_16_4 qmaxcut_18_4 qmaxcut_20_4; do
  f="$DATA/$m.txt"
  [ -f "$f" ] || { echo "SKIP,$m,missing" | tee -a "$OUT"; continue; }
  ./spmv_driver "$f" 200 | tee -a "$OUT"
done 2> "$OUTDIR/a100_fix_plan_${SLURM_JOB_ID}.log"

# addendum: the atomic count for the SpMSpM contention claim.  atomicAdd
# with an unused return value compiles to RED (reduction) ops, so the
# earlier op_atom metric read 0 for the HM baseline — count op_red too.
NCU=$(command -v ncu || true)
if [ -n "$NCU" ]; then
  make spmspm_driver ARCH=sm_80 || exit 4
  "$NCU" --csv --metrics \
    l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum \
    -k "regex:gather_flat|hm_structured|diaq_product" \
    ./spmspm_driver "$DATA/heis_16_5.txt" 3 --no-cusparse \
    > "$OUTDIR/ncu_atomics_heis16_${SLURM_JOB_ID}.csv" 2>&1
  "$NCU" --csv --metrics \
    l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum \
    -k "regex:gather_flat|hm_structured|diaq_product" \
    ./spmspm_driver "$DATA/BeH_12_6.txt" 3 --no-cusparse \
    > "$OUTDIR/ncu_atomics_beh12_${SLURM_JOB_ID}.csv" 2>&1
fi

echo "done -> $OUT"
