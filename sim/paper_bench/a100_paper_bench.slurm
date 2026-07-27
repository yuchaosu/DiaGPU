#!/bin/bash
# ===========================================================================
# a100_paper_bench.slurm — full SpMV+SpMSpM paper sweep on the A100 (c33):
# both drivers (ours didx/gather/gopt/auto, CSR-zskip, row-dense, cuSPARSE,
# HamSim/libdiaq fp64+fp32, HM atomic) + Drawloom rows via dia_to_mtx.
# Output: /mnt/beegfs/ysu34/paper_bench/a100_all_<jobid>.csv (+ plan log).
# submit:  cd sim/paper_bench && mkdir -p logs && sbatch a100_paper_bench.slurm
# ===========================================================================
#SBATCH -J paper_bench_a100
#SBATCH -p a100
#SBATCH -w c33
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH -o logs/paper_bench_a100_%j.out
set -u
REPO=/home/ysu34/DiaGPU
cd "$REPO/sim/paper_bench" || exit 2

make clean && make ARCH=sm_80 || exit 3

DATA=/mnt/beegfs/ysu34/hamlib/dia_e2e
OUTDIR=/mnt/beegfs/ysu34/paper_bench
mkdir -p "$OUTDIR"

# every non-imag matrix staged in dia_e2e; per-matrix failures (e.g. SpMSpM
# fill-in OOM on the giants) are contained by the drivers and logged as such
MATS=$(ls "$DATA"/*.txt 2>/dev/null | grep -v imag | xargs -n1 basename | sed 's/\.txt$//' | sort | tr '\n' ' ')
echo "matrices: $MATS"

DATA="$DATA" MATS="$MATS" \
OUT="$OUTDIR/a100_all_${SLURM_JOB_ID}.csv" ITERS=200 SPITERS=50 \
DRAWLOOM_BIN=/mnt/beegfs/ysu34/drawloom_ae/drawloom_ae/Drawloom_code/myfloat \
D2MTX="$REPO/sim/suite/bin/dia_to_mtx" \
bash run_all.sh 2> "$OUTDIR/a100_plan_${SLURM_JOB_ID}.log"

# --------------- ncu mechanism profiles (representative subset) -----------
# One CSV per (matrix, driver): DRAM%/L1/L2/SM%/occupancy/long-scoreboard for
# every kernel variant, plus the global-atomic request count that backs the
# "0 vs millions of atomics" SpMSpM claim.  Driver runs with 3 iters — ncu
# replays each launch per metric pass, so keep the workload tiny.
NCU=$(command -v ncu || true)
if [ -n "$NCU" ]; then
  MET="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
  MET="$MET,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
  MET="$MET,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct"
  MET="$MET,sm__throughput.avg.pct_of_peak_sustained_elapsed"
  MET="$MET,sm__warps_active.avg.pct_of_peak_sustained_active"
  MET="$MET,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
  MET="$MET,smsp__inst_executed.sum"
  MET="$MET,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"
  for m in heis_16_5 tfim_12_4 BeH_12_6 BH_20_7; do
    f="$DATA/$m.txt"; [ -f "$f" ] || continue
    "$NCU" --csv --metrics "$MET" \
      -k "regex:didx|gather|tile_vec4|spmv_dia|csr_scalar|diaq_spmv" \
      ./spmv_driver "$f" 3 > "$OUTDIR/ncu_spmv_${m}_${SLURM_JOB_ID}.csv" 2>&1
    "$NCU" --csv --metrics "$MET" \
      -k "regex:gather_flat|hm_structured|diaq_product" \
      ./spmspm_driver "$f" 3 --no-cusparse > "$OUTDIR/ncu_spmspm_${m}_${SLURM_JOB_ID}.csv" 2>&1
  done
else
  echo "WARN: ncu not found on node; mechanism profiles skipped"
fi

echo "done -> $OUTDIR/a100_all_${SLURM_JOB_ID}.csv"
