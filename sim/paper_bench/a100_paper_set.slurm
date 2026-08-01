#!/bin/bash
# ===========================================================================
# a100_paper_set.slurm — the CURRENT protocol, reproduced on A100 (sm_80):
#   * matrices_paper.txt only (q16-20 + SS), REPS=3 medians
#   * trimmed drivers (didx single scheme, diaq fp32, honest cuSPARSE,
#     int32 guards, fixed sym kernel) rebuilt for sm_80 in bin_sm80/
#     (separate dir: the H100 REPS job uses the sm_90a binaries in PB)
#   * ncu: OUR ablation kernels (didx / stream / csr_scalar / didx_sym)
#     AND every baseline (cuSPARSE SpMV+SpGEMM, diaq SpMV+product, HM,
#     gather_flat/_sym, Drawloom) on the representative 4-matrix set
#   * outputs: *_a100_paperset.csv + ncu_abl_a100/  (f_C reused from H100 —
#     it is a matrix property, not a hardware measurement)
# submit:  cd sim/paper_bench && sbatch a100_paper_set.slurm
# ===========================================================================
#SBATCH -J a100_paper_set
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/a100_paper_set_%j.out
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
REPO=/home/ysu34/DiaGPU
PB=$REPO/sim/paper_bench
BIN=$PB/bin_sm80
B=/mnt/beegfs/ysu34/paper_bench
LIST=$PB/matrices_paper.txt
ARCH=sm_80
ITERS=${ITERS:-50}; SPITERS=${SPITERS:-100}; STEPS=${STEPS:-1000}; REPS=${REPS:-3}
DRAW=/mnt/beegfs/ysu34/drawloom_ae/drawloom_ae/Drawloom_code/myfloat   # original sm_80 build
D2M=$REPO/sim/suite/bin/dia_to_mtx
mkdir -p "$BIN" "$B/ncu_abl_a100"
echo "host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 2

echo "== build ($ARCH -> $BIN) =="
NV="nvcc -O3 -std=c++17 -arch=$ARCH --expt-relaxed-constexpr"
NVML="-lnvidia-ml"
( cd "$PB" && \
  $NV prep_driver.cu   "$REPO/spmspm/paper_hm_kernel.cu" -o "$BIN/prep_driver"   -lcusparse -lnvidia-ml && \
  $NV spmspm_driver.cu "$REPO/spmspm/paper_hm_kernel.cu" -o "$BIN/spmspm_driver" -lcusparse -lnvidia-ml -Xcompiler -fopenmp && \
  $NV spmv_driver.cu -o "$BIN/spmv_driver" -lcusparse -lnvidia-ml && \
  $NV e2e_driver.cu  -o "$BIN/e2e_driver"  -lcusparse -lnvidia-ml && \
  $NV cusparse_fix_bench.cu -o "$BIN/cusparse_fix_bench" -lcusparse -lnvidia-ml && \
  $NV spgemm_bench.cu       -o "$BIN/spgemm_bench"       -lcusparse -lnvidia-ml && \
  $NV didx_sym_bench.cu     -o "$BIN/didx_sym_bench" -lnvidia-ml && \
  $NV amort_bench.cu        -o "$BIN/amort_bench"    -lcusparse -lnvidia-ml ) || { echo build_fail; exit 3; }

# ---- stage 1: REPS x full pipeline ----------------------------------------
PC=$B/prep_a100_paperset.csv;   [ -f $PC ] || echo "# $(date -Is) A100 $ARCH iters=$ITERS PREPCSV,..." > $PC
SC=$B/spmspm_a100_paperset.csv; [ -f $SC ] || echo "# $(date -Is) A100 iters=$SPITERS honest-cusparse" > $SC
CC=$B/cusparse_fix_a100.csv;    [ -f $CC ] || echo "# $(date -Is) A100 honest true-nonzero cuSPARSE SpMV" > $CC
EC=$B/e2e_a100_paperset.csv;    [ -f $EC ] || echo "# $(date -Is) A100 steps=$STEPS dynamic-K HamLib --sym" > $EC
for rep in $(seq 1 $REPS); do
for f in $(cat "$LIST"); do
  m=$(basename "$f" .txt)
  echo "== [$rep] $m =="
  timeout 3600 stdbuf -oL "$BIN/prep_driver" "$f" "$ITERS"        | tee -a $PC
  timeout 3600 stdbuf -oL "$BIN/spmspm_driver" "$f" "$SPITERS"    | tee -a $SC
  timeout 1800 stdbuf -oL "$BIN/cusparse_fix_bench" "$f" "$ITERS" | tee -a $CC
  timeout 1800 stdbuf -oL "$BIN/spmv_driver" "$f" "$ITERS"        | tee -a $B/spmv_sweep_a100.csv
  case "$f" in */dia_e2e/*)
    timeout 5400 stdbuf -oL "$BIN/e2e_driver" "$f" "$STEPS" --sym | tee -a $EC ;;
  esac
done
done

# ---- stage 1b: Drawloom timing for the NEW SS matrices (old sweeps predate them)
if [ -x "$DRAW" ] && [ -x "$D2M" ]; then
  for m in ecology2 atmosmodd Lin; do
    f=/mnt/beegfs/ysu34/hamlib/dia_suitesparse/$m.txt; mtx=$B/_${m}_dl.mtx
    "$D2M" "$f" "$mtx" >/dev/null 2>&1 || { echo "SKIP drawloom $m"; continue; }
    for rep in $(seq 1 $REPS); do
      ms=$(OMP_NUM_THREADS=16 timeout 1800 "$DRAW" -filename "$mtx" 2>/dev/null | sed -n 's/.*drawloom time:\s*\([0-9.]*\)\s*ms.*/\1/p' | head -1)
      [ -n "$ms" ] && echo "SPMVCSV,$f,na,na,na,na,drawloom,$ms,na" | tee -a $B/spmv_sweep_a100.csv
    done
    rm -f "$mtx"
  done
fi

# ---- stage 2: ncu — ours ablation + ALL baselines --------------------------
MET="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,gpu__time_duration.sum"
NCU=$(command -v ncu || true)
if [ -n "$NCU" ]; then
for m in heis_20_5 tfim_18_5 qmaxcut_18_4 ecology2; do
  f=/mnt/beegfs/ysu34/hamlib/dia_e2e/$m.txt; [ -f "$f" ] || f=/mnt/beegfs/ysu34/hamlib/dia_suitesparse/$m.txt
  echo "== ncu $m =="
  for kre in spmv_didx spmv_gather spmv_csr_scalar; do   # ours + ablation arms
    PREP_ALL_VARIANTS=1 timeout 1800 ncu --csv --metrics "$MET" --kernel-name "regex:$kre" -c 1 \
      "$BIN/prep_driver" "$f" 3 > $B/ncu_abl_a100/${m}_${kre}.csv 2>/dev/null && echo "  ok $kre" || echo "  FAIL $kre"
  done
  timeout 1800 ncu --csv --metrics "$MET" --kernel-name "regex:didx_sym" -c 1 \
    "$BIN/didx_sym_bench" "$f" 3 > $B/ncu_abl_a100/${m}_didx_sym.csv 2>/dev/null && echo "  ok didx_sym" || echo "  FAIL didx_sym"
  timeout 1800 ncu --csv --metrics "$MET" -c 12 "$BIN/cusparse_fix_bench" "$f" 2 \
    > $B/ncu_abl_a100/${m}_cusparse_spmv.csv 2>/dev/null && echo "  ok cusparse_spmv" || echo "  FAIL cusparse_spmv"
  timeout 1800 ncu --csv --metrics "$MET" --kernel-name "regex:diaq" -c 1 \
    "$BIN/spmv_driver" "$f" 3 > $B/ncu_abl_a100/${m}_diaq_spmv.csv 2>/dev/null && echo "  ok diaq_spmv" || echo "  FAIL diaq_spmv"
  for kre in gather_flat_kernel gather_flat_sym hm_structured diaq_product; do
    timeout 1800 ncu --csv --metrics "$MET" --kernel-name "regex:$kre" -c 1 \
      "$BIN/spmspm_driver" "$f" 3 > $B/ncu_abl_a100/${m}_${kre}.csv 2>/dev/null && echo "  ok $kre" || echo "  FAIL $kre"
  done
  timeout 1800 ncu --csv --metrics "$MET" -c 40 "$BIN/spgemm_bench" "$f" 2 \
    > $B/ncu_abl_a100/${m}_cusparse_spgemm.csv 2>/dev/null && echo "  ok cusparse_spgemm" || echo "  FAIL cusparse_spgemm"
  if [ -x "$DRAW" ] && [ -x "$D2M" ]; then
    mtx=$B/ncu_abl_a100/_$m.mtx
    "$D2M" "$f" "$mtx" >/dev/null 2>&1 && \
      OMP_NUM_THREADS=16 timeout 1800 ncu --csv --metrics "$MET" -c 40 "$DRAW" -filename "$mtx" \
        > $B/ncu_abl_a100/${m}_drawloom.csv 2>/dev/null && echo "  ok drawloom" || echo "  FAIL drawloom"
    rm -f "$mtx"
  else echo "  WARN drawloom binary/converter missing -> N/A"; fi
done
else echo "WARN: ncu not found on node; profiles skipped"; fi
# ---- stage 2b: amortization sweep T(N)=prep+upload+N*kernel ---------------
AM=$B/amort_a100.csv
[ -f $AM ] || echo "# $(date -Is) A100 measured T(N)  AMORTCSV,file,variant,N,total_ms" > $AM
for f in $(cat "$LIST"); do
  m=$(basename "$f" .txt)
  grep -q "/$m.txt,didx,10000" $AM && continue
  timeout 2400 "$BIN/amort_bench" "$f" | tee -a $AM || echo "AMORTFAIL,$m" | tee -a $AM
done

# ---- stage 3: energy sweep (ENERGY=1 hook in tms(); J/apply net of idle) --
E=$B/energy_a100.csv
[ -f $E ] || echo "# $(date -Is) A100 ENERGY sweep: ENERGYRAW,J_net,avg_W,idle_W,iters then the variant row it belongs to" > $E
for f in $(cat "$LIST"); do
  m=$(basename "$f" .txt); echo "== energy $m ==" | tee -a $E
  ENERGY=6 timeout 3600 stdbuf -oL "$BIN/prep_driver"   "$f" 20 | tee -a $E
  ENERGY=6 timeout 3600 stdbuf -oL "$BIN/spmspm_driver" "$f" 20 | tee -a $E
done
echo "== A100 PAPER SET DONE $(date -Is)"