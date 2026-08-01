#!/bin/bash
# ===========================================================================
# run_ncu_shared.sh — H100 (partition h100/c29): per-kernel shared-mem profile.
# One ncu pass per named kernel; long-format ncu_shared_h100.csv
# (matrix,kernel,metric,value), same shape as run_ncu.sh / plot_ncu.py.
#
# Kernels:
#   SpMSpM (spmspm_ablation, one binary launches all levels):
#     hm_structured_sparse_matmul_kernel  L0 baseline (global atomicAdd)
#     gather_meta_kernel                  L1 smem accumulation, atomic-free
#     gather_flat_kernel                  L2/L3 pair-metadata staged in smem
#     gather_flat_sym_kernel              L4 symmetric upper-half
#   SpMV (spmv_driver) — negative control, expected ~0 shared traffic:
#     cuda_spmv_dia, spmv_dia_zeroskip
#
# Metrics: shared bytes ld/st (+/sec), shared-pipe %peak, bank conflicts,
# static/dynamic smem per block; plus global-ld bytes, DRAM bytes, global
# atomics/red, L1/L2 hit — so shared traffic can be argued as REPLACING
# global traffic, not just as raw activity. Metric names vary across ncu
# versions -> each is probed against `ncu --query-metrics` and dropped
# (with a note) if unsupported on this install.
#
#   usage (on an h100 node, e.g. c29): bash sim/suite/hpc/run_ncu_shared.sh
#          MATS="heis_16_5 O2_16_5" bash run_ncu_shared.sh   # subset
# ===========================================================================
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
REPO=${REPO:-/home/ysu34/DiaGPU}
DIA=${DIA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}
PB=$REPO/sim/paper_bench
BIN=${BIN:-$PB}
OUT=${OUT:-/mnt/beegfs/ysu34/paper_bench}
PY=${PY:-python3}
ARCH=${ARCH:-sm_90a}
ITERS=${ITERS:-3}
TIMEOUT=${NCU_TIMEOUT:-600}
mkdir -p "$BIN" "$OUT"
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) arch=$ARCH out=$OUT"
command -v ncu >/dev/null 2>&1 || { echo "ncu not found"; exit 1; }
ncu --version | head -1

echo "== build ($ARCH) =="
( cd "$PB" && make spmv_driver ARCH=$ARCH ) || { echo build_fail_spmv; exit 1; }
nvcc -O3 -std=c++17 -arch=$ARCH $REPO/sim/spmspm_ablation.cu $REPO/spmspm/paper_hm_kernel.cu \
  -o $BIN/spmspm_ablation -Xcompiler -fopenmp || { echo build_fail_abl; exit 1; }

# ---- probe metric availability on this ncu/GPU (names differ by version) ----
WANT="\
smsp__sass_data_bytes_mem_shared_op_ld.sum \
smsp__sass_data_bytes_mem_shared_op_st.sum \
smsp__sass_data_bytes_mem_shared_op_ld.sum.per_second \
smsp__sass_data_bytes_mem_shared_op_st.sum.per_second \
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
l1tex__t_requests_pipe_lsu_mem_global_op_red.sum \
dram__bytes_read.sum \
l1tex__t_sector_hit_rate.pct \
lts__t_sector_hit_rate.pct"
AVAIL=$OUT/_ncu_avail.txt
ncu --query-metrics 2>/dev/null | awk '{print $1}' | sort -u > "$AVAIL"
METRICS="launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic"
for m in $WANT; do
  if grep -qx "${m%%.*}" "$AVAIL"; then METRICS="$METRICS,$m"
  else echo "  metric unsupported on this ncu, dropped: $m"; fi
done

PROF=$OUT/ncu_shared_h100.csv
# PROF_APPEND=1 appends to an existing CSV (make-up passes) instead of truncating
[ "${PROF_APPEND:-0}" = 1 ] && [ -f "$PROF" ] || echo "matrix,kernel,metric,value" > $PROF

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

# representative set; falls back to first 3 matrices found if none exist
MATS=${MATS:-"heis_16_5 tfim_16_5 fermi_16_5 qmaxcut_16_4 O2_16_5"}
found=0; for m in $MATS; do [ -f "$DIA/$m.txt" ] && found=1; done
if [ "$found" = 0 ]; then
  MATS=$(cd "$DIA" && ls *.txt 2>/dev/null | grep -v imag | sed 's/\.txt$//' | head -3)
  echo "  default MATS not found in $DIA -> using: $MATS"
fi

SPMSPM_KERNELS=${SPMSPM_KERNELS-"hm_structured_sparse_matmul_kernel gather_meta_kernel gather_flat_kernel gather_flat_sym_kernel"}
SPMV_KERNELS=${SPMV_KERNELS-"cuda_spmv_dia spmv_csr"}   # csr_zskip launches spmv_csr_* kernels

echo "== ncu shared-mem profile -> $PROF =="
for m in $MATS; do p=$DIA/$m.txt; [ -f "$p" ] || { echo "  skip $m (missing)"; continue; }
  any=0
  for kre in $SPMSPM_KERNELS; do
    timeout $TIMEOUT ncu --csv --metrics "$METRICS" --kernel-name "regex:$kre" -c 1 \
      $BIN/spmspm_ablation "$p" $ITERS > $OUT/_ncu_raw.csv 2>/dev/null && { emit_ncu "$m" "$OUT/_ncu_raw.csv"; any=1; }
  done
  for kre in $SPMV_KERNELS; do
    timeout $TIMEOUT ncu --csv --metrics "$METRICS" --kernel-name "regex:$kre" -c 1 \
      $PB/spmv_driver "$p" $ITERS > $OUT/_ncu_raw.csv 2>/dev/null && { emit_ncu "$m" "$OUT/_ncu_raw.csv"; any=1; }
  done
  [ "$any" = 0 ] && echo "$m,ncu_skipped,timeout_or_error,-" >> $PROF
  echo "  profiled $m"
done
rm -f $OUT/_ncu_raw.csv $AVAIL
echo "== done: $PROF ($(( $(wc -l < $PROF) - 1 )) rows) =="
