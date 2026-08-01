#!/bin/bash
# ===========================================================================
# spmspm_cusparse_fix.sh — rerun spmspm_driver with the fixed (true-nonzero)
# dia_to_csr so the cuSPARSE SpGEMM baseline stops paying for interior zeros.
# Ours rows are regenerated too (harmless fresh remeasure; DIA path unchanged).
# Idempotent: matrices already present in the output CSV are skipped.
#   bash spmspm_cusparse_fix.sh            (interactive)
#   sbatch spmspm_cusparse_fix.sh          (safety net; same file)
# ===========================================================================
#SBATCH -J spmspm_cufix
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH -o logs/spmspm_cufix_%j.out
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
REPO=/home/ysu34/DiaGPU
PB=$REPO/sim/paper_bench
HAM=/mnt/beegfs/ysu34/hamlib/dia_e2e
SSD=/mnt/beegfs/ysu34/hamlib/dia_suitesparse
C=/mnt/beegfs/ysu34/paper_bench/spmspm_cusparse_fix_h100.csv
ITERS=${ITERS:-100}
cd "$PB" || exit 2
echo "host=$(hostname) $(date -Is)"

# heartbeat guard: if someone else is appending, wait (slurm twin safety)
while [ -f "$C" ] && [ $(( $(date +%s) - $(stat -c %Y "$C") )) -lt 1800 ] \
      && [ "${FORCE:-0}" != 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; do
  echo "csv active; waiting 10 min"; sleep 600
done

nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr spmspm_driver.cu \
  "$REPO/spmspm/paper_hm_kernel.cu" -o spmspm_driver -lcusparse -Xcompiler -fopenmp || exit 3
[ -f "$C" ] || echo "# $(date -Is) host=$(hostname) iters=$ITERS spmspm rerun with TRUE-NONZERO cuSPARSE CSR (dia_to_csr drop_zeros fix)" > "$C"

for f in $(ls "$HAM"/*.txt "$SSD"/*.txt 2>/dev/null | grep -v imag | sort); do
  # done = the LAST variant (cusparse_spgemm) landed; partial rows alone rerun
  grep -q ",$f,.*,cusparse_spgemm," "$C" && continue
  echo "== $(basename "$f" .txt)"
  timeout 3600 stdbuf -oL ./spmspm_driver "$f" "$ITERS" | tee -a "$C"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "SKIP(rc=$rc),$(basename "$f")"
done
echo "== DONE: $(grep -c 'SPMSPMCSV' "$C") rows"
