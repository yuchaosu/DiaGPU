#!/bin/bash
# ===========================================================================
# cusparse_fix.slurm — safety net for the honest (true-nonzero) cuSPARSE
# rebench: finishes whatever matrices cusparse_fix_h100.csv is still missing.
# Idempotent; waits while an interactive run is still appending (heartbeat).
# submit:  cd sim/paper_bench && sbatch cusparse_fix.slurm
# ===========================================================================
#SBATCH -J cusparse_fix
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o logs/cusparse_fix_%j.out
set -u
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
REPO=/home/ysu34/DiaGPU
PB=$REPO/sim/paper_bench
HAM=/mnt/beegfs/ysu34/hamlib/dia_e2e
SSD=/mnt/beegfs/ysu34/hamlib/dia_suitesparse
C=/mnt/beegfs/ysu34/paper_bench/cusparse_fix_h100.csv
cd "$PB" || exit 2
echo "host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name --format=csv,noheader || exit 2

# wait while another writer is actively appending to the fix CSV
while [ -f "$C" ] && [ $(( $(date +%s) - $(stat -c %Y "$C") )) -lt 1800 ]; do
  echo "fix csv still being written ($(date -Is)); waiting 10 min"; sleep 600
done

nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr cusparse_fix_bench.cu \
  -o cusparse_fix_bench -lcusparse || exit 3
[ -f "$C" ] || echo "# $(date -Is) host=$(hostname) honest true-nonzero cuSPARSE rebench, iters=50, ALG2" > "$C"

for f in $(ls "$HAM"/*.txt "$SSD"/*.txt 2>/dev/null | grep -v imag | sort); do
  grep -q "^PREPCSV,$f," "$C" && continue
  echo "== $(basename "$f" .txt)"
  timeout 2400 stdbuf -oL ./cusparse_fix_bench "$f" 50 | tee -a "$C" || echo "SKIP,$(basename "$f")"
done
echo "== DONE: $(grep -c '^PREPCSV' "$C") rows"
