#!/bin/bash
# ===========================================================================
# gen_hamlib.slurm — Phase 0 of the hamlib.csv pipeline.
# Generate the MISSING even-q matrices (q18-28 scalable; q12-24 molecular)
# from the LOCAL HamLib HDF5 sources into dia_oom/. Serial, on the a100 node
# (c33) but CPU/RAM-bound (qiskit sparse Pauli build). 125 GB cap: the biggest
# q28 banded / O2_24 fill-in builds may OOM -> logged FAILED, not fatal.
#
# Keys verified against each dataset's `nqubits` HDF5 attribute (2026-07-12).
# Portal is NOT used (it fuzzy-matches the wrong molecules); local files only.
#
# submit:  sbatch sim/suite/hpc/gen_hamlib.slurm
# ===========================================================================
#SBATCH -J gen_hamlib
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH -o logs/gen_hamlib_%j.out
# CPU-only (qiskit sparse build) — deliberately NOT the a100 node, so it runs in
# parallel with the GPU bench job. Cluster RealMemory is misconfigured (all 1MB)
# so we can't request RAM; the biggest q28-banded/O2_24 builds may OOM whatever
# node this lands on -> logged FAILED (expected; needs turin128 the user excluded).
set -u

HAMLIB=/mnt/beegfs/ysu34/hamlib
OUT=$HAMLIB/dia_oom
GEN=$HAMLIB/fetchham_sparse_dia.py
export HAMLIB_KMAX=40
mkdir -p "$OUT" "$(dirname "$0")/logs"
cd "$HAMLIB" || exit 2

echo "host=$(hostname)  job=$SLURM_JOB_ID  start=$(date)"
free -g | awk '/Mem/{print "RAM total="$2"G"}'

# tag  local_hdf5(relative to $HAMLIB)  key
# Ordered cheap/likely-success first, RAM-risky (q28 banded, O2_24, vib-q24) last.
MANIFEST=$(cat <<'EOF'
c2h      c2h/all-vib-c2h.hdf5        enc_gray_dvalues_8-8-8-8
c2h      c2h/all-vib-c2h.hdf5        enc_gray_dvalues_16-16-8-8
c2h      c2h/all-vib-c2h.hdf5        enc_gray_dvalues_16-16-16-16
hnc      hnc/all-vib-hnc.hdf5        enc_gray_dvalues_8-8-8-8
hnc      hnc/all-vib-hnc.hdf5        enc_gray_dvalues_16-16-8-8
hnc      hnc/all-vib-hnc.hdf5        enc_gray_dvalues_16-16-16-16
Li2      chemistry/Li2/Li2.hdf5      ham_BK14
O2       O2/O2.hdf5                  ham_BK22
fermi    fermi/fermi.hdf5            fh-graph-1D-grid-nonpbc-qubitnodes_Lx-9_U-2_enc-jw
fermi    fermi/fermi.hdf5            fh-graph-1D-grid-nonpbc-qubitnodes_Lx-10_U-2_enc-jw
maxcut   maxcut/maxcut.hdf5          complbipart-n-22_a-11_b-11
maxcut   maxcut/maxcut.hdf5          complbipart-n-24_a-12_b-12
qmaxcut  qmaxcut/qmaxcut.hdf5        graph-n-22
qmaxcut  qmaxcut/qmaxcut.hdf5        graph-n-24
fermi    fermi/fermi.hdf5            fh-graph-1D-grid-nonpbc-qubitnodes_Lx-12_U-2_enc-jw
O2       O2/O2.hdf5                  ham_BK24
c2h      c2h/all-vib-c2h.hdf5        enc_unary_dvalues_8-8-4-4
hnc      hnc/all-vib-hnc.hdf5        enc_unary_dvalues_8-8-4-4
maxcut   maxcut/maxcut.hdf5          complbipart-n-26_a-13_b-13
maxcut   maxcut/maxcut.hdf5          complbipart-n-28_a-14_b-14
qmaxcut  qmaxcut/qmaxcut.hdf5        graph-n-26
qmaxcut  qmaxcut/qmaxcut.hdf5        graph-n-28
fermi    fermi/fermi.hdf5            fh-graph-1D-grid-nonpbc-qubitnodes_Lx-14_U-2_enc-jw
EOF
)

ok=0; fail=0; skip=0
while read -r tag rel key; do
  [ -z "$tag" ] && continue
  echo "======================================================================"
  echo "[$(date +%H:%M:%S)] tag=$tag  key=$key"
  HAMLIB_TAG="$tag" python3.11 "$GEN" "$HAMLIB/$rel" "$key" 1.2 1000 "$OUT"
  rc=$?
  if   [ "$rc" -eq 0 ]; then echo "OK: $tag $key"; ok=$((ok+1))
  elif [ "$rc" -eq 7 ]; then echo "SKIP (K>KMAX): $tag $key"; skip=$((skip+1))
  else echo "FAILED (rc=$rc, maybe host-OOM): $tag $key"; fail=$((fail+1)); fi
done <<< "$MANIFEST"

echo "======================================================================"
echo "done=$(date)  OK=$ok  SKIP=$skip  FAILED=$fail"
echo "dia_oom now has:"
ls -1 "$OUT"/*.txt 2>/dev/null | xargs -n1 basename 2>/dev/null | sort | grep -vE 'imag|csr' | tail -40
