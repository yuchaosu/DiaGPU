#!/bin/bash
# Convert the remaining expanded-coverage Hamiltonians to dia .txt + .csr.npz.
# Each: HAMLIB_TAG=<fam> python3.11 fetchham_sparse_dia.py <hdf5> <key> 1.2 1000 dia_expand
set -u
cd /mnt/beegfs/ysu34/hamlib
mkdir -p dia_expand
run(){ tag=$1; hdf5=$2; key=$3
  echo "############ $tag <= $key"
  HAMLIB_TAG=$tag python3.11 fetchham_sparse_dia.py "$hdf5" "$key" 1.2 1000 dia_expand 2>&1 \
    | grep -E "num_qubits|non-zero diag|taylor_conv|saved to:|saved CSR|Error|Traceback|MemoryError" | head -8
  echo
}
run O2   O2/O2.hdf5              "ham_BK16"
run O2   O2/O2.hdf5              "ham_BK20"
run c2h  c2h/all-vib-c2h.hdf5    "enc_stdbinary_dvalues_16-16-8-8"
run c2h  c2h/all-vib-c2h.hdf5    "enc_stdbinary_dvalues_16-16-16-16"
run hnc  hnc/all-vib-hnc.hdf5    "enc_stdbinary_dvalues_16-16-8-8"
run hnc  hnc/all-vib-hnc.hdf5    "enc_stdbinary_dvalues_16-16-16-16"
run Li2  chemistry/Li2/Li2.hdf5  "ham_BK14"
echo "ALL_CONVERSIONS_DONE"
ls -la dia_expand/*.txt | awk '{print $5, $9}'
