#!/usr/bin/env python3.11
"""Export a HamLib Hamiltonian's Pauli terms in the Z^z X^x symplectic form for the
GPU SpMSpM. Operator term = c * Z^z X^x  with c = coeff * (-i)^phase (qiskit's phase
convention). Writes: line1 "T nq"; then T lines "x z cr ci" (x,z integers).
"""
import sys, re, csv
import numpy as np, h5py
from qiskit.quantum_info import SparsePauliOp

HAM = "/mnt/beegfs/ysu34/hamlib"
def find_key(family, q):
    for row in csv.reader(open(f"{HAM}/dia_e2e/generated_matrices.csv")):
        if len(row) > 3 and row[0] == family and row[3] == str(q):
            return f"{HAM}/{row[1]}", row[2]
    return f"{HAM}/{family}/{family}.hdf5", f"graph-1D-grid-nonpbc-qubitnodes_Lx-{q}_h-1"

def load_op(family, q):
    fn, key = find_key(family, q)
    def gen(term):
        idx = [(m.group(1), int(m.group(2))) for m in re.finditer(r'([A-Z])(\d+)', term)]
        return ''.join(next((c for c, i in idx if i == k), 'I') for k in range(max(i for _, i in idx) + 1))
    with h5py.File(fn, 'r') as f:
        pat = r'\(?([\d.-]+(?:[+-][\d.]+j)?)\)? \[([^\]]+)\]'
        ms = re.findall(pat, f[key][()].decode())
        labels = [gen(m[1]) for m in ms]; co = [complex(m[0]).real for m in ms]
        L = max(map(len, labels)); labels = [p + 'I' * (L - len(p)) for p in labels]
    return SparsePauliOp(labels, co).simplify()

PAULI_DIR = "/mnt/beegfs/ysu34/hamlib/pauli"   # term files are data -> beegfs, not the repo

if __name__ == "__main__":
    import os
    family, q = sys.argv[1], int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else f"{PAULI_DIR}/pauli_{family}_{q}.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    spo = load_op(family, q)
    pl = spo.paulis; nq = pl.num_qubits
    Z = pl.z; X = pl.x; ph = pl.phase
    co = spo.coeffs
    # operator term = coeff * (-i)^(phase + #Y) * Z^z X^x, where #Y = qubits with z&x set
    # (Y = -i*ZX, so each Y contributes a (-i) vs the bare Z^z X^x). Verified vs qiskit.
    ny = np.array([int(np.sum(Z[t] & X[t])) for t in range(len(spo))])
    fac = (-1j) ** ((ph + ny) % 4)
    c = co * fac
    with open(out, "w") as f:
        f.write(f"{len(spo)} {nq}\n")
        for t in range(len(spo)):
            xi = sum(int(X[t][b]) << b for b in range(nq))
            zi = sum(int(Z[t][b]) << b for b in range(nq))
            f.write(f"{xi} {zi} {c[t].real:.17g} {c[t].imag:.17g}\n")
    print(f"wrote {out}: {len(spo)} terms, nq={nq}")
