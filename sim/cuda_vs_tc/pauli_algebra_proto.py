#!/usr/bin/env python3.11
"""Prototype: compute H^k by Pauli algebra (P_a P_b = phase * P_{a^b}) instead of
assembling the matrix. Answers:
  (1) does term-count of H^k stay manageable? (grows with q, not n=2^q?)
  (2) does it dodge the dense fill that OOMs SpMSpM at q20 / walls operator-build at q>=16?
  (3) is it correct? (H^k Pauli  vs  (assembled H)^k, at small q)

Uses qiskit SparsePauliOp.dot + .simplify (the exact symplectic product). This measures
FEASIBILITY of the Pauli-basis SpMSpM/operator-build before committing to a CUDA kernel.
"""
import sys, re, time, glob, csv
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

K = 6                       # Taylor order in the operator build
def study(family, q, verify):
    H = load_op(family, q); nq = H.paulis.num_qubits; n = 1 << nq
    terms = [len(H)]; t0 = time.time()
    Hk = H
    for k in range(2, K + 1):
        Hk = Hk.dot(H).simplify()
        terms.append(len(Hk))
    dt = time.time() - t0
    # dense comparison for H^2 (what SpMSpM actually builds)
    H2 = H.dot(H).simplify()
    line = f"{family} q{q} (n={n}): H terms={terms[0]:>5}  H^k terms k=1..{K}: {terms}"
    print(line)
    print(f"     H^2 Pauli terms={len(H2):>6}   vs dense H^2 storage ~ (diags*n);  build {dt:.2f}s for chain to k={K}")
    if verify:
        import scipy.sparse as sp
        Hm = H.to_matrix(sparse=True)
        H2m_alg = H2.to_matrix(sparse=True)
        H2m_dir = (Hm @ Hm)
        err = abs((H2m_alg - H2m_dir)).max()
        # assembled H^2 diagonal count & nnz (what dense-DIA SpMSpM stores)
        C = H2m_dir.tocoo(); off = C.col.astype(np.int64) - C.row.astype(np.int64)
        print(f"     CORRECT: max|H^2_alg - H^2_direct| = {err:.2e}   "
              f"(assembled H^2: {len(np.unique(off))} diagonals, {C.nnz} nnz; "
              f"Pauli rep is {C.nnz/len(H2):.0f}x fewer terms than nnz)")

print("=== Pauli-algebra operator build: term-count growth ===")
for fam in ["heis", "tfim"]:
    for q in [8, 10, 12, 14]:
        try: study(fam, q, verify=(q <= 10))
        except Exception as e: print(f"{fam} q{q}: {e}")
    print()
