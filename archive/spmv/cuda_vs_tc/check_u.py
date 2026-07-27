#!/usr/bin/env python3.11
"""Validate a GPU-built U_step (Pauli terms) by fidelity: apply U_step^num_steps to a
state, compare to exact expm(-i H t). Usage: check_u.py <H_file> <U_file> <family> <q> [steps]
"""
import sys, numpy as np
from scipy.sparse.linalg import expm_multiply
from check_pauli import load_terms, to_matrix
from export_pauli import load_op

Hf, Uf, fam, q = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 500
t = 1.2
H = load_op(fam, q); Hs = H.to_matrix(sparse=True); n = 1 << H.num_qubits
nqU, ux, uz, uc = load_terms(Uf)
U = to_matrix(nqU, ux, uz, uc).tocsr()
rng = np.random.default_rng(0)
psi = rng.standard_normal(n) + 1j*rng.standard_normal(n); psi /= np.linalg.norm(psi)
exact = expm_multiply(-1j*Hs*t, psi)
p = psi.copy()
for _ in range(steps): p = U @ p
fid = abs(np.vdot(exact, p))**2 / np.vdot(p, p).real
print(f"{fam} q{q}: U_step {len(ux)} terms, {steps} steps -> fidelity={fid:.8f}  infid={1-fid:.2e}")
