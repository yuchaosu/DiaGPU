#!/usr/bin/env python3.11
"""Verify the ASSEMBLED-chain truncation algorithm (drop whole diagonals with
max|val| < thr/|c_k|) preserves fidelity. Mirrors e2e_spmspm.cu's prune_diag, but in
numpy/scipy so we can compare U_step^steps against exact expm(-iHt). Also uses the
adaptive K from the 1-norm convergence bound, like the kernel.
"""
import sys, numpy as np
from math import factorial
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from export_pauli import load_op

def trunc_diag(M, atol):
    """drop entries whose diagonal (offset=col-row) has max|val| < atol."""
    M = M.tocoo(); off = (M.col.astype(np.int64) - M.row.astype(np.int64))
    a = np.abs(M.data)
    order = np.argsort(off, kind="stable"); offs = off[order]; aa = a[order]
    uoff, idx = np.unique(offs, return_index=True)
    mx = np.maximum.reduceat(aa, idx)                     # max |val| per offset
    keepmap = {int(o): (m >= atol) for o, m in zip(uoff, mx)}
    keep = np.array([keepmap[int(o)] for o in off])
    return coo_matrix((M.data[keep], (M.row[keep], M.col[keep])), shape=M.shape).tocsr()

def adaptive_K(h1, dt, tol):
    nrm = dt*h1; K=0; t=1.0
    while K < 40:
        tn = t*nrm/(K+1)
        if tn < tol: break
        K += 1; t = tn
    return max(K, 2)

def run(fam, q, t=1.2, steps=500, conv_tol=1e-10):
    H = load_op(fam, q); n = 1 << H.num_qubits
    Hs = H.to_matrix(sparse=True).tocsr()
    h1 = abs(Hs).sum(axis=0).max()                        # ||H||_1 = max column abs-sum
    dt = t/steps; K = adaptive_K(h1, dt, conv_tol)
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(n) + 1j*rng.standard_normal(n); psi /= np.linalg.norm(psi)
    exact = expm_multiply(-1j*Hs*t, psi)
    print(f"\n=== {fam} q{q}: ||H||_1={h1:.1f}, dt={dt:.2e} -> adaptive K={K} ===")
    print(f"   {'thr':>7} {'fidelity':>12} {'infidelity':>11}   (whole-diagonal truncation)")
    ck = [(-1j*dt)**k/factorial(k) for k in range(K+1)]
    for thr in [0.0, 1e-8, 1e-6, 1e-4, 1e-3]:
        # U_step = sum_k c_k H^k, truncating each H^k's diagonals at thr/|c_k|
        U = ck[0]*coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n,n)).tocsr()
        Pk = Hs.copy()
        for k in range(1, K+1):
            if thr > 0:
                Pk = trunc_diag(Pk, thr/abs(ck[k]))
                if Pk.nnz == 0: break
            U = U + ck[k]*Pk
            if k < K: Pk = (Pk @ Hs).tocsr()
        p = psi.copy()
        for _ in range(steps): p = U @ p
        fid = abs(np.vdot(exact, p))**2 / np.vdot(p, p).real
        tag = "  <- no truncation" if thr == 0 else ""
        print(f"   {thr:>7.0e} {fid:>12.8f} {1-fid:>11.2e}{tag}")

for fam, q in [("heis",10), ("tfim",10), ("BeH",8)]:
    run(fam, q)
