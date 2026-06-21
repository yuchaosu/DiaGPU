#!/usr/bin/env python3.11
"""Validate GPU Pauli SpMSpM: build matrices from the term files and compare
H^2_gpu against (assembled H)^2. Term = c * Z^z X^x; column r -> row r^x with
value c*(-1)^popcount(z & (r^x)).
"""
import sys
import numpy as np
from scipy.sparse import coo_matrix

def load_terms(fn):
    lines = open(fn).read().split("\n")
    T, nq = map(int, lines[0].split())
    xs, zs, cs = [], [], []
    for l in lines[1:1+T]:
        x, z, cr, ci = l.split()
        xs.append(int(x)); zs.append(int(z)); cs.append(float(cr) + 1j*float(ci))
    return nq, np.array(xs, dtype=np.int64), np.array(zs, dtype=np.int64), np.array(cs)

def to_matrix(nq, xs, zs, cs):
    n = 1 << nq
    r = np.arange(n, dtype=np.int64)
    rows_all, cols_all, vals_all = [], [], []
    for x, z, c in zip(xs, zs, cs):
        col = r
        row = r ^ int(x)
        pc = np.array([bin(int(z) & int(rr)).count("1") & 1 for rr in row])
        v = c * np.where(pc == 1, -1.0, 1.0)
        rows_all.append(row); cols_all.append(col); vals_all.append(v)
    return coo_matrix((np.concatenate(vals_all),
                       (np.concatenate(rows_all), np.concatenate(cols_all))),
                      shape=(n, n)).tocsr()

Hf, Cf = sys.argv[1], sys.argv[2]
nq, hx, hz, hc = load_terms(Hf)
H = to_matrix(nq, hx, hz, hc)
nq2, cx, cz, cc = load_terms(Cf)
Cgpu = to_matrix(nq2, cx, cz, cc)
Cref = (H @ H)
diff = abs((Cgpu - Cref))
mx = diff.max() if diff.nnz else 0.0
den = abs(Cref).max()
print(f"nq={nq}  H^2_gpu terms={len(cx)}  max|C_gpu - H@H| = {mx:.2e}  (rel {mx/max(den,1e-30):.2e})")
print("  PASS" if mx < 1e-4 * max(den, 1) else "  FAIL")
