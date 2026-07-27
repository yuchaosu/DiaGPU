#!/usr/bin/env python3.11
"""Does compression help SpMSpM? Compute C = H*H and measure whether its OUTPUT
diagonals are single-valued (compressible) the way H's are. This is the make-or-break
for the bitmap-AND SpMSpM idea and for fitting the q20 fill-in without OOM.
"""
import sys, glob
import numpy as np
from scipy.sparse import coo_matrix

D = "/mnt/beegfs/ysu34/hamlib/dia_e2e"

def load_sparse(name):
    fn = sorted(glob.glob(f"{D}/{name}_*.txt"))[0]
    lines = open(fn).read().split("\n")
    n = int(lines[0].split()[1])
    rs, cs, vs = [], [], []
    for l in lines[1:]:
        if ":" not in l: continue
        off_s, rest = l.split(":", 1); d = int(off_s)
        for j, t in enumerate(rest.split()):
            v = float(t)
            if v == 0.0: continue
            r, c = (j, j + d) if d >= 0 else (j - d, j)
            rs.append(r); cs.append(c); vs.append(v)
    return fn, n, coo_matrix((vs, (rs, cs)), shape=(n, n))

def diag_stats(M, n, label):
    M = M.tocoo()
    off = M.col.astype(np.int64) - M.row.astype(np.int64)
    av = np.abs(M.data)
    # round to kill fp noise from the matmul
    avr = np.round(av, 10)
    uoff = np.unique(off)
    ndiag = len(uoff)
    nnz = M.nnz
    single = 0; multi = 0; sv_nnz = 0; mv_nnz = 0
    # group by offset
    order = np.argsort(off, kind="stable")
    off_s = off[order]; avr_s = avr[order]
    idx = np.searchsorted(off_s, uoff, side="right")
    start = 0
    for e in idx:
        seg = avr_s[start:e]
        if len(np.unique(seg)) == 1: single += 1; sv_nnz += len(seg)
        else: multi += 1; mv_nnz += len(seg)
        start = e
    dense_dia_bytes = ndiag * n * 4
    csr_bytes = nnz * 8
    # compressed estimate: single-val diag -> 1 float + posbitmap(n/8) + signbitmap(n/8); multi -> dense len*4 (~n*4)
    comp = single * (4 + 2*(n//8)) + multi * n * 4
    print(f"  {label:14s} n={n} ndiag={ndiag} nnz={nnz} nnz/row={nnz/n:.1f}")
    print(f"     single-value diags={single}/{ndiag} ({100*single/ndiag:.0f}%)  holding {100*sv_nnz/max(nnz,1):.0f}% of nnz")
    print(f"     dense-DIA={dense_dia_bytes/1e6:.1f}MB  CSR={csr_bytes/1e6:.1f}MB  compressed~={comp/1e6:.1f}MB"
          f"  ({csr_bytes/comp:.2f}x vs CSR, {dense_dia_bytes/comp:.1f}x vs dense-DIA)")

for name in ["heis_10","heis_12","tfim_10","maxcut_10","BeH_8","BeH_10"]:
    try:
        fn, n, H = load_sparse(name)
    except IndexError:
        continue
    print(f"\n=== {name} ===")
    diag_stats(H, n, "H")
    C = (H @ H).tocoo()
    diag_stats(C, n, "C = H*H")
