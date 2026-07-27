#!/usr/bin/env python3.11
"""Is truncating small Pauli coefficients worth it, and what does it cost in fidelity?
For each family: coefficient distribution, how #terms and #DIAGONALS shrink with the
drop threshold, and (small q) the evolution fidelity |<psi_full|psi_trunc>|^2 at t=1.2.
"""
import sys, numpy as np
from scipy.linalg import expm
from export_pauli import load_op

def diag_count(spo):
    # distinct offsets = distinct (xmask - 2*b) over terms; cheap proxy: build sparse, count offsets
    M = spo.to_matrix(sparse=True).tocoo()
    return len(np.unique(M.col.astype(np.int64) - M.row.astype(np.int64))), M.nnz

def analyze(family, q, do_fid):
    spo = load_op(family, q)
    c = np.abs(spo.coeffs.real)
    print(f"\n=== {family} q{q}: {len(spo)} terms ===")
    print(f"   |coeff|: min={c.min():.2e} max={c.max():.2e}  median={np.median(c):.2e}")
    base_d, base_nnz = (diag_count(spo) if do_fid or q<=12 else (None,None))
    H = spo.to_matrix() if do_fid else None
    if do_fid:
        t = 1.2
        psi = np.random.default_rng(0).standard_normal(1<<q) + 1j*np.random.default_rng(1).standard_normal(1<<q)
        psi /= np.linalg.norm(psi)
        Uf = expm(-1j*H*t); pf = Uf@psi
    for thr in [1e-8,1e-6,1e-4,1e-3,1e-2]:
        keep = c >= thr
        nt = int(keep.sum())
        if nt==0: continue
        sub = spo[keep] if hasattr(spo,'__getitem__') else None
        line = f"   thr={thr:.0e}: terms {nt:>4}/{len(spo)} ({100*nt/len(spo):.0f}%)"
        if q<=12:
            d,nnz = diag_count(sub)
            line += f"  diagonals {d}/{base_d}  nnz {nnz}/{base_nnz}"
        if do_fid:
            Ut = expm(-1j*sub.to_matrix()*t); pt = Ut@psi
            fid = abs(np.vdot(pf,pt))**2
            line += f"  fidelity={fid:.6f}  infid={1-fid:.2e}"
        print(line)

# spin (all-O(1) coeffs) vs molecular (long tail). Fidelity only at small q.
analyze("heis", 10, do_fid=True)
analyze("tfim", 10, do_fid=True)
analyze("BeH", 8, do_fid=True)
analyze("BeH", 12, do_fid=False)
analyze("O2", 16, do_fid=False)
