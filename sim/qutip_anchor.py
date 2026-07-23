#!/usr/bin/env python3.11
"""qutip_anchor.py — correctness anchor: validate our fixed-K Taylor evolution
against the recognized tool (QuTiP sesolve) on the SAME matrix our kernel uses.

Loads the DIA .txt (the exact H the CUDA kernel reads), builds the same initial
vector (xhash, matching e2e_speedup.cu), evolves e^{-iHt}psi0 three ways:
  - QuTiP sesolve            (recognized-tool reference, fp64 adaptive)
  - scipy expm_multiply      (independent high-accuracy cross-check)
  - our fixed-K Taylor loop  (the scheme our CUDA kernel implements), fp64 & fp32
and reports pairwise fidelity |<a|b>|. Small q suffices to validate the scheme;
kernel exactness (zero-skip == dense) is checked separately in the C++ harness.

usage: python3.11 qutip_anchor.py <dia.txt> [final_time=1.2] [steps=1000] [K=6]
"""
import sys, numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

def load_dia(path):
    with open(path) as f:
        hdr = f.readline().split()
        N = int(hdr[1])
        rows, cols, data = [], [], []
        for line in f:
            if ':' not in line: continue
            off_s, vals_s = line.split(':', 1)
            o = int(off_s)
            vals = np.fromstring(vals_s, sep=' ')
            L = vals.size
            if o >= 0:                      # p=row: (row=p, col=p+o)
                r = np.arange(L); c = r + o
            else:                           # p=col: (row=p+|o|, col=p)
                c = np.arange(L); r = c - o
            rows.append(r); cols.append(c); data.append(vals)
    H = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N)).tocsr()
    return N, H

def xhash(i):                               # matches e2e_speedup.cu xhash (uint32)
    h = (np.uint32(i) * np.uint32(2246822519))
    h ^= h >> np.uint32(13); h = h * np.uint32(0x85ebca6b); h ^= h >> np.uint32(16)
    return (h & np.uint32(0xFFFF)).astype(np.float64) / 65536.0 * 2.0 - 1.0

def taylor(H, psi0, dt, steps, K, dtype):
    psi = psi0.astype(dtype)
    Hc = H.astype(np.float64)
    for _ in range(steps):
        term = psi.copy(); acc = psi.copy()
        for k in range(1, K+1):
            term = (np.array(-1j*dt/k, dtype=dtype)) * (Hc @ term).astype(dtype)
            acc = acc + term
        psi = acc
    return psi

def fid(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    path = sys.argv[1]
    ftime = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    dt = ftime / steps
    N, H = load_dia(path)
    idx = np.arange(N)
    psi0 = (xhash(idx) + 1j*xhash(idx + N))
    psi0 /= np.linalg.norm(psi0)
    maxasym = abs(H - H.T).max() if H.nnz else 0.0
    print(f"# {path}  N={N} nnz={H.nnz} K={K} steps={steps} dt={dt:.4g} max|H-H^T|={maxasym:.1e}")

    # reference A: scipy expm_multiply (independent, high accuracy)
    psi_expm = expm_multiply(-1j * H * ftime, psi0)
    # reference B: QuTiP sesolve (recognized tool)
    psi_q = None
    try:
        from qutip import Qobj, sesolve
        res = sesolve(Qobj(H), Qobj(psi0.reshape(-1, 1)), [0.0, ftime],
                      options={"atol": 1e-10, "rtol": 1e-8})
        psi_q = np.asarray(res.states[-1].full()).flatten()
    except Exception as e:
        print(f"# QuTiP sesolve skipped: {e}")
    # our scheme
    psi_t64 = taylor(H, psi0, dt, steps, K, np.complex128)
    psi_t32 = taylor(H, psi0, dt, steps, K, np.complex64)

    print(f"QANCHOR,{path},{N},{K},{steps}," +
          (f"{fid(psi_q, psi_t64):.10f}," if psi_q is not None else "NA,") +
          (f"{fid(psi_q, psi_expm):.10f}," if psi_q is not None else "NA,") +
          f"{fid(psi_expm, psi_t64):.10f},{fid(psi_expm, psi_t32):.10f}")
    print("  columns: fid(qutip,taylor64), fid(qutip,expm), fid(expm,taylor64), fid(expm,taylor32)")

if __name__ == "__main__":
    main()
