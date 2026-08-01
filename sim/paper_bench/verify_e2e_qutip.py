#!/usr/bin/env python3.11
"""verify_e2e_qutip.py — external triangle verification of the e2e evolution.

Our pipeline: 1000 steps of the K-term Taylor expansion of e^{-iH dt},
dt=1e-3, fp32  =>  psi_ours ~ e^{-iH * 1.0} psi0.
Two INDEPENDENT references (float64, true-nonzero CSR):
  ref1 = scipy.sparse.linalg.expm_multiply(-1j*H, psi0)      (Krylov)
  ref2 = qutip.sesolve(H, psi0, [0,1])  rtol=1e-10 atol=1e-12 (adaptive ODE)
Reported: fidelity |<ref|ours>|^2, L2 relative error, and ref1-vs-ref2
distance (the references' own error bar — our error is only meaningful
above it).

Inputs per matrix: <dia.txt> + e2e_state_v0.bin / e2e_state_final.bin from
`e2e_driver <dia.txt> 1000 --sym --dump-state` (fp32, re-plane then im-plane).

usage: python3.11 verify_e2e_qutip.py <dia.txt> <v0.bin> <final.bin>
Pinned: qutip 5.3.0, scipy 1.17.1 (record with results).
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


def load_dia_csr(path):
    with open(path) as f:
        head = f.readline().split()
        n = int(head[1])
        rows_i, cols_i, vals = [], [], []
        for line in f:
            off_s, _, rest = line.partition(":")
            d = int(off_s)
            v = np.array(rest.split(), dtype=np.float64)
            p = np.nonzero(v)[0]
            if p.size == 0:
                continue
            r = p if d >= 0 else p - d
            c = r + d
            rows_i.append(r); cols_i.append(c); vals.append(v[p])
    r = np.concatenate(rows_i); c = np.concatenate(cols_i); v = np.concatenate(vals)
    return sp.csr_matrix((v, (r, c)), shape=(n, n)), n


def load_state(path, n):
    x = np.fromfile(path, dtype=np.float32)
    assert x.size == 2 * n, f"{path}: {x.size} != 2n"
    return x[:n].astype(np.float64) + 1j * x[n:].astype(np.float64)


def main():
    dia, v0f, pf = sys.argv[1], sys.argv[2], sys.argv[3]
    H, n = load_dia_csr(dia)
    psi0 = load_state(v0f, n)
    ours = load_state(pf, n)
    t = 1.0

    ref1 = expm_multiply(-1j * t * H.tocsc(), psi0)

    ref2 = None
    try:
        import qutip
        Hq = qutip.Qobj(H)
        p0 = qutip.Qobj(psi0[:, None])
        opt = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 10**7, "progress_bar": False}
        res = qutip.sesolve(Hq, p0, [0.0, t], options=opt)
        ref2 = res.states[-1].full().ravel()
    except Exception as e:  # qutip failure must not kill the scipy comparison
        print(f"# qutip arm failed: {e}", file=sys.stderr)

    def fid(a, b):
        return abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)

    def rel(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    name = dia.split("/")[-1].replace(".txt", "")
    line = (f"QUTIPVER,{name},n={n},fid_scipy={fid(ours, ref1):.12f},"
            f"rel_scipy={rel(ours, ref1):.3e}")
    if ref2 is not None:
        line += (f",fid_qutip={fid(ours, ref2):.12f},rel_qutip={rel(ours, ref2):.3e},"
                 f"ref_vs_ref={rel(ref1, ref2):.3e}")
    print(line)


if __name__ == "__main__":
    main()
