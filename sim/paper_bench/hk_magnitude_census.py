#!/usr/bin/env python3
"""hk_magnitude_census.py — magnitude structure of the WHOLE power chain
H^2..H^K (the SpMSpM/opbuild chain), f64 CPU, diagonal offset algebra.

Two chains propagate side by side:
  EXACT   P_k = P_{k-1} * H                       (f64, no storage of C beyond
                                                   the diagonal dict)
  TRIMMED T_k = trim( T_{k-1} * H ),  trim = drop |v| < tau*max|T| entries,
          then drop all-zero diagonals            (tau default 1e-8: below the
                                                   fp32 resolution of the max)
Per k we report, for EXACT: realized diagonals D, stored, nnz, dust fractions
|v| < eps*max among nnz (eps 1e-6/1e-8/1e-12), droppable diagonals @1e-8;
for TRIMMED: D, stored, nnz and max-relative deviation from EXACT.
A memory guard stops a chain when the next product would exceed MEMGB.

usage: python3 hk_magnitude_census.py <dia.txt> [K=4] [tau=1e-8] [MEMGB=20]
"""
import sys, numpy as np

def load_dia_full(path):
    with open(path) as f:
        hd = f.readline().split(); n = int(hd[1])
        F = {}
        for line in f:
            o_s, vals = line.split(":", 1)
            o = int(o_s)
            v = np.fromstring(vals, dtype=np.float64, sep=" ")
            a = np.zeros(n)
            if o >= 0: a[:n - o] = v
            else:      a[-o:] = v
            F[o] = a
    return n, F

def dia_mul(P, H, n, membytes):
    """P*H as dict offset->array; None if projected size exceeds membytes."""
    outoffs = {oi + oj for oi in P for oj in H if abs(oi + oj) < n}
    if len(outoffs) * n * 8 > membytes: return None
    out = {d: np.zeros(n) for d in outoffs}
    for oi, a in P.items():
        if abs(oi) >= n: continue
        for oj, b in H.items():
            if abs(oi + oj) >= n: continue
            acc = out[oi + oj]
            if oi >= 0: acc[:n - oi] += a[:n - oi] * b[oi:]
            else:       acc[-oi:]    += a[-oi:] * b[:n + oi]
    # zero out the out-of-band ends (rows outside the diagonal's valid range
    # are already zero because inputs are zero there)
    return out

def stats(P, n, label, k, ref=None):
    mx = max((float(np.abs(a).max()) for a in P.values()), default=0.0)
    D = sum(1 for a in P.values() if np.any(a != 0))
    stored = sum(n - abs(d) for d, a in P.items() if np.any(a != 0))
    nz = dust6 = dust8 = dust12 = 0
    drop = droplen = 0
    for d, a in P.items():
        v = np.abs(a[a != 0.0])
        if v.size == 0: continue
        nz += v.size
        dust6 += int((v < 1e-6 * mx).sum())
        dust8 += int((v < 1e-8 * mx).sum())
        dust12 += int((v < 1e-12 * mx).sum())
        if v.max() < 1e-8 * mx: drop += 1; droplen += n - abs(d)
    line = (f"  {label} k={k}: D={D} stored={stored} nnz={nz} max={mx:.3e}"
            f" dust<1e-6:{dust6/max(nz,1):.3f} <1e-8:{dust8/max(nz,1):.3f}"
            f" <1e-12:{dust12/max(nz,1):.3f} dropdiag={drop}({droplen/max(stored,1):.3f})")
    if ref is not None:
        err = 0.0; rmx = max((float(np.abs(a).max()) for a in ref.values()), default=0.0)
        for d in set(P) | set(ref):
            a = P.get(d); b = ref.get(d)
            if a is None: err = max(err, float(np.abs(b).max()))
            elif b is None: err = max(err, float(np.abs(a).max()))
            else: err = max(err, float(np.abs(a - b).max()))
        line += f" err_rel={err/max(rmx,1e-300):.2e}"
    print(line, flush=True)

def trim(P, n, tau):
    mx = max((float(np.abs(a).max()) for a in P.values()), default=0.0)
    out = {}
    for d, a in P.items():
        a = np.where(np.abs(a) < tau * mx, 0.0, a)
        if np.any(a != 0): out[d] = a
    return out

def main():
    path = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    memgb = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
    membytes = memgb * 1e9
    n, H = load_dia_full(path)
    name = path.split("/")[-1].replace(".txt", "")
    print(f"== {name}: n={n} D={len(H)} K={K} tau={tau:g} guard={memgb}GB", flush=True)
    P = dict(H); T = trim(dict(H), n, tau)
    exact_alive = True
    for k in range(2, K + 1):
        if exact_alive:
            Pn = dia_mul(P, H, n, membytes)
            if Pn is None:
                print(f"  EXACT k={k}: memory guard tripped "
                      f"(projected diagonals exceed {memgb}GB) — exact chain stops", flush=True)
                exact_alive = False
            else:
                P = {d: a for d, a in Pn.items() if np.any(a != 0)}
                stats(P, n, "EXACT", k)
        Tn = dia_mul(T, H, n, membytes)
        if Tn is None:
            print(f"  TRIM  k={k}: memory guard tripped — trimmed chain stops", flush=True)
            break
        T = trim(Tn, n, tau)
        stats(T, n, "TRIM ", k, ref=P if exact_alive else None)
    print(f"== {name} done", flush=True)

if __name__ == "__main__":
    main()
