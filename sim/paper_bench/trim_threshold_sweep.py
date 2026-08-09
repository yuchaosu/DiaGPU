#!/usr/bin/env python3
"""trim_threshold_sweep.py — find the right relative trim threshold tau for
SpMV / SpMSpM: drop |H_ij| < tau * max|H|, measure what survives and what it
costs in accuracy.  CPU f64 only (threshold study; GPU timing runs later on
trimmed DIA files through the unchanged drivers).

Per (matrix, tau):
  keep_nnz  = surviving true nonzeros / original true nonzeros
  keep_D    = diagonals with >=1 survivor (didx smem table + plan size)
  spmv_err  = ||(H - Ht) x||_inf / ||H x||_inf   (exact; x ~ U[-0.5,0.5))
  spmspm    = C_t - C = -(Ht*Delta + Delta*H), Delta = dropped part (sparse):
              exact err_max = max|E| / max|C-ish|, err_fro relative, computed
              pair-by-pair without materializing C  (only for D <= DMAX_EXACT;
              wider matrices get the analytic bound
              ||E||_inf <= ||Delta||_inf * (||H||_inf + ||Ht||_inf) row-sum).
usage: python3 trim_threshold_sweep.py <dia.txt> [...]
"""
import sys, numpy as np

TAUS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
DMAX_EXACT = 1500          # D^2 pair sweep affordable below this

def load(path):
    with open(path) as f:
        hd = f.readline().split(); n = int(hd[1]); F = {}
        for line in f:
            o_s, vals = line.split(":", 1); o = int(o_s)
            v = np.fromstring(vals, dtype=np.float64, sep=" ")
            a = np.zeros(n)
            if o >= 0: a[:n - o] = v
            else:      a[-o:] = v
            F[o] = a
    return n, F

def rowsum_norm(F, n):
    """induced inf-norm = max row abs-sum, from full-length diag arrays."""
    s = np.zeros(n)
    for a in F.values(): s += np.abs(a)
    return float(s.max())

def pair_sweep(A, B, n, EM):
    """accumulate E += A*B diag-products into per-diag max/fro tallies."""
    for oi, a in A.items():
        if not np.any(a): continue
        for oj, b in B.items():
            if not np.any(b): continue
            d = oi + oj
            if abs(d) >= n: continue
            if oi >= 0: c = a[:n - oi] * b[oi:]
            else:       c = a[-oi:] * b[:n + oi]
            acc = EM.setdefault(d, np.zeros(n))
            if oi >= 0: acc[:n - oi] += c
            else:       acc[-oi:] += c

def census(path):
    n, F = load(path)
    name = path.split("/")[-1].replace(".txt", "")
    mx = max(float(np.abs(a).max()) for a in F.values())
    nnz0 = sum(int((a != 0).sum()) for a in F.values())
    D0 = len(F)
    rng = np.random.RandomState(20260801)
    x = rng.random_sample(n) - 0.5
    y = np.zeros(n)
    for o, a in F.items():
        if o >= 0: y[:n - o] += a[:n - o] * x[o:]
        else:      y[-o:] += a[-o:] * x[:n + o]
    ynorm = float(np.abs(y).max())
    exact_C = D0 <= DMAX_EXACT
    print(f"== {name}: n={n} D={D0} nnz={nnz0} max|H|={mx:.3e} "
          f"spmspm_err={'exact' if exact_C else 'BOUND-only'}", flush=True)
    Hn = rowsum_norm(F, n)
    if exact_C:                      # reference max|C| for relative error
        CM = {}
        pair_sweep(F, F, n, CM)
        maxC = max(float(np.abs(a).max()) for a in CM.values())
        froC = float(np.sqrt(sum((a * a).sum() for a in CM.values())))
        del CM
    for tau in TAUS:
        thr = tau * mx
        Ht = {}; Dl = {}
        for o, a in F.items():
            keep = np.where(np.abs(a) >= thr, a, 0.0)
            drop = a - keep
            if np.any(keep): Ht[o] = keep
            if np.any(drop): Dl[o] = drop
        nnz1 = sum(int((a != 0).sum()) for a in Ht.values())
        if not Dl:
            print(f"  tau={tau:g}: NO-OP (nothing below threshold)", flush=True)
            continue
        yt = np.zeros(n)
        for o, a in Dl.items():
            if o >= 0: yt[:n - o] += a[:n - o] * x[o:]
            else:      yt[-o:] += a[-o:] * x[:n + o]
        spmv_err = float(np.abs(yt).max()) / max(ynorm, 1e-300)
        line = (f"  tau={tau:g}: keep_nnz={nnz1/nnz0:.4f} keep_D={len(Ht)}/{D0}"
                f" spmv_relerr={spmv_err:.2e}")
        Dn = rowsum_norm(Dl, n); Htn = rowsum_norm(Ht, n)
        line += f" spmspm_bound={(Dn*(Hn+Htn))/max(Hn*Hn,1e-300):.2e}"
        if exact_C:
            EM = {}
            pair_sweep(Ht, Dl, n, EM); pair_sweep(Dl, F, n, EM)
            emax = max(float(np.abs(a).max()) for a in EM.values())
            efro = float(np.sqrt(sum((a * a).sum() for a in EM.values())))
            line += f" spmspm_maxrel={emax/max(maxC,1e-300):.2e} fro_rel={efro/max(froC,1e-300):.2e}"
        print(line, flush=True)
    print(f"== {name} done", flush=True)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        census(p)
