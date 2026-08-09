#!/usr/bin/env python3
"""spmspm_magnitude_census.py — do C = H*H entries include many NEGLIGIBLE
(~1e-8-relative) values, or is the fill-in made of honest-sized numbers?

CPU/numpy only (float64 accumulation; H read as the stored fp32), diagonal
offset algebra identical to gather_flat:
    C[r, r+d] = sum_{oi+oj=d} H[r, r+oi] * H[r+oi, r+d]
Per matrix, over each realized C diagonal d (accumulated then discarded —
nothing is stored):
  stored   = sum of full diagonal lengths n-|d|   (our C_stored convention)
  touched  = positions where >=1 pair of TRUE-nonzero H entries contributes
             (structural zeros excluded — they are what f_C already counts)
  nz       = |C| > 0 in f64;  touched - nz = perfect cancellations
  tiny(eps)= among touched, |C| < eps * max|C|    for eps = 1e-4..1e-12
  droppable diagonals: max over the diagonal < 1e-8 * max|C|
usage: python3 spmspm_magnitude_census.py <dia.txt> [...]
"""
import sys, numpy as np

def load_dia_full(path):
    """dict offset -> full-length-n float64 array F[o][r] = H[r][r+o]."""
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

def census(path):
    n, F = load_dia_full(path)
    offs = sorted(F)
    nzmask = {o: F[o] != 0.0 for o in offs}
    pairs = {}
    for oi in offs:
        for oj in offs:
            pairs.setdefault(oi + oj, []).append((oi, oj))
    EPS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    stored = touched = nz = 0
    tiny = np.zeros(len(EPS), dtype=np.int64)
    diag_max = {}
    # pass 1 needs max|C| for relative thresholds -> two passes would double
    # the work; instead collect per-diagonal (acc, cnt) stats in one pass and
    # only histogram magnitudes per diagonal, then combine with global max.
    per_d = []
    for d, pl in sorted(pairs.items()):
        acc = np.zeros(n); cnt = np.zeros(n, dtype=np.int32)
        for oi, oj in pl:
            a = F[oi]; b = F[oj]; am = nzmask[oi]; bm = nzmask[oj]
            if oi >= 0:
                acc[:n - oi] += a[:n - oi] * b[oi:]
                cnt[:n - oi] += am[:n - oi] & bm[oi:]
            else:
                acc[-oi:] += a[-oi:] * b[:n + oi]
                cnt[-oi:] += am[-oi:] & bm[:n + oi]
        lo, hi = max(0, -d), n - max(0, d)
        v = np.abs(acc[lo:hi]); c = cnt[lo:hi]
        t = c > 0
        stored += n - abs(d); touched += int(t.sum()); nz += int((v > 0).sum())
        mx = float(v.max()) if hi > lo else 0.0
        # magnitudes of touched entries, kept as a small histogram vs later max
        tv = v[t]
        per_d.append((d, mx, np.sort(tv) if tv.size else tv))
    maxC = max((m for _, m, _ in per_d), default=0.0)
    drop_diags = sum(1 for _, m, _ in per_d if m < 1e-8 * maxC)
    drop_len = sum(n - abs(d) for d, m, _ in per_d if m < 1e-8 * maxC)
    for _, _, tv in per_d:
        if tv.size:
            tiny += np.searchsorted(tv, np.array(EPS) * maxC)
    name = path.split("/")[-1].replace(".txt", "")
    print(f"== {name}: n={n} D={len(offs)} -> C diags={len(pairs)} maxC={maxC:.3e}")
    print(f"   stored={stored}  touched={touched} ({touched/stored:.3f} of stored)"
          f"  nz={nz}  perfect_cancel={touched - nz}")
    for e, tcount in zip(EPS, tiny):
        print(f"   |C| < {e:g}*maxC : {tcount}  ({tcount/max(touched,1):.4f} of touched)")
    print(f"   droppable diagonals (max < 1e-8*maxC): {drop_diags}/{len(pairs)}"
          f"  stored-share {drop_len/max(stored,1):.4f}", flush=True)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        census(p)
