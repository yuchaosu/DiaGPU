#!/usr/bin/env python3
"""opbuild_trim_verify.py — accuracy of the OPERATOR-BUILD path under trim.

U(dt) = sum_{k=0}^{K} (-i dt H)^k / k!   built via the diagonal SpMSpM chain
(H real => U = U_re + i U_im with even/odd powers), then applied 1000 times.
Reference state: fp64 STEPWISE Taylor evolution of the fp32 file H (SpMV only,
no fill-in) — mathematically identical to applying the exact U 1000x, so any
deviation is chain/trim error, not Taylor truncation.

Variants per matrix:
  exact   : file H, exact chain              (feasible only while diags fit)
  trimin  : trim(file,1e-6) input, exact chain
  chain   : trim(file,1e-6) input + per-step 1e-8 relative trim of each power
Products are streamed per output diagonal (two passes: max, then keep) so
memory = survivors only; a diag-count guard stops infeasible chains honestly.

usage: python3 opbuild_trim_verify.py <dia.txt> [KOVERRIDE]
"""
import sys, numpy as np

TAU_IN = 1e-6; TAU_CHAIN = 1e-8; STEPS = 1000; DT = 0.0012
GUARD_DIAGS = 40000            # per power level (x n x 8B each)

def load_f32(path):
    with open(path) as f:
        n = int(f.readline().split()[1]); F = {}
        for line in f:
            o_s, vals = line.split(":", 1); o = int(o_s)
            v = np.fromstring(vals, dtype=np.float64, sep=" ")
            a = np.zeros(n)
            if o >= 0: a[:n - o] = v
            else:      a[-o:] = v
            F[o] = a
    return n, F

def spmv(H, x, n):
    y = np.zeros(n)
    for o, a in H.items():
        if o >= 0: y[:n - o] += a[:n - o] * x[o:]
        else:      y[-o:] += a[-o:] * x[:n + o]
    return y

def prodstream(P, H, n, tau):
    """P*H streamed per output diagonal; per-step relative trim tau (0=off).
    Returns dict or None if survivors exceed GUARD_DIAGS."""
    pairs = {}; npairs = 0
    for oi in P:
        for oj in H:
            d = oi + oj
            if abs(d) < n: pairs.setdefault(d, []).append((oi, oj)); npairs += 1
    if npairs > 3_000_000: return None       # compute guard (CPU study)
    def diag(d):
        acc = np.zeros(n)
        for oi, oj in pairs[d]:
            a = P[oi]; b = H[oj]
            if oi >= 0: acc[:n - oi] += a[:n - oi] * b[oi:]
            else:       acc[-oi:] += a[-oi:] * b[:n + oi]
        return acc
    mx = 0.0
    for d in pairs: mx = max(mx, float(np.abs(diag(d)).max()))
    thr = tau * mx
    out = {}
    for d in pairs:
        a = diag(d)
        if tau > 0: a = np.where(np.abs(a) < thr, 0.0, a)
        if np.any(a): out[d] = a
        if len(out) > GUARD_DIAGS: return None
    return out

def build_U(H, n, K, tau):
    """-> (U_re, U_im, diags, nnz) or None on guard."""
    Ure = {0: np.ones(n)}; Uim = {}
    P = {o: a.copy() for o, a in H.items()}          # H^1
    coef = -DT                                        # (-i dt)^1 -> imag part
    for o, a in P.items(): Uim[o] = coef * a
    for k in range(2, K + 1):
        P = prodstream(P, H, n, tau)
        if P is None: return None
        c = (DT ** k) / np.prod(np.arange(1, k + 1, dtype=float))
        ph = (-1j) ** k
        tgt = Ure if abs(ph.real) > 0.5 else Uim
        s = ph.real if abs(ph.real) > 0.5 else ph.imag
        for o, a in P.items():
            t = tgt.setdefault(o, np.zeros(n)); t += (s * c) * a
    diags = len(set(Ure) | set(Uim))
    nnz = sum(int((a != 0).sum()) for a in list(Ure.values()) + list(Uim.values()))
    return Ure, Uim, diags, nnz

def apply_U(Ure, Uim, psi, n):
    re, im = psi.real.copy(), psi.imag.copy()
    yr = spmv(Ure, re, n) - spmv(Uim, im, n)
    yi = spmv(Ure, im, n) + spmv(Uim, re, n)
    return yr + 1j * yi

def main():
    path = sys.argv[1]
    n, F = load_f32(path)
    K = 5
    try:
        for l in open(path + ".meta"):
            if l.startswith("n_taylor"): K = int(l.split()[1])
    except FileNotFoundError: pass
    if len(sys.argv) > 2: K = int(sys.argv[2])
    name = path.split("/")[-1].replace(".txt", "")
    mx = max(float(np.abs(a).max()) for a in F.values())
    print(f"== {name}: n={n} D={len(F)} K={K}", flush=True)
    rng = np.random.RandomState(20260801)
    psi0 = rng.random_sample(n) - 0.5 + 1j * (rng.random_sample(n) - 0.5)
    psi0 /= np.linalg.norm(psi0)
    # reference: stepwise Taylor with the FILE matrix (isolates chain error)
    psi = psi0.copy()
    for _ in range(STEPS):
        term = psi; acc = psi.copy()
        for k in range(1, K + 1):
            t2 = spmv(F, term.real, n) + 1j * spmv(F, term.imag, n)
            term = (-1j * DT / k) * t2; acc = acc + term
        psi = acc
    ref = psi; rn = float(np.linalg.norm(ref))
    Ft = {o: np.where(np.abs(a) >= TAU_IN * mx, a, 0.0) for o, a in F.items()}
    Ft = {o: a for o, a in Ft.items() if np.any(a)}
    print(f"   input: file D={len(F)}  trim({TAU_IN:g}) D={len(Ft)}", flush=True)
    for tag, H, tau in (("exact-chain(file)", F, 0.0),
                        ("trim-input", Ft, 0.0),
                        (f"trim-input+chain({TAU_CHAIN:g})", Ft, TAU_CHAIN)):
        r = build_U(H, n, K, tau)
        if r is None:
            print(f"   U[{tag}]: INFEASIBLE (> {GUARD_DIAGS} diagonals at some power)", flush=True)
            continue
        Ure, Uim, dg, nnz = r
        p = psi0.copy()
        for _ in range(STEPS): p = apply_U(Ure, Uim, p, n)
        fid = abs(np.vdot(ref, p)) ** 2 / (rn ** 2 * float(np.linalg.norm(p)) ** 2)
        err = float(np.linalg.norm(ref - p)) / rn
        print(f"   U[{tag}]: diags={dg} nnz={nnz}  after {STEPS} applies: "
              f"infid={1 - fid:.2e} state_relerr={err:.2e}", flush=True)
    print(f"== {name} done", flush=True)

if __name__ == "__main__":
    main()
