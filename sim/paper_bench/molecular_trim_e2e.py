#!/usr/bin/env python3
"""molecular_trim_e2e.py — the missing measurement: full-protocol 1000-step
f64 Taylor evolution of the CLEAN molecular rebuilds, trimmed at tau=1e-6 and
1e-8, vs the untrimmed file. Reports final-state infidelity + relerr, i.e.
the true end-to-end cost of each trim tier (matrix perturbation only; all
arithmetic f64). K from the .meta n_taylor, dt=0.0012, seed 20260801.
usage: python3 molecular_trim_e2e.py <dia.txt>
"""
import sys, numpy as np

def load(path):
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

def evolve(H, psi0, n, K, dt=0.0012, steps=1000):
    psi = psi0.copy()
    for _ in range(steps):
        term = psi; acc = psi.copy()
        for k in range(1, K + 1):
            t2 = spmv(H, term.real, n) + 1j * spmv(H, term.imag, n)
            term = (-1j * dt / k) * t2; acc = acc + term
        psi = acc
    return psi

path = sys.argv[1]
n, F = load(path)
K = 5
try:
    for l in open(path + ".meta"):
        if l.startswith("n_taylor"): K = int(l.split()[1])
except FileNotFoundError: pass
mx = max(float(np.abs(a).max()) for a in F.values())
rng = np.random.RandomState(20260801)
psi0 = rng.random_sample(n) - 0.5 + 1j * (rng.random_sample(n) - 0.5)
psi0 /= np.linalg.norm(psi0)
name = path.split("/")[-1].replace(".txt", "")
print(f"== {name}: n={n} D={len(F)} K={K}", flush=True)
ref = evolve(F, psi0, n, K); rn = float(np.linalg.norm(ref))
for tau in (1e-6, 1e-8):
    T = {o: np.where(np.abs(a) >= tau * mx, a, 0.0) for o, a in F.items()}
    T = {o: a for o, a in T.items() if np.any(a)}
    p = evolve(T, psi0, n, K)
    fid = abs(np.vdot(ref, p)) ** 2 / (rn ** 2 * float(np.linalg.norm(p)) ** 2)
    err = float(np.linalg.norm(ref - p)) / rn
    print(f"   TRIMEVOL,{name},tau={tau:g},D={len(T)}/{len(F)},infid={1 - fid:.3e},state_relerr={err:.3e}", flush=True)
print(f"== {name} done", flush=True)
