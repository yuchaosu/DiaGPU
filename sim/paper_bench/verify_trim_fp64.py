#!/usr/bin/env python3.11
"""verify_trim_fp64.py — rebuild H in complex128 DIRECTLY from the HamLib
HDF5 Pauli terms (independent of the fp32 DIA pipeline) and use it as ground
truth to verify the fp32 trim findings.

Pauli matrix element (m = X|Y flip mask, zy = Z|Y sign mask):
    <r^m| P |r> = coeff * i^{|Y|} * (-1)^{parity(r & zy)}
Build TWICE with different term orders: entries that are true zeros carry
order-DEPENDENT ~1e-16 summation residue (ghost), genuine small entries are
order-INDEPENDENT and match the fp32 file to ~1e-7.

Checks per matrix:
  A. healthy entries (|file| >= 1e-3 max): max rel diff file vs fp64  (fp32 cast level?)
  B. ghost candidates (0 < |file| < 1e-6 max): fp64 magnitude quantiles,
     order-flip ratio |Ha-Hb|/|Ha|, count exactly-0 in fp64
  C. trim-vs-truth: err(x) = ||(Ha - M) x||inf / ||Ha x||inf for
     M = file untrimmed and M = trim(file, tau); trim is FREE iff its error
     stays at the untrimmed (fp32-cast) level.
usage: python3.11 verify_trim_fp64.py  (matrix list is built in)
"""
import re, numpy as np, h5py

HAM = "/mnt/beegfs/ysu34/hamlib"
PORTAL = "/mnt/beegfs/ysu34/hamlib_portal_hdf5"
CASES = [
    # key=None -> auto-match among `cands` by fp64-vs-file healthy-value agreement
    (f"{HAM}/dia_e2e/bh_16_5.txt", f"{HAM}/BH/BH.hdf5", None,
     lambda ks: [k for k in ks if "Lx-8_" in k]),
    (f"{HAM}/dia_e2e/vibbf3_18_3.txt",
     f"{PORTAL}/chemistry/vibrational/all-vib-bf3.hdf5",
     "mu_y_prime_enc_gray_dvalues_8-8-8-8-8-8", None),
    (f"{HAM}/dia_e2e/O2_16_5.txt", f"{HAM}/O2/O2.hdf5", None,
     lambda ks: [k for k in ks if k.endswith("16")]),
]
TERM = re.compile(r'\(?([\d.eE+\-]+(?:[+-][\d.eE]+j)?)\)? \[([^\]]*)\]')
OPQ = re.compile(r'([XYZ])(\d+)')

def parse_terms(hdf5, key, nq, rev=True):
    """rev=True: qiskit SparsePauliOp convention (index j -> bit nq-1-j),
    rev=False: direct little-endian (older build scripts)."""
    with h5py.File(hdf5, 'r') as f:
        txt = f[key][()].decode()
    terms = []
    for m in TERM.finditer(txt):
        if not OPQ.search(m.group(2)):
            continue   # identity term: the generator's regex drops it (label non-empty)
        c = complex(m.group(1)); xy = 0; zy = 0; ny = 0
        for om in OPQ.finditer(m.group(2)):
            j = int(om.group(2))
            if j >= nq: raise ValueError("index exceeds qubit count")
            b = 1 << ((nq - 1 - j) if rev else j); p = om.group(1)
            if p in 'XY': xy |= b
            if p in 'ZY': zy |= b
            if p == 'Y': ny += 1
        terms.append((c * (1j ** ny), xy, zy))
    return terms

def parity(v):
    v = v.copy()
    for s in (16, 8, 4, 2, 1): v ^= v >> s
    return (v & 1).astype(np.float64)

def build(terms, n):
    r = np.arange(n, dtype=np.int64)
    H = {}
    for c, xy, zy in terms:
        amp = c * (1.0 - 2.0 * parity(r & zy))
        d = (r ^ xy) - r
        for dv in np.unique(d):
            sel = d == dv
            a = H.setdefault(int(dv), np.zeros(n, dtype=np.complex128))
            a[sel] += amp[sel]
    return H

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

def healthy_diff(F, H, n, mx):
    hd = 0.0
    for o in F:
        f = F[o]; a = H.get(o, np.zeros(n))
        h = np.abs(f) >= 1e-3 * mx
        if h.any(): hd = max(hd, float(np.abs(np.real(f[h]) - np.real(a[h])).max()) / mx)
    return hd

def run(dia, hdf5, key, candfn):
    n, F = load_f32(dia)
    name = dia.split("/")[-1]
    fmx = max(float(np.abs(a).max()) for a in F.values())
    if key is None:                          # auto-match by value agreement
        with h5py.File(hdf5, 'r') as f:
            cands = candfn(list(f.keys()))
        best = (1e9, None, None, None)
        done = False
        for k in cands:
            for rev in (True, False):
                try:
                    t = parse_terms(hdf5, k, n.bit_length() - 1, rev)
                except ValueError:           # candidate has more qubits than n
                    break
                Ht = build(t, n)
                hd = healthy_diff(F, Ht, n, fmx)
                if hd < best[0]: best = (hd, f"{k}[rev={rev}]", t, Ht)
                if hd < 1e-5: done = True; break
            if done: break
        hd, key, terms, Ha = best
        if key is None or hd > 1e-5:
            print(f"[{name}] NO KEY MATCH (best diff {hd:.2e} @ {key}) — skipping"); return
        print(f"[{name}] matched key={key} (healthy diff {hd:.2e})")
    else:
        terms = parse_terms(hdf5, key, n.bit_length() - 1)
        Ha = build(terms, n)
    Hb = build(list(reversed(terms)), n)
    imax = max(float(np.abs(a.imag).max()) for a in Ha.values())
    Ha = {o: a.real.copy() for o, a in Ha.items()}
    Hb = {o: a.real.copy() for o, a in Hb.items()}
    mx = max(float(np.abs(a).max()) for a in Ha.values())
    print(f"== {name}: key={key} terms={len(terms)} fp64_D={sum(1 for a in Ha.values() if np.abs(a).max()>0)}"
          f" file_D={len(F)} max|H|={mx:.3e} max_imag={imax:.1e}")
    # A. healthy entries
    hd = gm = 0.0
    ghost_f64 = []; ghost_flip = []; ghost_zero = 0; nghost = 0
    for o in F:
        f = F[o]; a = Ha.get(o, np.zeros(n)); b = Hb.get(o, np.zeros(n))
        h = np.abs(f) >= 1e-3 * mx
        if h.any(): hd = max(hd, float(np.abs(f[h] - a[h]).max()) / mx)
        g = (np.abs(f) > 0) & (np.abs(f) < 1e-6 * mx)
        if g.any():
            nghost += int(g.sum())
            ghost_f64.append(np.abs(a[g]))
            ghost_zero += int((a[g] == 0.0).sum())
            nz = a[g] != 0
            if nz.any(): ghost_flip.append(np.abs(a[g][nz] - b[g][nz]) / np.abs(a[g][nz]))
    print(f"   A healthy: max|file-fp64|/max = {hd:.2e}   (fp32 cast level ~1e-7)")
    if nghost:
        gf = np.concatenate(ghost_f64); fl = np.concatenate(ghost_flip) if ghost_flip else np.array([0.])
        print(f"   B ghosts n={nghost}: fp64 |v|/max med={np.median(gf)/mx:.1e}"
              f" p99={np.percentile(gf,99)/mx:.1e}  exactly0={ghost_zero}"
              f"  order-flip med={np.median(fl):.2f}")
    else:
        print("   B ghosts: none in file")
    # C. trim vs fp64 truth — FULL E2E: 1000-step Taylor evolution (protocol
    # dt=0.0012, K = n_taylor from the .meta), all in f64 so every difference
    # is the MATRIX perturbation (fp32 cast / trim), not kernel arithmetic.
    K = 5
    try:
        for l in open(dia + ".meta"):
            if l.startswith("n_taylor"): K = int(l.split()[1])
    except FileNotFoundError:
        pass
    dt = 0.0012; steps = 1000
    rng = np.random.RandomState(20260801)
    psi0 = rng.random_sample(n) - 0.5 + 1j * (rng.random_sample(n) - 0.5)
    psi0 /= np.linalg.norm(psi0)

    def evolve(H):
        psi = psi0.copy()
        for _ in range(steps):
            term = psi; acc = psi.copy()
            for k in range(1, K + 1):
                t2 = spmv(H, term.real, n) + 1j * spmv(H, term.imag, n)
                term = (-1j * dt / k) * t2
                acc = acc + term
            psi = acc
        return psi

    ref = evolve(Ha); rnorm = float(np.linalg.norm(ref))
    def report(tag, M, D):
        p = evolve(M)
        fid = abs(np.vdot(ref, p))**2 / (rnorm**2 * float(np.linalg.norm(p))**2)
        err = float(np.linalg.norm(ref - p)) / rnorm
        print(f"   C e2e {tag}: K={K} infid={1-fid:.2e} state_relerr={err:.2e} D={D}/{len(F)}", flush=True)
        return 1 - fid
    i0 = report("untrimmed-file", F, len(F))
    for tau in (1e-4, 1e-5, 1e-6):
        T = {o: np.where(np.abs(a) >= tau * mx, a, 0.0) for o, a in F.items()}
        T = {o: a for o, a in T.items() if np.any(a)}
        it = report(f"trim tau={tau:g}", T, len(T))
        print(f"      -> {'FREE (within fp32-cast noise)' if it <= 3*max(i0,1e-14) else 'LOSSY beyond cast noise'}", flush=True)

if __name__ == "__main__":
    for dia, h5, key, candfn in CASES:
        run(dia, h5, key, candfn)
