#!/usr/bin/env python3.11
"""Visualize why wide molecular Hamiltonians fight spatial sparse formats.

Reads the DIA .txt (lines "offset: v0 v1 ..."), reconstructs (row,col) of every
nonzero, and draws:
  (a) full spy of a narrow band (heis) vs a wide molecular matrix (BeH)
  (b) a zoom showing the within-band scatter (XOR/Pauli structure)
  (c) per-16x16-tile occupancy heatmap (why "drop zero tiles" doesn't help)
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/mnt/beegfs/ysu34/hamlib/dia_e2e"
def load_coords(name):
    import glob
    fn = sorted(glob.glob(f"{D}/{name}_*.txt"))[0]
    lines = open(fn).read().split("\n")
    n = int(lines[0].split()[1])
    rs, cs = [], []
    for l in lines[1:]:
        if ":" not in l: continue
        off_s, rest = l.split(":", 1); d = int(off_s)
        for j, t in enumerate(rest.split()):
            if t == "0" or float(t) == 0.0: continue
            if d >= 0: r, c = j, j + d
            else:      r, c = j - d, j
            rs.append(r); cs.append(c)
    return fn, n, np.array(rs), np.array(cs)

heis = load_coords("heis_12")   # narrow band, n=4096
beh  = load_coords("BeH_12")    # wide molecular, n=4096
print("heis", heis[1], len(heis[2]), "nnz")
print("BeH ", beh[1],  len(beh[2]),  "nnz")

fig, ax = plt.subplots(2, 3, figsize=(16, 10))

def spy(a, name, n, r, c, ms):
    a.scatter(c, r, s=ms, c="#1f77b4", marker=".", linewidths=0)
    a.set_xlim(0, n); a.set_ylim(n, 0); a.set_aspect("equal")
    a.set_title(name); a.set_xlabel("col"); a.set_ylabel("row")

# (a) full spy
spy(ax[0,0], f"heis q12 (narrow band, {len(heis[2])} nnz)", heis[1], heis[2], heis[3], 0.5)
spy(ax[1,0], f"BeH q12 (wide molecular, {len(beh[2])} nnz)", beh[1], beh[2], beh[3], 0.5)

# (b) zoom 0..256 to show within-band scatter
def zoom(a, name, r, c, w=256):
    m = (r < w) & (c < w)
    a.scatter(c[m], r[m], s=8, c="#d62728", marker="s", linewidths=0)
    a.set_xlim(0, w); a.set_ylim(w, 0); a.set_aspect("equal")
    a.set_title(name + f"  (zoom {w}x{w})"); a.set_xlabel("col"); a.set_ylabel("row")
    for g in range(0, w+1, 16): a.axhline(g, color="k", lw=0.2, alpha=.4); a.axvline(g, color="k", lw=0.2, alpha=.4)
zoom(ax[0,1], "heis q12", heis[2], heis[3])
zoom(ax[1,1], "BeH q12",  beh[2],  beh[3])

# (c) 16x16 tile occupancy: fraction of tiles in each macro-region that are non-empty,
#     and the density WITHIN non-empty tiles (the MMA-useful fraction).
def tilestats(a, name, n, r, c, B=16):
    nt = (n + B - 1) // B
    cnt = np.zeros((nt, nt), dtype=np.int32)
    np.add.at(cnt, (r // B, c // B), 1)
    nonempty = cnt > 0
    occ = nonempty.mean()
    within = cnt[nonempty].mean() / (B*B)   # avg density inside non-empty tiles
    im = a.imshow(cnt, cmap="viridis", aspect="equal")
    a.set_title(f"{name}: 16x16 tiles\nnon-empty={occ*100:.1f}%  within-tile density={within*100:.1f}%")
    a.set_xlabel("tile col"); a.set_ylabel("tile row")
    plt.colorbar(im, ax=a, fraction=0.046, label="nnz in tile (max %d)" % (B*B))
tilestats(ax[0,2], "heis q12", heis[1], heis[2], heis[3])
tilestats(ax[1,2], "BeH q12",  beh[1],  beh[2],  beh[3])

fig.suptitle("Why wide molecular matrices fight spatial sparse formats\n"
             "narrow band (top) = contiguous & fillable; wide molecular (bottom) = XOR/Pauli scatter",
             fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparsity_patterns.png")
fig.savefig(out, dpi=130); print("wrote", out)
