#!/usr/bin/env python3
"""pair_energy.py — pair ENERGYRAW rows with the variant row they precede.

The ENERGY=1 hook in tms() prints  ENERGYRAW,J_net,avg_W,idle_W,iters
immediately BEFORE the benchmark prints its own PREPCSV/SPMSPMCSV row, so
each ENERGYRAW pairs with the next data row.  J_net is net Joules PER APPLY:
(avg_W - idle_W) * wall / iters, idle re-measured at each probe start.

Validity rules:
  * avg_W <= idle_W: load too short/light for the 25 ms NVML polling -> N/A.
  * implied net power J_net/kernel_s < 20 W: the idle baseline was captured
    on a still-warm GPU, wiping out the net signal (hits long low-power
    kernels like diaq SpMSpM, and sub-5us kernels) -> N/A.  20 W is a
    heuristic floor: H100 NVL true idle is ~60-90 W, so a real running
    kernel always adds far more than 20 W over a correct baseline.

Sources: SpMV = PREPCSV spmv rows; SpMSpM = SPMSPMCSV rows (the dedicated
driver; the PREPCSV spmspm duplicates are skipped).  cusparse_spgemm has no
hook (own timing path) -> no energy column for it.

usage: python3 pair_energy.py [energy_csv ...]   (default: H100 file)
out:   results/paper_tables/energy_main.csv
"""
import csv, os, sys, collections, statistics, math

_here = os.path.dirname(os.path.abspath(__file__))
FILES = sys.argv[1:] or ["/mnt/beegfs/ysu34/paper_bench/energy_h100.csv"]
OUT = os.path.join(_here, "..", "..", "results", "paper_tables")
PAPER = {l.strip().split("/")[-1][:-4] for l in open(os.path.join(_here, "matrices_paper.txt")) if l.strip()}

def matkey(p): return p.split("/")[-1].replace(".txt", "")

# (matrix, phase, variant) -> [J_net]  (None entries = measured-but-invalid)
cells = collections.defaultdict(list)
for path in FILES:
    pend = None                      # last unconsumed ENERGYRAW
    for line in open(path):
        r = line.rstrip("\n").split(",")
        if r[0] == "ENERGYRAW":
            pend = (float(r[1]), float(r[2]), float(r[3]))
            continue
        if r[0] == "PREPCSV" and len(r) >= 9 and r[4] == "spmv":
            key = (matkey(r[1]), "spmv", r[5]); kms = float(r[8])
        elif r[0] == "SPMSPMCSV" and len(r) >= 8:
            key = (matkey(r[1]), "spmspm", r[6]); kms = float(r[7])
        else:
            if r[0] in ("PREPCSV", "SPMSPMCSV"):
                pend = None          # spmspm-in-prep duplicate consumes its raw
            continue
        if pend is None: continue
        j, avg, idle = pend; pend = None
        net_w = j / (kms * 1e-3) if kms > 0 else 0.0
        cells[key].append(j if (avg > idle and net_w >= 20.0) else None)

def med(k):
    xs = [x for x in cells.get(k, []) if x is not None]
    return statistics.median(xs) if xs else None

SPMV_V = ["didx", "didx_sym", "gopt", "cusparse_csr"]
SPM_V = ["ours_flat", "ours_sym", "hm_atomic", "diaq_fp32"]
os.makedirs(OUT, exist_ok=True)
rows, ratios = [], []
for m in sorted(PAPER):
    sv = {v: med((m, "spmv", v)) for v in SPMV_V}
    sp = {v: med((m, "spmspm", v)) for v in SPM_V}
    if all(v is None for v in list(sv.values()) + list(sp.values())): continue
    ours = sv["didx_sym"] if sv["didx_sym"] is not None else sv["didx"]
    rat = sv["cusparse_csr"] / ours if (ours and sv["cusparse_csr"]) else None
    if rat: ratios.append(rat)
    fmt = lambda x: f"{x:.3e}" if x is not None else "N/A"
    rows.append([m, "didx_sym" if sv["didx_sym"] is not None else "didx",
                 fmt(ours), fmt(sv["didx"]), fmt(sv["didx_sym"]), fmt(sv["cusparse_csr"]),
                 f"{rat:.2f}" if rat else "N/A",
                 fmt(sp["ours_flat"]), fmt(sp["ours_sym"]), fmt(sp["hm_atomic"]), fmt(sp["diaq_fp32"])])
path = os.path.join(OUT, "energy_main.csv")
with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["matrix", "ours_mode", "spmv_ours_J", "spmv_didx_J", "spmv_didx_sym_J",
                "spmv_cusparse_J", "cusparse_over_ours", "spmspm_flat_J", "spmspm_sym_J",
                "spmspm_hm_J", "spmspm_diaq_J"])
    w.writerows(rows)
print(f"wrote {path} ({len(rows)} rows)")
if ratios:
    g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"SpMV energy, cuSPARSE/ours: geomean {g:.2f}x over {len(ratios)} matrices, "
          f"min {min(ratios):.2f} max {max(ratios):.2f}")
