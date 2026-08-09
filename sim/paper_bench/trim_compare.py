#!/usr/bin/env python3
"""trim_compare.py — ABLATION TIER A0 table: our kernels on DIA_TRIM=1e-8
matrices (the *_trim corpus) vs our kernels untrimmed AND vs the official
baselines (which always come from the UNTRIMMED corpus — user decision:
baselines never touch trimmed data).

Columns per matrix:
  D_untrim/D_trim, spmv ours untrim/trim ms + trim_gain, new-vs-cusparse,
  spmspm flat & sym untrim/trim + gains, e2e ours untrim/trim.
Medians over all reps. 'ours' = didx_sym where present else didx (mainline).
out: results/paper_tables/trim_ablation.csv
"""
import csv, os, statistics, collections, math

B = "/mnt/beegfs/ysu34/paper_bench"
_here = os.path.dirname(os.path.abspath(__file__))
import sys
PLAT = sys.argv[1] if len(sys.argv) > 1 else "h100"   # h100 | a100
OUT = os.path.join(_here, "..", "..", "results", "paper_tables", f"trim_ablation_{PLAT}.csv")
PAPER = {l.strip().split("/")[-1][:-4] for l in open(os.path.join(_here, "matrices_paper.txt")) if l.strip()}

def key(p): return p.split("/")[-1].replace(".txt", "")
def med(xs): return statistics.median(xs) if xs else None

def collect_prep(files):
    spmv = collections.defaultdict(list); Dv = {}
    for f in files:
        p = os.path.join(B, f)
        if not os.path.exists(p): continue
        for r in csv.reader(open(p)):
            if not r or r[0] != "PREPCSV" or len(r) < 10 or r[4] != "spmv": continue
            m = key(r[1]); spmv[(m, r[5])].append(float(r[8])); Dv[m] = int(r[3])
    return spmv, Dv

def collect_spm(files):
    d = collections.defaultdict(list)
    for f in files:
        p = os.path.join(B, f)
        if not os.path.exists(p): continue
        for r in csv.reader(open(p)):
            if r and r[0] == "SPMSPMCSV" and len(r) >= 8:
                d[(key(r[1]), r[6])].append(float(r[7]))
    return d

def collect_e2e(files):
    d = collections.defaultdict(list)
    for f in files:
        p = os.path.join(B, f)
        if not os.path.exists(p): continue
        for r in csv.reader(open(p)):
            if r and r[0] == "E2EEVCSV":
                d[key(r[1])].append(float(r[9]))
    return d

if PLAT == "a100":
    U_PREP = ["prep_a100_paperset.csv"]; T_PREP = ["prep_a100_trim.csv"]
    U_SPM = ["spmspm_a100_paperset.csv"]; T_SPM = ["spmspm_a100_trim.csv"]
    U_E2E = ["e2e_a100_paperset.csv"]; T_E2E = ["e2e_a100_trim.csv"]
    CUSP = "cusparse_fix_a100.csv"
else:
    U_PREP = ["prep_h100_268657.csv", "prep_h100_c29resume.csv", "prep_h100_c29patch.csv"]
    T_PREP = ["prep_h100_trim.csv"]
    U_SPM = ["spmspm_cusparse_fix_h100.csv"]; T_SPM = ["spmspm_h100_trim.csv"]
    U_E2E = ["e2e_diaq_h100_268662.csv", "e2e_diaq_h100_c29resume.csv", "e2e_diaq_h100_c29patch.csv"]
    T_E2E = ["e2e_h100_trim.csv"]
    CUSP = "cusparse_fix_h100.csv"
sp_u, D_u = collect_prep(U_PREP)
sp_t, D_t = collect_prep(T_PREP)
spm_u = collect_spm(U_SPM)
spm_t = collect_spm(T_SPM)
e2e_u = collect_e2e(U_E2E)
e2e_t = collect_e2e(T_E2E)
cusp = collections.defaultdict(list)
for r in csv.reader(open(os.path.join(B, CUSP))):
    if r and r[0] == "PREPCSV": cusp[key(r[1])].append(float(r[8]))

def ours(sp, m):
    s = med(sp.get((m, "didx_sym"), []))
    return (s, "sym") if s is not None else (med(sp.get((m, "didx"), [])), "didx")

rows = []
gains_v = []; gains_s = []
for m in sorted(PAPER):
    ou, mu = ours(sp_u, m); ot, mt = ours(sp_t, m)
    fu = med(spm_u.get((m, "ours_flat"), [])); ft = med(spm_t.get((m, "ours_flat"), []))
    su = med(spm_u.get((m, "ours_sym"), []));  st = med(spm_t.get((m, "ours_sym"), []))
    eu = med(e2e_u.get(m, [])); et = med(e2e_t.get(m, []))
    cu = med(cusp.get(m, []))
    if ou is None and ft is None: continue
    gv = ou / ot if (ou and ot) else None
    gf = fu / ft if (fu and ft) else None
    gs = su / st if (su and st) else None
    ge = eu / et if (eu and et) else None
    if gv: gains_v.append((gv, m))
    if gf: gains_s.append((gf, m))
    f = lambda x, p=4: (f"%.{p}f" % x) if x is not None else ""
    rows.append([m, D_u.get(m, ""), D_t.get(m, ""), f(ou, 6), f(ot, 6), f(gv, 2),
                 f(cu, 6), f(cu / ot, 2) if (cu and ot) else "",
                 f(fu), f(ft), f(gf, 2), f(su), f(st), f(gs, 2), f(eu, 1), f(et, 1), f(ge, 2)])

with open(OUT, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["matrix", "D_untrim", "D_trim", "spmv_ours_untrim", "spmv_ours_trim",
                "spmv_gain", "cusparse_untrim", "new_speedup_vs_cusp",
                "spmspm_flat_untrim", "spmspm_flat_trim", "flat_gain",
                "spmspm_sym_untrim", "spmspm_sym_trim", "sym_gain",
                "e2e_untrim", "e2e_trim", "e2e_gain"])
    w.writerows(rows)
print(f"wrote {OUT} ({len(rows)} rows)")
for tag, g in (("SpMV", gains_v), ("SpMSpM-flat", gains_s)):
    xs = [x for x, _ in g]
    if xs:
        gm = math.exp(sum(math.log(x) for x in xs) / len(xs))
        top = sorted(g, reverse=True)[:3]
        print(f"{tag} trim gain: geomean {gm:.3f}x over {len(xs)}; top: "
              + ", ".join(f"{m} {x:.2f}x" for x, m in top))
