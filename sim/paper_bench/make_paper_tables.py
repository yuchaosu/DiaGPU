#!/usr/bin/env python3
"""make_paper_tables.py — merge the H100 CSV corpus into paper-ready tables.

Protocol decisions encoded here (2026-08-01):
  * MAIN TABLES = the locked paper set only (matrices_paper.txt: q16-20 + SS);
    all other matrices stay in raw CSVs for the boundary narrative.
  * REPS policy: every timing cell = MEDIAN over all recorded runs of the
    same (matrix, variant); `spread` = (max-min)/median over those runs.
  * SpMV = didx single scheme; 'stream'/'csr_zskip' excluded; diaq fp32 only.
  * cuSPARSE columns come ONLY from the honest true-nonzero rebench files.
  * ours_sym valid only where the driver cross-check <= 1e-2 (absolute; note
    Lin-style large-value matrices pass at driver level; ablation column uses
    value-scale-normalized judgement separately).
  * Drawloom == DASP (user confirmation).

usage: python3 make_paper_tables.py [outdir]
"""
import csv, os, sys, math, glob, collections, statistics

B = "/mnt/beegfs/ysu34/paper_bench"
_here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "..", "..", "results", "paper_tables")
os.makedirs(OUT, exist_ok=True)
PAPER = {l.strip().split("/")[-1][:-4] for l in open(os.path.join(_here, "matrices_paper.txt")) if l.strip()}
SS_MATS = {os.path.basename(p)[:-4] for p in glob.glob("/mnt/beegfs/ysu34/hamlib/dia_suitesparse/*.txt")}

def matkey(p): return p.split("/")[-1].replace(".txt", "")
def med(xs): return statistics.median(xs)
def spread(xs): return (max(xs) - min(xs)) / med(xs) if len(xs) > 1 and med(xs) > 0 else 0.0

# ---------- collect ----------
didx = collections.defaultdict(list)          # m -> [ms]
dsym = collections.defaultdict(list)          # m -> [ms]  (mainline symmetric mode)
meta = {}                                     # m -> (n, D)
for f in ["prep_h100_268657.csv", "prep_h100_c29resume.csv", "prep_h100_c29patch.csv",
          "spmv_variants_h100.csv"]:
    p = os.path.join(B, f)
    if not os.path.exists(p): continue
    for r in csv.reader(open(p)):
        if not r or r[0] != "PREPCSV" or len(r) < 10: continue
        m = matkey(r[1])
        if r[4] == "spmv" and r[5] == "didx":
            didx[m].append(float(r[8])); meta[m] = (int(r[2]), int(r[3]))
        if r[4] == "spmv" and r[5] == "didx_sym":
            dsym[m].append(float(r[8])); meta.setdefault(m, (int(r[2]), int(r[3])))

cusp_spmv = collections.defaultdict(list); cusp_cfg = {}
for r in csv.reader(open(os.path.join(B, "cusparse_fix_h100.csv"))):
    if r and r[0] == "PREPCSV":
        m = matkey(r[1]); cusp_spmv[m].append(float(r[8])); cusp_cfg[m] = r[5]

fc = {}
for r in csv.reader(open(os.path.join(B, "cfill_h100.csv"))):
    if r and r[0] == "CFILL": fc[matkey(r[1])] = r[6]

sweep_l = collections.defaultdict(lambda: collections.defaultdict(list))
for f in ["h100_l4_sweep_268133.csv", "spmv_sweep_extra_h100.csv"]:
    p = os.path.join(B, f)
    if not os.path.exists(p): continue
    for r in csv.reader(open(p)):
        if r and r[0] == "SPMVCSV" and len(r) > 8:
            try: sweep_l[matkey(r[1])][r[6]].append(float(r[7]))
            except ValueError: pass
sweep = {m: {v: med(xs) for v, xs in d.items()} for m, d in sweep_l.items()}

spm = collections.defaultdict(lambda: collections.defaultdict(list))   # m -> var -> [(ms, check)]
spm_meta = {}
for r in csv.reader(open(os.path.join(B, "spmspm_cusparse_fix_h100.csv"))):
    if not r or r[0] != "SPMSPMCSV": continue
    m = matkey(r[1])
    spm_meta[m] = (int(r[2]), int(r[4]), int(r[5]))
    if r[6] == "diaq_fp64": continue
    spm[m][r[6]].append((float(r[7]), float(r[8])))

e2e = collections.defaultdict(lambda: collections.defaultdict(list))   # m -> arm -> [ms]
e2e_rel = {}
for f in ["e2e_diaq_h100_268662.csv", "e2e_diaq_h100_c29resume.csv", "e2e_diaq_h100_c29patch.csv"]:
    p = os.path.join(B, f)
    if not os.path.exists(p): continue
    for r in csv.reader(open(p)):
        if not r: continue
        if r[0] == "E2EEVCSV":
            m = matkey(r[1]); e2e[m]["ours"].append(float(r[9]))
            if float(r[10]) > 0: e2e[m]["cusp"].append(float(r[10]))
            e2e_rel[m] = r[12]
        if r[0] == "E2EEVDIAQCSV":
            e2e[matkey(r[1])]["diaq"].append(float(r[7]))

# ---------- emit ----------
def w(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")

rows = []
for m in sorted(PAPER):
    if m not in didx: continue
    dm = med(didx[m]); n, D = meta[m]
    sm = med(dsym[m]) if dsym[m] else float("nan")
    # MAINLINE (2026-08-01): didx_sym on symmetric H, didx otherwise
    ours = sm if sm == sm else dm
    mode = "didx_sym" if sm == sm else "didx"
    cu = med(cusp_spmv[m]) if cusp_spmv[m] else float("nan")
    dl = sweep.get(m, {}).get("drawloom", float("nan"))
    dq = sweep.get(m, {}).get("diaq_fp32", float("nan"))
    rows.append([m, n, D, fc.get(m, ""), mode, f"{ours:.6f}", f"{dm:.6f}",
                 f"{sm:.6f}" if sm == sm else "N/A", f"{spread(didx[m]):.3f}", len(didx[m]),
                 cusp_cfg.get(m, ""), f"{cu:.6f}", f"{spread(cusp_spmv[m]):.3f}",
                 f"{cu/ours:.2f}" if cu == cu else "",
                 f"{dl:.6f}" if dl == dl else "", f"{dq:.6f}" if dq == dq else ""])
w("spmv_main.csv",
  ["matrix", "n", "D", "f_C", "ours_mode", "ours_ms_med", "didx_ms_med", "didx_sym_ms_med",
   "didx_spread", "reps", "cusparse_cfg", "cusparse_ms_med", "cusparse_spread",
   "speedup_vs_cusparse", "drawloom_dasp_ms", "diaq_fp32_ms"], rows)

rows = []
for m in sorted(PAPER):
    if m not in spm: continue
    d = spm[m]; n, Cd, Cst = spm_meta[m]
    def cell(var):
        xs = d.get(var)
        return (med([x[0] for x in xs]), max(x[1] for x in xs), len(xs)) if xs else None
    o, s, h, dq, c = cell("ours_flat"), cell("ours_sym"), cell("hm_atomic"), cell("diaq_fp32"), cell("cusparse_spgemm")
    sym_ok = s and s[1] <= 1e-2
    rows.append([m, n, Cd, Cst, fc.get(m, ""),
                 f"{o[0]:.4f}" if o else "OOM",
                 f"{s[0]:.4f}" if sym_ok else ("N/A(nonsym)" if s else "N/A"),
                 f"{h[0]:.4f}" if h else "N/A(int32)",
                 f"{dq[0]:.4f}" if dq else "N/A",
                 f"{c[0]:.4f}" if c else "N/A",
                 f"{c[0]/o[0]:.2f}" if (o and c) else "", o[2] if o else 0])
w("spmspm_main.csv",
  ["matrix", "n", "Cd", "C_stored", "f_C", "ours_flat_ms_med", "ours_sym_ms_med(L4)",
   "hm_atomic_ms_med", "diaq_fp32_ms_med", "cusparse_spgemm_ms_med", "spgemm_vs_flat", "reps"], rows)

rows = []
for m in sorted(PAPER - SS_MATS):
    if m not in e2e or not e2e[m]["ours"]: continue
    o = med(e2e[m]["ours"])
    cu = med(e2e[m]["cusp"]) if e2e[m]["cusp"] else float("nan")
    dq = med(e2e[m]["diaq"]) if e2e[m]["diaq"] else float("nan")
    rows.append([m, f"{o:.1f}", f"{spread(e2e[m]['ours']):.3f}", len(e2e[m]["ours"]),
                 f"{cu:.1f}" if cu == cu else "N/A", f"{dq:.1f}" if dq == dq else "N/A",
                 f"{cu/o:.2f}" if cu == cu else "", e2e_rel.get(m, "")])
w("e2e_main.csv", ["matrix", "ours_ms_med", "ours_spread", "reps", "cusparse_ms_med",
                   "diaq_ms_med", "speedup_vs_cusparse", "relerr_vs_cusp"], rows)
print("done.")
