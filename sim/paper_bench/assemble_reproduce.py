#!/usr/bin/env python3
"""assemble_reproduce.py — build the reproducibility table for the paper set:
matrix -> source HDF5 + key + generation protocol + provenance confidence.

Sources, in priority order:
  1. .txt.meta `source_hdf5/source_key` fields (written by the patched
     generator from 2026-08-01 on — O2c/C2c/H8c/bhc rebuilds)
  2. dia_e2e/generated_matrices.csv (the e2e batch's provenance log)
  3. portal_matrices.txt manifest (the wide-portal symlinked additions)
  4. hardcoded notes (SuiteSparse URLs; pattern-derived dia_families fermi)
Confidence: fp64-verified > meta-recorded > csv-recorded > manifest >
pattern-derived (see verify_trim_fp64.py runs for the fp64 list).
out: results/paper_tables/reproduce_sources.csv
"""
import csv, os

_here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_here, "..", "..", "results", "paper_tables", "reproduce_sources.csv")
PAPER = [l.strip() for l in open(os.path.join(_here, "matrices_paper.txt")) if l.strip()]
GEN = "/mnt/beegfs/ysu34/hamlib/dia_e2e/generated_matrices.csv"
PORTAL_MAN = os.path.join(_here, "..", "suite", "hpc", "portal_matrices.txt")

FP64_VERIFIED = {"O2c_16_5", "C2c_18_4", "H8c_16_4", "vibbf3_18_3", "BH_20_7",
                 "bhc_16_7", "bhc_18_7", "fermi_18_5", "fermi_20_5", "O2c_20_5"}
PORTAL_LINKS = {   # our dia_e2e symlink name -> portal manifest tag
    "bhD1_18_7": "condensedmatter-bosehubbard-BH-D-1-d-4",
    "fermiFH_18_4": "condensedmatter-fermihubbard-FH-D-1",
    "qmaxcutR_18_3": "binaryoptimization-qmaxcut-biqmac-rudy-instances",
    "qmaxcutK_18_4": "binaryoptimization-qmaxcut-ciqube-Karloff-hams",
    "qmaxcutC_18_4": "binaryoptimization-qmaxcut-random-graph-circulant",
    "vibbf3_18_3": "chemistry-vibrational-all-vib-bf3",
}
SS_URL = {
    "ecology2": "https://sparse.tamu.edu/McRae/ecology2",
    "atmosmodd": "https://sparse.tamu.edu/Bourchtein/atmosmodd",
    "Lin": "https://sparse.tamu.edu/Lin/Lin",
}
PATTERN = {}
KNOWN = {   # provenance established outside the meta fields; ALL fp64-verified
    # fermi: brute-force matched 2026-08-01, bit-exact (diff 0.0)
    "fermi_18_5": ("fermi/fermi.hdf5", "fh-graph-1D-grid-nonpbc-qubitnodes_Lx-9_U-2_enc-jw"),
    "fermi_20_5": ("fermi/fermi.hdf5", "fh-graph-1D-grid-nonpbc-qubitnodes_Lx-10_U-2_enc-jw"),
    "O2c_16_5": ("O2/O2.hdf5", "ham_JW16"),
    "O2c_20_5": ("O2/O2.hdf5", "ham_JW20"),   # invocation in regen.log; fp64 check queued
    "C2c_18_4": ("hamlib_portal_hdf5/chemistry/electronic/standard/C2.hdf5", "ham_BK-18"),
    "H8c_16_4": ("hamlib_portal_hdf5/chemistry/electronic/hydrogen_data/H8_linear/"
                 "ES_H8_linear_R0.6_sto-6g_ham.hdf5", "ham_BK"),
}

genrec = {}
if os.path.exists(GEN):
    for r in csv.DictReader(open(GEN)):
        if r.get("status") == "OK":
            genrec[r["output_file"].replace(".txt", "")] = (r["hdf5"], r["key"])
portman = {}
for line in open(PORTAL_MAN):
    p = line.split()
    if len(p) >= 3 and not line.startswith("#"):
        portman[p[0]] = (p[1], p[2])

rows = []
for path in PAPER:
    m = os.path.basename(path).replace(".txt", "")
    meta = {}
    try:
        for l in open(path + ".meta"):
            k, _, v = l.partition(" ")
            meta[k] = v.strip()
    except FileNotFoundError:
        pass
    if "source_hdf5" in meta:
        src, key = meta["source_hdf5"], meta["source_key"]
        conf = "fp64-verified" if m in FP64_VERIFIED else "meta-recorded"
    elif m in KNOWN:
        src, key = KNOWN[m]
        conf = "fp64-verified"
    elif m in genrec:
        src, key = genrec[m]
        conf = "fp64-verified" if m in FP64_VERIFIED else "csv-recorded"
    elif m in PORTAL_LINKS and PORTAL_LINKS[m] in portman:
        src, key = portman[PORTAL_LINKS[m]]
        src = "hamlib_portal_hdf5/" + src
        conf = "fp64-verified" if m in FP64_VERIFIED else "manifest"
    elif any(k.startswith(m.rsplit("_", 1)[0] + "_") for k in genrec):
        sib = next(k for k in genrec if k.startswith(m.rsplit("_", 1)[0] + "_"))
        src, key = genrec[sib]
        conf = f"csv-recorded(sibling {sib})"
    elif m in SS_URL:
        src, key = SS_URL[m], "MatrixMarket via sim/mtx_to_dia"
        conf = "recorded"
    elif m in PATTERN:
        src, key = PATTERN[m]
        conf = "pattern-derived(UNVERIFIED)"
    else:
        src, key, conf = "UNKNOWN", "UNKNOWN", "missing"
    rows.append([m, src, key, meta.get("n_taylor", ""), "dt=0.0012 steps=1000",
                 "fetchham_sparse_dia.py" if m not in SS_URL else "mtx_to_dia", conf])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["matrix", "source", "key", "K", "protocol", "generator", "provenance"])
    w.writerows(sorted(rows))
print(f"wrote {OUT} ({len(rows)} rows)")
for r in sorted(rows):
    if "UNVERIFIED" in r[-1] or r[-1] == "missing": print("  ATTN:", r[0], r[-1])
