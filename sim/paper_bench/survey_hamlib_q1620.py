#!/usr/bin/env python3.11
"""survey_hamlib_q1620.py — structural census of ALL HamLib instances with
q in [16,20], from the local portal mirror. NO matrix build, NO GPU: every
characteristic is derived from the Pauli term labels.

  q          = 1 + max qubit index in any label
  flip mask  = bits of X/Y qubits in a term; the term populates the diagonal
               offsets { sum_{i in m} s_i 2^i : s_i = +-1 }  (m=0 -> {0})
  D          = |union of offsets over terms|  (upper set: coefficient
               cancellation could only shrink it)
  bandwidth  = max |offset|
Early abort: the byte scan stops the moment an index > 19 appears, so
out-of-range (q>20) instances cost only their prefix read.
Caps (censored, flagged in the row): dataset > 200 MB skipped; a single mask
with weight > 16, or offset-union > 2^17, stops D enumeration (D reported as
lower bound, cens=D).

usage:   python3.11 survey_hamlib_q1620.py [--workers 6] [--portal DIR]
output:  appends to hamlib_q1620_census.csv (one row per kept instance);
         done-file hamlib_q1620_census.done makes reruns incremental.
"""
import sys, os, re, glob, csv, itertools
from multiprocessing import Pool

PORTAL = "/mnt/beegfs/ysu34/hamlib_portal_hdf5"
OUTDIR = "/mnt/beegfs/ysu34/paper_bench"
WORKERS = 6
args = sys.argv[1:]
if "--portal" in args: PORTAL = args[args.index("--portal")+1]
if "--workers" in args: WORKERS = int(args[args.index("--workers")+1])
CSVP = os.path.join(OUTDIR, "hamlib_q1620_census.csv")
DONE = os.path.join(OUTDIR, "hamlib_q1620_census.done")

LBL = re.compile(rb'\[([^\]]*)\]')
OPQ = re.compile(rb'([XYZ])(\d+)')

def offsets_of_mask(m):
    bits = [1 << i for i in range(20) if m >> i & 1]
    if not bits: return {0}
    return {sum(s) for s in itertools.product(*[(b, -b) for b in bits])}

def scan_key(buf):
    """-> (q, nterms, masks, aborted_q_gt20)"""
    qmax = -1; nterms = 0; masks = set()
    for lm in LBL.finditer(buf):
        nterms += 1; m = 0
        for om in OPQ.finditer(lm.group(1)):
            i = int(om.group(2))
            if i > 19: return (None, 0, None, True)
            if i > qmax: qmax = i
            if om.group(1) != b'Z': m |= 1 << i
        masks.add(m)
    return (qmax + 1, nterms, masks, False)

def process_file(rel):
    import h5py
    out = []
    try:
        with h5py.File(os.path.join(PORTAL, rel), 'r') as f:
            for key in f:
                d = f[key]
                if getattr(d, "nbytes", 0) > 200 * 1024 * 1024:
                    out.append([rel, key, -1, 0, 0, 0, 0, "toolarge"]); continue
                try: buf = d[()]
                except Exception: continue
                if not isinstance(buf, bytes): continue
                q, nterms, masks, aborted = scan_key(buf)
                if aborted or q is None or not (16 <= q <= 20): continue
                offs = set(); cens = ""
                for m in masks:
                    if bin(m).count("1") > 16 or len(offs) > 131072:
                        cens = "D"; break
                    offs |= offsets_of_mask(m)
                bw = max((abs(o) for o in offs), default=0)
                out.append([rel, key, q, nterms, len(masks), len(offs), bw, cens])
    except Exception as e:
        out.append([rel, "<FILEERROR>", -1, 0, 0, 0, 0, str(e)[:60]])
    return rel, out

def main():
    os.nice(19)
    done = set()
    if os.path.exists(DONE): done = set(l.strip() for l in open(DONE))
    files = sorted(os.path.relpath(p, PORTAL)
                   for p in glob.glob(f"{PORTAL}/**/*.hdf5", recursive=True))
    todo = [f for f in files if f not in done]
    print(f"{len(files)} files, {len(todo)} to scan", flush=True)
    new_csv = not os.path.exists(CSVP)
    with open(CSVP, "a", newline="") as fc, open(DONE, "a") as fd, Pool(WORKERS, os.nice, (19,)) as pool:
        w = csv.writer(fc)
        if new_csv:
            w.writerow(["file", "key", "q", "nterms", "nmasks", "D", "bandwidth", "censored"])
        for i, (rel, rows) in enumerate(pool.imap_unordered(process_file, todo)):
            w.writerows(rows); fc.flush()
            fd.write(rel + "\n"); fd.flush()
            if (i + 1) % 50 == 0: print(f"[{i+1}/{len(todo)}] {rel}", flush=True)
    print("census done", flush=True)

if __name__ == "__main__":
    main()
