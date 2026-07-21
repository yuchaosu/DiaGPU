#!/usr/bin/env python3.11
"""
build_hamlib_csv.py — orchestrate the full hamlib.csv benchmark sweep.

For every matrix in dia_e2e/ + dia_oom/ passing the qubit filter (q>=18 for
scalable families, q>=12 for molecular), run:
  SpMV : ours DIA zero-skip vs tensor-core, dense-CUDA, cuSPARSE, Drawloom
  SpMSpM: ours gather_flat vs HM(Haque), cuSPARSE SpGEMM
join provenance (family + HDF5 key) and write one row per matrix.

OOM-stop: per family, sizes are visited ascending; once a kernel OOMs at some q
its cells are N/A for that and all larger q of the family (no silent blanks).

Binaries + paths come from env (set by bench_hamlib.slurm):
  SPMV_BIN SPMSPM_BIN DIA2MTX_BIN DRAWLOOM_BIN  (DRAWLOOM_BIN optional)
  DIA_E2E DIA_OOM OUTDIR  GEN_MANIFEST_CSV
"""
import os, re, sys, subprocess, glob, csv, shutil, time

E2E   = os.environ.get("DIA_E2E","/mnt/beegfs/ysu34/hamlib/dia_e2e")
OOM   = os.environ.get("DIA_OOM","/mnt/beegfs/ysu34/hamlib/dia_oom")
OUT   = os.environ.get("OUTDIR","/mnt/beegfs/ysu34/hamlib_bench")
SPMV  = os.environ["SPMV_BIN"]
SPMSPM= os.environ["SPMSPM_BIN"]
D2MTX = os.environ.get("DIA2MTX_BIN","")
DRAW  = os.environ.get("DRAWLOOM_BIN","")      # may be empty/missing
GEN_CSV = os.environ.get("GEN_MANIFEST_CSV","")
TMP   = os.environ.get("MTX_TMP","/mnt/beegfs/ysu34/hamlib_bench/_mtx_tmp")
ITERS = os.environ.get("ITERS","200")
SPITERS = os.environ.get("SPITERS","30")

MOLECULAR = {"B2","O2","BeH","Li2","c2h","hnc"}
DRAW_MAX_NNZ   = 1.5e8    # a .mtx above this is >~2 GB text — skip Drawloom
SPMSPM_MAX_N   = 1<<22    # C=H*H fill-in explodes past here for banded/fill-in

os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)

def parse_name(path):
    b=os.path.basename(path)[:-4]
    m=re.match(r"^(.*)_(\d+)_(\d+)$", b)
    if not m: return None
    return m.group(1), int(m.group(2)), int(m.group(3))   # fam, q, K

def header_nd(path):
    with open(path) as f: h=f.readline().split()
    return int(h[1]), int(h[3])   # N, D

def structural_class(D):
    return "diagonal" if D==1 else ("banded" if D<=128 else "fill-in")

def run(cmd, timeout=3600, env=None):
    try:
        p=subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:
        return 1, "", str(e)

# ---- provenance lookup: filename -> (family, key, hdf5, status) ----
prov={}
def load_prov():
    gm=os.path.join(E2E,"generated_matrices.csv")
    if os.path.exists(gm):
        for r in csv.DictReader(open(gm)):
            f=r.get("output_file","")
            if f: prov[f]=(r.get("family",""),r.get("key",""),r.get("hdf5",""),r.get("status",""))
    # DIAMOND provenance (workload_file, hdf5_key, hdf5_file, key_status)
    dp="/home/ysu34/DIAMOND/diamond/isca/hamlib_provenance.csv"
    if os.path.exists(dp):
        for r in csv.DictReader(open(dp)):
            f=r.get("workload_file","")
            if f and f not in prov:
                prov[f]=(r.get("family",""),r.get("hdf5_key",""),r.get("hdf5_file",""),r.get("key_status",""))
    if GEN_CSV and os.path.exists(GEN_CSV):   # newly generated: file,family,key,hdf5
        for r in csv.DictReader(open(GEN_CSV)):
            prov[r["file"]]=(r["family"],r["key"],r.get("hdf5",""),"generated")
load_prov()

def prov_of(fname, fam):
    if fname in prov:
        p=prov[fname]; return p[0] or fam, p[1] or "-", p[2] or "-", p[3] or "-"
    return fam, "-", "-", "unlogged"

# ---- per-kernel runners; return dict or None(=oom) ----
def spmv(path):
    rc,out,err=run([SPMV,path,ITERS,"--csv"], timeout=5400)
    for ln in out.splitlines():
        if ln.startswith("CSV,"):
            c=ln.split(",")
            # CSV,file,n,D,fill,nnz,zsv,tc,dense,zs,csp,zs_vs_tc,zs_vs_dense,zs_vs_csp,tc_relerr,eff_bw,skip
            return dict(n=c[2],D=c[3],fill=c[4],nnz=c[5],zsv=c[6],tc=c[7],dense=c[8],zs=c[9],
                        csp=c[10],zs_tc=c[11],zs_dense=c[12],zs_csp=c[13],relerr=c[14],eff_bw=c[15],skip=c[16])
    return None   # OOM / crash

def spmspm(path, N):
    if N>SPMSPM_MAX_N: return "toolarge"
    rc,out,err=run([SPMSPM,path,SPITERS,"--csv"], timeout=5400)
    for ln in out.splitlines():
        if ln.startswith("SPCSV,"):
            c=ln.split(",")
            # SPCSV,file,n,Hd,nnzH,Cd,nnzC,ours,hm,cusp,ours_vs_hm,ours_vs_cusp
            return dict(Cd=c[5],nnzC=c[6],ours=c[7],hm=c[8],cusp=c[9],o_hm=c[10],o_csp=c[11])
    return None   # OOM

def drawloom(path, nnz):
    if not DRAW or not os.path.exists(DRAW): return ("na","drawloom_unavailable")
    if nnz>DRAW_MAX_NNZ: return ("na","mtx_too_large")
    mtx=os.path.join(TMP,"cur.mtx")
    rc,_,_=run([D2MTX,path,mtx], timeout=1800)
    if rc!=0 or not os.path.exists(mtx): return ("na","mtx_fail")
    env=dict(os.environ, OMP_NUM_THREADS="16")
    rc,out,err=run([DRAW,"-filename",mtx], timeout=1800, env=env)
    try: os.remove(mtx)
    except: pass
    for ln in (out+err).splitlines():
        m=re.search(r"drawloom time:\s*([\d.]+)\s*ms", ln)
        if m: return (m.group(1),"")
    return ("na","drawloom_fail")

# ---- enumerate + filter ----
# EXCLUDE: comma-separated <fam>_<q> stems to skip (e.g. matrices still being
# generated in a concurrent job — avoids reading a file mid-write).
EXCLUDE=set(x for x in os.environ.get("EXCLUDE","").split(",") if x)
mats=[]
for d,tag in [(E2E,"dia_e2e"),(OOM,"dia_oom")]:
    for p in glob.glob(os.path.join(d,"*.txt")):
        if p.endswith(".imag.txt"): continue
        nm=parse_name(p)
        if not nm: continue
        fam,q,K=nm
        if f"{fam}_{q}" in EXCLUDE: continue
        thr = 12 if fam in MOLECULAR else 18
        if q<thr: continue
        mats.append((fam,q,K,tag,p))
mats.sort(key=lambda x:(x[0],x[1],x[2]))

# ---- sweep with per-family per-kernel OOM-stop ----
COLS=["set","output_file","family","structural_class","qubits","N","D","nnz","fill_pct",
      "source_hdf5","source_key","provenance_status",
      "spmv_zeroskip_ms","zeroskip_variant","spmv_tc_ms","spmv_dense_ms","spmv_cusparse_ms","spmv_drawloom_ms",
      "spmv_zs_vs_tc","spmv_zs_vs_dense","spmv_zs_vs_cusparse","spmv_zs_vs_drawloom","spmv_tc_relerr",
      "spmspm_ours_ms","spmspm_hm_ms","spmspm_cusparse_ms","spmspm_ours_vs_hm","spmspm_ours_vs_cusparse",
      "spmv_eff_bw_GBs","na_reason","gpu","driver"]
GPU=os.environ.get("GPU_NAME","A100-80GB-PCIe"); DRV=os.environ.get("DRIVER","580.82.07")
oom_spmv=set(); oom_spmspm=set()   # families whose kernel already OOM'd
rows=[]
t0=time.time()
for fam,q,K,tag,p in mats:
    fname=os.path.basename(p)
    N,D=header_nd(p)
    pf_fam,key,hdf5,status=prov_of(fname,fam)
    na=[]
    # SpMV
    if fam in oom_spmv:
        sv=None; na.append("spmv_oom(family-stopped)")
    else:
        sv=spmv(p)
        if sv is None: oom_spmv.add(fam); na.append("spmv_oom")
    # SpMSpM
    if D==1:
        sp="diag"
    elif fam in oom_spmspm:
        sp=None; na.append("spmspm_oom(family-stopped)")
    else:
        sp=spmspm(p,N)
        if sp is None: oom_spmspm.add(fam); na.append("spmspm_oom")
    nnz = sv["nnz"] if sv else "-"
    # Drawloom
    if sv:
        dl,dlreason=drawloom(p,float(sv["nnz"]))
        if dl=="na" and dlreason: na.append("drawloom_"+dlreason)
    else:
        dl,dlreason="na","spmv_oom"
    def g(d,k): return d.get(k,"na") if isinstance(d,dict) else "na"
    sp_ours = sp["ours"] if isinstance(sp,dict) else ("diag" if sp=="diag" else "na")
    row={
      "set":tag,"output_file":fname,"family":pf_fam,"structural_class":structural_class(D),
      "qubits":q,"N":N,"D":D,"nnz":nnz,"fill_pct":(sv["fill"] if sv else "-"),
      "source_hdf5":hdf5,"source_key":key,"provenance_status":status,
      "spmv_zeroskip_ms":g(sv,"zs"),"zeroskip_variant":g(sv,"zsv"),
      "spmv_tc_ms":g(sv,"tc"),"spmv_dense_ms":g(sv,"dense"),"spmv_cusparse_ms":g(sv,"csp"),
      "spmv_drawloom_ms":dl,
      "spmv_zs_vs_tc":g(sv,"zs_tc"),"spmv_zs_vs_dense":g(sv,"zs_dense"),
      "spmv_zs_vs_cusparse":g(sv,"zs_csp"),
      "spmv_zs_vs_drawloom":(f"{float(sv['zs'])>0 and float(dl)>0 and float(dl)/float(sv['zs']) or -1:.4f}" if (sv and dl not in ('na',) ) else "na"),
      "spmv_tc_relerr":g(sv,"relerr"),
      "spmspm_ours_ms":sp_ours,"spmspm_hm_ms":g(sp,"hm"),"spmspm_cusparse_ms":g(sp,"cusp"),
      "spmspm_ours_vs_hm":g(sp,"o_hm"),"spmspm_ours_vs_cusparse":g(sp,"o_csp"),
      "spmv_eff_bw_GBs":g(sv,"eff_bw"),
      "na_reason":(";".join(na) if na else "-"),"gpu":GPU,"driver":DRV,
    }
    rows.append(row)
    print(f"[{time.time()-t0:6.0f}s] {fname:22} q{q} D{D} "
          f"zs={g(sv,'zs')} vs_tc={g(sv,'zs_tc')} vs_csp={g(sv,'zs_csp')} dl={dl} "
          f"spgemm_ours={sp_ours} vs_hm={g(sp,'o_hm')}", flush=True)

with open(os.path.join(OUT,"hamlib.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=COLS); w.writeheader()
    for r in rows: w.writerow(r)
print(f"\nwrote {os.path.join(OUT,'hamlib.csv')}  ({len(rows)} rows)")
