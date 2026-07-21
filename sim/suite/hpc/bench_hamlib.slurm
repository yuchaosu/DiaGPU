#!/bin/bash
# ===========================================================================
# bench_hamlib.slurm — Phase 1/2 of the hamlib.csv pipeline (GPU, on c33/A100).
# Builds all SpMV/SpMSpM benches + Drawloom, runs the orchestrator over every
# q-filtered matrix, does a small ncu deep-profile subset, writes to
# /mnt/beegfs/ysu34/hamlib_bench/.  submit AFTER gen (afterok) or standalone.
#
# submit: sbatch --dependency=afterok:<genjob> sim/suite/hpc/bench_hamlib.slurm
# ===========================================================================
#SBATCH -J bench_hamlib
#SBATCH -p a100
#SBATCH -w c33
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 4-00:00:00
#SBATCH -o logs/bench_hamlib_%j.out
set -u
cd /home/ysu34/DiaGPU || exit 2
REPO=/home/ysu34/DiaGPU
SP=$REPO/sim
BIN=/mnt/beegfs/ysu34/hamlib_bench/bin
OUT=/mnt/beegfs/ysu34/hamlib_bench
DRAW=/mnt/beegfs/ysu34/drawloom_ae/drawloom_ae/Drawloom_code/myfloat
mkdir -p "$BIN" "$OUT" "$(dirname "$0")/logs"

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "== build benches (sm_80) =="
nvcc -O3 -std=c++17 -arch=sm_80 $SP/spmv_dia_vs_tc.cu $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $BIN/spmv_dia_vs_tc -lcusparse || exit 3
nvcc -O3 -std=c++17 -arch=sm_80 $SP/spmspm_hamlib.cu $REPO/spmspm/paper_hm_kernel.cu -o $BIN/spmspm_pure -lcusparse -Xcompiler -fopenmp || exit 3
g++  -O2 -std=c++17 $SP/dia_to_mtx.cpp -o $BIN/dia_to_mtx || exit 3
nvcc -O3 -std=c++17 -arch=sm_80 $SP/spmv_tc_ablation.cu $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $BIN/spmv_tc_ablation || exit 3
nvcc -O3 -std=c++17 -arch=sm_80 $SP/spmspm_ablation.cu $REPO/spmspm/paper_hm_kernel.cu -o $BIN/spmspm_ablation -Xcompiler -fopenmp || exit 3
[ -x "$DRAW" ] || echo "WARN: Drawloom binary missing at $DRAW -> drawloom cells N/A"
echo "  builds OK"

echo "== provenance keys for newly generated matrices (from gen log) =="
python3.11 - <<'PY'
import glob,re,csv,os
rows=[]
for lg in sorted(glob.glob("/home/ysu34/DiaGPU/sim/suite/hpc/logs/gen_hamlib_*.out")):
    tag=key=None
    for ln in open(lg,errors='ignore'):
        m=re.search(r'tag=(\S+)\s+key=(\S+)',ln)
        if m: tag,key=m.group(1),m.group(2)
        s=re.search(r'saved to:\s+(\S+\.txt)',ln)
        if s and tag and key: rows.append((os.path.basename(s.group(1)),tag,key))
with open("/mnt/beegfs/ysu34/hamlib_bench/gen_keys.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["file","family","key","hdf5"])
    for fn,tg,ky in rows: w.writerow([fn,tg,ky,""])
print(f"  gen_keys.csv: {len(rows)} generated matrices")
PY

echo "== run orchestrator =="
export SPMV_BIN=$BIN/spmv_dia_vs_tc SPMSPM_BIN=$BIN/spmspm_pure DIA2MTX_BIN=$BIN/dia_to_mtx
export DRAWLOOM_BIN=$DRAW OUTDIR=$OUT MTX_TMP=$OUT/_mtx_tmp
export GEN_MANIFEST_CSV=$OUT/gen_keys.csv
export GPU_NAME="A100-80GB-PCIe" DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
python3.11 $REPO/sim/suite/hpc/build_hamlib_csv.py

echo "== TC + SpMSpM ablations (paper) =="
# TC ablation: T0 full / T1 sym / T2 ztile / T3 sym+ztile — narrow+wide subset where full recon fits
echo "TCABL,file,n,D,Dsym,kept_ztile,kept_symztile,t0_full,t1_sym,t2_ztile,t3_symztile,s1,s2,s3,relerr" \
  | sed 's/,/,/g' > $OUT/tc_ablation.csv
for m in heis_16_5 heis_18_5 heis_20_6 tfim_18_5 qmaxcut_18_4 fermi_16_5 bh_18_5 O2_16_5; do
  for d in dia_e2e dia_oom; do p=/mnt/beegfs/ysu34/hamlib/$d/$m.txt; [ -f "$p" ] || continue
    timeout 1200 $BIN/spmv_tc_ablation "$p" 200 --csv 2>/dev/null | grep '^TCABL,' | sed 's/^TCABL,//' >> $OUT/tc_ablation.csv; break; done
done
# SpMSpM ablation: L0 HM / L1 meta / L2 flat-ILP1 / L3 flat-ILP4 / L4 flat_sym — narrow-band, SpMSpM-feasible
echo "file,n,Hd,nnzC,L0_hm,L1_meta,L2_flat1,L3_flat4,L4_sym,s1,s2,s3,s4,relerr" > $OUT/spmspm_ablation.csv
for m in heis_14_5 heis_16_5 heis_18_5 tfim_16_5 tfim_18_5 qmaxcut_16_4 qmaxcut_18_4 fermi_14_5 fermi_16_5; do
  p=/mnt/beegfs/ysu34/hamlib/dia_e2e/$m.txt; [ -f "$p" ] || continue
  timeout 1200 $BIN/spmspm_ablation "$p" 30 --csv 2>/dev/null | grep '^SPABL,' | sed 's/^SPABL,//' >> $OUT/spmspm_ablation.csv
done

echo "== ncu profile ALL sweep matrices (per user) =="
NCU_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
PROF=$OUT/ncu_profile.csv
echo "matrix,kernel,metric,value" > $PROF
# every q-filtered matrix; one ncu pass profiles all kernels of spmv_dia_vs_tc.
# 600s per-call timeout guards the largest; timeouts logged as ncu_skipped.
emit_ncu(){ # $1=matrix-tag $2=raw-csv -> append (matrix,kernel,metric,value)
  python3.11 - "$1" "$2" "$PROF" <<'PY'
import sys,csv
mat,raw,prof=sys.argv[1],sys.argv[2],sys.argv[3]
try: rows=list(csv.reader(open(raw)))
except: sys.exit(0)
hi=[i for i,r in enumerate(rows) if "Metric Name" in r]
if not hi: sys.exit(0)
h=rows[hi[0]]
try: kn=h.index("Kernel Name"); mn=h.index("Metric Name"); mv=h.index("Metric Value")
except: sys.exit(0)
with open(prof,"a") as f:
    for r in rows[hi[0]+1:]:
        if len(r)>max(kn,mn,mv) and r[mn]:
            k=r[kn].split('(')[0].split('<')[0][:44]
            f.write(f"{mat},{k},{r[mn]},{r[mv]}\n")
PY
}
for d in dia_e2e dia_oom; do for p in /mnt/beegfs/ysu34/hamlib/$d/*.txt; do
  case "$p" in *.imag.txt) continue;; esac
  b=$(basename "$p" .txt); fam=$(echo "$b"|sed -E 's/_[0-9]+_[0-9]+$//'); q=$(echo "$b"|sed -E 's/.*_([0-9]+)_[0-9]+$/\1/')
  case "$fam" in B2|O2|BeH|Li2|c2h|hnc) thr=12;; *) thr=18;; esac
  [ "$q" -ge "$thr" ] 2>/dev/null || continue
  # one ncu pass per OUR named kernel (-c 1 = a single launch each). cuSPARSE's
  # kernels are library-internal names -> not profiled here (its timing is in hamlib.csv).
  any=0
  for kre in tc_spmv_regdirect cuda_spmv_dia spmv_dia_zeroskip; do
    timeout 600 ncu --csv --metrics "$NCU_METRICS" --kernel-name "regex:$kre" -c 1 \
        $BIN/spmv_dia_vs_tc "$p" 3 > $OUT/_ncu_raw.csv 2>/dev/null && { emit_ncu "$b" "$OUT/_ncu_raw.csv"; any=1; }
  done
  [ "$any" = 0 ] && echo "$b,ncu_skipped,timeout_or_error,-" >> $PROF
done; done
rm -f $OUT/_ncu_raw.csv
echo "  ncu_profile.csv rows: $(wc -l < $PROF)"

echo "== meta.txt =="
{ echo "hamlib_bench meta"; echo "date=$(date)"; echo "host=$(hostname) job=$SLURM_JOB_ID";
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader;
  echo "git_commit=$(git -C $REPO rev-parse HEAD 2>/dev/null)";
  echo "git_branch=$(git -C $REPO rev-parse --abbrev-ref HEAD 2>/dev/null)";
  echo "spmv_iters=200 spmspm_iters=30  x=deterministic hash (i*2654435761 &1023)/1024-0.5";
  echo "build: nvcc -O3 -std=c++17 -arch=sm_80 (A100; NOT the sm_90 suite default / H100=c29)";
  echo "TIMINGS ARE A100 (fresh) — do NOT merge with older H100 memory numbers.";
  echo "SpMV: ours=DIA zero-skip (dia1B D<=256 / dia2B D>256, int64 rowptr); baselines TC, dense-CUDA, cuSPARSE CSR-ALG2, Drawloom(Zenodo 17709956, fp32 sm_80).";
  echo "SpMSpM: ours=gather_flat; baselines HM(Haque, Algorithms 2024), cuSPARSE SpGEMM.";
  echo "q filter: scalable families q>=18; molecular {B2,O2,BeH,Li2,c2h,hnc} q>=12.";
  echo "ncu: ALL q-filtered matrices profiled (per user), one pass per matrix over spmv_dia_vs_tc kernels; 600s/call timeout -> ncu_skipped rows.";
  echo "tc_ablation.csv: T0 full-TC / T1 +symmetric(half recon) / T2 +zero-tile-skip / T3 +both[bestTC]; bit-exact; measured heis_18 T3=2.5x vs full-TC.";
  echo "spmspm_ablation.csv: L0 HM(Haque) / L1 +atomic-free(meta) / L2 +flat-pairs(ILP1) / L3 +adaptive-ILP4 / L4 +symmetric(flat_sym); bit-exact; measured heis_16 L4=7.5x vs HM.";
  echo "c2h/hnc: LEFT OUT — vibrational ||H||_1~1e6 -> Taylor K>KMAX at dt=0.0012 (non-convergent evolution).";
} > $OUT/meta.txt

echo "done=$(date)"
ls -la $OUT/*.csv $OUT/meta.txt
