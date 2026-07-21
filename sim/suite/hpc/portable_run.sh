#!/bin/bash
# ===========================================================================
# portable_run.sh — ONE self-contained bootstrap to run the whole plan on a
# fresh GPU cluster (e.g. H100). Does everything: dependency check, download
# baselines (Drawloom, QuEST), build all harnesses + baselines, verify data,
# and run NEEDTORUN Phase 1 (P1.1-P1.5). No hardcoded cluster paths.
#
# Configure via env (all have sensible defaults):
#   REPO   DiaGPU checkout root         (auto-detected from this script's path)
#   DATA   HamLib DIA .txt matrices dir (REQUIRED — see "data" step for how to get)
#   WORK   scratch for downloads/bins   (default $REPO/_work)
#   ARCH   sm target                    (default: auto from GPU; H100->sm_90a)
#   STEPS  which steps to run           (default: check download build data run)
#   SMOKE  1 => tiny q8/q10 matrix set  (fast end-to-end validation)
#   BUILD_QUEST 1 => also build QuEST GPU (P2.3 baseline; default 1, non-fatal)
#
# Examples:
#   DATA=/data/hamlib/dia bash portable_run.sh                 # full, H100 auto
#   ARCH=sm_90a DATA=/data/dia SMOKE=1 bash portable_run.sh    # quick smoke
#   STEPS=check bash portable_run.sh                           # deps only
# ===========================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
WORK=${WORK:-$REPO/_work}
DATA=${DATA:-/mnt/beegfs/ysu34/hamlib/dia_e2e}         # override on the new platform
ARCH=${ARCH:-auto}
STEPS=${STEPS:-"check download build data run"}
SMOKE=${SMOKE:-0}
BUILD_QUEST=${BUILD_QUEST:-1}
BIN=$WORK/bin; SRC=$WORK/src; OUT=${OUT:-$REPO/results}
mkdir -p "$BIN" "$SRC" "$OUT"
DRAW_ZIP_URL="https://zenodo.org/records/17709956/files/drawloom_ae.zip?download=1"
DRAW_MD5="925551551df2f653c06bbcd537036dbb"
QUEST_URL="https://github.com/QuEST-Kit/QuEST.git"

say(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok(){  printf '   \033[1;32mOK\033[0m %s\n' "$*"; }
warn(){ printf '   \033[1;33mWARN\033[0m %s\n' "$*"; }
die(){ printf '   \033[1;31mFAIL\033[0m %s\n' "$*"; exit 1; }
have(){ command -v "$1" >/dev/null 2>&1; }
step(){ case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

# ---- ARCH auto-detect (compute cap -> sm_XX; H100 9.0 -> sm_90a) ----
detect_arch(){
  local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  case "$cc" in
    9.0) echo sm_90a;; 8.9) echo sm_89;; 8.6) echo sm_86;; 8.0) echo sm_80;;
    7.5) echo sm_75;; 7.0) echo sm_70;; *) echo sm_80;; esac
}
[ "$ARCH" = auto ] && ARCH=$(detect_arch)
ARCHNUM=${ARCH#sm_}                                    # e.g. 90a, 80
GENCODE="arch=compute_${ARCHNUM},code=sm_${ARCHNUM}"   # Drawloom-style
CUDA_CC=${ARCHNUM%a}                                   # for QuEST CMAKE_CUDA_ARCHITECTURES (90,80)

echo "REPO=$REPO  DATA=$DATA  WORK=$WORK  ARCH=$ARCH  STEPS='$STEPS'  SMOKE=$SMOKE"

# ============================ 1. dependency check ============================
if step check; then
  say "dependency check"
  have nvcc || die "nvcc (CUDA toolkit) not found"; ok "nvcc $(nvcc --version|grep -oE 'release [0-9.]+'|head -1)"
  have g++  || die "g++ not found";                 ok "g++ $(g++ -dumpversion)"
  have git  || warn "git not found (needed to clone QuEST)"
  have cmake|| warn "cmake not found (needed to build QuEST)"
  have curl || have wget || warn "curl/wget not found (needed to download Drawloom)"
  have python3.11 && PY=python3.11 || PY=python3
  $PY -c 'import qiskit,h5py' 2>/dev/null && ok "$PY + qiskit + h5py (matrix gen)" || warn "$PY qiskit/h5py missing (only needed to GENERATE matrices)"
  have ncu && ok "ncu (Nsight Compute; optional profiling)" || warn "ncu not found (profiling optional)"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/   GPU: /'
  # cuSPARSE link test
  echo 'int main(){return 0;}' > $SRC/_t.cu
  nvcc -arch=$ARCH -lcusparse $SRC/_t.cu -o $BIN/_t 2>/dev/null && ok "cuSPARSE links (-arch=$ARCH)" || die "cuSPARSE link test failed for -arch=$ARCH"
  rm -f $BIN/_t $SRC/_t.cu
fi

# ============================ 2. download baselines ==========================
DRAW_DIR=$WORK/drawloom_ae; DRAW_BIN=$DRAW_DIR/drawloom_ae/Drawloom_code/myfloat
QUEST_DIR=$WORK/QuEST
if step download; then
  say "download Drawloom (Zenodo)"
  if [ -d "$DRAW_DIR/drawloom_ae/Drawloom_code" ]; then ok "Drawloom already present"
  else
    dl=$WORK/drawloom_ae.zip
    if [ ! -s "$dl" ] || [ "$(md5sum "$dl"|cut -d' ' -f1)" != "$DRAW_MD5" ]; then
      have curl && curl -fL -C - --retry 5 -o "$dl" "$DRAW_ZIP_URL" || wget -c -O "$dl" "$DRAW_ZIP_URL" || warn "Drawloom download failed"
    fi
    if [ -s "$dl" ] && [ "$(md5sum "$dl"|cut -d' ' -f1)" = "$DRAW_MD5" ]; then
      mkdir -p "$DRAW_DIR"; unzip -qo "$dl" -d "$DRAW_DIR" && ok "Drawloom extracted (md5 verified)"
    else warn "Drawloom zip missing/bad md5 -> Drawloom baseline will be N/A"; fi
  fi
  say "download QuEST (GitHub, P2.3 baseline)"
  if [ -d "$QUEST_DIR/quest" ]; then ok "QuEST already present"
  elif have git; then timeout 180 git clone --depth 1 "$QUEST_URL" "$QUEST_DIR" 2>&1 | tail -1 && ok "QuEST cloned" || warn "QuEST clone failed"
  else warn "git missing -> skip QuEST"; fi
fi

# ============================ 3. build ======================================
if step build; then
  say "build DiaGPU harnesses (-arch=$ARCH)"
  B(){ nvcc -O3 -std=c++17 -arch=$ARCH "$@"; }
  B $REPO/sim/spmv_dia_vs_tc.cu   $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $BIN/spmv_dia_vs_tc   -lcusparse                     || die "build spmv_dia_vs_tc"
  B $REPO/sim/spmspm_hamlib.cu    $REPO/spmspm/paper_hm_kernel.cu            -o $BIN/spmspm_pure       -lcusparse -Xcompiler -fopenmp || die "build spmspm_pure"
  B $REPO/sim/spmv_tc_ablation.cu $REPO/spmv/src/tc_spmv_regdirect_kernel.cu -o $BIN/spmv_tc_ablation                                || die "build spmv_tc_ablation"
  B $REPO/sim/spmspm_ablation.cu  $REPO/spmspm/paper_hm_kernel.cu            -o $BIN/spmspm_ablation   -Xcompiler -fopenmp            || die "build spmspm_ablation"
  g++ -O2 -std=c++17 $REPO/sim/dia_to_mtx.cpp -o $BIN/dia_to_mtx || die "build dia_to_mtx"
  ok "5 harnesses built"

  if [ -d "$DRAW_DIR/drawloom_ae/Drawloom_code" ]; then
    say "build Drawloom myfloat ($GENCODE)"
    ( cd "$DRAW_DIR/drawloom_ae/Drawloom_code"
      # patch the Makefile's hardcoded sm_80 to the target arch, then build fp32
      sed -i "s#arch=compute_[0-9a]*,code=sm_[0-9a]*#$GENCODE#g" Makefile
      make clean >/dev/null 2>&1; make myfloat FLAGS="-D f32" >/dev/null 2>&1
    ) && [ -x "$DRAW_BIN" ] && ok "Drawloom myfloat built" || warn "Drawloom build failed -> baseline N/A"
  fi

  if [ "$BUILD_QUEST" = 1 ] && [ -d "$QUEST_DIR/quest" ] && have cmake; then
    say "build QuEST — P2.3 baseline (GPU, CPU fallback)"
    # QuEST v4 GPU fails to compile on CUDA 13.x (Thrust). If the default nvcc is
    # >=13, auto-pick an installed CUDA<=12 toolkit (verified: 12.5 builds it).
    QCUDA=${QUEST_CUDA:-}
    cv=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+' | grep -oE '[0-9]+')
    if [ -z "$QCUDA" ] && [ "${cv:-13}" -ge 13 ]; then
      for c in /usr/local/cuda-12.5 /usr/local/cuda-12.4 /usr/local/cuda-12.3 /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-11.8; do
        [ -x "$c/bin/nvcc" ] && { QCUDA=$c; break; }; done
      [ -n "$QCUDA" ] && warn "default CUDA=$cv.x (QuEST-v4 GPU needs <=12) -> using $QCUDA/bin/nvcc for QuEST"
    fi
    QF="-DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_CC -DCMAKE_BUILD_TYPE=Release"
    [ -n "$QCUDA" ] && QF="$QF -DCMAKE_CUDA_COMPILER=$QCUDA/bin/nvcc -DCUDAToolkit_ROOT=$QCUDA"
    qok=0
    ( cd "$QUEST_DIR" && rm -rf build && cmake -B build $QF >/dev/null 2>&1 && cmake --build build -j 8 >/dev/null 2>&1 ) \
      && { ok "QuEST GPU build OK ($QUEST_DIR/build/libQuEST.so)"; qok=1; } \
      || warn "QuEST GPU build failed -> trying CPU-only"
    if [ $qok = 0 ]; then
      ( cd "$QUEST_DIR" && rm -rf build && cmake -B build -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 && cmake --build build -j 8 >/dev/null 2>&1 ) \
        && ok "QuEST CPU build OK (set QUEST_CUDA=/path/to/cuda-12 for GPU, or 'pip install cuquantum-python')" \
        || warn "QuEST build failed entirely (P2.3 optional; Phase 1 unaffected)"
    fi
  fi
fi

# ============================ 4. data check =================================
if step data; then
  say "data check — HamLib DIA matrices"
  n=$(ls "$DATA"/*.txt 2>/dev/null | grep -v imag | wc -l)
  if [ "$n" -gt 0 ]; then ok "$n .txt matrices in $DATA"
  else
    warn "no matrices in DATA=$DATA. To provide them:"
    echo "     (a) copy from the source cluster:  rsync -a user@src:/mnt/beegfs/ysu34/hamlib/dia_e2e/  $DATA/"
    echo "     (b) or regenerate (needs qiskit+h5py+HamLib HDF5): see sim/suite/hpc/gen_hamlib.slurm"
    [ "$STEPS" = "${STEPS/run/}" ] || die "cannot run without matrices"
  fi
fi

# ============================ 5. run Phase 1 ================================
if step run; then
  say "run NEEDTORUN Phase 1 (P1.1-P1.5)"
  export REPO DIA="$DATA" OUT ARCH PLAN_BIN="$BIN"
  export DRAWLOOM_BIN="${DRAWLOOM_BIN:-$DRAW_BIN}"    # external override wins
  if [ "$SMOKE" = 1 ]; then
    export MATS="heis_10_5 tfim_10_4 fermi_10_4 qmaxcut_10_4 maxcut_10_4" MATS_SP="heis_8_4 tfim_8_4 fermi_8_4"
  fi
  bash "$HERE/run_plan.sh"
  say "DONE — results in $OUT"
  ls -la "$OUT"/*.csv 2>/dev/null | awk '{print "   "$9" ("$5" B)"}'
fi
