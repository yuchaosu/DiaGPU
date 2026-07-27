# Build all diagonal-kernel benchmark/ablation drivers into bin/.
# Sources live in the parent sim/ dir; their #include paths resolve relative to
# the source file, so compiling from here is fine.
NVCC    ?= nvcc
ARCH    ?= sm_90
NVFLAGS  = -O3 -std=c++17 -arch=$(ARCH)
CXX     ?= g++

SIM   = ..
SPMV  = ../../spmv/src
SPMSPM= ../../spmspm
BIN   = bin

ALL = $(BIN)/spmv_e2e $(BIN)/spmspm_pure $(BIN)/spmspm_e2e $(BIN)/spmspm_sym \
      $(BIN)/fidelity $(BIN)/square_test $(BIN)/dia_to_mtx $(BIN)/dia_storehalf

all: $(ALL)

# --- SpMV: pure-kernel timing + full Krylov e2e (ours TF32 vs cuSPARSE) ---
$(BIN)/spmv_e2e: $(SIM)/e2e_speedup.cu ../../archive/spmv/tc_spmv_regdirect_kernel.cu
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcusparse

# --- SpMSpM pure C=H*H: ours(gather_meta) vs HM vs cuSPARSE vs CPU ---
$(BIN)/spmspm_pure: $(SIM)/spmspm_hamlib.cu $(SPMSPM)/paper_hm_kernel.cu
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp $^ -o $@ -lcusparse

# --- SpMSpM e2e operator-build U=sum c_k H^k: ours vs HM vs cuSPARSE ---
$(BIN)/spmspm_e2e: $(SIM)/e2e_spmspm.cu $(SPMSPM)/paper_hm_kernel.cu
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp $^ -o $@ -lcusparse

# --- SpMSpM symmetric (store-half + compute-half) vs full ---
$(BIN)/spmspm_sym: $(SIM)/spmspm_sym.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# --- fidelity (TF32 vs FP64, correctness gate) ---
$(BIN)/fidelity: $(SIM)/fidelity.cu ../../archive/spmv/tc_spmv_regdirect_kernel.cu
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcusparse

# --- squaring ablation (chain vs square) ---
$(BIN)/square_test: $(SIM)/square_test.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# --- helpers: export .mtx (Drawloom), store-half verifier ---
$(BIN)/dia_to_mtx: $(SIM)/dia_to_mtx.cpp
	$(CXX) -O2 -std=c++17 $< -o $@
$(BIN)/dia_storehalf: $(SIM)/dia_storehalf.cpp
	$(CXX) -O2 -std=c++17 $< -o $@

clean:
	rm -f $(BIN)/*
.PHONY: all clean
