# ============================================================
# Makefile for DiagSpMM Framework
# ============================================================

NVCC       = nvcc
NVCC_FLAGS = -std=c++17 -O2
# Adjust -arch to match your GPU (e.g., sm_80, sm_86, sm_90, sm_120)
ARCH       = -arch=sm_90
# NVTX3 is header-only in CUDA >=12; older toolkits need -lnvToolsExt
LIBS       = -lcusparse
HAZEL_INCLUDES = -I ../../conda/conda_envs/cudadev/nsight-compute-2024.3.2/host/target-linux-x64/nvtx/include

# Tuning: max registers per thread (0 = unlimited, try 32/48/64)
# Lower values increase occupancy but may spill to local memory.
# Usage: make MAXREG=32
MAXREG     ?= 0

TARGET     = test_diag
SRCS       = test_dia.cu diag_kernel.cu
HEADERS    = diag_types.cuh diag_host_preprocess.cuh diag_kernel.cuh

# Build register-limit flag only if MAXREG > 0
ifneq ($(MAXREG),0)
  REG_FLAG = -maxrregcount=$(MAXREG)
else
  REG_FLAG =
endif

# Paper algorithms (standalone, no NVTX/cuSPARSE dependency)
PAPER_TARGET = paper_alg
PAPER_SRCS   = paper_algorithms.cu paper_hm_kernel.cu

# Benchmark comparison (DiagSpMM vs Paper-HM vs cuSPARSE)
BENCH_TARGET = bench_compare
BENCH_SRCS   = bench_compare.cu paper_hm_kernel.cu diag_kernel.cu diag_rowtiled_kernel.cu diag_batched_kernel.cu hm_optimized_kernel.cu
BENCH_HEADERS = $(HEADERS) paper_hm.cuh diag_rowtiled_kernel.cuh diag_batched_kernel.cuh hm_optimized_kernel.cuh

# DIA SpMV (standalone)
SPMV_TARGET  = test_spmv
SPMV_SRCS    = test_spmv.cu dia_spmv.cu
SPMV_HEADERS = dia_spmv.cuh

.PHONY: all clean run hazel reginfo paper run_paper bench run_bench spmv run_spmv

all: $(TARGET)

paper: $(PAPER_TARGET)

$(PAPER_TARGET): $(PAPER_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(REG_FLAG) $(PAPER_SRCS) -o $(PAPER_TARGET)

run_paper: $(PAPER_TARGET)
	./$(PAPER_TARGET)

bench: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_SRCS) $(BENCH_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(REG_FLAG) $(BENCH_SRCS) $(LIBS) -o $(BENCH_TARGET)

benchh: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_SRCS) $(BENCH_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(REG_FLAG) $(BENCH_SRCS) $(LIBS) $(HAZEL_INCLUDES) -o $(BENCH_TARGET)

run_bench: $(BENCH_TARGET)
	./$(BENCH_TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(REG_FLAG) $(SRCS) $(LIBS) -o $(TARGET)

hazel: $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(REG_FLAG) $(SRCS) $(LIBS) $(HAZEL_INCLUDES) -o $(TARGET)

# Show register usage per kernel (useful for tuning)
reginfo: $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(REG_FLAG) --ptxas-options=-v $(SRCS) $(LIBS) -o $(TARGET) 2>&1 | grep -E "registers|smem"

run: $(TARGET)
	./$(TARGET)

spmv: $(SPMV_TARGET)

$(SPMV_TARGET): $(SPMV_SRCS) $(SPMV_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(REG_FLAG) $(SPMV_SRCS) -o $(SPMV_TARGET)

run_spmv: $(SPMV_TARGET)
	./$(SPMV_TARGET)

clean:
	rm -f $(TARGET) $(PAPER_TARGET) $(BENCH_TARGET) $(SPMV_TARGET)
