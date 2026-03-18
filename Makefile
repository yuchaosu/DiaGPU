# ============================================================
# Makefile for DiagSpMM Framework
# ============================================================

NVCC       = nvcc
NVCC_FLAGS = -std=c++17 -O2
# Adjust -arch to match your GPU (e.g., sm_80, sm_86, sm_90, sm_120)
ARCH       = -arch=sm_86
# NVTX3 is header-only in CUDA >=12; older toolkits need -lnvToolsExt
LIBS       = -lcusparse
HAZEL_INCLUDES = -I ../../conda/conda_envs/cudadev/nsight-compute-2024.3.2/host/target-linux-x64/nvtx/include

TARGET     = test_diag
SRCS       = test_dia.cu diag_kernel.cu
HEADERS    = diag_types.cuh diag_host_preprocess.cuh diag_kernel.cuh

.PHONY: all clean run hazel

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(SRCS) $(LIBS) -o $(TARGET)

hazel: $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(SRCS) $(LIBS) $(HAZEL_INCLUDES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
