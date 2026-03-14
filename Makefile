# ============================================================
# Makefile for DiagSpMM Framework
# ============================================================

NVCC       = nvcc
NVCC_FLAGS = -std=c++17 -O2
# Adjust -arch to match your GPU (e.g., sm_80, sm_86, sm_90, sm_120)
ARCH       = -arch=sm_86
# NVTX3 is header-only in CUDA >=12; older toolkits need -lnvToolsExt
LIBS       =

TARGET     = test_diag
SRCS       = test_driver.cu diag_kernel.cu
HEADERS    = diag_types.cuh diag_host_preprocess.cuh diag_kernel.cuh

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(SRCS) $(LIBS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
