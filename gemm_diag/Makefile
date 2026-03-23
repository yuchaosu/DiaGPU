NVCC       = nvcc
NVCC_FLAGS = -std=c++17 -O2
ARCH       = -arch=sm_86
TARGET     = gemm_diag

.PHONY: all clean run

all: $(TARGET)

$(TARGET): gemm_diag.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
