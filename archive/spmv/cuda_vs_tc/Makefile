# CUDA-core vs tensor-core diagonal SpMV/SpMM benchmark.
NVCC   ?= nvcc
ARCH   ?= sm_90
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH)
SPMV    = ../../spmv/src

bench: bench.cu kernels.cuh $(SPMV)/tc_spmv_regdirect_kernel.cu
	$(NVCC) $(NVFLAGS) bench.cu $(SPMV)/tc_spmv_regdirect_kernel.cu -o bench -lcusparse

bench_comp: bench_comp.cu kernels.cuh compressed_dia.cuh
	$(NVCC) $(NVFLAGS) bench_comp.cu -o bench_comp -lcusparse

clean:
	rm -f bench bench_comp
.PHONY: clean
