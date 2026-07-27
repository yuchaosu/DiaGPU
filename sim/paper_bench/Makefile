# Paper benchmark drivers.  ARCH=sm_90a for H100, sm_89 for local 4060 Ti.
ARCH ?= sm_89
NVCC ?= nvcc
FLAGS = -O3 -arch=$(ARCH) --std=c++17 --expt-relaxed-constexpr

all: spmv_driver spmspm_driver

spmv_driver: spmv_driver.cu common.cuh ../dia_io.hpp \
             ../../spmv/src/cuda_dia_kernels.cuh ../../spmv/src/spmv_gather.cuh
	$(NVCC) $(FLAGS) -o $@ $< -lcusparse

spmspm_driver: spmspm_driver.cu common.cuh ../dia_io.hpp \
               ../../spmspm/gather_flat.cuh ../../spmspm/paper_hm.cuh \
               ../../spmspm/paper_hm_kernel.cu
	$(NVCC) $(FLAGS) -o $@ $< ../../spmspm/paper_hm_kernel.cu -lcusparse

clean:
	rm -f spmv_driver spmspm_driver
