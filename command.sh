# choose your arch: sm_80 (A100), sm_86 (RTX 30xx), sm_89 (RTX 40xx), sm_90 (H100)
nvcc -O3 -std=c++17 -arch=sm_86 src/DiaGPU.cu src/main.cpp -Iinclude -o output/DiaGPU