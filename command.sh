# choose your arch: sm_80 (A100), sm_86 (RTX 30xx), sm_89 (RTX 40xx), sm_90 (H100), sm_89 (a5000adas)
nvcc -O3 -std=c++17 -arch=sm_86 src/DiaGPU.cu src/main.cpp -Iinclude -o output/DiaGPU

nvcc -O3 -std=c++17 -arch=sm_90 src/DiaGPU.cu src/bench_power_events.cpp -Iinclude -lcusparse -o output/bench_power_events

./output/bench_power_events 4096 5

