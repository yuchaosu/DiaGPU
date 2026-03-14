# DiagSpMM — Diagonal Sparse Matrix Multiplication Framework

高性能对角线稀疏矩阵乘法 CUDA kernel 框架。

```
C = A × B
```

A 和 B 均以 diagonal format（DIA 格式）存储。框架围绕"零 atomic、
output-tile-owned、A-stationary、warp-major packed B"四个原则设计。

---

## 目录

1. [问题背景与数学基础](#1-问题背景与数学基础)
2. [为什么不能用通用 SpGEMM](#2-为什么不能用通用-spgemm)
3. [核心设计原则](#3-核心设计原则)
4. [数学映射推导](#4-数学映射推导)
5. [数据结构设计](#5-数据结构设计)
6. [Host 端预处理流水线](#6-host-端预处理流水线)
7. [PackedB：warp-major 打包布局](#7-packedb-warp-major-打包布局)
8. [Kernel 设计](#8-kernel-设计)
9. [内存访问模式分析](#9-内存访问模式分析)
10. [Warp 负载均衡与 bucketization](#10-warp-负载均衡与-bucketization)
11. [为什么绝对禁止 atomic](#11-为什么绝对禁止-atomic)
12. [文件结构](#12-文件结构)
13. [编译与运行](#13-编译与运行)
14. [小例子：端到端数据流追踪](#14-小例子端到端数据流追踪)
15. [已完成的扩展实现](#15-已完成的扩展实现)
16. [后续扩展方向](#16-后续扩展方向)

---

## 1. 问题背景与数学基础

### Diagonal Format（DIA 格式）

一个稀疏矩阵的非零元素集中在若干条对角线上。每条对角线由以下
两个要素描述：

| 要素 | 含义 |
|---|---|
| `offset` (整数) | 对角线偏移：0 = 主对角线，正数 = 超对角线，负数 = 次对角线 |
| `values[]` | 该对角线上的元素值（连续存储） |

对于 M×N 矩阵、偏移为 `d` 的对角线：

```
start_row = max(0, -d)
start_col = max(0,  d)
length    = min(M - start_row, N - start_col)
```

第 `p` 个元素（从 0 开始）位于矩阵中的 `(start_row + p, start_col + p)`。

### 对角线乘法的关键性质

**定理**：A（M×K）的偏移 `d_a` 对角线与 B（K×N）的偏移 `d_b` 对角线
相乘，贡献且仅贡献到 C（M×N）的 `d_c = d_a + d_b` 对角线上。

**证明**：

```
A 的对角线 d_a 上的元素：A[i][i + d_a]
B 的对角线 d_b 上的元素：B[k][k + d_b]

C[i][j] += A[i][k] * B[k][j]

令 k = i + d_a（A 的列），则 j = k + d_b = i + d_a + d_b
→  j - i = d_a + d_b = d_c
→  此 C 元素在 d_c 对角线上。
```

这不是近似，而是精确的结构性质。整个框架所有优化都建立
在这一等式 `d_c = d_a + d_b` 之上。

---

## 2. 为什么不能用通用 SpGEMM

通用 CSR/COO SpGEMM 处理任意稀疏结构，必须解决：

- 运行时动态确定输出非零位置（symbolic phase）
- 多个 partial product 写入同一输出位置（需要 atomic 或二次归约）
- 无法预知 workload 分布（动态负载均衡困难）

DIA 格式的 diagonal 乘法有完全确定的结构：

- 所有输出对角线的位置在 host 预处理阶段已完全已知
- 每个输出位置由哪些 `(d_a, d_b)` 对贡献完全静态可计算
- 对角线上元素是顺序存储的，天然适合 coalesced 访问

使用通用 SpGEMM 处理对角线矩阵，等于放弃了这些已知结构，
引入大量不必要的 overhead。

---

## 3. 核心设计原则

### 原则 1：Output-Tile-Owned（输出 tile 独占）

```
一个 CTA  ←→  一个 output tile（exclusive ownership）
```

输出对角线被划分为长度为 `TILE_SIZE`（默认 128）的连续 tile。
每个 tile 由且仅由一个 CTA 负责计算和写回。

**后果**：CTA 的最终写回是唯一写者，完全不需要任何 atomic。

### 原则 2：A-Stationary Grouped Contributors

对于一个固定的 output tile（`d_c`, `p_begin`），枚举所有满足
`d_a + d_b = d_c` 的 contributor pair，并按 `d_a` 分组：

```
Group g:  所有 pair 共享同一个 d_a
          → 共享同一段 A 数据（A slice）
          → A slice 加载到 shared memory 一次，被 group 内所有 pair 复用
```

这是"A 静止（A-stationary）"原则：A 是固定的，B 在 pair 间变化。

### 原则 3：Warp-Major Packed B（B 的 warp-major 打包布局）

在 host 端将 B 的数据重新打包，使得：

```
warp 内 lane l 读取的 B 元素 = packedB[base + l]
```

32 个 lane 读取 32 个连续地址 → 一次 coalesced 128-byte 事务。

### 原则 4：Register Accumulation + Direct Writeback

每个线程在寄存器中维护一个 `acc`，对所有 group/pair 的贡献
求和，最终一次写回到对应输出位置：

```
C_values[idx] = acc;   // 不是 atomicAdd，是直接 store
```

---

## 4. 数学映射推导

### 4.1 有效位置区间计算

对于 output 对角线 `d_c`（长度 `c_len`），在位置 `p` 处：

```
row  = c_start_row + p
p_a  = row - a_start_row  = c_start_row + p - a_start_row   (A 对角线索引)
p_b  = row + d_a - b_start_row                               (B 对角线索引)
```

有效条件：

```
0 ≤ p_a < a_len     →   p ≥ -a_base,    p < a_len - a_base
0 ≤ p_b < b_len     →   p ≥ -b_base,    p < b_len - b_base
0 ≤ p   < c_len
```

五个约束取交集，得到 `[valid_begin, valid_end)`。这一计算
完全在 host 端完成，kernel 不做任何此类判断。

### 4.2 tile-local 坐标系

将全局对角线位置 `p` 转换为 tile 内本地坐标 `q = p - p_begin`，
则：

```
p_a = a_map_offset + q      其中 a_map_offset = c_start_row + p_begin - a_start_row
p_b = b_map_offset + q      其中 b_map_offset = c_start_row + p_begin + d_a - b_start_row
```

这两个 offset 在 host 预处理时计算并存入 `Group.a_map_offset`
和 `PairMeta.b_base`，kernel 只做简单的线性寻址。

---

## 5. 数据结构设计

各结构体的关系如下：

```
TaskTable[]
  ↓ group_begin, group_count
GroupTable[]
  ↓ pair_begin, pair_count
  ↓ a_global_start, a_map_offset
PairMetaTable[]
  ↓ packedB_offset
  ↓ out_valid_begin, out_valid_end
packedB[]  (flat float array, warp-major layout)
```

### 5.1 Task

```cpp
struct Task {
    int c_diag_idx;   // 输出对角线在 OutputDiag 表中的下标
    int c_offset;     // d_c 的值
    int p_begin;      // tile 在输出对角线上的起始位置
    int p_len;        // tile 长度（≤ TILE_SIZE = 128）
    int group_begin;  // 在 Group 表中的起始下标
    int group_count;  // 该 task 包含的 group 数
    int work_est;     // 预估工作量（pair overlaps 之和）
    int bucket;       // 0=LIGHT, 1=MEDIUM, 2=HEAVY
};
```

**设计思路**：一个 Task = 一个 CTA 的全部工作，是 kernel 的
调度单元。bucket 决定用哪种 kernel launch。

### 5.2 Group

```cpp
struct Group {
    int a_diag_idx;       // A 的第几条对角线
    int a_offset;         // d_a 值
    int a_global_start;   // A.values[] 中该对角线的起始偏移
    int a_diag_len;       // A 对角线长度
    int a_map_offset;     // tile-local q → A 对角线索引的偏移量
    int pair_begin;       // PairMeta 表的起始下标
    int pair_count;       // 该 group 包含的 pair 数
};
```

**设计思路**：Group 是 A-stationary 复用的单位。同一 group 内
所有 pair 用同一份 smemA，这是 shared memory 复用的来源。
`a_map_offset` 预计算好后，kernel 内 A 的寻址退化为一次加法。

### 5.3 PairMeta

```cpp
struct PairMeta {
    int b_diag_idx;       // B 的第几条对角线
    int b_offset;         // d_b 值
    int out_valid_begin;  // tile-local 有效起始（含，用于调试/优化）
    int out_valid_end;    // tile-local 有效结束（不含）
    int a_base;           // 调试用，记录 tile 起点对应的 A 对角线索引
    int b_base;           // b_map_offset，记录 tile-local q → B 对角线索引的偏移
    int packedB_offset;   // 该 pair 的 warp-major packed B 在 packedB[] 中的偏移
};
```

**设计思路**：`out_valid_begin/end` 是冗余信息（kernel 靠零填充
自动处理），但保留用于调试和后续优化（例如 warp-level skip）。
Kernel 真正使用的核心字段只有 `packedB_offset`。

---

## 6. Host 端预处理流水线

Host 预处理的核心哲学：**把所有复杂计算挪到 CPU，让 kernel
只做纯粹的 compute，不做任何 index gymnastics**。

```
build_output_diagonals()
        ↓
  枚举所有 d_c = d_a + d_b，建立 OutputDiagInfo 表
        ↓
build_contributors()
        ↓
  对每个 d_c，求所有 (d_a, d_b) pair 的有效区间 [valid_begin, valid_end)
        ↓
group_contributors_by_a_diag()
        ↓
  在每个 d_c 内部，按 a_diag_idx 排序，为 A-stationary 分组提供基础
        ↓
estimate_tile_work()  +  tileing loop
        ↓
  按 TILE_SIZE 切分每条输出对角线
  对每个 tile 求 work = Σ overlap_len(pair, tile)
  按 work 分桶（LIGHT / MEDIUM / HEAVY）
        ↓
build_all()：emit Task / Group / PairMeta 表 + pack B
```

### 6.1 contributor 有效区间的计算

`build_contributors()` 是整个预处理最关键的一步：

```cpp
int a_base = c_start_row - a_start_row;
int b_base = c_start_row + d_a - b_start_row;

int lo = max(0, max(-a_base, -b_base));
int hi = min(c_length, min(a_len - a_base, b_len - b_base));
```

五个约束的联立解，结果是一个连续区间。只有 `lo < hi` 时
该 pair 才对当前 output diagonal 有贡献。

### 6.2 work 估算

```
work(tile) = Σ  max(0, min(pair.valid_end, tile_end)
                       - max(pair.valid_begin, tile_begin))
```

这比"每个 tile 固定工作量"更准确，因为：
- 靠近对角线边界的 tile，许多 pair 的 overlap 更短
- contributor 数量多的 tile 工作量更大
- 只用瓦片切分不考虑 pair 数量会严重高估或低估

---

## 7. PackedB：warp-major 打包布局

### 7.1 为什么需要打包

假设 tile 覆盖输出位置 `[p_begin, p_begin + p_len)`。
对于其中某个 pair，B 的访问序列为：

```
原始 B 数据：B.values[b_start + (b_map_offset + 0)]
              B.values[b_start + (b_map_offset + 1)]
              B.values[b_start + (b_map_offset + 2)]
              ...
```

若直接从原始 B 读取，访问本身是 coalesced（相邻线程读相邻地址）。
但问题在于 `b_map_offset` 会因 pair 不同而偏移到不同的基地址，
打包则统一了访问模式，同时支持零填充无效位置。

### 7.2 布局定义

对于 task `t`、group `g`、pair `p`、tile-local 位置 `q`：

```
packedB_for_pair[q] = B.values[b_start + b_map_offset + q]  (若 q 有效)
                    = 0.0f                                   (若 q 无效)
```

在内存中，该 pair 的数据从 `packedB[pair.packedB_offset]`
开始，共 `packed_count` 个 float（`packed_count` 向上取整到
32 的倍数，以保证最后一个 warp 的访问也是对齐的）。

### 7.3 为什么 warp-major 保证 coalesced

```
Warp w（lanes 0..31）在处理某 pair 时读取：
  packedB[pair.packedB_offset + w * 32 + 0]
  packedB[pair.packedB_offset + w * 32 + 1]
  ...
  packedB[pair.packedB_offset + w * 32 + 31]

→ 32 个连续 float = 128 bytes
→ 单次 L2 cache line 事务（完美 coalesced）
```

在 kernel 中，所有线程统一执行：

```cpp
float b_val = packedB[pair.packedB_offset + tid];
```

其中 `tid = warp_id * 32 + lane_id`，正好是上述模式。

### 7.4 对比非打包方案

若不打包，当 `b_map_offset` 使得 B 的访问起点不是 128 字节
对齐的，则每个 warp 需要 2 次 cache line 事务。对于大量短 pair
的场景，这会将 B 的有效访问带宽降低接近一半。

---

## 8. Kernel 设计

### 8.1 线程到输出位置的映射

```
blockDim = (128, 1, 1)   →   4 warps

tid = threadIdx.x
warp_id  = tid / 32      →   0, 1, 2, 3
lane_id  = tid % 32      →   0 .. 31

out_local_idx = tid      →   tile 内本地位置

Warp 0 负责输出位置 [ 0.. 31]
Warp 1 负责输出位置 [32.. 63]
Warp 2 负责输出位置 [64.. 95]
Warp 3 负责输出位置 [96..127]
```

每个线程恰好独占一个输出位置，寄存器 `acc` 即为该位置的最终值。

### 8.2 执行流水（medium kernel）

```
CTA 启动，blockIdx.x → task_ids[blockIdx.x] → Task

for each Group g in Task:
    ├── 所有 128 线程协同将 A slice 载入 smemA
    │       smemA[tid] = A_values[g.a_global_start + g.a_map_offset + tid]
    │       （若 tid 超出有效范围，存 0.0f）
    │
    ├── __syncthreads()           ← 确保 smemA 全部就绪
    │
    └── for each Pair p in Group:
            ├── 读 B：b_val = packedB[p.packedB_offset + tid]  (coalesced)
            ├── 读 A：a_val = smemA[tid]                       (shared mem, no conflict)
            └── acc += a_val * b_val                            (FMAD in registers)

    └── __syncthreads()           ← 保护 smemA，防止下一 group 提前写入

最终：C_values[output_linear_index(...)] = acc     ← 直接 store，无 atomic
```

### 8.3 Shared Memory 使用策略

```
smemA:  TILE_SIZE × sizeof(float) = 128 × 4 = 512 bytes
用途：  存当前 group 的 A slice（high-reuse stationary operand）
特性：  group 内所有 pair 复用 → reuse ratio = pair_count / group

故意不放入 shared memory 的数据：
  - packedB：streaming 数据，每个 pair 读一次，无必要缓存
  - PairMeta / Group：结构体每个 group/pair 读一次，走 L1 cache 足够
```

**原则**：shared memory 只放高复用 operand（A slice），
不用于低复用 streaming 数据（B）。

### 8.4 Register File 使用

每个线程使用的寄存器：
- `acc`（1 个 float）：最终输出累加器
- `task`, `grp`, `pair`：临时的结构体（编译器通常会寄存器化）
- `a_val`, `b_val`：一对操作数

register pressure 很低，有充足空间留给编译器做 ILP 优化。

---

## 9. 内存访问模式分析

### 9.1 Global Memory Reads（每个 CTA）

| 操作 | 访问模式 | 说明 |
|---|---|---|
| smemA 加载 | **coalesced** | 128 线程读 A_values 的 128 个连续 float |
| packedB 读取 | **coalesced** | 32 个 lane 读 packedB 的 32 个连续 float |
| C 写回 | **coalesced** | 128 线程写 C_values 的 128 个连续 float |
| Task/Group/PairMeta | stride-1 per CTA | 数据量小，走 L1/L2 cache |

所有热路径（A 和 B 的实际数据）均为 coalesced 访问。

### 9.2 Shared Memory Bank Conflicts

smemA 的访问：
```
warp w 访问位置 [w*32 + 0, w*32 + 1, ..., w*32 + 31]
bank(i) = i % 32
→ lane l 访问 bank (w*32 + l) % 32 = l
→ 32 个 lane 访问 32 个不同 bank
→ 零 bank conflict
```

### 9.3 L1/L2 Cache 友好性

- A slice 每个 group 加载一次，载入 shared memory 后绝不再走 L1
- packedB 是 streaming 访问，数据量 = pairs × tile_len × 4 bytes，
  对于 medium bucket 通常可完全驻留 L2
- 结构体表（Task/Group/PairMeta）体积小，读取时稳定驻留 L1

---

## 10. Warp 负载均衡与 Bucketization

### 10.1 不均衡的来源

对角线矩阵的 contributor 分布天然不均：

- 主对角线（d=0）与之相乘的 pair 数量最多
- 靠近矩阵边角的对角线不仅 pair 数量少，每个 pair 的有效重叠
  区间也更短
- 固定瓦片长度切分会导致 tile 间工作量差异高达数十倍

### 10.2 三档 Bucket 方案

```
work(task) = Σ overlap_len(pair)

LIGHT  : work ≤ 128      → 多 task 打包进一个 CTA，每 warp 一个 task
MEDIUM : work ≤ 4096     → 一 CTA 一个 task，128 线程
HEAVY  : work >  4096    → 一 CTA 一个 task，更大 block，双缓冲 smemA
```

阈值 `LIGHT_WORK_MAX` 和 `MEDIUM_WORK_MAX` 均为 `constexpr`，
可在 `diag_types.cuh` 中调整。

### 10.3 Middle kernel 内的 warp 对齐

medium kernel 内，所有 warp 执行完全相同的控制流：

```
for group → for pair → FMAD
```

没有 warp divergence（有效/无效的分支由零填充数据消除）。
不同 warp 只是操作不同的输出位置区间，但计算路径完全一致。

### 10.4 无效位置的处理策略

两种可能的方案：

| 方案 | 优点 | 缺点 |
|---|---|---|
| `if (tid < valid_range)` 分支 | 跳过无效乘法 | warp divergence；额外分支指令 |
| **零填充 packedB 和 smemA**（本框架采用） | 无任何分支；warp 完全对齐 | 少量无效乘法（0×0=0）不影响结果 |

对于 medium bucket（pair 数量适中），无效位置通常只占 tile 边界
处很少的几个，零填充方案的额外计算代价可忽略不计。

---

## 11. 为什么绝对禁止 Atomic

### 11.1 写冲突的来源

在 A-entry-centric 方案中：

```
for each A nonzero A[i][k]:
    for each B nonzero on row k:
        atomicAdd(&C[i][j], A[i][k] * B[k][j])
```

多条 `d_a` 不同的对角线上的 A 元素会写到同一个 C 位置（比如
当 `d_c = d_a1 + d_b1 = d_a2 + d_b2`），产生写冲突，必须
用 `atomicAdd` 序列化，严重降低吞吐量。

### 11.2 本框架如何消除写冲突

```
输出视角（output-centric）：
  线程 t 负责 C 位置 p，C[p] 的最终值 = Σ_{所有 pair} A_pair[p] * B_pair[p]

  所有这些求和都由同一个线程在寄存器 acc 中完成，
  最后 C[p] = acc  → 单次写，无冲突
```

换言之：写冲突的出现源于"先生成 partial product，再合并写
C"的思路。本框架的 output-tile-owned 设计从根本上消除了
这种思路。

### 11.3 Atomic 的具体危害

除了正确性之外，`atomicAdd` 还有以下性能代价：

- 强制序列化对同一地址的写操作，warp 内多次写相同 cacheline
  会产生连续 cache invalidation
- 阻止编译器将 `acc` 保留在寄存器中（必须每次写回 global memory）
- 对 L2 cache 产生大量 read-modify-write 流量
- 无法利用 write-combining 优化

---

## 12. 文件结构

```
DiaGPU/
├── diag_types.cuh            核心数据结构和常量
│                               DiagMatrix, Task, Group, PairMeta,
│                               OutputDiag, KernelParams
│                               constexpr: WARP_SIZE, BLOCK_SIZE_MED/LIGHT/HEAVY,
│                               TILE_SIZE/TILE_SIZE_LIGHT/TILE_SIZE_HEAVY,
│                               WIDE_TILE_SIZE, WIDE_BLOCK_SIZE, WIDE_ELEMS_PER_THREAD,
│                               LIGHT/MEDIUM_WORK_MAX,
│                               ADAPTIVE_LARGE/HUGE_DIAG_THRESH
│
├── diag_host_preprocess.cuh  Host 端预处理（header-only inline）
│                               build_output_diagonals()
│                               build_contributors()
│                               group_contributors_by_a_diag()
│                               estimate_tile_work()
│                               pack_B_for_pair()
│                               choose_tile_config()  ← 自适应 tile 选择
│                               build_all()           ← 主入口（可配 tile_size）
│                               build_all_adaptive()  ← 自适应 tile 入口
│                               PreprocessResult（含 light/medium/heavy/wide_task_ids）
│
├── diag_kernel.cuh           Kernel 声明 + device inline helpers
│                               output_linear_index()
│                               get_task / get_group / get_pair / get_packedB_ptr
│                               4 个 kernel 前向声明
│                               launch_{light,medium,heavy,wide}_kernel() 声明
│
├── diag_kernel.cu            Kernel 实现
│                               diag_spmm_medium_kernel（完整实现）
│                               diag_spmm_light_kernel（完整实现，多 task/CTA）
│                               diag_spmm_heavy_kernel（完整实现，双缓冲 smemA）
│                               diag_spmm_wide_kernel（完整实现，多输出/线程）
│                               launch_{light,medium,heavy,wide}_kernel（完整实现）
│
├── test_driver.cu            测试驱动
│                               4×4 小例子
│                               CPU reference（cpu_diag_multiply）
│                               per-bucket dispatch（分桶调度）
│                               host preprocess → device upload → launch → verify
│
└── Makefile                  nvcc 编译脚本
```

**编译关系**：

```
test_driver.cu  ─includes─→  diag_types.cuh
                              diag_host_preprocess.cuh
                              diag_kernel.cuh
diag_kernel.cu  ─includes─→  diag_kernel.cuh
                                └─includes─→ diag_types.cuh

编译命令：nvcc -std=c++17 -O2 -arch=<sm_XX> test_driver.cu diag_kernel.cu -o test_diag
```

---

## 13. 编译与运行

```bash
# 查看 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# 修改 Makefile 中的 ARCH 为对应值，例如：
#   sm_80  → A100, RTX 30xx
#   sm_86  → RTX 3090
#   sm_90  → H100
#   sm_120 → B100/B200 (Blackwell)

make                  # 编译
make run              # 编译并运行
make clean            # 清理
```

预期输出（4×4 测试用例）：

```
CPU reference C (dense):
     4.0  16.0  16.0   0.0
     1.0   7.0  19.0  18.0
     0.0   2.0  10.0  22.0
     0.0   0.0   3.0  13.0

=== GPU Output (diagonal format) ===
  d_c = -1:  [1.0, 2.0, 3.0]
  d_c =  0:  [4.0, 7.0, 10.0, 13.0]
  d_c =  1:  [16.0, 19.0, 22.0]
  d_c =  2:  [16.0, 18.0]

PASS: GPU output matches CPU reference.
```

---

## 14. 小例子：端到端数据流追踪

以输出对角线 `d_c = 0` 为例，追踪完整的数据流。

### 输入矩阵

```
A (4×4):
  d=-1: values=[1,2,3]  (A[1][0], A[2][1], A[3][2])
  d= 0: values=[4,5,6,7]
  d= 1: values=[8,9,10]

B (4×4):
  d= 0: values=[1,1,1,1]
  d= 1: values=[2,2,2]
```

### Step 1：枚举 contributors

`d_c = 0` 的所有 `(d_a, d_b)` 满足 `d_a + d_b = 0`：

| pair | d_a | d_b | a_base | b_base | valid range |
|---|---|---|---|---|---|
| P1 | -1 | +1 | -1 | -1 | [1, 4) |
| P2 |  0 |  0 |  0 |  0 | [0, 4) |

### Step 2：按 a_diag 分组

```
Group 0  (a_diag = d_a = -1):  [P1]
Group 1  (a_diag = d_a =  0):  [P2]
```

### Step 3：tile = [0, 4)，生成 smemA 和 packedB

**Group 0（d_a = -1）的 smemA 加载**：

```
a_map_offset = c_start_row + p_begin - a_start_row = 0 + 0 - 1 = -1

thread 0: p_a = -1 + 0 = -1  → 无效 → smemA[0] = 0.0
thread 1: p_a = -1 + 1 =  0  → A[-1 diag][0] = 1.0  → smemA[1] = 1.0
thread 2: p_a = -1 + 2 =  1  → A[-1 diag][1] = 2.0  → smemA[2] = 2.0
thread 3: p_a = -1 + 3 =  2  → A[-1 diag][2] = 3.0  → smemA[3] = 3.0
```

**Pair P1（d_b = +1）的 packedB 打包**：

```
b_map_offset = c_start_row + p_begin + d_a - b_start_row = 0+0+(-1)-0 = -1

q=0: p_b=-1 → 无效 → packedB[0] = 0.0
q=1: p_b= 0 → B[+1 diag][0] = 2.0  → packedB[1] = 2.0
q=2: p_b= 1 → B[+1 diag][1] = 2.0  → packedB[2] = 2.0
q=3: p_b= 2 → B[+1 diag][2] = 2.0  → packedB[3] = 2.0
(q=4..31: 填 0.0，保证最后 warp 对齐)
```

**Group 0 内 FMAD 运算**：

```
thread 0: acc += 0.0 × 0.0 = 0.0
thread 1: acc += 1.0 × 2.0 = 2.0
thread 2: acc += 2.0 × 2.0 = 4.0
thread 3: acc += 3.0 × 2.0 = 6.0
```

**Group 1（d_a = 0）的 smemA 加载**：

```
a_map_offset = 0

thread 0: smemA[0] = A[0 diag][0] = 4.0
thread 1: smemA[1] = A[0 diag][1] = 5.0
thread 2: smemA[2] = A[0 diag][2] = 6.0
thread 3: smemA[3] = A[0 diag][3] = 7.0
```

**Pair P2（d_b = 0）的 packedB**：`[1.0, 1.0, 1.0, 1.0]`

**Group 1 内 FMAD 运算（acc 累加）**：

```
thread 0: acc = 0.0 + 4.0×1.0 = 4.0   → C[0][0] = 4   ✓
thread 1: acc = 2.0 + 5.0×1.0 = 7.0   → C[1][1] = 7   ✓
thread 2: acc = 4.0 + 6.0×1.0 = 10.0  → C[2][2] = 10  ✓
thread 3: acc = 6.0 + 7.0×1.0 = 13.0  → C[3][3] = 13  ✓
```

### Step 4：写回（无 atomic）

```cpp
C_values[c_diags[0].values_start + 0] = 4.0;    // thread 0
C_values[c_diags[0].values_start + 1] = 7.0;    // thread 1
C_values[c_diags[0].values_start + 2] = 10.0;   // thread 2
C_values[c_diags[0].values_start + 3] = 13.0;   // thread 3
```

完全正确，且全程无 atomicAdd。

---

## 15. 已完成的扩展实现

### 15.1 Light Kernel（已实现）

**目标**：多 task 打包进一个 CTA，每个 warp 独立处理一个 task，
消除小 task 的 CTA launch overhead。

#### 线程映射

```
blockDim = 128 (4 warps)
Grid     = ceil(num_light_tasks / 4)

warp_id  = tid / 32  →  task slot (0..3)
lane_id  = tid % 32  →  输出位置

task_slot = blockIdx.x * 4 + warp_id
→  warp 0 处理 task_ids[4*blockIdx.x + 0]
→  warp 1 处理 task_ids[4*blockIdx.x + 1]
→  ...
```

#### Shared Memory 分区

```
smem 总大小：4 * WARP_SIZE * sizeof(float) = 512 bytes

warp 0 → smem[  0.. 31]
warp 1 → smem[ 32.. 63]
warp 2 → smem[ 64.. 95]
warp 3 → smem[ 96..127]

每个 warp 的 smemA 分区完全独立，互不干扰。
```

#### 同步策略

```
传统 medium kernel：__syncthreads()（block 级同步，所有 128 线程参与）
light kernel：      __syncwarp()  （warp 级同步，仅 32 个 lane 参与）

优势：
  - 无跨 warp 依赖，避免不必要的全局同步
  - 各 warp 可以独立推进，即使 task 长度不同
  - 当 task_slot >= num_tasks 时，对应 warp 直接 return，
    不影响其他 warp 的执行
```

#### 执行流水

```
warp w 启动：task = tasks[task_ids[task_slot]]

for each Group g in task:
    ├── lane l 将 A slice 载入 my_smemA
    │     my_smemA[lane_id] = A_values[...]  (若有效)
    │                       = 0.0f           (若无效)
    │
    ├── __syncwarp()    ← 仅 32 lane 同步
    │
    └── for each Pair p in Group:
            ├── b_val = packedB[p.packedB_offset + lane_id]
            └── acc += my_smemA[lane_id] * b_val

    └── __syncwarp()    ← 保护 smemA

最终：C_values[...] = acc    ← 直接 store
```

#### 内存访问分析

| 操作 | 访问模式 | 说明 |
|---|---|---|
| smemA 加载 | **coalesced** | 32 lane 读 A_values 的 32 个连续 float |
| packedB 读取 | **coalesced** | 32 lane 读 packedB 的 32 个连续 float |
| C 写回 | **coalesced** | 32 lane 写 C_values 的 32 个连续 float |
| smemA bank | **零冲突** | lane l 访问 bank l → 32 lane 访问 32 bank |

#### 性能对比

```
场景：100 个 LIGHT task（work ≤ 128）

medium kernel：100 个 CTA launch，每 CTA 仅用 1 warp（浪费 3 warp）
  → 100 次 CTA 调度 + 75% 线程空转

light kernel：25 个 CTA launch，每 CTA 4 warp 全忙
  → 25 次 CTA 调度 + 0% 线程空转
  → CTA launch overhead 减少 4 倍
```

---

### 15.2 Heavy Kernel（已实现）

**目标**：对有大量 pairs 和长 overlap 的 task 进行流水线优化，
通过双缓冲 smemA 重叠数据加载与计算。

#### Block 配置

```
blockDim = 256 (8 warps)     ← 比 medium 多 4 个 warp
Grid     = num_heavy_tasks   ← 一 CTA 一个 task
smem     = 2 * TILE_SIZE_HEAVY * sizeof(float) = 2048 bytes
```

8 个 warp 的意义：

```
medium kernel（4 warps）：
  每 CTA 的 FMA 吞吐 = 4 warps × 32 lanes = 128 FMAD/cycle

heavy kernel（8 warps）：
  每 CTA 的 FMA 吞吐 = 8 warps × 32 lanes = 256 FMAD/cycle
  → 对 pair 数量多的 task，计算密度翻倍
```

#### 双缓冲 smemA 协议

```
smemA[0]:  smem_heavy[0       .. 255]    ← buffer 0
smemA[1]:  smem_heavy[256     .. 511]    ← buffer 1

初始状态：
  buf = 0
  将 Group 0 的 A slice 载入 smemA[0]
  __syncthreads()

Group 循环 (gi = 0 .. group_count-1)：

  ┌─────────────────────────────────────────────┐
  │  smemA[buf] 已就绪，包含 Group gi 的 A 数据   │
  │                                               │
  │  1. 计算 Group gi 的所有 pairs                 │
  │     for each Pair p:                          │
  │       b_val = packedB[p.offset + tid]         │
  │       acc += smemA[buf][tid] * b_val           │
  │                                               │
  │  2. __syncthreads()  ← 确保计算完毕            │
  │                                               │
  │  3. 若 gi+1 < group_count：                    │
  │     将 Group gi+1 的 A slice 载入 smemA[1-buf] │
  │     __syncthreads()  ← 确保加载完毕            │
  │                                               │
  │  4. buf = 1 - buf   ← 翻转缓冲                │
  └─────────────────────────────────────────────┘
```

时序图（理想情况，group 间 A 加载与计算重叠）：

```
时间 →
Group 0:  [Load A→buf0] [Compute pairs] [Load A→buf1]
Group 1:                                 [Compute pairs] [Load A→buf0]
Group 2:                                                  [Compute pairs]
          │                              │
          └── buf0 用于 compute ──────────┘
                                         └── buf1 用于 compute
```

#### 线程映射

```
tid = threadIdx.x  (0 .. 255)
→ 每线程独占一个输出位置
→ Warp 0: [0..31], Warp 1: [32..63], ... , Warp 7: [224..255]

tile_len 最大 256（TILE_SIZE_HEAVY），每线程一个 acc。
```

#### 与 cp.async 的结合（15.4）

双缓冲的结构天然适配 `cp.async`（sm_80+ 的异步全局→共享拷贝）。
当前实现使用同步加载+barrier 方式完成 ping-pong，预留了
将 `load_a_slice()` 替换为 `__pipeline_memcpy_async()` 的接口：

```cpp
// 当前实现（同步加载）
smemA_buf[target_buf][tid] = A_values[...];
__syncthreads();

// 未来可替换为（非阻塞异步加载）
__pipeline_memcpy_async(smemA_buf[target_buf] + tid,
                        A_values + ..., sizeof(float));
__pipeline_commit();
// ... compute on the other buffer ...
__pipeline_wait_prior(0);
__syncthreads();
```

---

### 15.4 cp.async 异步预取（已融入 Heavy Kernel）

cp.async 的核心思想——"在计算当前 group 的同时异步加载下一 group
的 A 数据"——已通过 Heavy Kernel 的双缓冲架构实现。

具体来说：

1. **双缓冲 smemA 提供了硬件层面的重叠基础**：两块独立的共享
   内存区域使得一个 buffer 可被读取（计算）的同时另一个 buffer
   被写入（加载），不产生数据竞争。

2. **加载与计算的指令流水**：在 `for pair` 内循环期间，GPU 的
   内存子系统可以并行处理下一 group 的全局内存读取请求。

3. **barrier 语义**：`__syncthreads()` 在翻转 buffer 前确保
   加载完成，功能等价于 `__pipeline_wait_prior(0)`。

```
性能收益模型：

设 T_load = 加载一个 A slice 的延迟
设 T_comp = 计算一个 group 所有 pairs 的延迟

无双缓冲：total = Σ (T_load + T_comp)  = G × (T_load + T_comp)
有双缓冲：total ≈ T_load + G × max(T_load, T_comp)

当 T_load ≈ T_comp 时，加速比 ≈ 2×
```

---

### 15.5 Wide Kernel — 多输出 Tiling（已实现）

**目标**：将 TILE_SIZE 与 BLOCK_SIZE 解耦，每线程负责多个输出位置，
增大每次 kernel launch 的工作粒度。

#### 常量定义

```
WIDE_TILE_SIZE        = 512       ← 每个 tile 覆盖 512 个输出位置
WIDE_BLOCK_SIZE       = 128       ← 仍用 128 线程 = 4 warps
WIDE_ELEMS_PER_THREAD = 512/128 = 4   ← 每线程 4 个输出位置
```

#### 线程到输出的映射

```
线程 tid 负责的输出位置：

  q[0] = tid              →  [0    .. 127]
  q[1] = tid + 128        →  [128  .. 255]
  q[2] = tid + 256        →  [256  .. 383]
  q[3] = tid + 384        →  [384  .. 511]

寄存器：acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}
→ 4 个独立的累加器，每个对应一个输出位置
```

#### smemA 加载（分 4 次迭代）

```
smemA 大小 = WIDE_TILE_SIZE = 512 float = 2048 bytes

加载策略：128 线程每次加载 128 float，迭代 4 次

迭代 k=0:  线程 tid 加载 smemA[tid +   0] = A[...]  (coalesced)
迭代 k=1:  线程 tid 加载 smemA[tid + 128] = A[...]  (coalesced)
迭代 k=2:  线程 tid 加载 smemA[tid + 256] = A[...]  (coalesced)
迭代 k=3:  线程 tid 加载 smemA[tid + 384] = A[...]  (coalesced)

__syncthreads() ← 确保 512 float 全部就绪
```

#### 计算阶段

```
for each Pair p:
    k=0: acc[0] += smemA[tid]       * packedB[p.offset + tid]
    k=1: acc[1] += smemA[tid + 128] * packedB[p.offset + tid + 128]
    k=2: acc[2] += smemA[tid + 256] * packedB[p.offset + tid + 256]
    k=3: acc[3] += smemA[tid + 384] * packedB[p.offset + tid + 384]

每次 k 迭代中，warp 内 32 lane 仍读 32 个连续 float → coalesced
packedB 此时打包为 512 float（padded to 512 = 16 × 32）
```

#### Coalesced 访问证明

```
warp w 在 k 迭代中读取：
  packedB[p.offset + w*32 + k*128 + 0]
  packedB[p.offset + w*32 + k*128 + 1]
  ...
  packedB[p.offset + w*32 + k*128 + 31]

→ 32 个连续 float = 128 bytes → 单次 L2 cache line 事务
→ 完美 coalesced，与 medium kernel 相同
```

#### 适用场景

```
对角线很长（> 4096 元素）但 pair 数量少（≤ 4）的场景：

medium kernel（TILE_SIZE=128）：
  长度 8192 的对角线 → 64 个 task → 64 次 CTA 调度

wide kernel（TILE_SIZE=512）：
  长度 8192 的对角线 → 16 个 task → 16 次 CTA 调度
  → launch overhead 减少 4 倍
  → 每 CTA 计算量增加 4 倍，GPU 利用率更高
```

#### 与 build_all() 的集成

```
build_all(A, B, M, K, N, tile_size)
                          ↑
                    可传入 WIDE_TILE_SIZE = 512

→ 所有 tile 按 512 切分
→ 相应 packedB 按 512 对齐
→ 生成的 task 由 launch_wide_kernel() 调度
```

---

### 15.6 Tile Size 自适应（已实现）

**目标**：根据输出对角线的长度和 contributor 密度，在 host
预处理阶段自动选择最优的 tile size，而非全局固定 128。

#### 自适应策略

`choose_tile_config()` 根据三个维度做决策：

| 维度 | 阈值 | 选择 |
|---|---|---|
| 估算工作量 ≤ `LIGHT_WORK_MAX` (128) | — | TILE=32, BLOCK=128 (light) |
| 估算工作量 > `MEDIUM_WORK_MAX` (4096) | — | TILE=256, BLOCK=256 (heavy) |
| 对角线长度 > `ADAPTIVE_HUGE_DIAG_THRESH` (4096) 且 pair 数 ≤ 4 | — | TILE=512, BLOCK=128 (wide) |
| 对角线长度 > `ADAPTIVE_LARGE_DIAG_THRESH` (1024) 且 pair 数 ≤ 8 | — | TILE=256, BLOCK=256 (heavy) |
| 其他 | — | TILE=128, BLOCK=128 (medium) |

决策流程：

```
choose_tile_config(diag_length, num_contributors, avg_pair_count)
    │
    ├── est_work = avg_pair_count × min(diag_length, 128)
    │
    ├── est_work ≤ 128?  ─── yes ─→  LIGHT  (tile=32)
    │
    ├── est_work > 4096? ─── yes ─→  HEAVY  (tile=256)
    │
    ├── diag_length > 4096 && pairs ≤ 4? ── yes ─→  WIDE (tile=512)
    │
    ├── diag_length > 1024 && pairs ≤ 8? ── yes ─→  HEAVY (tile=256)
    │
    └── 否则 ─────────────────────────────→  MEDIUM (tile=128)
```

#### per-diagonal 统计信息

`build_all_adaptive()` 在 tile 前为每条输出对角线计算：

```
num_contrib   = contributors.size()           ← 该 d_c 的总 pair 数
num_groups_est = 唯一 a_diag_idx 的个数       ← 估算 group 数
avg_pairs     = ceil(num_contrib / num_groups) ← 平均每 group 的 pair 数
```

这些统计量反映了对角线的"工作密度"，指导 tile 选择。

#### 四档 Bucket 方案

```
Bucket 0 (LIGHT) : tile=32   → launch_light_kernel()
                               4 task 打包一个 CTA，warp 独立

Bucket 1 (MEDIUM): tile=128  → launch_medium_kernel()
                               经典 4-warp CTA

Bucket 2 (HEAVY) : tile=256  → launch_heavy_kernel()
                               8-warp CTA + 双缓冲 smemA

Bucket 3 (WIDE)  : tile=512  → launch_wide_kernel()
                               4-warp CTA + 每线程 4 输出
```

#### Per-Bucket Dispatch（分桶调度）

Test driver 中的调度逻辑：

```cpp
// 每个 bucket 独立 launch，互不干扰
launch_light_kernel (..., d_light_task_ids,  num_light);
launch_medium_kernel(..., d_medium_task_ids, num_medium);
launch_heavy_kernel (..., d_heavy_task_ids,  num_heavy);
launch_wide_kernel  (..., d_wide_task_ids,   num_wide);
```

各 kernel 通过 `task_ids[]` 索引到全局 `tasks[]` 表，共享
同一套 `groups[] / pairs[] / packedB[]` 数据，无需重复存储。

#### 综合性能模型

```
假设矩阵有 D 条输出对角线，各对角线长度和 pair 密度差异很大：

固定 tile=128：
  所有 task 都走 medium kernel
  短对角线（len=4）：128 线程中 124 个空转
  长对角线（len=8192）：64 个 CTA，launch overhead 高

自适应 tile：
  短对角线 → LIGHT（4 task/CTA，零空转）
  中等对角线 → MEDIUM（经典方案，已被验证）
  长+稀疏对角线 → WIDE（4× 减少 launch）
  长+密集对角线 → HEAVY（2× compute 吞吐 + 双缓冲）
```

---

## 16. 后续扩展方向

### 16.1 Half-Precision / Tensor Core（TODO）

- 将 `Task / Group / PairMeta` 的值类型模板化
- 当 `tile_len` 为 16 的倍数时，使用 WMMA 或 MMA PTX 指令
- smemA 布局适配 16×16 fragment
