# H100 (c29) prep / e2e / shared-mem ncu — 数据说明

2026-07-30/31 在 c29 (H100 NVL 95GB, sm_90a, CUDA /usr/local/cuda) 完成的三套数据。原始任务 268657 (prep) 与 268662 (e2e) 被 root 于 2026-07-30 09:12 清场杀死，其后经交互续跑 (`resume_c29.sh`)、int32 补丁补跑 (`patch_c29.sh`) 与 SLURM 安全网 (`finish_c29.slurm`, job 268940) 三段接力补全。所有矩阵均达到终态：实测值、带原因的 N/A、或真实 OOM 墙。

## 文件与合并规则

| 文件 (beegfs paper_bench/) | 内容 |
|---|---|
| `prep_h100_268657.csv` | 原始任务已完成的 24 矩阵（不可变） |
| `prep_h100_c29resume.csv` | 续跑 21 矩阵 |
| `prep_h100_c29patch.csv` | int32 补丁后补跑的 15 矩阵 |
| `e2e_diaq_h100_268662.csv` | 原始 7 矩阵（BeH_12_6 行有效但以 resume 为准） |
| `e2e_diaq_h100_c29resume.csv` | 续跑 44 矩阵 |
| `e2e_diaq_h100_c29patch.csv` | 补丁补跑 11 矩阵（含 O2_20_8 stdbuf 重跑） |
| `ncu_shared_h100.csv` | shared-mem profile，364 行 = 6 kernel × 5 代表矩阵 × ~12 metric |

合并规则：同一矩阵出现在多个文件时取 **patch > resume > 原始**（patch 是完整重测；resume 中 heis_18_5、qmaxcut_18/20_4、tfim_18_4/5、O2_16_* 只有部分行）。`PREPCSV` 每 (file,variant) 取**最后一行**。SuiteSparse 的 `ours_sym` 行一律无效（非对称矩阵，upper-half kernel 不适用）。

## N/A 格子清单（全部有因）

| 格子 | 原因 |
|---|---|
| prep `hm_atomic`: BH_20_7, O2_16_5/7, heis_22/24_*, tfim_22/24_5 | C nnz > 2^31，HM 基线 `total_nz`/`diag_starts` 为 int32，架构性放不下（stderr 有 `hm_atomic skipped` 注记与精确 nnz） |
| prep `cusparse_csr`: BH_24_8, O2_20_5/8 | H nnz > 2^31，cuSPARSE 32I 索引拒绝 |
| prep spmspm 全套: BH_24_8, O2_20_5/8 | ours_flat 的 C 超过 95GB 显存，真实 OOM 墙（`prep_driver.cu:238 out of memory`） |
| prep `ours_sym` kernel 时间: O2_16_5/7（prep-only 行, kernel_ms=-1） | gather_flat_sym 已知宽带崩溃；plan/upload 时间有效 |
| e2e OPBUILD `-1,-1,-1` 行: heis_18+、tfim_18+、qmaxcut_18+、O2、BH 大矩阵 | opbuild 墙点行：stderr 注记区分 `int32 index limit` 与 `OOM`，U_diags/U_stored 列为精确值，本身是 fill-in 论据 |
| e2e EV 行 cusp 列 -1: BH_24_8, O2_20_5/8 | cuSPARSE 臂自动跳过（H nnz > 2^31） |
| ~~prep `csr_zskip`: BH_24_8, O2_20_5/8 无效~~ **已撤回**：行有效 | csr_zskip 构建时零跳过，真非零数 ≤ 2^31（护栏未触发即为证明），row_ptr 未溢出。护栏保留（真非零超限时才 N/A）。注意 2.48-2.89B 是**存储** nnz（含对角线内部零），真非零远小于它 |
| prep `cusparse_csr_64i`: BH_24_8, O2_20_5/8（2026-07-31 补测） | 32I 不可能时的诚实 cuSPARSE 基线：kernel 16.6/19.0/19.1 ms vs didx 2.0/2.26/2.25 ms（**didx 8.2-8.5×**），且 64I 转换 prep 高达 16 秒 |
| sweep `drawloom`: q≥22 全缺（heis/tfim_22/24, BH_24_8, O2_20_*，共 9 个） | 原 sweep 静默丢弃；2026-08-01 复测 heis_22_5（mtx 转换成功，n=4.2M/nnz=176M）**Drawloom 本体 core dump**——其加载/运行路径在该规模崩溃，9 格标 "crash at scale"（已验证实例：heis_22_5），不是未测 |

## 驱动补丁（本分支工作区）

`prep_driver.cu`：HM 块前未计时 size_t 预判（>2^31 跳过）；`total_nz*4` 加 size_t 强转（修 int 乘法溢出）；cuSPARSE 块按 H.nnz 自动跳过。`e2e_driver.cu`：cuSPARSE 臂同款自动跳过；full/sym 两个 opbuild 的 `Unnz>INT32_MAX` 折入墙点 -1 行路径；k 循环中间幂 `Cnnz>INT32_MAX` 按 truncate 语义 break。对已测尺寸行为零改变（交叉验证：BH_20_7 ours_sym 19.84ms vs 268133 的 19.59ms）。

## SpMSpM 诚实基线与大 q 反转（2026-08-01）

`spmspm_cusparse_fix_h100.csv`（真非零 CSR 喂 cuSPARSE，iters=100，60 矩阵全覆盖）与 `cfill_h100.csv`（f_C = C=H·H 对角线内真非零率，cfill_bench 设备端计数）。结论：ours_flat 49 胜 8 负，几何平均 27.5×（sym 33.7×）；**败区二维刻画：C_stored ≳ 3B 且 f_C ≲ 0.33（heis/tfim q≥22-24），或 f_C ≲ 0.02 的极稀 C（flowmeter0、B2_14 边缘）**。大倍数区含 cuSPARSE ~2ms 小问题开销地板的贡献，报告时按家族×q 画衰减曲线，不要只报几何平均。物理机制：稠密对角 C 的物化字节 = stored×4B，heis_24 光输出写就 62GB>19ms，超过 cuSPARSE 全程 5.95ms——输入侧跳零救不了，出路是输出表示（真非零对角/Pauli 基）。旧"全胜"是双重假象：零填充 CSR 使 cuSPARSE SpGEMM 虚慢 2-13×（heis_18 达 13×），且旧 sweep 只有 41 矩阵有 cuSPARSE 行——败区 19 个矩阵因驱动在 HM/sym/diaq 处崩溃而静默缺失。本轮修复：gather_flat_sym 的 MAXP 分块（宽带 sym 崩溃根因，np>MAXP 越界写 shared，已修，解锁宽带 L4）、spmspm_driver 的 HM int32 预判/diaq u32 守卫/OOM 优雅降级。

## shared-mem ncu 摘要（论文用）

HM 基线 kernel shared 字节 = **0**（纯全局 atomicAdd）；`gather_meta`/`gather_flat` ≈ **7100 万字节** shared 读取、每 block 8KB 静态、bank conflict 率 ~0.08%；`gather_flat_sym` 恰好减半（3644 万，上半三角自洽）；`cuda_spmv_dia` 与 `spmv_csr_scalar` 均为 0（SpMV 不依赖 shared 的阴性对照）。论证"shared 有用"的主证据是 L0→L1 ablation（atomics 34M→0）+ 流量替代，此表为支撑数据。

## 复现

```bash
# 单矩阵，与全表相同命令行（ITERS=50 / STEPS=1000, sm_90a）
./prep_driver <dia.txt> 50
./e2e_driver_sm_90a <dia.txt> 1000 --sym     # HamLib；SuiteSparse 去掉 --sym
bash sim/suite/hpc/run_ncu_shared.sh          # MATS=... 可选子集
# 断点接力（按数据文件终态判定，幂等）
sbatch sim/paper_bench/finish_c29.slurm
```

注意：长跑单矩阵时用 `stdbuf -oL` 防 timeout 丢缓冲 stdout（O2_20_8 教训）。
