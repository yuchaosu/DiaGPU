# 论文实验完整规格（H100，2026-08-01 定稿）

评测协议、矩阵集、方法配置、基线公平性、消融与验证的唯一权威记录。数据文件在 beegfs `paper_bench/`，表格出口 `results/paper_tables/`，运行入口 `run_paper_set.sh`。

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA H100 NVL 95830 MiB（节点 c29，compute capability 9.0，独占运行） |
| 编译 | nvcc（/usr/local/cuda），`-O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr` |
| 精度 | fp32 值 + 计算（didx/gather 家族无 TC）；diaq 基线 fp32 |
| profiling | Nsight Compute（节点自带 ncu），指标见消融节 |
| 验证栈 | python3.11 + QuTiP 5.3.0 + SciPy 1.17.1（外部三角验证，进行中） |

## 矩阵集（`matrices_paper.txt`，25 个，其余一律不跑）

- **HamLib q16-20（22 个）**：heis/tfim/fermi/qmaxcut/maxcut/tsp 各 q∈{16,18,20} 档 + BH_20_7 + O2_16_5/16_7/20_5/20_8（O2 属域边界：D=1243-3359 违反 D=poly(q) 判据，主表照测、叙事归 boundary）。紧凑 DIA 格式（offsets 升序可空洞、无 padding、size_t starts），来源 fetchham_sparse_dia.py。
- **SuiteSparse 通用层（3 个）**：ecology2（n=1M，D=7，5 点 stencil，对称）、atmosmodd（n=1.27M，D=7，7 点，数值非对称）、Lin（n=256k，D=7，对称）。判据：n∈[65k,1.3M]、≥95% nnz 落于 ≤256 条对角线、填充 ≥0.2；tmt_sym 经筛落选（D=3843）。转换器 `sim/mtx_to_dia.cpp`（对称 MM 镜像展开）。
- 每矩阵附带：输入填充 f（真非零/存储）、f_C（C=H·H 对角线内真非零率，`cfill_bench` 设备端计数，`cfill_h100.csv`）。

## 方法配置（全部单方案，无自动派发）

- **SpMV = didx**：两遍 O(stored) plan（计数→前缀和→散射），元素级零跳过，逐元素 1B（D≤256）/2B 对角槽位，偏移表每 block 装入 shared（D×4B），列号计算式重建。计时 = 预热 10 + 50 次迭代均值；plan/upload/kernel 三列分离；`amort_applies` 列 = plan+upload 摊销所需 apply 次数。
- **SpMSpM = gather_flat（+对称时 gather_flat_sym/L4）**：C 结构由偏移代数 host 端预知；每 C 对角线一张 GPair 表（基址/移位/长度预算好），pair 元数据按 MAXP 分块进 shared（2026-08-01 修复：sym 版补上分块，宽带 np>MAXP 不再越界），自适应 ILP，全程零全局原子。100 次迭代。
- **e2e = Taylor-Horner 时间演化**（HamLib 专属，SS 不跑）：dt=1e-3、1000 步、K 动态取自文件名（SS 若跑默认 K=6）、每臂独立 20 步预热（消除冷启动不公）、复数以实/虚双平面表示；operator-build（U=Σc_kH^k）full+sym 双轨，超限输出带 U_diags/U_stored 的 `-1` 墙点行（stderr 注记区分 int32 索引墙/OOM）。

## 基线与公平性章程

| 基线 | 配置 | 公平性要点 |
|---|---|---|
| cuSPARSE SpMV | CSR 32I，SPMV_CSR_ALG2 | **真非零 CSR**（dia_to_csr 默认 drop_zeros，2026-08-01 修复；旧零填充数据全部作废重测）；转换/上传不计入 kernel 时间；真非零>2^31 时用 64I（本集未触发） |
| cuSPARSE SpGEMM | SPGEMM_DEFAULT，fp32，time-only | 同上真非零输入；它只产真非零 C 而我们物化稠密对角槽位——口径对我们不利，如实报 |
| Drawloom（**即 DASP**，用户确认） | 官方 AE 二进制，mtx 输入，取其自报 kernel 时间 | q≥22 其加载/运行崩溃（已验证实例 heis_22_5 core dump），表内标 crash-at-scale |
| diaq/HamSim | 融合复数 SpMV kernel + SpMSpM product kernel，**仅 fp32**，device-resident（其最优情形） | e2e 同管线换 kernel，setup 单列 |
| HM（Haque 原子散射） | 原kernel | C nnz>2^31 属其 int32 布局架构上限，规范 N/A 注记 |

通用规则：无 silent SKIP（每格 = 实测或带因 N/A）；GPU 独占串行；长任务 stdbuf 行缓冲防截断丢数；合并优先级 patch > resume > 原始、同文件取末次；机制主张须有 registered prediction（含被证伪的：didx_t 表大小✗、w4 MLP✗、w4p L2 逐出✗、sym 在 heis 的缓存复用✗——全部入机制节）。

## 消融（SpMV，全部相对 didx 的单变量拆解）

| 级 | 对照 | 隔离贡献 | 状态 |
|---|---|---|---|
| A1 | didx w/o 零跳过（stream） | 跳零 ≈ 1/f：heis 2.1×、O2_16 5× | 计时✅ + ncu 字节账（进行中） |
| A2 | didx w/o 索引压缩（czskip，4B 列号） | 槽位编码的字节价值：+13-73% | 同上 |
| A3 | nv1×2 vs nv2 融合 | 复数双平面共享矩阵读取 | sweep 已含 |
| sym | didx_sym（上半存储双位置读） | 对称红利：tfim 类 1.5×，heis 类无效（大偏移 L2 复用失败） | 计时✅ + ncu（进行中） |
| 对手解剖 | Drawloom/DASP 的 ncu 面板 | 与我们并排的 DRAM%/字节/stall 结构 | 进行中 |

ncu 指标集：DRAM 吞吐%、dram bytes read/write、L1/L2 命中、SM%、long-scoreboard stall、occupancy、时长；另有 shared/atomics 面板（`ncu_shared_h100.csv`：HM 原子 60-197 万 vs 我们 0；shared HM=0 vs gather≈71MB，sym 减半）。代表集：heis_20_5、tfim_18_5、qmaxcut_18_4、ecology2。

## 验证体系

1. **kernel 级**：结构等价变体逐位一致（didx_t/didx_g vs didx BITEXACT）；求和序变化的变体对 CPU double 参照取 ‖·‖∞ 归一误差 <1e-5（逐行相对误差在相消行会假阳性，弃用）。
2. **跨实现**：SpMSpM 各基线 vs ours_flat 的 maxdiff 校验列；e2e diaq（独立实现）vs ours 千步 relerr 7.7-9.2e-7（fp32 累积量级），在 24.8-28.9 亿 nnz 上交叉验证 didx 正确性。
3. **外部三角（进行中）**：6 矩阵子集（每家族 q16 代表 + heis_20），QuTiP sesolve（rtol=1e-10）+ scipy expm_multiply 双独立参照互检，报保真度与 L2 relerr；需 e2e_driver 加 --dump-state。
4. **对称性**：ours_sym 仅在数值对称矩阵有效（校验列 ≤1e-2），非对称自动 N/A（atmosmodd 等）。

## 产出文件地图

| 文件 | 内容 |
|---|---|
| `prep_h100_{268657,c29resume,c29patch}.csv` | SpMV/SpMSpM prep 分解 + 计时（PREPCSV 长表） |
| `spmspm_cusparse_fix_h100.csv` | SpMSpM 全变体（诚实 SpGEMM） |
| `cusparse_fix_h100.csv` | 诚实 cuSPARSE SpMV + 每矩阵 f（CSRNNZ 行） |
| `cfill_h100.csv` | f_C 全量 |
| `e2e_diaq_h100_{268662,c29resume,c29patch}.csv` | e2e 演化/opbuild |
| `spmv_variants_h100.csv` | didx_g/w/w4/w4p/s/t/sym/csr_warp 探索数据（52 行，不进主表） |
| `ncu_shared_h100.csv`、`ncu_abl/` | shared/atomics 面板、消融 ncu |
| `results/paper_tables/{spmv,spmspm,e2e}_main.csv` | 主表（make_paper_tables.py 再生） |

## 主结果速览与边界

域内（q16-20 局域 + SS）：SpMV didx 全胜（几何平均 2.61×，全 63 矩阵口径）；SpMSpM flat 域内全胜（27.5×，sym 33.7×）；e2e 2.9-9.2×。边界如实报：O2_20 SpMV 0.42×（D 千级域外）；SpMSpM 大 q 反转（q≥22，C_stored≳3B 且 f_C≲0.33，输出物化下界超对手全程——Pauli 基续作的动机）；maxcut 上与 diaq 1-2% 噪声级平手。
