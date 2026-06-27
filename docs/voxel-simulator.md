# Voxel 仿真器技术文档

> 本文档根据论文 *Exploring the Efficiency of 3D-Stacked AI Chip Architecture for LLM Inference with Voxel*（Liu et al., arXiv:2604.26821v1, 2026）整理，汇总文中与 **Voxel** 仿真框架相关的全部技术信息。

---

## 1. 概述

**Voxel** 是一个面向 3D 堆叠 AI 芯片的、**编译器感知（compiler-aware）** 的 **端到端仿真框架**，用于探索 LLM 推理场景下 3D AI 芯片的效率。

### 1.1 定位与动机

3D 堆叠 AI 芯片具有独特的分布式架构特征：

- 每个 AI core 通过高密度 TSV 总线直连其上方堆叠的 DRAM bank
- 各 core 之间通过 NoC 互联
- DRAM 访问延迟非均匀，取决于物理距离与 NoC 拥塞

现有开源仿真工具无法同时满足以下需求（见论文 Table 1）：

| 能力 | 典型现有工具 | Voxel |
|------|-------------|-------|
| 参数化 Core/DRAM | 部分支持 | 支持 |
| 多样化 NoC 拓扑 | 部分支持 | 支持 |
| 分布式 DRAM 非均匀访问 | 不支持 | 支持 |
| 自定义计算范式 | 极少 | 支持 |
| 自定义 tile-to-core 映射 | 部分 | 支持 |
| 自定义 tensor-to-bank 映射 | 不支持 | 支持 |
| 整模型 LLM 快速仿真 | 部分 | 支持 |

Voxel 填补了这一空白，支持 **软硬件协同探索（hardware/software co-exploration）**。

### 1.2 核心能力摘要

- 通过编程接口让 ML 编译器自定义模型执行计划（分块、映射、计算范式）
- 基于执行计划构建 **执行事件图（execution graph）**，以事件驱动方式仿真全部硬件组件
- 动态学习并复用重复的 DRAM 访问模式，加速含数千算子的 LLM 工作流仿真
- 支持热/功率密度约束建模
- 经真实硅片 emulator 验证，误差 0.24%–6.8%
- 论文作者计划 **开源 Voxel** 及研究结果

---

## 2. 设计目标

Voxel 的四个主要设计目标（§3.1）：

| 目标 | 说明 |
|------|------|
| **Software awareness** | 纳入 tiling、mapping、compute paradigm 等软件因素对性能的影响，揭示 3D AI 芯片的硬件潜力 |
| **Accurate behaviors** | 细粒度建模每个 DRAM bank、AI core 及 NoC 的行为 |
| **Fast simulation** | 低仿真开销，支持数百 core、数百 bank 的大规模芯片快速评估 |
| **Reliable statistics** | 仿真结果可被真实 AI 芯片实验验证 |

---

## 3. 系统架构

### 3.1 总体流程（Figure 4）

```
DL Model + 3D AI Chip Specification
        ↓
   DL Compiler Program（compute paradigm, tile/data mapping, ...）
        ↓
   §3.3 软件接口 & 执行图生成
        ↓
   Execution Graph（compute / copy_data / sync 事件 + 依赖边）
        ↓
   §3.4 端到端仿真
   ├── AI Cores（Matrix Unit, Vector Unit, SRAM, ALU）
   ├── NoC Links
   └── DRAM Channels
        ↓
   Performance Results
```

Voxel 的三大阶段：

1. **软件接口**：ML 编译器决定模型如何在芯片上分区与执行
2. **执行图构建与遍历**：将决策转化为事件图，在对应硬件组件上仿真
3. **验证**：与基于真实 AI 芯片的 emulator 交叉验证（§3.5）

### 3.2 目标硬件模型

Voxel 仿真的 3D AI 芯片典型结构：

- **计算层**：AI core 网格，经 NoC 互联
- **存储层**：DRAM bank 网格堆叠在 core/NoC 之上，可多层堆叠以扩展容量
- **每个 AI core 内部**：
  - 本地 SRAM（scratchpad）
  - Vector Unit（通用向量运算）
  - Matrix Unit（如 systolic array，高吞吐 MatMul）
- **互连**：
  - 垂直 TSV 总线：core ↔ 其上方 DRAM bank
  - 水平 NoC：core ↔ core，以及 core ↔ 远端 DRAM

---

## 4. 软件接口（§3.3）

Voxel 通过编程接口让 ML 编译器指定优化后的执行计划，并据此构建执行图。

### 4.1 三个基础 API

#### `compute(op_tile, core_id)`

- **功能**：指定某个算子 tile 在哪个 core 上执行
- **`op_tile`**：任意张量算子（MatMul、elementwise、融合算子等）的分块；包含其操作的输入/输出张量片段
- **`core_id`**（可选）：目标 core；若输入不在 core 的 SRAM 中，将按需从 DRAM bank 访问
- **最小要求**：对每个算子 tile 调用 `compute()` 即可生成合法执行图

#### `copy_data(src_tensor, dest_tensor)`

- **功能**：定义张量片段的初始放置或运行时拷贝（预取）
- **`dest_tensor`**：目标数据类型（如 BF16）、shape、位置（DRAM bank 或 core SRAM）
- **`src_tensor`**：同 shape 的源张量片段（运行时拷贝），或 `NULL`（初始放置）

#### `sync()`

- **功能**：定义函数间的 barrier 同步

### 4.2 复合集体通信函数

Voxel 提供复合函数以简化 core 间集体通信，内部由多个 `copy_data()` 和 `compute()` 组成：

| 函数 | 行为 |
|------|------|
| `allReduce()` | 多 core 间搬运 partial result + 本地归约 |
| `reduceScatter()` | 类似 allReduce，但将各 partial result 拷贝到指定 core |
| `allGather()` | 通过 `copy_data()` 将各输入拼接到连续输出 tile |
| `broadcast` | 支持（论文引用 NCCL 风格接口 [43]） |
| 更多 | 可扩展 |

### 4.3 执行图生成

- **节点**：单个 core、DRAM bank 或 NoC link 上的执行事件（`compute()` / `copy_data()` / `sync()`）
- **有向边**：事件间依赖关系
- **兼容性**：仅需对编译器（如 XLA [5,6,9,35,69,71]）做简单修改，使其能指定分块算子的 shape

### 4.4 通过接口探索的软件空间

- Tile 分块策略
- Tile-to-core 映射
- Tensor-to-bank 映射
- 计算范式（SPMD、dataflow、compute-shift 等）

---

## 5. 端到端仿真引擎（§3.4）

### 5.1 规模与挑战

- 3D AI 芯片可能有 **数百至数千** 个硬件组件
- 单个 LLM 程序含 **数百个算子**，总计可达 **数百万** 个执行事件
- Voxel 通过 **合并相同事件与硬件组件** 降低仿真时间

### 5.2 执行图遍历

- 按时间顺序 chronologically 遍历事件图
- 每个事件在其数据依赖满足的最早时刻下发到对应组件：
  - `compute()` → AI core
  - `copy_data()` → NoC；若涉及 DRAM，则同时下发到对应 DRAM channel
- 全部事件处理完毕后输出端到端性能统计

### 5.3 AI Core 仿真

- **底层引擎**：[ScaleSim v3](https://arxiv.org/abs/2308.11030) [49]
- **建模对象**：Vector Unit、Matrix Unit（systolic array）、SRAM
- **按需数据访问**：若 `compute()` 的输入不在本地 SRAM，Voxel 按数据消费顺序创建并发 `copy_data()` 事件并路由到对应硬件
- **计算复用**：对 shape 相同的 tile 复用计算代价（LLM 中大量相同算子分块为相同 tile）

### 5.4 NoC 仿真

- **职责**：core-to-core 与 core-to-DRAM 通信
- **链路争用**：多个传输共用同一 link 时共享带宽
- **开销计算**：基于传输数据量、可用带宽、NoC hop 数
- **假设**：
  - Core-to-core：NoC 带宽 **严格低于** SRAM 读带宽
  - Core-to-DRAM：综合 NoC 与 DRAM 仿真报告整体性能

### 5.5 分布式 DRAM 仿真

#### 基本模型

- 每个 **DRAM channel** 含一个或多个共享同一 TSV 总线的 bank
- `copy_data()` 到达 channel 后分解为一系列 memory request：
  - 地址
  - 操作类型（read / write）
  - 到达时间戳
- 每个 request 访问一个 DRAM burst
- 每个 channel 维护 **按到达时间排序的优先级队列**（Figure 5 步骤 1）；同时间戳按 event index 排序
- **底层引擎**：[Ramulator 2.0](https://arxiv.org/abs/2308.11030) [39]

#### DRAM 访问 trace 合并加速（Figure 5）

显式仿真所有 request 代价极高。例：30B 参数 LLM 以 BF16 读权重约产生 **5 亿次 DRAM read**（burst 宽度 128 B），在现代 64 线程服务器上需 **数周**。

**加速方法：利用重复访问模式**

1. **Match Key 生成（步骤 2）**：对 channel 上待服务的 trace，计算每个 request 地址与其前一个 request 地址的 **bit-wise XOR**，得到 match key 列表
   - Match key 编码 row/column 转换信息
   - Channel 内 DRAM 时序仅取决于 bank/row/column 变化模式，而非具体地址

2. **Trace 匹配与缓存（步骤 3）**：match key 列表相同的 trace 具有相同 DRAM 时序，可复用缓存结果

3. **Mismatch 处理（步骤 4）**：当前 trace 与参考 trace 不匹配时：
   - 标记 divergent request 及其前后各 **N** 个 request（N = DRAM 内部队列大小，**默认 N = 32**）
   - 仅对 tagged request 块调用 Ramulator 仿真
   - 每块前 N 个 request 用于 warm-up DRAM 状态
   - 非 tagged request 复用缓存延迟

4. **缓存命中率**：LLM 40+ 重复层 × 64+ DRAM channel，每模式可重复 2,560 次；实验 hit rate **99.91%**（每结果复用约 1,100 次）

5. **Refresh 处理**：缓存无法捕获 refresh 影响；Voxel 跟踪当前 undergoing refresh 的地址范围，若 incoming request 命中 active refresh set，将其到达时间推迟到 refresh 结束

### 5.6 热/功率密度建模

3D 堆叠加剧热挑战；Voxel 在功率密度超限时 **延长执行事件时间**。

| 项目 | 说明 |
|------|------|
| 功率密度 | 基于并发事件的功耗与对应硬件组件面积 |
| 事件能耗 | 来自各组件级仿真器 |
| 功率模型来源 | AI core [49]、DRAM [65]、SRAM [65]、NoC [21] |
| 面积模型 | [11, 18, 42, 66] |
| 用户配置 | 可指定芯片最大功率密度 |
| 超限处理 | 按 `(超出功率 / 最大功率)` 比例降低 AI core 频率，更新执行时间 |
| **默认功率密度上限** | **0.7 W/mm²**（保守值，保证 < 85°C，避免温度 induced DRAM refresh penalty [17]） |
| 未来工作 | 更复杂的空间-时间热网络（论文称当前为简单合理模型以降低开销） |

---

## 6. 仿真器验证（§3.5）

### 6.1 验证背景

- 市场上 **无商用 3D AI 芯片**
- 作者基于现有 AI 芯片构建 **emulator**

### 6.2 Emulator 平台：Graphcore IPU Mk2 [28]

| 规格 | 数值 |
|------|------|
| 总 core 数 | 1,472 |
| 互联总带宽 | 7.8 TB/s |
| 分布式 SRAM 总带宽 | 62 TB/s |
| 总容量 | 896 MB |

- 1,472 core 全互联 + 分布式 per-core SRAM，可模拟分布式 stacked DRAM

### 6.3 Emulator 配置

- **960 core** 作为 "AI cores"
- **512 core** 模拟分布式 "DRAM banks"
- AI core 以 burst 粒度从指定 "DRAM bank" 取数

### 6.4 验证方法

1. 配置 Voxel 匹配 emulator 硬件参数与 overlap 约束
2. 比较 Voxel **Simulated Time** vs emulator **Emulated Time**
3. Emulator 平均比 Voxel 快 **12.7%**（因用 SRAM 模拟 DRAM，任意访问模式均可满带宽）
4. 进一步验证：提取各模型中重复的 transformer block，在 Ramulator 2.0 上跑完整 DRAM trace（**不使用** trace 合并），将 DRAM 延迟 replay 到 emulator trace 上
5. **Emulated Time with DRAM Latencies** 与 Voxel 在总时间与 DRAM 访问时间分解上高度吻合

### 6.5 验证结果（Figure 6）

| 模型 | 误差范围 |
|------|----------|
| Llama2-13B, Gemma2-27B, OPT-30B, Llama3-70B, DiT-XL | **0.24% – 6.8%** |

---

## 7. 可配置设计参数

### 7.1 论文主要探索的设计决策（Table 2）

| 类别 | 参数 | 说明 | 默认值 |
|------|------|------|--------|
| 软件 | Computation Paradigm | 每 core 计算与 core 间通信模式 | **Compute-shift** |
| 软件 | Tile-to-Core Mapping | 算子 tile 到 core 的映射 | **Dimension-ordered** |
| 软件 | Tensor-to-DRAM Bank Mapping | 张量 tile 到 bank 的映射 | **Software-aware** |
| 硬件 | NoC Topology | 片上互联拓扑 | **2D mesh** |
| 硬件 | DRAM Bandwidth | 总 DRAM 带宽 | **12 TB/s** |
| 硬件 | Number of Cores | core 总数 | **256** |
| 硬件 | Systolic Array Size | 每 core systolic array 宽度 | **32** |
| 硬件 | Core Group Size | core group 大小（组内可合并 DRAM 访问） | **8** |
| 硬件 | NoC Link Bandwidth | 每条 NoC link 带宽 | **32 B/cycle** |
| 硬件 | Per-Core SRAM Size | 每 core SRAM 容量 | **2048 KB (2 MB)** |

### 7.2 其他可配置参数（Table 3）

| 参数 | 默认值 | 参数 | 默认值 |
|------|--------|------|--------|
| DRAM layer count | 8 | DRAM Capacity | 192 GB |
| Number of DRAM banks per layer | 16 | Frequency (DRAM & AI core) | 1.6 GHz |
| DRAM timing (tCL-tRCD-tRP-tRAS) | 14-14-14-34 | Power Density Limit | 0.7 W/mm² |
| Batchsize | 32 | Sequence length | 2048 |
| DRAM interface size | 128 B | Precision | BF16 |
| DRAM burst length | Vary with BW | | |

### 7.3 默认芯片面积分解（Table 4，bottom die）

| 组件 | 面积 |
|------|------|
| Systolic arrays | 260 mm² |
| SRAMs | 433 mm² |
| TSVs | 18.4 mm² |
| Other | 91.2 mm² |
| **合计（约）** | **~802.6 mm²**（baseline 受 **850 mm² per-die** 面积限制 [2]） |

### 7.4 默认 Baseline 配置说明

- 面积约束：850 mm² per-die
- 优化目标：高 decode 性能（3D AI 芯片主要面向 memory-bound LLM decoding）
- 高 DRAM 带宽 12 TB/s，对 prefill 不在 Pareto 最优曲线上

---

## 8. 依赖与子模块集成

| 组件 | 工具/模型 | 用途 |
|------|-----------|------|
| AI Core | **ScaleSim v3** [49] | Vector/Matrix Unit、SRAM 周期级仿真 |
| DRAM | **Ramulator 2.0** [39] | 分布式 DRAM 请求级时序仿真 |
| NoC 功率/面积 | **ORION 2.0** [21] | NoC 功率与面积模型 |
| DRAM/SRAM 功率 | **ReGate 相关模型** [65] | 组件能耗 |
| 面积 | OpenRAM [11]、TSV [18]、NoC [42]、LLMCompass [66] 等 | 芯片面积估算 |

---

## 9. 使用 Voxel 进行设计空间探索

### 9.1 评估工作负载

论文使用 Voxel 仿真以下模型的 **prefill** 与 **decode** 阶段：

- Llama2-13B [56]
- Gemma2-27B [55]
- OPT-30B [67]
- Llama3-70B [36]
- DiT-XL [47]（100 iterations）

模型均可放入评估 3D AI 芯片的 DRAM。各计算范式的分区/执行计划按对应 SOTA 方法优化 [5, 35, 51, 52]。

### 9.2 探索方法

- 利用 Voxel 低仿真开销，快速评估 expansive hardware design space
- **Pareto 前沿搜索**（Figure 7）：
  - 将面积约束离散为多个几何阈值
  - 每阈值用 **coordinate descent** [59] 迭代最小化执行时间的几何平均
  - 识别 area-performance Pareto 最优前沿

### 9.3 Voxel 输出的性能分解

Voxel 可报告并分解以下开销（论文实验中广泛使用）：

- **NoC overhead**（core 间通信）
- **DRAM row-buffer conflict overhead**
- **DRAM access overhead**
- **Total / decode / prefill time**（cycles 或 cycle/token）
- **Component-level energy breakdown**（DRAM/TSV/NoC/SRAM/VU/SA 动静态能耗）
- **Utilization metrics**（SA spatial utilization、DRAM bandwidth utilization）

---

## 10. 论文中通过 Voxel 得到的关键发现（仿真结论）

以下结论均由 Voxel 实验得出，体现仿真器的分析能力：

### 10.1 计算范式（§4.1）

- 不同范式性能差异可达 **1.84×**
- **Compute-shift** 最优：最大化 tile 计算、NoC 通信、DRAM 访问的重叠
- SPMD 的 NoC overhead 可占执行时间 **49.08%**
- Dataflow 相对 SPMD prefill 平均快 **35.70%**；Compute-shift 相对 SPMD 平均快 **46.73%**，相对 dataflow 平均快 **17.74%**

### 10.2 NoC 与映射（§4.2）

- **Dimension-ordered mapping** 可将 NoC-bound 工作负载延迟降低最多 **57.48%**
- 配合 dimension-ordered mapping，mesh / torus / all-to-all 性能接近；**2D mesh + dimension-ordered mapping** 为 area 开销最低的近最优选择
- Decode 对 NoC 带宽不敏感；Prefill 在 link 带宽低于 **32 B/cycle** 时性能下降

### 10.3 Tensor-to-bank 映射（§4.3）

- 单纯扩 DRAM 带宽可能因 row-buffer conflict 导致利用率下降；16 TB/s 时 conflict overhead 可达 **43.35%**
- **Software-aware placement** 相对 uniform placement 平均降低 conflict overhead **80.68%**（最高 **80.7%**）
- Software-aware 策略在 ultra-high 带宽下仍保持 ≤ **14.8%** decode conflict overhead

### 10.4 Core scaling 与 Core Group（§4.4）

- 单纯增加 core 数会加剧 row-buffer conflict，降低 DRAM 带宽利用率
- **Core group + request tracker**（Figure 13）：组内同步 DRAM 请求，避免 execution desync 导致的 row thrashing
- Group size **8** 对 1024-core 芯片 decode 最多快 **57%**（平均 **42%**）
- SA 大于 32×32 收益递减（spatial underutilization）

### 10.5 SRAM scaling（§4.5）

- Memory-bound decode：更大 SRAM 提升预取窗口；**8 MB/core** 可饱和 DRAM 带宽
- Compute-bound prefill：SRAM 32× 仅带来 **35.7%** 提升；0.5 MB 时 FLOPS 利用率已平均 **67%**

### 10.6 能效（§4.6）

- 扩 DRAM 带宽改善 memory-bound decode 能效（缩短执行时间，TSV 静态能耗增幅小）
- 扩 core 数对 prefill 能效 benefit 有限；过多 core 反而降低整体能效

---

## 11. Voxel 支持的计算范式（Figure 8）

Voxel 通过软件接口部署以下代表性计算范式：

| 范式 | 特点 | Voxel 中的实现要点 |
|------|------|-------------------|
| **SPMD** | 算子分多独立 task，partial result 后归约 | 归约导致高 NoC 开销 |
| **Dataflow** | 少量 core 跑单算子，多算子 pipeline；microbatch 流经 `copy_data()` | 减少 DRAM 流量，重叠计算与通信 |
| **Compute-shift** | 全 chip 跑单算子；共享 tensor 在 core ring 上 circular shift | 几乎消除 NoC overhead，节省 SRAM 用于预取 |

---

## 12. Voxel 相关优化技术（论文提出，可在仿真中评估）

### 12.1 Software-aware Tensor-to-Bank Placement

- 从执行图检测并发访问：
  - 单算子或融合算子内所有 tensor 并发
  - 相邻算子间：前算子输出与后算子某输入并发
- 将并发访问的 tensor 映射到 disjoint bank，最小化 row-buffer conflict

### 12.2 Core Group + Request Tracker（Figure 13）

- 将物理相邻 core 分组
- 硬件 request tracker 选择性 stall 超前 core 的 DRAM 请求
- 规则：core 的 `(i+1)`th 访问不得早于组内所有 core 的 `i`th 访问被 dispatch
- 默认 group size：**8**

---

## 13. 与其他仿真器对比（Table 1 完整）

| 能力 | TimeLoop | ScaleSim | ONNXim | LLMCompass | AccelSim | Neurocube | NicePIM | NeuroSim | H2-LLM | **Voxel** |
|------|:--------:|:--------:|:------:|:----------:|:--------:|:---------:|:-------:|:--------:|:------:|:---------:|
| Parameterizable Cores/DRAM | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 是 |
| Diverse NoC Topologies | 是 | 是 | 是 | 是 | 是 | 否 | 否 | 否 | 否 | 是 |
| Non-Uniform Distributed DRAM | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 是 |
| Custom Tensor Tiling | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 是 |
| Custom Compute Paradigms | 否 | 否 | 否 | 否 | 是 | 否 | 否 | 否 | 是 | 是 |
| Custom Tile-to-Core Mapping | 是 | 是 | 否 | 是 | 否 | 是 | 是 | 是 | 是 | 是 |
| Custom Tensor-to-Bank Mapping | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 是 |
| Fast Simulation for Entire LLM | 是 | 是 | 是 | 是 | 否 | 否 | 是 | 是 | 是 | 是 |

---

## 14. 已知限制与未来工作

| 项目 | 说明 |
|------|------|
| 热模型 | 当前为简单功率密度模型；作者计划开发复杂 spatial-temporal thermal network |
| DRAM refresh | Trace 缓存不捕获 refresh；需额外 tracking 逻辑 |
| 3D 芯片可用性 | 无商用 3D AI 芯片，验证依赖 IPU emulator |
| 开源状态 | 论文声明将开源 Voxel 及研究结果（截至文档编写时以论文信息为准） |

---

## 15. 参考文献（Voxel 直接引用）

- [5] Cai et al., ISCA'23 — Inter-layer scheduling for tiled accelerators
- [9] Google XLA
- [14] He et al., OSDI'25 — WaferLLM / compute-shift
- [21] Kahng et al., DATE'09 — ORION 2.0 NoC model
- [28] Knowles, HCS'21 — Graphcore Colossus Mk2 IPU
- [35] Liu et al., SOSP'24 — T10 inter-core connected processor
- [39] Luo et al., Ramulator 2.0
- [43] NVIDIA NCCL
- [49] Raj et al., ISPASS'25 — ScaleSim v3
- [65] Xue & Huang, MICRO'25 — ReGate power models
- [66] Zhang et al., LLMCompass

---

## 16. 论文元信息

| 字段 | 内容 |
|------|------|
| 标题 | Exploring the Efficiency of 3D-Stacked AI Chip Architecture for LLM Inference with Voxel |
| 作者 | Yiqi Liu, Noelle Crawford, Michael Wang, Jilong Xue, Jian Huang (UIUC) |
| arXiv | 2604.26821v1 [cs.AR], 29 Apr 2026 |
| 本地文件 | `Exploring the efficiency of 3D-stacked AI chip architecture for LLM inference with voxel.pdf` |
