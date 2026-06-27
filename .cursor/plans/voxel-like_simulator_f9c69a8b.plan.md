---
name: voxel-like simulator
overview: 从零实现一个与论文 Voxel 能力一致的高性能 3D 堆叠 AI 芯片仿真框架：Python + NumPy/numba 框架层，集成真实的 ScaleSim v3（AI core）、Ramulator 2.0（分布式 DRAM）与 BookSim 2.0（高精度 NoC，分层 + 模式缓存），执行图引擎、trace 合并加速、热/能耗（DSENT）建模与 Pareto 搜索。分阶段交付，先打通核心闭环。
todos:
  - id: p0
    content: "Phase 0: 脚手架 pyproject/requirements + chip/config.py 表达 Table 2/3/4 默认值"
    status: completed
  - id: p1
    content: "Phase 1: 实现 api (compute/copy_data/sync + collectives) 与 graph/builder 执行图"
    status: completed
  - id: p2
    content: "Phase 2: chip/topology (mesh/torus/all-to-all) + mapping 策略 + noc_sim 分层模型 (Tier-A 解析)"
    status: completed
  - id: p3
    content: "Phase 3: 集成 ScaleSim v3 + Ramulator 2.0 + BookSim 2.0 后端 (构建脚本 + 解析 + 缓存 + 降级桩)"
    status: completed
  - id: p4
    content: "Phase 4: sim/engine 事件驱动遍历 + dram_sim per-channel 优先队列 + 端到端统计"
    status: completed
  - id: p5
    content: "Phase 5: DRAM trace_coalesce (XOR match-key, N=32, numba) + refresh + NoC 流量模式缓存"
    status: completed
  - id: p6
    content: "Phase 6: thermal 节流 + energy 分解 (DSENT NoC + ReGate) + explore/pareto 坐标下降搜索"
    status: completed
  - id: p7
    content: "Phase 7: models/llm 五个模型 prefill/decode 图 + paradigms 三种计算范式"
    status: completed
  - id: p8
    content: "Phase 8: 单元/端到端/合并正确性测试 + 自洽性趋势复现 + README"
    status: completed
isProject: false
---

## 目标

实现一个名为 `voxelsim` 的 Python 包，复刻 [docs/voxel-simulator.md](docs/voxel-simulator.md) 描述的全部能力：软件接口、执行图、端到端事件驱动仿真、分布式 DRAM、trace 合并加速、热/能耗、Pareto 搜索；通过子进程集成真实组件级仿真器以对齐论文精度。

## 技术决策（已确认）

- 框架层：Python 3.10+ + NumPy，热点（trace 合并、XOR match key）用 `numba` JIT。
- AI core：集成真实 **ScaleSim v3**（`pip install scalesim`，GEMM 模式），按唯一 tile shape 缓存计算周期。
- DRAM：集成真实 **Ramulator 2.0**（C++ 构建，memory-trace 前端，subprocess + YAML 配置）。
- NoC 时序（精度升级）：**分层混合模型**，不再单纯依赖解析公式。
  - Tier-A 解析模型（默认）：hop 数 / 带宽共享 / 争用，用于 Pareto/DSE 海量快速评估。
  - Tier-B **BookSim 2.0**（Stanford，flit 级 cycle-accurate，已对 RTL 验证）：trace 驱动，对接执行图产出的「源/目的/数据量/注入时刻」，覆盖 mesh / torus（all-to-all 用 flattened-butterfly 近似或 1-hop 解析）。
  - 性能手段：把 DRAM 的 trace 合并思路复用到 NoC——按**结构化流量模式**（归一化 src/dst 偏移、volume、相对注入时刻、拓扑、路由）做 key 缓存 BookSim 结果，使 BookSim 调用数受限于「唯一模式数」而非事件数；LLM 层间通信高度重复，缓存命中率高。
  - BookSim 以 `libbooksim` 进程内或 subprocess 集成，避免每事件开销。
- NoC 功率/面积：用 **DSENT**（query-based，专为 BookSim/Garnet 配套）替代论文老旧的 ORION 2.0。
- 验证：无 3D/IPU 硬件，采用自洽性测试 + 与组件仿真器交叉对拖；额外用 Tier-B BookSim 校准 Tier-A 解析模型误差。

### NoC 仿真器选型对比（调研结论）

- **BookSim 2.0（选定 Tier-B）**：cycle-accurate，topology/路由最丰富，支持 trace 驱动，RTL 验证过，DSENT 功率配套；inline 较慢 → 用模式缓存化解。
- **Garnet（gem5）**：低/中负载更快但微架构保真度低，需构建整个 gem5，过重，不选。
- **Noxim**：SystemC，2D mesh 为主，依赖 SystemC，不选（拓扑覆盖不足）。
- **DSENT**：仅功率/面积（非时序），与 BookSim 配套，用于能耗阶段。

## 架构总览

```mermaid
flowchart TD
    Model["LLM 工作负载构建<br/>models/llm.py"] --> Plan["计算范式生成<br/>SPMD/Dataflow/Compute-shift"]
    Plan --> API["软件接口<br/>compute/copy_data/sync + collectives"]
    API --> Graph["执行图构建<br/>graph/builder.py"]
    Graph --> Engine["事件驱动引擎<br/>sim/engine.py"]
    Cfg["芯片配置 Table 2/3/4<br/>chip/config.py"] --> Engine
    Map["映射策略<br/>tile-to-core / tensor-to-bank"] --> Engine
    Engine --> CoreSim["AI core: ScaleSim v3<br/>+ tile 计算缓存"]
    Engine --> NoCSim["NoC 分层模型<br/>Tier-A 解析 / Tier-B BookSim 2.0"]
    NoCSim --> NoCCache["流量模式缓存<br/>结构化 pattern key"]
    Engine --> DramSim["分布式 DRAM: Ramulator 2.0<br/>per-channel 优先队列 + trace 合并"]
    DramSim --> Coalesce["XOR match-key 缓存<br/>numba, N=32"]
    Engine --> Thermal["热/功率密度节流"]
    Engine --> Energy["能耗分解 (DSENT NoC + ReGate)"]
    Engine --> Stats["端到端性能/能耗统计"]
    Stats --> Pareto["explore/pareto.py<br/>坐标下降面积约束搜索"]
```



## 模块布局

- `voxelsim/api/`：`ops.py`（Tensor/TensorPart/OpTile）、`interface.py`（`compute`/`copy_data`/`sync`）、`collectives.py`（allReduce/reduceScatter/allGather/broadcast）、`program.py`（录制调用）
- `voxelsim/graph/`：`events.py`（Compute/CopyData/Sync 事件节点）、`builder.py`（依赖边构建）
- `voxelsim/chip/`：`config.py`（Table 2/3/4 默认值）、`topology.py`（mesh/torus/all-to-all + 路由/hop 计算）、`mapping.py`（sequential、dimension-ordered；uniform、interleave、software-aware）
- `voxelsim/sim/`：`engine.py`（按时间遍历）、`core_sim.py`、`noc_sim.py`（分层模型 + 模式缓存调度）、`dram_sim.py`（per-channel 优先队列）、`trace_coalesce.py`、`refresh.py`、`thermal.py`、`energy.py`
- `voxelsim/backends/`：`scalesim_backend.py`、`ramulator_backend.py`、`booksim_backend.py`（NoC 高精度后端）、`dsent_backend.py`（NoC 功率/面积）
- `voxelsim/models/`：`llm.py`（transformer prefill/decode 算子图）、`paradigms.py`
- `voxelsim/explore/pareto.py`、`voxelsim/cli.py`
- `third_party/`（克隆并构建 ramulator2、booksim2、dsent）、`configs/`（默认配置）、`tests/`

## 分阶段实现

### Phase 0 — 脚手架与配置

- `pyproject.toml`/`requirements.txt`（numpy、numba、scalesim、pyyaml、pandas、pytest）
- `chip/config.py`：用 dataclass 表达 Table 2/3/4 全部参数与默认值（compute-shift、dim-ordered、software-aware、2D mesh、12TB/s、256 cores、SA 32、group 8、32B/cycle、2MB SRAM、DRAM 8 层×16 bank、1.6GHz、tCL-tRCD-tRP-tRAS=14-14-14-34、0.7W/mm²、batch 32、seq 2048、128B 接口、BF16）

### Phase 1 — 软件接口 + 执行图（§4）

- `api/`：实现三个基础函数与复合集体通信；`Program` 记录事件序列
- `graph/builder.py`：节点=（core/bank/link 上的事件），有向边=数据依赖；保证仅靠 `compute()` 即可生成合法图

### Phase 2 — 硬件映射与 NoC 解析模型（§4.2）

- `chip/topology.py`：三种拓扑的相邻关系、路由、hop 数（torus wraparound、all-to-all=1 hop）；与 BookSim 拓扑参数（k-ary n-mesh/cube、flattened-butterfly）对齐
- `chip/mapping.py`：tile-to-core（sequential / dimension-ordered=MeshGEMM）、tensor-to-bank（uniform / interleave size-based / software-aware 并发检测）
- `sim/noc_sim.py`：定义统一 NoC 接口 `estimate_transfers(transfers) -> latencies`，实现 Tier-A 解析模型（传输量 / 可用带宽 / hop / 链路共享争用；core-to-core 带宽 < SRAM 读带宽）；预留 Tier-B 切换钩子（Phase 3 接 BookSim）

### Phase 3 — 组件级后端集成（§5.3 / §5.5）

- `backends/scalesim_backend.py`：将 op tile 映射为 GEMM (m,n,k) topology，调用 ScaleSim v3，解析 `COMPUTE_REPORT.csv` 得到 cycles；按 tile shape 做 LRU 缓存
- `backends/ramulator_backend.py`：脚本化克隆+cmake 构建 `third_party/ramulator2`；生成 per-channel memory trace + YAML（注入 DRAM timing/组织），subprocess 运行并解析 latency 统计
- `backends/booksim_backend.py`（NoC 精度核心）：克隆+构建 `third_party/booksim2`；由 `chip/topology.py` 生成 BookSim 配置（topology/k/n/路由/VC/buffer/channel latency）+ open-loop 注入 trace（每包：注入时刻、src、dst、size→flits）；subprocess 或 libbooksim 运行，解析 per-packet/平均延迟；接入 `sim/noc_sim.py` 作为 Tier-B
- `backends/dsent_backend.py`：构建 `third_party/dsent`，用 QueryString 取 router/link 的 Energy/Area，供能耗阶段使用
- 提供降级桩：任一组件不可用时回退到解析模型/经验公式，保证闭环可跑

### Phase 4 — 事件驱动引擎（§5.2 / §5.4 / §5.5）

- `sim/engine.py`：按时间顺序遍历，事件在依赖满足的最早时刻下发；`compute`→core，`copy_data`→NoC(+DRAM channel)
- `sim/dram_sim.py`：每 channel 按到达时间的优先级队列（同时间按 event index），request 拆成 burst
- 输出端到端 total/decode/prefill 时间与各项 overhead 分解

### Phase 5 — Trace 合并加速（§5.5）

- `sim/trace_coalesce.py`（numba）：对每 channel trace 计算相邻地址 XOR 得 match key；命中复用缓存延迟；mismatch 标记 divergent ± N（N=32）窗口，仅对 tagged 块跑 Ramulator，前 N 个 warm-up
- `sim/refresh.py`：跟踪 active refresh 地址区间，命中则推迟到 refresh 结束
- DRAM 目标：复现论文 ~99.91% 命中率量级的加速
- NoC 流量模式缓存：在 `sim/noc_sim.py` 中对一组并发传输生成**结构化 pattern key**（归一化 src/dst 偏移、volume 量化、相对注入时刻、拓扑、路由），命中则复用 BookSim 延迟结果，未命中才实跑 BookSim；使 Tier-B 在重复 LLM 层上的实跑次数受限于唯一模式数
- 一致性校验：抽样对「缓存命中复用」与「实跑 BookSim」结果对拖，确保模式 key 不引入显著误差

### Phase 6 — 热 / 能耗 / 设计空间探索（§5.6 / §4.6 / §9.2）

- `sim/thermal.py`：按组件面积+并发功率算功率密度，超 0.7W/mm² 时按比例降频并延长事件
- `sim/energy.py`：AI core/DRAM/SRAM/NoC 动静态能耗分解（NoC 用 **DSENT** router/link 模型 + ReGate 风格 core/DRAM 模型，面积模型 OpenRAM 等）
- `explore/pareto.py`：面积阈值离散 + 坐标下降最小化执行时间几何均值

### Phase 7 — LLM 工作负载 + 计算范式（§9.1 / §11）

- `models/llm.py`：参数化 transformer，构建 Llama2-13B / Gemma2-27B / OPT-30B / Llama3-70B / DiT-XL 的 prefill 与 decode 算子图
- `models/paradigms.py`：SPMD（独立 task+reduce）、Dataflow（microbatch pipeline + copy_data）、Compute-shift（ring circular shift）三种执行计划生成器

### Phase 8 — 验证与文档

- `tests/`：API/图/映射/拓扑单元测试；端到端 smoke（小模型小芯片）；trace 合并正确性（合并 vs 全量一致）
- NoC 精度校准：在若干典型流量（ring shift、allreduce、uniform/transpose）上对 Tier-A 解析 vs Tier-B BookSim 对拖，量化解析模型误差并记录适用边界
- 自洽性检查：复现论文趋势（如 compute-shift 优于 SPMD、software-aware 降低 row-conflict、core group 提升、mesh+dim-ordered 近最优）
- `README.md` 使用说明 + CLI 示例；与 `docs/voxel-simulator.md` 章节对应表

## 主要风险

- ScaleSim/Ramulator/BookSim/DSENT 的构建与网络拉取（已确认环境可装 C++ 依赖；提供降级桩兜底）
- Python 事件循环在百万级事件下的性能 → 依赖 tile 计算复用 + DRAM trace 合并 + NoC 流量模式缓存 + numba 热点加速
- **BookSim cycle-accurate 开销**：是 NoC 精度的主要性能风险 → 靠流量模式缓存把实跑次数压到唯一模式数；Tier-A 仍作为 DSE 默认，Tier-B 仅在精度敏感/校准时启用
- 将 transformer 算子准确映射到 ScaleSim GEMM、Ramulator 地址布局、BookSim 注入 trace，需在 Phase 3/7 对齐
- all-to-all 在 BookSim 无原生拓扑 → 用 flattened-butterfly 近似或保留 1-hop 解析，需在 Phase 3 确认口径
- 无真实硬件，精度只能做自洽与组件对拖，不能完全复现论文 6.8% 误差结论

## 建议默认（除非你另行指定）

- 包名 `voxelsim`；Python 3.10+；测试用 pytest
- 第三方仿真器置于 `third_party/`（ramulator2 / booksim2 / dsent）并提供 `scripts/setup_backends.sh` 一键构建
- 默认配置文件放 `configs/default.yaml`，对应 Table 2/3
- NoC 默认 **Tier-A 解析**（兼顾性能），通过配置 `noc.backend: booksim` 切到 **Tier-B 高精度**；两者共享同一 `estimate_transfers` 接口与流量模式缓存

