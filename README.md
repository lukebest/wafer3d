# wafer3d / voxelsim

[English](README.en.md)

面向 **3D 堆叠 AI 芯片** 的编译器感知端到端仿真器，设计参考 Voxel 论文（详见 [docs/voxel-simulator.md](docs/voxel-simulator.md)）。

## 功能特性

- **软件接口**：`compute()`、`copy_data()`、`sync()` 及集合通信（`allReduce`、`allGather` 等）
- **执行图**自动生成与依赖追踪
- **双层 NoC 模型**：Tier-A 解析模型（默认）+ Tier-B BookSim 2.0 周期精确仿真，含流量模式缓存；按 SYNC region 批量估计链路竞争
- **AI 核心**：[SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) v3（GEMM），不可用时降级为解析模型
- **DRAM**：Ramulator 2.0 逐通道仿真 + XOR 匹配 trace 合并（numba 加速）；支持 tensor-to-bank 映射策略与 core group 请求合并
- **热节流**（功率密度约束）、**DSENT** NoC 能耗建模、**Pareto** 设计空间探索
- **LLM 工作负载**：Llama2-13B、Gemma2-27B、OPT-30B、Llama3-70B、DiT-XL
- **计算范式**：SPMD、Dataflow、Compute-shift（含通信/计算重叠建模）

## 论文实验复现

本仓库提供 Voxel 论文（arXiv:2604.26821）实验流程梳理、批量复现脚本与对照报告。

| 文档 | 说明 |
|------|------|
| [docs/voxel-experiments-reproduction.md](docs/voxel-experiments-reproduction.md) | 实验设计、参数扫掠与复现步骤 |
| [docs/voxel-reproduction-report.md](docs/voxel-reproduction-report.md) | 论文 vs simulator 趋势核对（PASS/PARTIAL/FAIL） |
| [docs/reproduction-results.json](docs/reproduction-results.json) | 原始 JSON 实验数据 |

### 快速复现

```bash
# 快速扫掠（约 10 分钟，16 cores / batch=4 / seq=128）
python scripts/run_reproduction.py --quick

# 完整 Table 2/3 规模（耗时数小时）
python scripts/run_reproduction.py --full
```

当前 quick 模式复现结果：**10/10 PASS**（计算范式、NoC 带宽、DRAM 带宽、SRAM、能效等核心趋势与论文一致）。

### 引擎建模要点

| 模块 | 说明 |
|------|------|
| NoC 批量竞争 | SYNC region 内批量估计传输，SPMD all-to-all 规约产生最高 NoC 占比 |
| tensor-to-bank | `MappingPlanner` 接入 `dram_sim` burst 寻址；software-aware 按并发张量分配不相交 bank 块 |
| DRAM 权重读 | `COMPUTE` 的 DRAM 输入触发隐式读请求（decode 阶段） |
| 三范式等价计算 | 8 路并行 `(m/8)` matmul，以通信模式区分范式性能 |
| 通信/计算重叠 | 关键路径 = 并行 compute + noc×serial_factor（SPMD 1.0 / dataflow 0.5 / shift 0.3） |
| core group | 组内同 row DRAM 请求合并，降低 row-buffer 冲突 |

已知局限：绝对 cycle 与论文 Figure 不可直接对比；IPU emulator（Figure 6）未集成；重叠模型为范式级近似。

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| pip / venv | 推荐使用 |

可选后端（通过 `scripts/setup_backends.sh` 构建）：

| 工具 | 用途 |
|------|------|
| `pip`、Python venv | SCALE-Sim（editable 安装） |
| `cmake`、`g++` | Ramulator 2.0 |
| `flex`、`bison`、`g++` | BookSim 2.0 |
| `g++`、`make` | DSENT |

若某后端未构建或运行失败，仿真器会自动降级到对应的解析模型，不影响基本功能。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 可选：构建 third-party 后端

在 `third_party/` 下构建 SCALE-Sim、Ramulator 2.0、BookSim 2.0 和 DSENT：

```bash
bash scripts/setup_backends.sh
```

或仅安装 SCALE-Sim：

```bash
pip install -e third_party/SCALE-Sim
```

构建成功后的路径：

| 后端 | 路径 |
|------|------|
| SCALE-Sim | `third_party/SCALE-Sim/`（editable Python 包） |
| Ramulator 2.0 | `third_party/ramulator2/build/ramulator2` |
| BookSim 2.0 | `third_party/booksim2/src/booksim` |
| DSENT | `third_party/dsent_standalone/dsent` |

SCALE-Sim 需应用 NumPy 兼容补丁（`scripts/patches/scalesim-numpy-max.patch`），`setup_backends.sh` 会自动处理。

Ramulator / BookSim / DSENT 路径在 [configs/default.yaml](configs/default.yaml) 中配置。DSENT 使用 [Desent_modification](https://github.com/gyb1325/Desent_modification) 分支，Linux 下需 `LDFLAGS="-no-pie"`。BookSim 需先安装 `flex` 和 `bison`。

## 快速开始

```bash
# 端到端仿真（解析 NoC + 解析 DRAM 降级）
python -m voxelsim.cli --model llama2-13b --stage prefill --layers 2

# 使用 BookSim 高精度 NoC（需先构建）
python -m voxelsim.cli --noc-backend booksim --model llama2-13b --stage prefill

# Pareto 前沿搜索
python -m voxelsim.cli --pareto --model llama2-13b
```

输出为 JSON，包含周期数、能耗及各组件分解。

> **注意**：CLI 的 `--layers` 参数当前通过 `build_program_for_paradigm` 构建单层执行图（快速 smoke）。全层仿真请使用 Python API 中的 `build_full_model_program(..., layers=N)`，详见 [docs/voxel-experiments-reproduction.md](docs/voxel-experiments-reproduction.md)。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config PATH` | `configs/default.yaml` | 芯片 / 系统 YAML 配置 |
| `--model NAME` | `llama2-13b` | 可选：`llama2-13b`、`gemma2-27b`、`opt-30b`、`llama3-70b`、`dit-xl` |
| `--stage STAGE` | `prefill` | `prefill` 或 `decode` |
| `--layers N` | `2` | 仿真的 Transformer 层数（当前为单层 smoke） |
| `--noc-backend BACKEND` | 来自配置文件 | `analytic` 或 `booksim` |
| `--pareto` | 关闭 | 运行 Pareto 设计空间探索 |

## Python API

```python
from voxelsim import ChipConfig, Program, SimulationEngine
from voxelsim.models.llm import LLAMA2_13B
from voxelsim.models.paradigms import build_program_for_paradigm

cfg = ChipConfig(num_cores=64)
prog = build_program_for_paradigm(cfg, LLAMA2_13B, stage="decode")
graph = prog.build_graph(cfg.num_cores)
stats = SimulationEngine(cfg).run(graph)
print(stats.total_cycles, stats.noc_overhead_cycles)
```

全层 decode 仿真示例：

```python
from voxelsim.chip.config import ChipConfig
from voxelsim.models.llm import LLAMA2_13B, build_full_model_program
from voxelsim.sim.engine import SimulationEngine

cfg = ChipConfig()
prog = build_full_model_program(
    LLAMA2_13B,
    seq_len=cfg.sequence_length,
    batch=cfg.batch_size,
    stage="decode",
    layers=LLAMA2_13B.num_layers,
    num_cores=cfg.num_cores,
)
stats = SimulationEngine(cfg).run(prog.build_graph(cfg.num_cores))
```

加载自定义芯片配置：

```python
from voxelsim.chip.config import ChipConfig

cfg = ChipConfig.from_yaml("configs/default.yaml")
```

## 仿真架构

```
深度学习模型 + 芯片规格
       ↓
Program（compute / copy_data / sync + 集合通信）
       ↓
执行图（事件 + 依赖边）
       ↓
SimulationEngine
  ├── 核心模型   （SCALE-Sim v3 或解析模型）
  ├── NoC 模型   （Tier-A 解析 或 Tier-B BookSim）
  ├── DRAM 模型  （Ramulator 2.0 逐通道 或解析模型）
  ├── 热模型     （功率密度节流）
  └── 能耗模型   （DSENT NoC + ReGate 风格分解）
       ↓
统计结果（周期、能耗、分项分解）
```

## 测试

```bash
pytest tests/ -q
```

## 项目结构

```
voxelsim/
  api/          # 软件接口（论文 §3.3）
  graph/        # 执行图
  chip/         # 配置、拓扑、映射
  sim/          # 引擎、NoC、DRAM、热、能耗
  backends/     # SCALE-Sim、Ramulator、BookSim、DSENT
  models/       # LLM 模型与计算范式
  explore/      # Pareto 搜索
configs/        # default.yaml、dsent_router.cfg
scripts/        # setup_backends.sh、run_reproduction.py
tests/
docs/           # 论文技术参考与复现文档
```

## 配置说明

详见 [configs/default.yaml](configs/default.yaml)。主要参数与论文 Table 2/3 对齐：

| 参数 | 默认值 |
|------|--------|
| 计算范式 | `compute_shift` |
| NoC 拓扑 | `mesh_2d` |
| DRAM 带宽 | 12 TB/s |
| 核心数 | 256 |
| NoC 后端 | `analytic`（Tier-B 使用 `booksim`） |
| 频率 | 1.6 GHz |
| 批大小 / 序列长度 | 32 / 2048 |

## 参考文献

- Liu et al., *Exploring the Efficiency of 3D-Stacked AI Chip Architecture for LLM Inference with Voxel*, arXiv:2604.26821, 2026 — 详见 [docs/voxel-simulator.md](docs/voxel-simulator.md)
- [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim)（v3，位于 `third_party/SCALE-Sim`）
- [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2)
- [BookSim 2.0](https://github.com/booksim/booksim2)

## 许可证

研究原型项目 — 论文引用见 `docs/voxel-simulator.md`。
