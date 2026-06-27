# wafer3d / voxelsim

[English](README.md)

面向 **3D 堆叠 AI 芯片** 的编译器感知端到端仿真器，设计参考 Voxel 论文（详见 [docs/voxel-simulator.md](docs/voxel-simulator.md)）。

## 功能特性

- **软件接口**：`compute()`、`copy_data()`、`sync()` 及集合通信（`allReduce`、`allGather` 等）
- **执行图**自动生成与依赖追踪
- **双层 NoC 模型**：Tier-A 解析模型（默认）+ Tier-B BookSim 2.0 周期精确仿真，含流量模式缓存
- **AI 核心**：ScaleSim v3（GEMM），不可用时降级为解析模型
- **DRAM**：Ramulator 2.0 逐通道仿真 + XOR 匹配 trace 合并（numba 加速）
- **热节流**（功率密度约束）、**DSENT** NoC 能耗建模、**Pareto** 设计空间探索
- **LLM 工作负载**：Llama2-13B、Gemma2-27B、OPT-30B、Llama3-70B、DiT-XL
- **计算范式**：SPMD、Dataflow、Compute-shift

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| pip / venv | 推荐使用 |

可选 C++ 后端（通过 `scripts/setup_backends.sh` 构建）：

| 工具 | 用途 |
|------|------|
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

### 可选：构建 C++ 后端

在 `third_party/` 下构建 Ramulator 2.0、BookSim 2.0 和 DSENT：

```bash
bash scripts/setup_backends.sh
```

构建成功后，二进制路径如下：

| 后端 | 路径 |
|------|------|
| Ramulator 2.0 | `third_party/ramulator2/build/ramulator2` |
| BookSim 2.0 | `third_party/booksim2/src/booksim` |
| DSENT | `third_party/dsent_standalone/dsent` |

路径在 [configs/default.yaml](configs/default.yaml) 中配置。DSENT 使用 [Desent_modification](https://github.com/gyb1325/Desent_modification) 分支，Linux 下需 `LDFLAGS="-no-pie"`。BookSim 需先安装 `flex` 和 `bison`。

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

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config PATH` | `configs/default.yaml` | 芯片 / 系统 YAML 配置 |
| `--model NAME` | `llama2-13b` | 可选：`llama2-13b`、`gemma2-27b`、`opt-30b`、`llama3-70b`、`dit-xl` |
| `--stage STAGE` | `prefill` | `prefill` 或 `decode` |
| `--layers N` | `2` | 仿真的 Transformer 层数 |
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
  ├── 核心模型   （ScaleSim v3 或解析模型）
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
  backends/     # ScaleSim、Ramulator、BookSim、DSENT
  models/       # LLM 模型与计算范式
  explore/      # Pareto 搜索
configs/        # default.yaml、dsent_router.cfg
scripts/        # setup_backends.sh
tests/
docs/           # voxel-simulator.md（论文技术参考）
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
- [ScaleSim v3](https://github.com/scalesim-project/scalesim-v3)
- [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2)
- [BookSim 2.0](https://github.com/booksim/booksim2)

## 许可证

研究原型项目 — 论文引用见 `docs/voxel-simulator.md`。
