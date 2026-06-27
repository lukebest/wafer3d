# wafer3d / voxelsim

[中文文档](README.zh-CN.md)

Compiler-aware end-to-end simulator for **3D-stacked AI chips**, modeled after the Voxel paper ([docs/voxel-simulator.md](docs/voxel-simulator.md)).

## Features

- **Software interface**: `compute()`, `copy_data()`, `sync()` + collectives (`allReduce`, `allGather`, …)
- **Execution graph** generation with dependency tracking
- **Tier-A / Tier-B NoC**: analytic model (default) + BookSim 2.0 (cycle-accurate) with traffic pattern cache
- **AI core**: [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) v3 (GEMM) with analytic fallback
- **DRAM**: Ramulator 2.0 per-channel simulation with XOR trace coalescing (numba)
- **Thermal** power-density throttling, **DSENT** NoC energy, Pareto design-space exploration
- **LLM workloads**: Llama2-13B, Gemma2-27B, OPT-30B, Llama3-70B, DiT-XL
- **Compute paradigms**: SPMD, Dataflow, Compute-shift

## Requirements

| Component | Version |
|-----------|---------|
| Python | ≥ 3.10 |
| pip / venv | recommended |

Optional backends (via `scripts/setup_backends.sh`):

| Tool | Used by |
|------|---------|
| `pip`, Python venv | SCALE-Sim (editable install) |
| `cmake`, `g++` | Ramulator 2.0 |
| `flex`, `bison`, `g++` | BookSim 2.0 |
| `g++`, `make` | DSENT |

If a backend binary is missing or fails to build, the simulator falls back to analytic models automatically.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Optional: third-party backends

Build SCALE-Sim, Ramulator 2.0, BookSim 2.0, and DSENT under `third_party/`:

```bash
bash scripts/setup_backends.sh
```

Or install SCALE-Sim only:

```bash
pip install -e third_party/SCALE-Sim
```

Expected paths after a successful build:

| Backend | Path |
|---------|------|
| SCALE-Sim | `third_party/SCALE-Sim/` (editable Python package) |
| Ramulator 2.0 | `third_party/ramulator2/build/ramulator2` |
| BookSim 2.0 | `third_party/booksim2/src/booksim` |
| DSENT | `third_party/dsent_standalone/dsent` |

SCALE-Sim requires a small NumPy compatibility patch (`scripts/patches/scalesim-numpy-max.patch`), applied automatically by `setup_backends.sh`.

Ramulator / BookSim / DSENT paths are configured in [configs/default.yaml](configs/default.yaml). DSENT uses the [Desent_modification](https://github.com/gyb1325/Desent_modification) fork with `LDFLAGS="-no-pie"` on Linux.

## Quick start

```bash
# End-to-end simulation (analytic NoC + analytic DRAM fallback)
python -m voxelsim.cli --model llama2-13b --stage prefill --layers 2

# High-precision NoC via BookSim (requires build)
python -m voxelsim.cli --noc-backend booksim --model llama2-13b --stage prefill

# Pareto frontier search
python -m voxelsim.cli --pareto --model llama2-13b
```

Output is JSON with cycle counts, energy, and per-component breakdown.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `configs/default.yaml` | Chip / system YAML |
| `--model NAME` | `llama2-13b` | `llama2-13b`, `gemma2-27b`, `opt-30b`, `llama3-70b`, `dit-xl` |
| `--stage STAGE` | `prefill` | `prefill` or `decode` |
| `--layers N` | `2` | Transformer layers to simulate |
| `--noc-backend BACKEND` | from config | `analytic` or `booksim` |
| `--pareto` | off | Run Pareto design-space exploration |

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

Load a custom chip configuration:

```python
from voxelsim.chip.config import ChipConfig

cfg = ChipConfig.from_yaml("configs/default.yaml")
```

## Simulation stack

```
DL model + chip spec
       ↓
Program (compute / copy_data / sync + collectives)
       ↓
Execution graph (events + dependencies)
       ↓
SimulationEngine
  ├── Core model   (SCALE-Sim v3 or analytic)
  ├── NoC model    (analytic Tier-A or BookSim Tier-B)
  ├── DRAM model   (Ramulator 2.0 per-channel or analytic)
  ├── Thermal      (power-density throttling)
  └── Energy       (DSENT NoC + ReGate-style breakdown)
       ↓
Stats (cycles, energy, breakdown)
```

## Tests

```bash
pytest tests/ -q
```

## Project layout

```
voxelsim/
  api/          # Software interface (§3.3)
  graph/        # Execution graph
  chip/         # Config, topology, mapping
  sim/          # Engine, NoC, DRAM, thermal, energy
  backends/     # SCALE-Sim, Ramulator, BookSim, DSENT
  models/       # LLM + compute paradigms
  explore/      # Pareto search
configs/        # default.yaml, dsent_router.cfg
scripts/        # setup_backends.sh
tests/
docs/           # voxel-simulator.md (paper reference, Chinese)
```

## Configuration

See [configs/default.yaml](configs/default.yaml). Key knobs match paper Table 2/3:

| Parameter | Default |
|-----------|---------|
| Computation paradigm | `compute_shift` |
| NoC topology | `mesh_2d` |
| DRAM bandwidth | 12 TB/s |
| Cores | 256 |
| NoC backend | `analytic` (`booksim` for Tier-B) |
| Frequency | 1.6 GHz |
| Batch size / seq len | 32 / 2048 |

## References

- Liu et al., *Exploring the Efficiency of 3D-Stacked AI Chip Architecture for LLM Inference with Voxel*, arXiv:2604.26821, 2026 — see [docs/voxel-simulator.md](docs/voxel-simulator.md)
- [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) (v3, in `third_party/SCALE-Sim`)
- [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2)
- [BookSim 2.0](https://github.com/booksim/booksim2)

## License

Research prototype — see paper references in `docs/voxel-simulator.md`.
