# wafer3d / voxelsim

**[中文文档（主 README）](README.md)**

Compiler-aware end-to-end simulator for **3D-stacked AI chips**, modeled after the Voxel paper ([docs/voxel-simulator.md](docs/voxel-simulator.md)).

> This is the English companion page. The canonical README is in Chinese: [README.md](README.md).

## Features

- **Software interface**: `compute()`, `copy_data()`, `sync()` + collectives (`allReduce`, `allGather`, …)
- **Execution graph** generation with dependency tracking
- **Tier-A / Tier-B NoC**: analytic model (default) + BookSim 2.0 (cycle-accurate) with traffic pattern cache; SYNC-region batch contention
- **AI core**: [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) v3 (GEMM) with analytic fallback
- **DRAM**: Ramulator 2.0 per-channel simulation with XOR trace coalescing (numba); tensor-to-bank mapping + core group request merging
- **Thermal** power-density throttling, **DSENT** NoC energy, Pareto design-space exploration
- **LLM workloads**: Llama2-13B, Gemma2-27B, OPT-30B, Llama3-70B, DiT-XL
- **Compute paradigms**: SPMD, Dataflow, Compute-shift (with communication/compute overlap modeling)

## Paper experiment reproduction

| Document | Description |
|----------|-------------|
| [docs/voxel-experiments-reproduction.md](docs/voxel-experiments-reproduction.md) | Experiment design, sweeps, and reproduction steps |
| [docs/voxel-reproduction-report.md](docs/voxel-reproduction-report.md) | Paper vs simulator trend check (PASS/PARTIAL/FAIL) |
| [docs/reproduction-results.json](docs/reproduction-results.json) | Raw JSON results |

```bash
python scripts/run_reproduction.py --quick   # ~10 min
python scripts/run_reproduction.py --full    # full Table 2/3 scale
```

Current quick-mode result: **10/10 PASS** on core paper trends.

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
python -m voxelsim.cli --model llama2-13b --stage prefill --layers 2
python -m voxelsim.cli --noc-backend booksim --model llama2-13b --stage prefill
python -m voxelsim.cli --pareto --model llama2-13b
```

Output is JSON with cycle counts, energy, and per-component breakdown.

> **Note**: CLI `--layers` currently builds a single-layer execution graph via `build_program_for_paradigm`. For full-layer runs use `build_full_model_program(..., layers=N)` in Python — see [docs/voxel-experiments-reproduction.md](docs/voxel-experiments-reproduction.md).

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `configs/default.yaml` | Chip / system YAML |
| `--model NAME` | `llama2-13b` | `llama2-13b`, `gemma2-27b`, `opt-30b`, `llama3-70b`, `dit-xl` |
| `--stage STAGE` | `prefill` | `prefill` or `decode` |
| `--layers N` | `2` | Transformer layers (single-layer smoke today) |
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

```python
from voxelsim.chip.config import ChipConfig

cfg = ChipConfig.from_yaml("configs/default.yaml")
```

## Simulation stack

```
DL model + chip spec → Program → Execution graph → SimulationEngine
  ├── Core (SCALE-Sim v3 or analytic)
  ├── NoC (analytic Tier-A or BookSim Tier-B)
  ├── DRAM (Ramulator 2.0 or analytic)
  ├── Thermal + Energy (DSENT NoC + breakdown)
  → Stats (cycles, energy, breakdown)
```

## Tests

```bash
pytest tests/ -q
```

## Project layout

```
voxelsim/   api/ graph/ chip/ sim/ backends/ models/ explore/
configs/    scripts/ (setup_backends.sh, run_reproduction.py)
tests/      docs/   (paper reference + reproduction guides)
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
