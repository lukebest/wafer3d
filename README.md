# wafer3d / voxelsim

Compiler-aware end-to-end simulator for **3D-stacked AI chips**, modeled after the Voxel paper ([docs/voxel-simulator.md](docs/voxel-simulator.md)).

## Features

- **Software interface**: `compute()`, `copy_data()`, `sync()` + collectives (`allReduce`, `allGather`, …)
- **Execution graph** generation with dependency tracking
- **Tier-A / Tier-B NoC**: analytic model (default) + BookSim 2.0 (cycle-accurate) with traffic pattern cache
- **AI core**: ScaleSim v3 (GEMM) with analytic fallback
- **DRAM**: Ramulator 2.0 per-channel simulation with XOR trace coalescing (numba)
- **Thermal** power-density throttling, **DSENT** NoC energy, Pareto design-space exploration
- **LLM workloads**: Llama2-13B, Gemma2-27B, OPT-30B, Llama3-70B, DiT-XL
- **Compute paradigms**: SPMD, Dataflow, Compute-shift

## Install

```bash
pip install -e ".[dev]"
# Optional: build C++ backends (Ramulator, BookSim, DSENT)
bash scripts/setup_backends.sh
```

## Quick start

```bash
# End-to-end simulation (analytic NoC + analytic DRAM fallback)
python -m voxelsim.cli --model llama2-13b --stage prefill --layers 2

# High-precision NoC via BookSim (requires build)
python -m voxelsim.cli --noc-backend booksim --model llama2-13b

# Pareto frontier search
python -m voxelsim.cli --pareto --model llama2-13b
```

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
  backends/     # ScaleSim, Ramulator, BookSim, DSENT
  models/       # LLM + compute paradigms
  explore/      # Pareto search
configs/        # default.yaml (Table 2/3)
scripts/        # setup_backends.sh
tests/
docs/           # voxel-simulator.md (reference)
```

## Configuration

See [configs/default.yaml](configs/default.yaml). Key knobs match paper Table 2/3:

| Parameter | Default |
|-----------|---------|
| Computation paradigm | compute_shift |
| NoC topology | mesh_2d |
| DRAM bandwidth | 12 TB/s |
| Cores | 256 |
| NoC backend | analytic (`booksim` for Tier-B) |

## License

Research prototype — see paper references in `docs/voxel-simulator.md`.
