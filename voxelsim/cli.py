"""Command-line interface for voxelsim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxelsim.chip.config import ChipConfig, default_config
from voxelsim.models.llm import MODELS, build_full_model_program
from voxelsim.models.paradigms import build_program_for_paradigm
from voxelsim.sim.engine import SimulationEngine
from voxelsim.explore.pareto import ParetoExplorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Voxel-like 3D AI chip simulator")
    parser.add_argument("--config", type=Path, default=None, help="YAML chip config")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="llama2-13b")
    parser.add_argument("--stage", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--layers", type=int, default=2, help="Transformer layers to simulate")
    parser.add_argument("--pareto", action="store_true", help="Run Pareto exploration")
    parser.add_argument("--noc-backend", choices=["analytic", "booksim"], default=None)
    args = parser.parse_args()

    if args.config:
        cfg = ChipConfig.from_yaml(args.config)
    else:
        cfg = default_config()

    if args.noc_backend:
        from voxelsim.chip.config import NoCBackend

        cfg.noc.backend = NoCBackend(args.noc_backend)

    model = MODELS[args.model]
    prog = build_program_for_paradigm(
        cfg,
        model,
        seq_len=cfg.sequence_length,
        batch=cfg.batch_size,
        stage=args.stage,
    )
    graph = prog.build_graph(cfg.num_cores)

    if args.pareto:
        explorer = ParetoExplorer(cfg)
        frontier = explorer.pareto_frontier(graph)
        print(json.dumps([f.__dict__ for f in frontier], indent=2, default=str))
        return

    engine = SimulationEngine(cfg)
    stats = engine.run(graph)
    print(
        json.dumps(
            {
                "model": model.name,
                "stage": args.stage,
                "total_cycles": stats.total_cycles,
                "compute_cycles": stats.compute_cycles,
                "noc_overhead_cycles": stats.noc_overhead_cycles,
                "dram_access_cycles": stats.dram_access_cycles,
                "row_conflict_overhead_cycles": stats.row_conflict_overhead_cycles,
                "thermal_penalty_cycles": stats.thermal_penalty_cycles,
                "energy_joules": stats.energy_joules,
                "num_events": stats.num_events,
                "energy_breakdown": stats.breakdown,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
