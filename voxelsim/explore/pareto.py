"""Pareto frontier search with coordinate descent under area constraints."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

from voxelsim.chip.config import ChipConfig
from voxelsim.graph.events import ExecutionGraph
from voxelsim.sim.stats import SimulationStats
from voxelsim.sim.engine import SimulationEngine


@dataclass
class DesignPoint:
    config: ChipConfig
    stats: SimulationStats
    area_mm2: float
    score: float


@dataclass
class ParetoPoint:
    area_mm2: float
    prefill_cycles: int
    decode_cycles_per_token: int
    config_snapshot: dict


class ParetoExplorer:
    """Multi-level area-constrained coordinate descent (Figure 7)."""

    def __init__(
        self,
        base_config: ChipConfig,
        run_fn: Callable[[ChipConfig, ExecutionGraph], SimulationStats] | None = None,
    ) -> None:
        self.base = base_config
        self._run_fn = run_fn

    def _evaluate(self, cfg: ChipConfig, graph: ExecutionGraph) -> SimulationStats:
        if self._run_fn:
            return self._run_fn(cfg, graph)
        engine = SimulationEngine(cfg)
        return engine.run(graph)

    def _area(self, cfg: ChipConfig) -> float:
        return cfg.area.total_mm2

    def coordinate_descent_step(
        self,
        cfg: ChipConfig,
        graph: ExecutionGraph,
        area_limit: float,
    ) -> DesignPoint:
        """Minimize geometric mean of execution time under area constraint."""
        knobs = [
            ("dram_bandwidth_tbps", [4, 8, 12, 16]),
            ("num_cores", [64, 128, 256, 512]),
            ("systolic_array_size", [16, 32, 64]),
            ("per_core_sram_kb", [512, 2048, 8192]),
        ]
        best_cfg = cfg
        best_stats = self._evaluate(cfg, graph)
        best_score = float(best_stats.total_cycles)

        for name, values in knobs:
            for v in values:
                trial = copy.deepcopy(cfg)
                if name == "dram_bandwidth_tbps":
                    trial.dram_bandwidth_tbps = float(v)
                elif name == "num_cores":
                    side = int(math.isqrt(v))
                    if side * side != v:
                        continue
                    trial.num_cores = v
                elif name == "systolic_array_size":
                    trial.systolic_array_size = v
                elif name == "per_core_sram_kb":
                    trial.per_core_sram_kb = v

                if self._area(trial) > area_limit:
                    continue
                stats = self._evaluate(trial, graph)
                score = float(stats.total_cycles)
                if score < best_score:
                    best_score = score
                    best_cfg = trial
                    best_stats = stats

        return DesignPoint(
            config=best_cfg,
            stats=best_stats,
            area_mm2=self._area(best_cfg),
            score=best_score,
        )

    def pareto_frontier(
        self,
        graph: ExecutionGraph,
        area_thresholds: list[float] | None = None,
    ) -> list[ParetoPoint]:
        if area_thresholds is None:
            area_thresholds = [400, 600, 850, 1000]

        frontier: list[ParetoPoint] = []
        for limit in area_thresholds:
            dp = self.coordinate_descent_step(self.base, graph, limit)
            frontier.append(
                ParetoPoint(
                    area_mm2=dp.area_mm2,
                    prefill_cycles=dp.stats.total_cycles,
                    decode_cycles_per_token=max(1, dp.stats.total_cycles // max(1, self.base.batch_size)),
                    config_snapshot=dp.config.to_dict(),
                )
            )
        return frontier
