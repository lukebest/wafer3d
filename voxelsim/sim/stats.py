"""Simulation statistics container."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulationStats:
    total_cycles: int = 0
    compute_cycles: int = 0
    noc_overhead_cycles: int = 0
    dram_access_cycles: int = 0
    row_conflict_overhead_cycles: int = 0
    thermal_penalty_cycles: int = 0
    num_events: int = 0
    energy_joules: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
