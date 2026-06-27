"""Energy breakdown model (DSENT NoC + ReGate-style core/DRAM)."""

from __future__ import annotations

from voxelsim.backends.dsent_backend import DSENTBackend
from voxelsim.chip.config import ChipConfig
from voxelsim.sim.stats import SimulationStats


class EnergyModel:
    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self.dsent = DSENTBackend(config)
        self.breakdown: dict[str, float] = {}

    def estimate_total(self, stats: SimulationStats) -> float:
        freq = self.config.frequency_hz
        time_s = stats.total_cycles / freq

        dsent = self.dsent.estimate_noc(self.config.num_cores)
        core_static = 0.3 * self.config.num_cores * time_s
        core_dynamic = stats.compute_cycles / freq * 0.5
        dram_static = 0.1 * self.config.dram.total_banks * time_s
        dram_dynamic = stats.dram_access_cycles / freq * 0.15
        noc_dynamic = stats.noc_overhead_cycles / freq * dsent.link_energy_pj * 1e-12
        sram = stats.total_cycles / freq * 0.05 * self.config.num_cores

        self.breakdown = {
            "core_static_j": core_static,
            "core_dynamic_j": core_dynamic,
            "dram_static_j": dram_static,
            "dram_dynamic_j": dram_dynamic,
            "noc_dynamic_j": noc_dynamic,
            "sram_j": sram,
            "tsv_j": dram_dynamic * 0.1,
        }
        return sum(self.breakdown.values())
