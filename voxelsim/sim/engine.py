"""End-to-end event-driven simulation engine."""

from __future__ import annotations

from voxelsim.api.ops import MemoryLocation
from voxelsim.chip.config import ChipConfig
from voxelsim.graph.events import EventKind, ExecutionGraph
from voxelsim.sim.core_sim import CoreSimulator
from voxelsim.sim.dram_sim import DramSimulator
from voxelsim.sim.noc_sim import NoCSimulator, NoCTransfer
from voxelsim.sim.refresh import RefreshTracker
from voxelsim.sim.thermal import ThermalModel
from voxelsim.sim.energy import EnergyModel


from voxelsim.sim.stats import SimulationStats


class SimulationEngine:
    """Traverse execution graph chronologically; dispatch to core/NoC/DRAM."""

    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self.core_sim = CoreSimulator(config)
        self.noc_sim = NoCSimulator(config)
        self.dram_sim = DramSimulator(config)
        self.refresh = RefreshTracker()
        self.thermal = ThermalModel(config)
        self.energy = EnergyModel(config)
        self._end_times: dict[int, int] = {}
        self._noc_overhead_total = 0

    def run(self, graph: ExecutionGraph) -> SimulationStats:
        stats = SimulationStats(num_events=len(graph.events))
        global_cycle = 0

        for event in graph.topological_order():
            ready = max((self._end_times.get(d, 0) for d in event.deps), default=0)
            start = ready

            if event.kind == EventKind.COMPUTE:
                result = self.core_sim.simulate_compute(event)
                duration = result.compute_cycles + result.stall_cycles
                stats.compute_cycles += duration
                event.end_cycle = start + duration

            elif event.kind == EventKind.COPY_DATA:
                duration = self._simulate_copy(event, start, stats)
                event.end_cycle = start + duration

            elif event.kind == EventKind.SYNC:
                duration = 1
                event.end_cycle = start + duration

            else:
                event.end_cycle = start

            # Thermal throttling
            penalty = self.thermal.apply_throttle(event, start, event.end_cycle)
            if penalty > 0:
                event.end_cycle += penalty
                stats.thermal_penalty_cycles += penalty

            self._end_times[event.event_id] = event.end_cycle
            global_cycle = max(global_cycle, event.end_cycle)

        stats.total_cycles = global_cycle
        stats.dram_access_cycles = self.dram_sim.total_dram_cycles
        stats.row_conflict_overhead_cycles = self.dram_sim.row_conflict_overhead
        stats.energy_joules = self.energy.estimate_total(stats)
        stats.breakdown = self.energy.breakdown
        return stats

    def _simulate_copy(
        self,
        event,
        start: int,
        stats: SimulationStats,
    ) -> int:
        duration = 0
        src_core = event.src.core_id if event.src and event.src.core_id is not None else 0
        dst_core = event.dest.core_id if event.dest and event.dest.core_id is not None else src_core

        # NoC leg for core-to-core or core-to-DRAM
        if event.src and event.src.location == MemoryLocation.SRAM:
            transfer = NoCTransfer(
                src_core=src_core,
                dst_core=dst_core,
                byte_size=event.byte_size,
                inject_cycle=start,
                event_id=event.event_id,
            )
            noc_results = self.noc_sim.estimate_transfers([transfer])
            if noc_results:
                duration += noc_results[0].latency_cycles
                stats.noc_overhead_cycles += noc_results[0].noc_overhead_cycles

        # DRAM leg
        if event.dest and event.dest.location == MemoryLocation.DRAM:
            dram_cycles = self.dram_sim.simulate_event_dram(event, start + duration)
            duration += dram_cycles

        return max(1, duration)
