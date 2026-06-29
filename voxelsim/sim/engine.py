"""End-to-end event-driven simulation engine."""

from __future__ import annotations

from voxelsim.api.ops import MemoryLocation
from voxelsim.chip.config import ChipConfig, ComputationParadigm
from voxelsim.chip.mapping import MappingPlanner
from voxelsim.graph.events import EventKind, ExecutionEvent, ExecutionGraph
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
        self._noc_latencies: dict[int, int] = {}
        self._noc_overheads: dict[int, int] = {}

    def run(self, graph: ExecutionGraph) -> SimulationStats:
        self._prepare_bank_mapping(graph)
        self._precompute_noc_regions(graph)

        stats = SimulationStats(num_events=len(graph.events))
        global_cycle = 0
        core_compute: dict[int, int] = {}

        for event in graph.topological_order():
            ready = max((self._end_times.get(d, 0) for d in event.deps), default=0)
            start = ready

            if event.kind == EventKind.COMPUTE:
                result = self.core_sim.simulate_compute(event)
                duration = result.compute_cycles + result.stall_cycles
                dram_read = self._simulate_compute_dram_reads(event, start, stats)
                duration += dram_read
                stats.compute_cycles += duration - dram_read
                event.end_cycle = start + duration
                cid = event.core_id or 0
                core_compute[cid] = core_compute.get(cid, 0) + duration

            elif event.kind == EventKind.COPY_DATA:
                duration, is_noc = self._simulate_copy(event, start, stats)
                event.end_cycle = start + duration
                if not is_noc:
                    cid = (event.dest.core_id if event.dest and event.dest.core_id is not None
                           else (event.src.core_id if event.src and event.src.core_id is not None else 0))
                    core_compute[cid] = core_compute.get(cid, 0) + duration

            elif event.kind == EventKind.SYNC:
                duration = 1
                event.end_cycle = start + duration

            else:
                event.end_cycle = start

            penalty = self.thermal.apply_throttle(event, start, event.end_cycle)
            if penalty > 0:
                event.end_cycle += penalty
                stats.thermal_penalty_cycles += penalty

            self._end_times[event.event_id] = event.end_cycle
            global_cycle = max(global_cycle, event.end_cycle)

        compute_critical_path = max(core_compute.values()) if core_compute else 0
        stats.total_cycles = self._apply_overlap(global_cycle, compute_critical_path, stats)
        stats.dram_access_cycles = self.dram_sim.total_dram_cycles
        stats.row_conflict_overhead_cycles = self.dram_sim.row_conflict_overhead
        stats.energy_joules = self.energy.estimate_total(stats)
        stats.breakdown = self.energy.breakdown
        return stats

    def _apply_overlap(
        self, makespan: int, compute_critical_path: int, stats: SimulationStats
    ) -> int:
        """Critical-path model with paradigm-aware NoC/compute overlap (paper A2).

        Total = compute_critical_path (parallel per-core compute + DRAM loads)
                + noc_serial, where noc_serial is the fraction of cumulative NoC
                overhead that remains on the critical path after overlap.

        SPMD's all-reduce is a serial barrier (no overlap -> factor 1.0).
        Dataflow pipelines overlap half its NoC with compute (factor 0.5).
        Compute-shift's 1-hop ring shift overlaps most (factor 0.3).

        Because the credit is a fraction of NoC overhead (not of compute), a
        lower-NoC paradigm always lands lower on the critical path, preserving
        the paper's compute-shift < dataflow < SPMD ordering.
        """
        serial_factor = {
            ComputationParadigm.SPMD: 1.0,
            ComputationParadigm.DATAFLOW: 0.5,
            ComputationParadigm.COMPUTE_SHIFT: 0.3,
        }.get(self.config.computation_paradigm, 1.0)
        noc_serial = int(stats.noc_overhead_cycles * serial_factor)
        component_total = compute_critical_path + noc_serial
        return max(component_total, makespan) if serial_factor >= 1.0 else max(component_total, compute_critical_path)

    def _prepare_bank_mapping(self, graph: ExecutionGraph) -> None:
        planner = MappingPlanner(self.config)
        concurrent = planner.detect_concurrent_tensors(graph)
        num_banks = self.config.dram.total_banks
        tensor_banks: dict[str, list[int]] = {}

        for ev in graph.events:
            if ev.kind != EventKind.COMPUTE or ev.op_tile is None:
                continue
            names = concurrent.get(ev.event_id, [])
            ev_tensors = ev.op_tile.inputs + ev.op_tile.outputs
            name_to_bank = {t.name: t.bank_id for t in ev_tensors}
            bank_ids = [name_to_bank.get(n) for n in names]
            for i, t in enumerate(ev_tensors):
                if t.name not in tensor_banks:
                    tensor_banks[t.name] = planner.map_tensor_to_banks(
                        t,
                        num_banks,
                        concurrent_tensors=names,
                        tensor_index=i,
                        concurrent_bank_ids=bank_ids,
                    )

        for ev in graph.copy_events():
            for t in (ev.src, ev.dest):
                if t is None or t.name in tensor_banks:
                    continue
                tensor_banks[t.name] = planner.map_tensor_to_banks(
                    t, num_banks, tensor_index=hash(t.name) % num_banks
                )

        self.dram_sim.set_tensor_banks(tensor_banks)

    def _precompute_noc_regions(self, graph: ExecutionGraph) -> None:
        """Batch NoC transfers within SYNC-bounded regions for link contention."""
        self._noc_latencies = {}
        self._noc_overheads = {}
        ordered = graph.topological_order()
        region: list[ExecutionEvent] = []

        def flush_region() -> None:
            transfers: list[NoCTransfer] = []
            event_ids: list[int] = []
            for ev in region:
                if ev.kind != EventKind.COPY_DATA or ev.src is None:
                    continue
                if ev.src.location != MemoryLocation.SRAM:
                    continue
                # SRAM->DRAM writes bypass the NoC (go straight to the DRAM
                # controller); only core-to-core SRAM->SRAM transfers contend.
                if ev.dest is not None and ev.dest.location == MemoryLocation.DRAM:
                    continue
                src_core = ev.src.core_id if ev.src.core_id is not None else 0
                dst_core = (
                    ev.dest.core_id
                    if ev.dest and ev.dest.core_id is not None
                    else src_core
                )
                transfers.append(
                    NoCTransfer(
                        src_core=src_core,
                        dst_core=dst_core,
                        byte_size=ev.byte_size,
                        inject_cycle=0,
                        event_id=ev.event_id,
                    )
                )
                event_ids.append(ev.event_id)

            if not transfers:
                region.clear()
                return

            results = self.noc_sim.estimate_transfers(transfers)
            for eid, res in zip(event_ids, results):
                self._noc_latencies[eid] = res.latency_cycles
                self._noc_overheads[eid] = res.noc_overhead_cycles
            region.clear()

        for ev in ordered:
            if ev.kind == EventKind.SYNC:
                flush_region()
                continue
            region.append(ev)
        flush_region()

    def _simulate_compute_dram_reads(
        self,
        event: ExecutionEvent,
        start: int,
        stats: SimulationStats,
    ) -> int:
        if event.op_tile is None:
            return 0
        core_id = event.core_id or 0
        total = 0
        for inp in event.op_tile.inputs:
            if inp.location != MemoryLocation.DRAM:
                continue
            byte_size = inp.byte_size(self.config.bytes_per_element)
            total += self.dram_sim.simulate_tensor_read(
                inp, start + total, event.event_id, core_id=core_id
            )
        return total

    def _simulate_copy(
        self,
        event: ExecutionEvent,
        start: int,
        stats: SimulationStats,
    ) -> tuple[int, bool]:
        """Return (duration, is_noc). is_noc marks core-to-core NoC transfers
        (counted in the NoC serial term); DRAM loads count toward compute path."""
        duration = 0
        is_noc = False

        if (
            event.src
            and event.src.location == MemoryLocation.SRAM
            and (event.dest is None or event.dest.location != MemoryLocation.DRAM)
        ):
            is_noc = True
            noc_lat = self._noc_latencies.get(event.event_id)
            if noc_lat is None:
                src_core = event.src.core_id if event.src.core_id is not None else 0
                dst_core = (
                    event.dest.core_id
                    if event.dest and event.dest.core_id is not None
                    else src_core
                )
                transfer = NoCTransfer(
                    src_core=src_core,
                    dst_core=dst_core,
                    byte_size=event.byte_size,
                    inject_cycle=start,
                    event_id=event.event_id,
                )
                results = self.noc_sim.estimate_transfers([transfer])
                if results:
                    duration += results[0].latency_cycles
                    stats.noc_overhead_cycles += results[0].noc_overhead_cycles
            else:
                duration += noc_lat
                stats.noc_overhead_cycles += self._noc_overheads.get(event.event_id, 0)

        if event.dest and event.dest.location == MemoryLocation.DRAM:
            dram_cycles = self.dram_sim.simulate_event_dram(
                event, start + duration, core_id=event.core_id
            )
            duration += dram_cycles
            is_noc = False

        if (
            event.src
            and event.src.location == MemoryLocation.DRAM
            and (event.dest is None or event.dest.location != MemoryLocation.DRAM)
        ):
            dram_cycles = self.dram_sim.simulate_tensor_read(
                event.src,
                start + duration,
                event.event_id,
                core_id=event.core_id or 0,
            )
            duration += dram_cycles
            is_noc = False

        return max(1, duration), is_noc
