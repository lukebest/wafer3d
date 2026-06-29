"""Distributed DRAM simulation with per-channel priority queues."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from voxelsim.backends.ramulator_backend import DramRequest, RamulatorBackend
from voxelsim.api.ops import TensorPart
from voxelsim.chip.config import ChipConfig
from voxelsim.graph.events import ExecutionEvent, EventKind
from voxelsim.sim.trace_coalesce import TraceCoalescer


@dataclass(order=True)
class QueuedRequest:
    arrival_cycle: int
    event_index: int
    request: DramRequest = field(compare=False)


@dataclass
class DramChannelState:
    channel_id: int
    queue: list[QueuedRequest] = field(default_factory=list)
    current_cycle: int = 0


class DramSimulator:
    """Per-channel priority queue DRAM simulation with trace coalescing."""

    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self.backend = RamulatorBackend(config)
        self.coalescer = TraceCoalescer(config.dram.timing.tCL)
        self.num_channels = config.dram.total_banks
        self.burst_bytes = config.dram.interface_bytes
        self.channels: dict[int, DramChannelState] = {
            i: DramChannelState(channel_id=i) for i in range(self.num_channels)
        }
        self.total_dram_cycles = 0
        self.row_conflict_overhead = 0
        self._tensor_banks: dict[str, list[int]] = {}

    def set_tensor_banks(self, mapping: dict[str, list[int]]) -> None:
        self._tensor_banks = mapping

    def banks_for_tensor(self, tensor: TensorPart) -> list[int]:
        if tensor.name in self._tensor_banks:
            return self._tensor_banks[tensor.name]
        if tensor.bank_id is not None:
            return [tensor.bank_id]
        return [0]

    def channel_for_bank(self, bank_id: int) -> int:
        return bank_id % self.num_channels

    def _bank_row(self, addr: int) -> tuple[int, int]:
        bank = self.backend.bank_id(addr)
        row = addr >> self.backend.ROW_SHIFT
        return bank, row

    def _core_group_merge(self, batch: list[DramRequest]) -> list[DramRequest]:
        """Merge same (core_group, bank, row) accesses — request tracker (Fig 13)."""
        group_size = max(1, self.config.core_group_size)
        if group_size <= 1 or not batch:
            return batch
        seen: set[tuple[int, int, int]] = set()
        merged: list[DramRequest] = []
        for req in sorted(batch, key=lambda r: (r.arrival_cycle, r.event_id)):
            cg = (req.core_id or 0) // group_size
            bank, row = self._bank_row(req.addr)
            key = (cg, bank, row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(req)
        return merged if merged else batch

    def _enqueue_bursts(
        self,
        *,
        byte_size: int,
        base_addr: int,
        banks: list[int],
        start_cycle: int,
        event_id: int,
        is_write: bool,
        core_id: int = 0,
    ) -> int:
        if not banks:
            banks = [0]
        num_bursts = max(1, (byte_size + self.burst_bytes - 1) // self.burst_bytes)
        channels_touched: set[int] = set()
        for i in range(num_bursts):
            bank = banks[i % len(banks)]
            ch = self.channel_for_bank(bank)
            addr = base_addr + i * self.burst_bytes + bank * (1 << 20)
            req = DramRequest(
                addr=addr,
                is_write=is_write,
                arrival_cycle=start_cycle + i,
                event_id=event_id,
                core_id=core_id,
            )
            heapq.heappush(
                self.channels[ch].queue,
                QueuedRequest(req.arrival_cycle, event_id, req),
            )
            channels_touched.add(ch)

        total = 0
        for ch in channels_touched:
            total += self.process_channel(ch)
        return total

    def enqueue_copy(
        self,
        event: ExecutionEvent,
        start_cycle: int,
        core_id: int = 0,
    ) -> int:
        if event.dest is None:
            return 0
        banks = self.banks_for_tensor(event.dest)
        return self._enqueue_bursts(
            byte_size=event.byte_size,
            base_addr=event.dest.base_addr or 0,
            banks=banks,
            start_cycle=start_cycle,
            event_id=event.event_id,
            is_write=event.src is None,
            core_id=core_id,
        )

    def simulate_tensor_read(
        self,
        tensor: TensorPart,
        start_cycle: int,
        event_id: int,
        *,
        core_id: int = 0,
    ) -> int:
        banks = self.banks_for_tensor(tensor)
        byte_size = tensor.byte_size(self.config.bytes_per_element)
        return self._enqueue_bursts(
            byte_size=byte_size,
            base_addr=tensor.base_addr or hash(tensor.name) % (1 << 24),
            banks=banks,
            start_cycle=start_cycle,
            event_id=event_id,
            is_write=False,
            core_id=core_id,
        )

    def process_channel(self, channel_id: int) -> int:
        ch = self.channels[channel_id]
        if not ch.queue:
            return 0

        batch: list[DramRequest] = []
        while ch.queue and len(batch) < 256:
            q = heapq.heappop(ch.queue)
            batch.append(q.request)

        latencies = self.coalescer.simulate_channel(channel_id, batch, self.backend)
        total = sum(latencies)
        analytic = self.backend._analytic_trace(batch)
        analytic_total = sum(r.latency_cycles for r in analytic)
        raw_conflict = max(0, total - analytic_total)

        merged = self._core_group_merge(batch)
        if len(merged) < len(batch):
            merged_lat = self.coalescer.simulate_channel(channel_id, merged, self.backend)
            merged_total = sum(merged_lat)
            merged_analytic = self.backend._analytic_trace(merged)
            merged_analytic_total = sum(r.latency_cycles for r in merged_analytic)
            merged_conflict = max(0, merged_total - merged_analytic_total)
            self.row_conflict_overhead += merged_conflict
        else:
            self.row_conflict_overhead += raw_conflict

        bw_scale = 12.0 / max(0.1, self.config.dram_bandwidth_tbps)
        total = int(total * bw_scale)
        self.total_dram_cycles += total
        return total

    def simulate_event_dram(
        self,
        event: ExecutionEvent,
        start_cycle: int,
        core_id: int | None = None,
    ) -> int:
        if event.kind != EventKind.COPY_DATA or event.dest is None:
            return 0
        cid = core_id if core_id is not None else (event.core_id or 0)
        return self.enqueue_copy(event, start_cycle, core_id=cid)
