"""Distributed DRAM simulation with per-channel priority queues."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from voxelsim.backends.ramulator_backend import DramRequest, RamulatorBackend
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

    def channel_for_bank(self, bank_id: int) -> int:
        return bank_id % self.num_channels

    def enqueue_copy(self, event: ExecutionEvent, start_cycle: int) -> list[DramRequest]:
        if event.dest is None:
            return []
        bank = event.dest.bank_id or 0
        ch = self.channel_for_bank(bank)
        elem_bytes = self.config.bytes_per_element
        num_bursts = max(1, (event.byte_size + self.burst_bytes - 1) // self.burst_bytes)
        reqs: list[DramRequest] = []
        for i in range(num_bursts):
            addr = (event.dest.base_addr or 0) + i * self.burst_bytes
            req = DramRequest(
                addr=addr,
                is_write=event.src is None,
                arrival_cycle=start_cycle + i,
                event_id=event.event_id,
            )
            heapq.heappush(
                self.channels[ch].queue,
                QueuedRequest(req.arrival_cycle, event.event_id, req),
            )
            reqs.append(req)
        return reqs

    def process_channel(self, channel_id: int) -> int:
        ch = self.channels[channel_id]
        if not ch.queue:
            return 0

        batch: list[DramRequest] = []
        while ch.queue and len(batch) < 256:
            q = heapq.heappop(ch.queue)
            batch.append(q.request)

        latencies = self.coalescer.simulate_channel(
            channel_id, batch, self.backend
        )
        total = sum(latencies)
        analytic = self.backend._analytic_trace(batch)
        analytic_total = sum(r.latency_cycles for r in analytic)
        self.row_conflict_overhead += max(0, total - analytic_total)
        self.total_dram_cycles += total
        return total

    def simulate_event_dram(self, event: ExecutionEvent, start_cycle: int) -> int:
        if event.kind != EventKind.COPY_DATA or event.dest is None:
            return 0
        if event.dest.bank_id is None:
            return 0
        self.enqueue_copy(event, start_cycle)
        ch = self.channel_for_bank(event.dest.bank_id)
        return self.process_channel(ch)
