"""Execution event nodes and graph container."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from voxelsim.api.ops import OpTile, TensorPart


class EventKind(str, Enum):
    COMPUTE = "compute"
    COPY_DATA = "copy_data"
    SYNC = "sync"


class ComponentKind(str, Enum):
    CORE = "core"
    DRAM_CHANNEL = "dram_channel"
    NOC_LINK = "noc_link"


@dataclass
class ExecutionEvent:
    event_id: int
    kind: EventKind
    component: ComponentKind
    component_id: int | str
    start_cycle: int = 0
    end_cycle: int = 0
    deps: list[int] = field(default_factory=list)
    op_tile: OpTile | None = None
    src: TensorPart | None = None
    dest: TensorPart | None = None
    core_id: int | None = None
    byte_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> int:
        return max(0, self.end_cycle - self.start_cycle)


@dataclass
class ExecutionGraph:
    events: list[ExecutionEvent]
    num_cores: int
    adjacency: dict[int, list[int]] = field(default_factory=dict)

    def topological_order(self) -> list[ExecutionEvent]:
        """Return events in dependency-respecting order (by event_id as tie-break)."""
        in_degree: dict[int, int] = {e.event_id: len(e.deps) for e in self.events}
        ready = sorted(
            (e for e in self.events if in_degree[e.event_id] == 0),
            key=lambda x: x.event_id,
        )
        order: list[ExecutionEvent] = []
        by_id = {e.event_id: e for e in self.events}
        children: dict[int, list[int]] = {e.event_id: [] for e in self.events}
        for e in self.events:
            for d in e.deps:
                children.setdefault(d, []).append(e.event_id)

        while ready:
            ready.sort(key=lambda x: x.event_id)
            e = ready.pop(0)
            order.append(e)
            for cid in children.get(e.event_id, []):
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    ready.append(by_id[cid])
        return order

    def compute_events(self) -> list[ExecutionEvent]:
        return [e for e in self.events if e.kind == EventKind.COMPUTE]

    def copy_events(self) -> list[ExecutionEvent]:
        return [e for e in self.events if e.kind == EventKind.COPY_DATA]
