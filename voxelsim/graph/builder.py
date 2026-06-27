"""Build execution graph from recorded program calls."""

from __future__ import annotations

from voxelsim.api.ops import MemoryLocation, OpTile, TensorPart
from voxelsim.graph.events import (
    ComponentKind,
    EventKind,
    ExecutionEvent,
    ExecutionGraph,
)


class ExecutionGraphBuilder:
    def __init__(self, num_cores: int, elem_bytes: int = 2) -> None:
        self.num_cores = num_cores
        self.elem_bytes = elem_bytes
        self.events: list[ExecutionEvent] = []
        self._last_on_core: dict[int, int] = {}
        self._tensor_producers: dict[str, int] = {}

    def add_compute(
        self,
        op_tile: OpTile,
        core_id: int | None,
        event_id: int,
    ) -> None:
        cid = core_id if core_id is not None else (event_id % self.num_cores)
        deps: list[int] = []
        if cid in self._last_on_core:
            deps.append(self._last_on_core[cid])
        for inp in op_tile.inputs:
            key = self._tensor_key(inp)
            if key in self._tensor_producers:
                pid = self._tensor_producers[key]
                if pid not in deps:
                    deps.append(pid)

        ev = ExecutionEvent(
            event_id=event_id,
            kind=EventKind.COMPUTE,
            component=ComponentKind.CORE,
            component_id=cid,
            deps=deps,
            op_tile=op_tile,
            core_id=cid,
        )
        self.events.append(ev)
        self._last_on_core[cid] = event_id
        for out in op_tile.outputs:
            self._tensor_producers[self._tensor_key(out)] = event_id

    def add_copy_data(
        self,
        src: TensorPart | None,
        dest: TensorPart,
        event_id: int,
    ) -> None:
        deps: list[int] = []
        if src is not None:
            key = self._tensor_key(src)
            if key in self._tensor_producers:
                deps.append(self._tensor_producers[key])

        byte_size = dest.shape.num_elements * self.elem_bytes
        component = ComponentKind.NOC_LINK
        component_id = "noc"

        if dest.location == MemoryLocation.DRAM and dest.bank_id is not None:
            component = ComponentKind.DRAM_CHANNEL
            component_id = dest.bank_id

        ev = ExecutionEvent(
            event_id=event_id,
            kind=EventKind.COPY_DATA,
            component=component,
            component_id=component_id,
            deps=deps,
            src=src,
            dest=dest,
            byte_size=byte_size,
        )
        self.events.append(ev)
        self._tensor_producers[self._tensor_key(dest)] = event_id

    def add_sync(
        self,
        core_ids: list[int] | None,
        event_id: int,
    ) -> None:
        deps = list(self._last_on_core.values())
        ev = ExecutionEvent(
            event_id=event_id,
            kind=EventKind.SYNC,
            component=ComponentKind.CORE,
            component_id=-1,
            deps=deps,
        )
        self.events.append(ev)

    def build(self) -> ExecutionGraph:
        adjacency: dict[int, list[int]] = {}
        for e in self.events:
            for d in e.deps:
                adjacency.setdefault(d, []).append(e.event_id)
        return ExecutionGraph(
            events=self.events,
            num_cores=self.num_cores,
            adjacency=adjacency,
        )

    @staticmethod
    def _tensor_key(t: TensorPart) -> str:
        loc = t.location.value
        bid = t.bank_id if t.bank_id is not None else -1
        cid = t.core_id if t.core_id is not None else -1
        return f"{t.name}:{loc}:{bid}:{cid}:{t.shape.dims}"
