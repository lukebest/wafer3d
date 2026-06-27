"""Program recording and Voxel-style software interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from voxelsim.api.ops import OpTile, TensorPart
from voxelsim.graph.builder import ExecutionGraphBuilder


@dataclass
class RecordedCall:
    kind: str
    args: tuple
    kwargs: dict


@dataclass
class Program:
    """Records compute/copy_data/sync calls from an ML compiler execution plan."""

    calls: list[RecordedCall] = field(default_factory=list)
    _event_counter: int = 0

    def next_event_id(self) -> int:
        eid = self._event_counter
        self._event_counter += 1
        return eid

    def compute(self, op_tile: OpTile, core_id: int | None = None) -> int:
        eid = self.next_event_id()
        self.calls.append(
            RecordedCall("compute", (op_tile, core_id), {"event_id": eid})
        )
        return eid

    def copy_data(
        self,
        src_tensor: TensorPart | None,
        dest_tensor: TensorPart,
    ) -> int:
        eid = self.next_event_id()
        self.calls.append(
            RecordedCall(
                "copy_data",
                (src_tensor, dest_tensor),
                {"event_id": eid},
            )
        )
        return eid

    def sync(self, core_ids: list[int] | None = None) -> int:
        eid = self.next_event_id()
        self.calls.append(
            RecordedCall("sync", (core_ids,), {"event_id": eid})
        )
        return eid

    def build_graph(self, num_cores: int) -> "ExecutionGraph":
        from voxelsim.graph.events import ExecutionGraph

        builder = ExecutionGraphBuilder(num_cores)
        for call in self.calls:
            eid = call.kwargs["event_id"]
            if call.kind == "compute":
                builder.add_compute(call.args[0], call.args[1], eid)
            elif call.kind == "copy_data":
                builder.add_copy_data(call.args[0], call.args[1], eid)
            elif call.kind == "sync":
                builder.add_sync(call.args[0], eid)
        return builder.build()


def run_plan(fn: Callable[[Program], None], num_cores: int) -> "ExecutionGraph":
    prog = Program()
    fn(prog)
    return prog.build_graph(num_cores)
