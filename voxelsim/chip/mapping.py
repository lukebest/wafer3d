"""Tile-to-core and tensor-to-bank mapping strategies."""

from __future__ import annotations

from dataclasses import dataclass

from voxelsim.api.ops import TensorPart
from voxelsim.chip.config import ChipConfig, TileToCoreMapping, TensorToBankMapping
from voxelsim.chip.topology import ChipTopology
from voxelsim.graph.events import EventKind, ExecutionEvent, ExecutionGraph


@dataclass
class TileMapping:
    tile_index: int
    core_id: int


@dataclass
class BankMapping:
    tensor_name: str
    bank_ids: list[int]


class MappingPlanner:
    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self.topology = ChipTopology(config)

    def map_tiles_to_cores(self, num_tiles: int) -> list[int]:
        """Return core_id for each tile index."""
        if self.config.tile_to_core_mapping == TileToCoreMapping.SEQUENTIAL:
            return [i % self.config.num_cores for i in range(num_tiles)]

        # Dimension-ordered: group tiles sharing data on same row/column
        side = self.config.grid_side
        mapping: list[int] = []
        for i in range(num_tiles):
            row = i // side
            col = i % side
            core_id = row * side + col
            mapping.append(core_id % self.config.num_cores)
        return mapping

    def map_tensor_to_banks(
        self,
        tensor: TensorPart,
        num_banks: int,
        *,
        concurrent_tensors: list[str] | None = None,
        tensor_index: int = 0,
        concurrent_bank_ids: list[int | None] | None = None,
    ) -> list[int]:
        strategy = self.config.tensor_to_bank_mapping

        if strategy == TensorToBankMapping.UNIFORM:
            # All tensors share the same bank subset -> high row conflicts.
            banks_used = max(1, min(8, num_banks // 4))
            return list(range(banks_used))

        if strategy == TensorToBankMapping.INTERLEAVE_SIZE:
            # Heuristic: consecutive tensors to disjoint banks, sized by tensor volume
            n_elems = tensor.shape.num_elements
            banks_needed = max(1, min(num_banks, n_elems // 4096))
            start = (tensor_index * banks_needed) % num_banks
            return [(start + i) % num_banks for i in range(banks_needed)]

        # Software-aware: place concurrent tensors on DISJOINT bank chunks. The
        # chunk for each tensor is chosen by the RANK of its explicit bank_id
        # among the concurrent set, so the paradigm builder's relative placement
        # intent is honored without the overlaps that arise from spreading each
        # tensor at its raw bank_id (e.g. bank_id 0 and 1 with chunk>1).
        if concurrent_tensors:
            chunk = max(1, num_banks // len(concurrent_tensors))
            if (
                concurrent_bank_ids is not None
                and tensor.name in concurrent_tensors
            ):
                pos = concurrent_tensors.index(tensor.name)
                order = sorted(
                    range(len(concurrent_tensors)),
                    key=lambda k: (
                        concurrent_bank_ids[k] is None,
                        concurrent_bank_ids[k] or 0,
                    ),
                )
                rank = order.index(pos)
                base = rank * chunk
            else:
                idx = (
                    concurrent_tensors.index(tensor.name)
                    if tensor.name in concurrent_tensors
                    else tensor_index
                )
                base = tensor.bank_id if tensor.bank_id is not None else idx * chunk
            return [(base + i) % num_banks for i in range(chunk)]
        base = tensor.bank_id if tensor.bank_id is not None else tensor_index
        return [base % num_banks]

    def detect_concurrent_tensors(self, graph: ExecutionGraph) -> dict[int, list[str]]:
        """Detect concurrent tensor accesses from execution graph (§4.3)."""
        concurrent: dict[int, list[str]] = {}
        for ev in graph.events:
            if ev.kind != EventKind.COMPUTE or ev.op_tile is None:
                continue
            names = [t.name for t in ev.op_tile.inputs] + [t.name for t in ev.op_tile.outputs]
            concurrent[ev.event_id] = list(dict.fromkeys(names))
        return concurrent

    def assign_banks_for_event(
        self,
        ev: ExecutionEvent,
        num_banks: int,
        concurrent: dict[int, list[str]],
    ) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        if ev.op_tile is None:
            return result
        names = concurrent.get(ev.event_id, [])
        ev_tensors = ev.op_tile.inputs + ev.op_tile.outputs
        name_to_bank = {t.name: t.bank_id for t in ev_tensors}
        bank_ids = [name_to_bank.get(n) for n in names]
        for i, t in enumerate(ev_tensors):
            result[t.name] = self.map_tensor_to_banks(
                t,
                num_banks,
                concurrent_tensors=names,
                tensor_index=i,
                concurrent_bank_ids=bank_ids,
            )
        return result
