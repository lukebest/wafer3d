"""NoC topology: mesh, torus, all-to-all routing and hop counts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from voxelsim.chip.config import ChipConfig, NoCTopology


@dataclass(frozen=True)
class CoreCoord:
    row: int
    col: int


class ChipTopology:
    """2D grid topology helpers aligned with BookSim k-ary n-mesh/cube params."""

    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self.num_cores = config.num_cores
        self.side = config.grid_side
        self.topology = config.noc_topology

    def core_to_coord(self, core_id: int) -> CoreCoord:
        return CoreCoord(row=core_id // self.side, col=core_id % self.side)

    def coord_to_core(self, coord: CoreCoord) -> int:
        return coord.row * self.side + coord.col

    def manhattan_hops(self, src: int, dst: int) -> int:
        if src == dst:
            return 0
        s = self.core_to_coord(src)
        d = self.core_to_coord(dst)
        return abs(s.row - d.row) + abs(s.col - d.col)

    def torus_hops(self, src: int, dst: int) -> int:
        if src == dst:
            return 0
        s = self.core_to_coord(src)
        d = self.core_to_coord(dst)
        dr = min(abs(s.row - d.row), self.side - abs(s.row - d.row))
        dc = min(abs(s.col - d.col), self.side - abs(s.col - d.col))
        return dr + dc

    def hops(self, src: int, dst: int) -> int:
        if self.topology == NoCTopology.ALL_TO_ALL:
            return 1 if src != dst else 0
        if self.topology == NoCTopology.TORUS_2D:
            return self.torus_hops(src, dst)
        return self.manhattan_hops(src, dst)

    def route_links(self, src: int, dst: int) -> list[tuple[int, int]]:
        """Return sequence of (from_core, to_core) hops for dimension-order routing."""
        if src == dst:
            return []
        if self.topology == NoCTopology.ALL_TO_ALL:
            return [(src, dst)]

        links: list[tuple[int, int]] = []
        cur = src
        s = self.core_to_coord(src)
        d = self.core_to_coord(dst)

        # Route along rows first (dimension-ordered)
        while s.row != d.row:
            if self.topology == NoCTopology.TORUS_2D:
                forward = (d.row - s.row) % self.side
                backward = (s.row - d.row) % self.side
                step = 1 if forward <= backward else -1
                next_row = (s.row + step) % self.side
            else:
                step = 1 if d.row > s.row else -1
                next_row = s.row + step
            next_coord = CoreCoord(row=next_row, col=s.col)
            nxt = self.coord_to_core(next_coord)
            links.append((cur, nxt))
            cur = nxt
            s = self.core_to_coord(cur)

        while s.col != d.col:
            if self.topology == NoCTopology.TORUS_2D:
                forward = (d.col - s.col) % self.side
                backward = (s.col - d.col) % self.side
                step = 1 if forward <= backward else -1
                next_col = (s.col + step) % self.side
            else:
                step = 1 if d.col > s.col else -1
                next_col = s.col + step
            next_coord = CoreCoord(row=s.row, col=next_col)
            nxt = self.coord_to_core(next_coord)
            links.append((cur, nxt))
            cur = nxt
            s = self.core_to_coord(cur)

        return links

    def booksim_params(self) -> dict[str, str | int]:
        """BookSim config parameters for subprocess backend."""
        k = self.side
        n = 2
        if self.topology == NoCTopology.MESH_2D:
            topo = "mesh"
        elif self.topology == NoCTopology.TORUS_2D:
            topo = "torus"
        else:
            topo = "mesh"  # all-to-all uses analytic 1-hop
        return {
            "topology": topo,
            "k": k,
            "n": n,
            "routing_function": "dim_order",
            "num_vcs": 2,
            "vc_buf_size": 8,
        }

    @staticmethod
    def grid_dims(num_cores: int) -> tuple[int, int]:
        side = int(math.isqrt(num_cores))
        if side * side != num_cores:
            raise ValueError(f"num_cores={num_cores} must be a perfect square")
        return side, side
