"""AI core simulation wrapper."""

from __future__ import annotations

from voxelsim.backends.scalesim_backend import CoreSimResult, ScaleSimBackend
from voxelsim.chip.config import ChipConfig
from voxelsim.graph.events import ExecutionEvent


class CoreSimulator:
    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self.backend = ScaleSimBackend(config)

    def simulate_compute(self, event: ExecutionEvent) -> CoreSimResult:
        if event.op_tile is None:
            return CoreSimResult(1, 0, 1.0, "empty")
        m, n, k = event.op_tile.gemm_dims()
        return self.backend.simulate_gemm(m, n, k)
