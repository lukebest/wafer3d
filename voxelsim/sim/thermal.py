"""Thermal / power density throttling model (§5.6)."""

from __future__ import annotations

from voxelsim.chip.config import ChipConfig
from voxelsim.graph.events import EventKind, ExecutionEvent


class ThermalModel:
    """Extend event times when power density exceeds limit."""

    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self.limit = config.power_density_limit_w_per_mm2
        self.chip_area_mm2 = config.area.total_mm2
        self.core_power_w = 0.5  # per core during compute
        self.dram_power_w = 0.2

    def event_power_w(self, event: ExecutionEvent, duration: int) -> float:
        if event.kind == EventKind.COMPUTE:
            return self.core_power_w
        if event.kind == EventKind.COPY_DATA:
            return self.dram_power_w * 0.5
        return 0.05

    def apply_throttle(
        self,
        event: ExecutionEvent,
        start: int,
        end: int,
    ) -> int:
        duration = max(1, end - start)
        power = self.event_power_w(event, duration)
        density = power / max(1e-6, self.chip_area_mm2 / self.config.num_cores)
        if density <= self.limit:
            return 0
        ratio = density / self.limit
        extra = int(duration * (ratio - 1.0))
        return max(0, extra)
