"""DRAM refresh tracking for requests targeting active refresh sets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RefreshWindow:
    start_addr: int
    end_addr: int
    end_cycle: int


class RefreshTracker:
    """Shift request arrival time if it hits an ongoing refresh region."""

    def __init__(self, refresh_interval_cycles: int = 7800) -> None:
        self.refresh_interval = refresh_interval_cycles
        self.active: list[RefreshWindow] = []
        self.last_refresh_start = 0

    def tick(self, cycle: int, memory_size: int) -> None:
        if cycle - self.last_refresh_start >= self.refresh_interval:
            chunk = memory_size // 8192
            start = (cycle // self.refresh_interval) * chunk
            self.active.append(
                RefreshWindow(
                    start_addr=start,
                    end_addr=start + chunk,
                    end_cycle=cycle + 512,
                )
            )
            self.last_refresh_start = cycle
        self.active = [w for w in self.active if w.end_cycle > cycle]

    def adjust_arrival(self, addr: int, arrival: int) -> int:
        for w in self.active:
            if w.start_addr <= addr < w.end_addr:
                return max(arrival, w.end_cycle)
        return arrival
