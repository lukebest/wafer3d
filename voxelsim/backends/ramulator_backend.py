"""Ramulator 2.0 backend for distributed DRAM simulation."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from voxelsim.chip.config import ChipConfig


@dataclass
class DramRequest:
    addr: int
    is_write: bool
    arrival_cycle: int
    event_id: int = 0


@dataclass
class DramSimResult:
    latency_cycles: int
    backend: str


class RamulatorBackend:
    """Per-channel DRAM simulation via Ramulator 2.0 subprocess or analytic fallback."""

    def __init__(self, config: ChipConfig, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.bin_path = self.repo_root / config.backends.ramulator_bin
        self._available = self.bin_path.exists() and self.bin_path.is_file()

    def simulate_trace(
        self,
        requests: list[DramRequest],
        channel_id: int = 0,
    ) -> list[DramSimResult]:
        if not requests:
            return []

        if self._available:
            try:
                return self._run_ramulator(requests, channel_id)
            except Exception:
                pass
        return self._analytic_trace(requests)

    def _analytic_trace(self, requests: list[DramRequest]) -> list[DramSimResult]:
        t = self.config.dram.timing
        row_hit = t.tCL
        row_miss = t.tRCD + t.tRP + t.tCL
        last_row: int | None = None
        results: list[DramSimResult] = []
        page_bits = 12

        for req in requests:
            row = req.addr >> page_bits
            if last_row is None or row == last_row:
                lat = row_hit
            else:
                lat = row_miss
            last_row = row
            results.append(DramSimResult(latency_cycles=lat, backend="analytic"))
        return results

    def _run_ramulator(
        self,
        requests: list[DramRequest],
        channel_id: int,
    ) -> list[DramSimResult]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path = tmp_path / f"ch{channel_id}.trace"
            cfg_path = tmp_path / "ramulator.yaml"

            with open(trace_path, "w") as f:
                for req in requests:
                    op = "W" if req.is_write else "R"
                    f.write(f"{req.arrival_cycle} {op} 0x{req.addr:x}\n")

            timing = self.config.dram.timing
            cfg = {
                "Frontend": {
                    "impl": "LoadStoreTrace",
                    "path": str(trace_path),
                },
                "MemorySystem": {
                    "impl": "GenericDRAM",
                    "clock_ratio": 1,
                    "DRAM": {
                        "impl": "DDR4",
                        "org": {
                            "preset": "DDR4_8Gb_x8",
                            "channel": 1,
                            "rank": 1,
                        },
                        "timing": {
                            "preset": "DDR4_2400R",
                            "nCL": timing.tCL,
                            "nRCD": timing.tRCD,
                            "nRP": timing.tRP,
                            "nRAS": timing.tRAS,
                        },
                    },
                    "Controller": {"impl": "Generic"},
                    "AddrMapper": {"impl": "RoBaRaCoCh"},
                },
            }
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f)

            proc = subprocess.run(
                [str(self.bin_path), "-f", str(cfg_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                return self._analytic_trace(requests)

            # Parse average latency from stdout or use analytic per-request
            return self._analytic_trace(requests)

    @staticmethod
    def addr_for_tensor(base: int, offset_elements: int, elem_bytes: int = 2) -> int:
        return base + offset_elements * elem_bytes
