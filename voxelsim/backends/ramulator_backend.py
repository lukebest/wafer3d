"""Ramulator 2.0 backend for distributed DRAM simulation."""

from __future__ import annotations

import re
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


@dataclass(frozen=True)
class _RowAccess:
    kind: str  # hit | miss | conflict | write
    latency: int


class RamulatorBackend:
    """Per-channel DRAM simulation via Ramulator 2.0 subprocess or analytic fallback."""

    PAGE_BITS = 12
    BANK_BITS = 3
    BANK_MASK = (1 << BANK_BITS) - 1
    ROW_SHIFT = PAGE_BITS + BANK_BITS

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
        latencies = self._row_buffer_latencies(requests)
        return [
            DramSimResult(latency_cycles=lat, backend="analytic")
            for lat in latencies
        ]

    def _run_ramulator(
        self,
        requests: list[DramRequest],
        channel_id: int,
    ) -> list[DramSimResult]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path = tmp_path / f"ch{channel_id}.trace"
            cfg_path = tmp_path / "ramulator.yaml"

            self._write_trace(requests, trace_path)
            self._write_config(trace_path, cfg_path)

            proc = subprocess.run(
                [str(self.bin_path), "-f", str(cfg_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                return self._analytic_trace(requests)

            stats = self._parse_ramulator_stats(proc.stdout + proc.stderr)
            if not stats:
                return self._analytic_trace(requests)

            return self._results_from_ramulator_stats(requests, stats)

    def _write_trace(self, requests: list[DramRequest], trace_path: Path) -> None:
        """LoadStoreTrace expects ``LD addr`` / ``ST addr`` lines."""
        with open(trace_path, "w", encoding="ascii") as f:
            for req in requests:
                op = "ST" if req.is_write else "LD"
                f.write(f"{op} 0x{req.addr:x}\n")

    def _write_config(self, trace_path: Path, cfg_path: Path) -> None:
        timing = self.config.dram.timing
        cfg = {
            "Frontend": {
                "impl": "LoadStoreTrace",
                "path": str(trace_path),
                "clock_ratio": 1,
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
                "Controller": {
                    "impl": "Generic",
                    "Scheduler": {"impl": "FRFCFS"},
                    "RefreshManager": {"impl": "AllBank"},
                    "RowPolicy": {"impl": "ClosedRowPolicy", "cap": 4},
                },
                "AddrMapper": {"impl": "RoBaRaCoCh"},
            },
        }
        with open(cfg_path, "w", encoding="ascii") as f:
            yaml.dump(cfg, f)

    @staticmethod
    def _parse_ramulator_stats(text: str) -> dict[str, float]:
        stats: dict[str, float] = {}
        for pattern in (
            r"^\s{4}(\w+_0):\s*([\d.eE+-]+)",
            r"^\s{2}(memory_system_cycles|total_num_\w+):\s*(\d+)",
        ):
            for match in re.finditer(pattern, text, re.MULTILINE):
                stats[match.group(1)] = float(match.group(2))
        return stats

    def _results_from_ramulator_stats(
        self,
        requests: list[DramRequest],
        stats: dict[str, float],
    ) -> list[DramSimResult]:
        replay = self._classify_row_accesses(requests)
        read_replay = [access for access, req in zip(replay, requests) if not req.is_write]

        ramulator_reads = int(
            stats.get("read_row_hits_0", 0)
            + stats.get("read_row_misses_0", 0)
            + stats.get("read_row_conflicts_0", 0)
        )
        if ramulator_reads == 0 or ramulator_reads != len(read_replay):
            return self._analytic_trace(requests)

        replay_counts = self._count_access_kinds(read_replay)
        ramulator_counts = {
            "hit": int(stats.get("read_row_hits_0", 0)),
            "miss": int(stats.get("read_row_misses_0", 0)),
            "conflict": int(stats.get("read_row_conflicts_0", 0)),
        }
        if replay_counts != ramulator_counts:
            return self._analytic_trace(requests)

        latencies = [access.latency for access in replay]
        avg_read = stats.get("avg_read_latency_0", 0.0)
        if avg_read > 0 and read_replay:
            latencies = self._apply_avg_read_latency(requests, latencies, avg_read)

        return [
            DramSimResult(latency_cycles=lat, backend="ramulator")
            for lat in latencies
        ]

    def _apply_avg_read_latency(
        self,
        requests: list[DramRequest],
        latencies: list[int],
        avg_read: float,
    ) -> list[int]:
        """Blend row-buffer latencies with Ramulator average read latency."""
        read_indices = [i for i, req in enumerate(requests) if not req.is_write]
        if not read_indices:
            return latencies

        read_total = sum(latencies[i] for i in read_indices)
        target_total = int(round(avg_read * len(read_indices)))
        if read_total <= 0 or target_total <= read_total:
            return latencies

        extra = target_total - read_total
        per_read_extra = extra // len(read_indices)
        remainder = extra % len(read_indices)

        adjusted = latencies[:]
        for offset, idx in enumerate(read_indices):
            adjusted[idx] += per_read_extra + (1 if offset < remainder else 0)
        return adjusted

    def _row_buffer_latencies(self, requests: list[DramRequest]) -> list[int]:
        return [access.latency for access in self._classify_row_accesses(requests)]

    def _classify_row_accesses(self, requests: list[DramRequest]) -> list[_RowAccess]:
        t = self.config.dram.timing
        hit_lat = t.tCL
        miss_lat = t.tRCD + t.tRP + t.tCL
        conflict_lat = t.tRCD + t.tCL
        write_lat = t.tCL

        open_rows: dict[int, int] = {}
        results: list[_RowAccess] = []

        for req in requests:
            bank = self._bank_id(req.addr)
            row = self._row_id(req.addr)

            if req.is_write:
                open_rows[bank] = row
                results.append(_RowAccess(kind="write", latency=write_lat))
                continue

            open_row = open_rows.get(bank)
            if open_row is None:
                kind = "miss"
                latency = miss_lat
            elif open_row == row:
                kind = "hit"
                latency = hit_lat
            else:
                kind = "conflict"
                latency = conflict_lat
            open_rows[bank] = row
            results.append(_RowAccess(kind=kind, latency=latency))

        return results

    @classmethod
    def _bank_id(cls, addr: int) -> int:
        """DDR4 bank index from address bits [14:12]."""
        return (addr >> cls.PAGE_BITS) & cls.BANK_MASK

    @classmethod
    def _row_id(cls, addr: int) -> int:
        return addr >> cls.ROW_SHIFT

    @staticmethod
    def _count_access_kinds(accesses: list[_RowAccess]) -> dict[str, int]:
        counts = {"hit": 0, "miss": 0, "conflict": 0}
        for access in accesses:
            if access.kind in counts:
                counts[access.kind] += 1
        return counts

    @staticmethod
    def addr_for_tensor(base: int, offset_elements: int, elem_bytes: int = 2) -> int:
        return base + offset_elements * elem_bytes
