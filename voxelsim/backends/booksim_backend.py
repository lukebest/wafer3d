"""BookSim 2.0 backend for Tier-B cycle-accurate NoC simulation."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from voxelsim.chip.config import ChipConfig, NoCTopology
from voxelsim.chip.topology import ChipTopology
from voxelsim.sim.noc_sim import NoCLatencyResult, NoCTransfer


@dataclass
class BookSimConfig:
    topology: str
    k: int
    n: int
    routing_function: str
    link_bandwidth: int
    injection_rate: float = 1.0


class BookSimBackend:
    """Run BookSim with open-loop trace injection; fall back to analytic on failure."""

    def __init__(self, config: ChipConfig, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.bin_path = self.repo_root / config.backends.booksim_bin
        self.topology = ChipTopology(config)
        self._available = self.bin_path.exists() and self.bin_path.is_file()

    def estimate_batch(self, transfers: list[NoCTransfer]) -> list[NoCLatencyResult]:
        if not transfers:
            return []

        if self.config.noc_topology == NoCTopology.ALL_TO_ALL:
            return self._all_to_all_analytic(transfers)

        if self._available:
            try:
                return self._run_booksim(transfers)
            except Exception:
                pass

        from voxelsim.sim.noc_sim import AnalyticNoCModel

        return AnalyticNoCModel(self.config).estimate_batch(transfers)

    def _all_to_all_analytic(self, transfers: list[NoCTransfer]) -> list[NoCLatencyResult]:
        bw = self.config.noc.link_bandwidth_bytes_per_cycle
        results: list[NoCLatencyResult] = []
        for t in transfers:
            if t.src_core == t.dst_core:
                lat = 1
            else:
                lat = max(1, (t.byte_size + bw - 1) // bw) + 1
            results.append(
                NoCLatencyResult(
                    transfer=t,
                    latency_cycles=lat,
                    noc_overhead_cycles=0,
                    backend="all_to_all_analytic",
                )
            )
        return results

    def _write_booksim_config(self, path: Path) -> None:
        params = self.topology.booksim_params()
        bw = self.config.noc.link_bandwidth_bytes_per_cycle
        content = f"""// Auto-generated BookSim config for voxelsim
topology = {params['topology']};
k = {params['k']};
n = {params['n']};
routing_function = {params['routing_function']};
num_vcs = {params['num_vcs']};
vc_buf_size = {params['vc_buf_size']};
packet_size = 1;
use_read_write = 0;
traffic = trace;
injection_rate = 0.5;
sim_type = latency;
sample_period = 1000;
injection_rate = 1.0;
"""
        path.write_text(content)

    def _write_trace(self, path: Path, transfers: list[NoCTransfer]) -> None:
        """BookSim trace format: cycle src dst size (simplified ad-hoc)."""
        with open(path, "w") as f:
            for t in transfers:
                flits = max(1, (t.byte_size + 31) // 32)
                f.write(f"{t.inject_cycle} {t.src_core} {t.dst_core} {flits}\n")

    def _run_booksim(self, transfers: list[NoCTransfer]) -> list[NoCLatencyResult]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = tmp_path / "booksim.cfg"
            trace = tmp_path / "trace.txt"
            self._write_booksim_config(cfg)
            self._write_trace(trace, transfers)

            proc = subprocess.run(
                [str(self.bin_path), str(cfg)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(tmp_path),
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr)

            # Parse average packet latency from BookSim output
            avg_lat = self._parse_latency(proc.stdout)
            per = max(1, avg_lat)
            return [
                NoCLatencyResult(
                    transfer=t,
                    latency_cycles=per,
                    noc_overhead_cycles=0,
                    backend="booksim",
                )
                for t in transfers
            ]

    @staticmethod
    def _parse_latency(stdout: str) -> int:
        for line in stdout.splitlines():
            if "packet latency average" in line.lower() or "average latency" in line.lower():
                parts = line.split("=")
                if len(parts) >= 2:
                    try:
                        return int(float(parts[-1].strip().split()[0]))
                    except ValueError:
                        continue
        return 8  # conservative default
