"""ScaleSim v3 backend for AI core cycle simulation with tile-shape cache."""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from voxelsim.chip.config import ChipConfig


@dataclass
class CoreSimResult:
    compute_cycles: int
    stall_cycles: int
    utilization: float
    backend: str


class ScaleSimBackend:
    """Wrap ScaleSim v3 GEMM mode; fall back to analytic SA model if unavailable."""

    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self._cache: dict[tuple[int, int, int], CoreSimResult] = {}
        self._available = self._probe()

    def _probe(self) -> bool:
        try:
            import scalesim  # noqa: F401

            return True
        except ImportError:
            return False

    def simulate_gemm(self, m: int, n: int, k: int) -> CoreSimResult:
        key = (m, n, k)
        if key in self._cache:
            return self._cache[key]

        if self._available:
            result = self._run_scalesim(m, n, k)
        else:
            result = self._analytic_gemm(m, n, k)

        self._cache[key] = result
        return result

    def _analytic_gemm(self, m: int, n: int, k: int) -> CoreSimResult:
        sa = self.config.systolic_array_size
        # Pad to SA dimensions (spatial underutilization)
        mp = ((m + sa - 1) // sa) * sa
        np_ = ((n + sa - 1) // sa) * sa
        kp = ((k + sa - 1) // sa) * sa
        useful = m * n * k
        total = mp * np_ * kp
        util = useful / max(1, total)
        cycles = (mp // sa) * (np_ // sa) * (kp // sa) * sa
        cycles = max(1, cycles)
        return CoreSimResult(
            compute_cycles=cycles,
            stall_cycles=0,
            utilization=util,
            backend="analytic",
        )

    def _run_scalesim(self, m: int, n: int, k: int) -> CoreSimResult:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                cfg = tmp_path / "scalesim.cfg"
                topo = tmp_path / "topo.csv"
                out = tmp_path / "out"

                sa = self.config.systolic_array_size
                sram_kb = self.config.per_core_sram_kb
                cfg.write_text(
                    f"""[general]
run_name = voxelsim_gemm

[architecture_presets]
ArrayHeight:    {sa}
ArrayWidth:     {sa}
IfmapSramSz:    {sram_kb}
FilterSramSz:   {sram_kb}
OfmapSramSz:    {sram_kb}
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Dataflow:       ws
Bandwidth :     {int(self.config.dram_bandwidth_bytes_per_cycle)}

[run_preset]
Interface: csv
"""
                )
                topo.write_text(f"Layer name, M, N, K,\nmatmul,{m},{n},{k},\n")

                from scalesim.scale_sim import scalesim

                sim = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=str(cfg),
                    topology=str(topo),
                    input_type_gemm=True,
                )
                sim.run_scale(top_path=str(out))

                report_dir = out / "voxelsim_gemm"
                compute_csv = report_dir / "COMPUTE_REPORT.csv"
                if compute_csv.exists():
                    with open(compute_csv, newline="") as f:
                        reader = csv.DictReader(f)
                        row = next(reader)
                        comp = int(float(row.get("Total Cycles", row.get("Compute cycles", 1))))
                        stall = int(float(row.get("Stall cycles", 0)))
                        util = float(row.get("Overall utilization %", row.get("Utilization", 50)))
                        return CoreSimResult(
                            compute_cycles=comp,
                            stall_cycles=stall,
                            utilization=util / 100.0 if util > 1 else util,
                            backend="scalesim",
                        )
        except Exception:
            pass
        return self._analytic_gemm(m, n, k)
