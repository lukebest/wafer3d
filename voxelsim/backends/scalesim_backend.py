"""ScaleSim v3 backend for AI core cycle simulation with tile-shape cache."""

from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from voxelsim.chip.config import ChipConfig


@dataclass
class CoreSimResult:
    compute_cycles: int
    stall_cycles: int
    utilization: float
    backend: str


class ScaleSimBackend:
    """Wrap SCALE-Sim (v3) GEMM mode; fall back to analytic SA model if unavailable."""

    # SCALE-Sim is intended for tile-level GEMMs; skip huge full-layer shapes.
    MAX_SCALESIM_DIM = 1024

    def __init__(self, config: ChipConfig, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.scalesim_root = self.repo_root / "third_party" / "SCALE-Sim"
        self.default_layout = self.scalesim_root / "layouts" / "conv_nets" / "test.csv"
        self._cache: dict[tuple[int, int, int], CoreSimResult] = {}
        self._available = self._probe()

    def _probe(self) -> bool:
        return self._ensure_scalesim_import()

    @classmethod
    def _ensure_scalesim_import(cls, scalesim_root: Path | None = None) -> bool:
        try:
            import scalesim  # noqa: F401

            return True
        except ImportError:
            pass

        if scalesim_root is None:
            repo_root = Path(__file__).resolve().parents[2]
            scalesim_root = repo_root / "third_party" / "SCALE-Sim"

        root_str = str(scalesim_root)
        if scalesim_root.is_dir() and root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            import scalesim  # noqa: F401

            return True
        except ImportError:
            return False

    def simulate_gemm(self, m: int, n: int, k: int) -> CoreSimResult:
        key = (m, n, k)
        if key in self._cache:
            return self._cache[key]

        if self._available and self._within_scalesim_limits(m, n, k):
            result = self._run_scalesim(m, n, k)
        else:
            result = self._analytic_gemm(m, n, k)

        self._cache[key] = result
        return result

    @classmethod
    def _within_scalesim_limits(cls, m: int, n: int, k: int) -> bool:
        return max(m, n, k) <= cls.MAX_SCALESIM_DIM

    def _analytic_gemm(self, m: int, n: int, k: int) -> CoreSimResult:
        sa = self.config.systolic_array_size
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
        if not self._ensure_scalesim_import(self.scalesim_root):
            return self._analytic_gemm(m, n, k)

        layout_path = self.default_layout
        if not layout_path.is_file():
            return self._analytic_gemm(m, n, k)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                cfg = tmp_path / "scalesim.cfg"
                topo = tmp_path / "topo.csv"
                out = tmp_path / "out"

                sa = self.config.systolic_array_size
                sram_kb = self.config.per_core_sram_kb
                bw = int(self.config.dram_bandwidth_bytes_per_cycle)
                cfg.write_text(
                    f"""[general]
run_name = voxelsim_gemm

[architecture_presets]
ArrayHeight:    {sa}
ArrayWidth:     {sa}
IfmapSramSzkB:    {sram_kb}
FilterSramSzkB:   {sram_kb}
OfmapSramSzkB:    {sram_kb}
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Dataflow:       ws
Bandwidth :     {bw}
ReadRequestBuffer: 128
WriteRequestBuffer: 128

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: CALC
UseRamulatorTrace: False
"""
                )
                topo.write_text(f"Layer name, M, N, K,\nmatmul,{m},{n},{k},\n")

                from scalesim.scale_sim import scalesim

                sim = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=str(cfg),
                    topology=str(topo),
                    layout=str(layout_path),
                    input_type_gemm=True,
                )
                sim.run_scale(top_path=str(out))

                report_dir = out / "voxelsim_gemm"
                compute_csv = report_dir / "COMPUTE_REPORT.csv"
                if compute_csv.exists():
                    return self._parse_compute_report(compute_csv)
        except Exception:
            pass
        return self._analytic_gemm(m, n, k)

    @staticmethod
    def _parse_compute_report(path: Path) -> CoreSimResult:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            raw = next(reader)
        row = {k.strip(): (v.strip() if v else v) for k, v in raw.items() if k and k.strip()}

        compute = int(
            float(
                row.get("Total Cycles")
                or row.get("Total Cycles (incl. prefetch)")
                or row.get("Compute cycles", 1)
            )
        )
        stall = int(float(row.get("Stall Cycles", row.get("Stall cycles", 0))))
        util_raw = row.get("Overall Util %", row.get("Overall utilization %", row.get("Utilization", 50)))
        util = float(util_raw)
        if util > 1:
            util /= 100.0

        return CoreSimResult(
            compute_cycles=max(1, compute),
            stall_cycles=stall,
            utilization=util,
            backend="scalesim",
        )
