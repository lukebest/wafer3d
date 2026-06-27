"""DSENT backend for NoC power/area estimation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from voxelsim.chip.config import ChipConfig


@dataclass
class DSENTResult:
    router_energy_pj: float
    link_energy_pj: float
    router_area_mm2: float
    link_area_mm2: float
    backend: str


class DSENTBackend:
    """Query DSENT for router/link energy and area; analytic fallback."""

    def __init__(self, config: ChipConfig, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.bin_path = self.repo_root / config.backends.dsent_bin
        self._available = self.bin_path.exists()

    def estimate_noc(
        self,
        num_routers: int,
        link_length_um: float = 500.0,
    ) -> DSENTResult:
        if self._available:
            try:
                return self._run_dsent(num_routers, link_length_um)
            except Exception:
                pass
        return self._analytic(num_routers, link_length_um)

    def _analytic(self, num_routers: int, link_length_um: float) -> DSENTResult:
        # ORION/DSENT-style rough estimates at 22nm
        router_e = 50.0  # pJ per flit
        link_e = 0.5 * link_length_um  # pJ per flit per um
        router_a = 0.05  # mm2
        link_a = 0.0001 * link_length_um  # mm2
        return DSENTResult(
            router_energy_pj=router_e * num_routers,
            link_energy_pj=link_e,
            router_area_mm2=router_a * num_routers,
            link_area_mm2=link_a,
            backend="analytic",
        )

    def _run_dsent(self, num_routers: int, link_length_um: float) -> DSENTResult:
        cfg_path = self.repo_root / "configs" / "dsent_router.cfg"
        if not cfg_path.exists():
            return self._analytic(num_routers, link_length_um)

        proc = subprocess.run(
            [str(self.bin_path), str(cfg_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return self._analytic(num_routers, link_length_um)

        energy = 50.0
        area = 0.05
        for line in proc.stdout.splitlines():
            if "Energy" in line and ">>" in line:
                try:
                    energy = float(line.split()[-1])
                except ValueError:
                    pass
            if "Area" in line and ">>" in line:
                try:
                    area = float(line.split()[-1])
                except ValueError:
                    pass

        return DSENTResult(
            router_energy_pj=energy * num_routers,
            link_energy_pj=0.5 * link_length_um,
            router_area_mm2=area * num_routers,
            link_area_mm2=0.0001 * link_length_um,
            backend="dsent",
        )
