"""Tests for ScaleSim backend integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from voxelsim.backends.scalesim_backend import ScaleSimBackend
from voxelsim.chip.config import default_config


def test_parse_compute_report_reads_scalesim_v3_columns() -> None:
    report = Path("third_party/SCALE-Sim/test/general/golden_trace_user_ws/COMPUTE_REPORT.csv")
    if not report.is_file():
        return

    result = ScaleSimBackend._parse_compute_report(report)
    assert result.backend == "scalesim"
    assert result.compute_cycles > 0
    assert 0.0 <= result.utilization <= 1.0


def test_simulate_gemm_uses_scalesim_when_third_party_available() -> None:
    scalesim_root = Path("third_party/SCALE-Sim")
    if not scalesim_root.is_dir():
        return

    backend = ScaleSimBackend(default_config())
    assert backend._available

    result = backend.simulate_gemm(64, 64, 64)
    assert result.backend == "scalesim"
    assert result.compute_cycles > 0


def test_simulate_gemm_falls_back_when_scalesim_missing() -> None:
    backend = ScaleSimBackend(default_config())
    with patch.object(backend, "_available", False):
        result = backend.simulate_gemm(32, 32, 32)
    assert result.backend == "analytic"
    assert result.compute_cycles > 0


def test_probe_passes_constructor_scalesim_root() -> None:
    custom_root = Path("/tmp/custom-wafer3d")
    with patch.object(ScaleSimBackend, "_ensure_scalesim_import") as mock_import:
        mock_import.return_value = False
        backend = ScaleSimBackend(default_config(), repo_root=custom_root)
    mock_import.assert_called_once_with(custom_root / "third_party" / "SCALE-Sim")
    assert backend.scalesim_root == custom_root / "third_party" / "SCALE-Sim"
