"""Tests for Ramulator backend integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voxelsim.backends.ramulator_backend import DramRequest, RamulatorBackend
from voxelsim.chip.config import default_config


RAMULATOR_STDOUT = """
MemorySystem:
  total_num_read_requests: 2
  Controller:
    read_row_hits_0: 1
    read_row_misses_0: 1
    read_row_conflicts_0: 0
    num_read_reqs_0: 2
    avg_read_latency_0: 28
"""


def test_parse_ramulator_stats_extracts_controller_metrics() -> None:
    stats = RamulatorBackend._parse_ramulator_stats(RAMULATOR_STDOUT)
    assert stats["read_row_hits_0"] == 1
    assert stats["read_row_misses_0"] == 1
    assert stats["avg_read_latency_0"] == 28
    assert stats["total_num_read_requests"] == 2


def test_run_ramulator_uses_parsed_stats_not_analytic_fallback() -> None:
    cfg = default_config()
    backend = RamulatorBackend(cfg)
    requests = [
        DramRequest(addr=0x1000, is_write=False, arrival_cycle=0),
        DramRequest(addr=0x1008, is_write=False, arrival_cycle=1),
    ]

    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = RAMULATOR_STDOUT
    proc.stderr = ""

    with patch.object(backend, "_available", True), patch(
        "voxelsim.backends.ramulator_backend.subprocess.run",
        return_value=proc,
    ):
        results = backend._run_ramulator(requests, channel_id=0)

    assert all(r.backend == "ramulator" for r in results)
    assert [r.latency_cycles for r in results] == [42, 14]


def test_run_ramulator_falls_back_when_subprocess_fails() -> None:
    cfg = default_config()
    backend = RamulatorBackend(cfg)
    requests = [DramRequest(addr=0x1000, is_write=False, arrival_cycle=0)]

    proc = MagicMock()
    proc.returncode = 1
    proc.stdout = ""
    proc.stderr = "error"

    with patch.object(backend, "_available", True), patch(
        "voxelsim.backends.ramulator_backend.subprocess.run",
        return_value=proc,
    ):
        results = backend._run_ramulator(requests, channel_id=0)

    assert results[0].backend == "analytic"


@pytest.mark.skipif(
    not Path("third_party/ramulator2/build/ramulator2").exists(),
    reason="Ramulator binary not built",
)
def test_simulate_trace_uses_ramulator_when_binary_available() -> None:
    cfg = default_config()
    backend = RamulatorBackend(cfg)
    requests = [
        DramRequest(addr=0x1000, is_write=False, arrival_cycle=0),
        DramRequest(addr=0x2000, is_write=False, arrival_cycle=1),
    ]
    results = backend.simulate_trace(requests, channel_id=0)
    assert results
    assert results[0].backend == "ramulator"
