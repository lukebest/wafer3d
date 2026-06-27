"""Tests for DRAM trace coalescing."""

import numpy as np

from voxelsim.backends.ramulator_backend import DramRequest, RamulatorBackend
from voxelsim.chip.config import ChipConfig
from voxelsim.sim.trace_coalesce import TraceCoalescer, compute_match_keys


def test_xor_match_keys():
    addrs = np.array([0x1000, 0x1100, 0x1200], dtype=np.uint64)
    keys = compute_match_keys(addrs)
    assert keys[1] == 0x1000 ^ 0x1100


def test_coalescer_hit_rate():
    cfg = ChipConfig()
    backend = RamulatorBackend(cfg)
    co = TraceCoalescer()
    reqs = [
        DramRequest(addr=0xA000 + i * 128, is_write=False, arrival_cycle=i)
        for i in range(8)
    ]
    co.simulate_channel(0, reqs, backend)
    co.simulate_channel(0, reqs, backend)
    assert co.hits >= 1
