"""Tests for NoC simulator."""

from voxelsim.chip.config import ChipConfig
from voxelsim.sim.noc_sim import NoCSimulator, NoCTransfer, PatternCache


def test_analytic_noc_latency():
    cfg = ChipConfig(num_cores=16)
    sim = NoCSimulator(cfg)
    transfers = [
        NoCTransfer(src_core=0, dst_core=3, byte_size=128, inject_cycle=0),
    ]
    results = sim.estimate_transfers(transfers)
    assert len(results) == 1
    assert results[0].latency_cycles >= 1


def test_pattern_cache_hit():
    cache = PatternCache()
    t = [NoCTransfer(0, 1, 64, 0), NoCTransfer(1, 2, 64, 10)]
    k1 = cache.pattern_key(t, "mesh_2d", "dim_order")
    cache.put(k1, 100)
    assert cache.get(k1) == 100
    assert cache.hits == 1
