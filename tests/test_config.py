"""Tests for chip configuration."""

from voxelsim.chip.config import ChipConfig, default_config


def test_default_config_loads():
    cfg = default_config()
    assert cfg.num_cores == 256
    assert cfg.dram_bandwidth_tbps == 12.0
    assert cfg.computation_paradigm.value == "compute_shift"


def test_grid_side():
    cfg = ChipConfig(num_cores=64)
    assert cfg.grid_side == 8


def test_to_dict_roundtrip_keys():
    cfg = default_config()
    d = cfg.to_dict()
    assert d["hardware"]["num_cores"] == 256
    assert d["software"]["computation_paradigm"] == "compute_shift"
