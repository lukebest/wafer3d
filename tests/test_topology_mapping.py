"""Tests for topology and mapping."""

from voxelsim.chip.config import ChipConfig, NoCTopology, TileToCoreMapping
from voxelsim.chip.topology import ChipTopology
from voxelsim.chip.mapping import MappingPlanner


def test_mesh_hops():
    cfg = ChipConfig(num_cores=16, noc_topology=NoCTopology.MESH_2D)
    topo = ChipTopology(cfg)
    assert topo.hops(0, 0) == 0
    assert topo.hops(0, 3) == 3
    assert topo.hops(0, 15) == 6


def test_torus_hops_wraparound():
    cfg = ChipConfig(num_cores=16, noc_topology=NoCTopology.TORUS_2D)
    topo = ChipTopology(cfg)
    assert topo.hops(0, 15) <= 4


def test_all_to_all_one_hop():
    cfg = ChipConfig(num_cores=16, noc_topology=NoCTopology.ALL_TO_ALL)
    topo = ChipTopology(cfg)
    assert topo.hops(0, 5) == 1


def test_dimension_ordered_mapping():
    cfg = ChipConfig(num_cores=16, tile_to_core_mapping=TileToCoreMapping.DIMENSION_ORDERED)
    planner = MappingPlanner(cfg)
    cores = planner.map_tiles_to_cores(16)
    assert len(cores) == 16
    assert cores[0] == 0
