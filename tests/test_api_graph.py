"""Tests for software API and execution graph."""

from voxelsim.api.ops import OpTile, make_tensor_part, MemoryLocation
from voxelsim.api.program import Program


def test_program_builds_graph():
    prog = Program()
    inp = make_tensor_part("a", (32, 32), location=MemoryLocation.DRAM, bank_id=0)
    w = make_tensor_part("w", (32, 32), location=MemoryLocation.DRAM, bank_id=1)
    out = make_tensor_part("o", (32, 32), location=MemoryLocation.SRAM, core_id=0)
    tile = OpTile("mm", "matmul", [inp, w], [out], gemm_m=32, gemm_n=32, gemm_k=32)
    prog.compute(tile, core_id=0)
    prog.copy_data(inp, out)
    prog.sync([0])
    graph = prog.build_graph(num_cores=4)
    assert len(graph.events) == 3
    assert len(graph.compute_events()) == 1
    assert len(graph.copy_events()) == 1


def test_topological_order_respects_deps():
    prog = Program()
    a = make_tensor_part("a", (8, 8), location=MemoryLocation.DRAM, bank_id=0)
    b = make_tensor_part("b", (8, 8), location=MemoryLocation.SRAM, core_id=1)
    prog.copy_data(a, b)
    tile = OpTile("t", "matmul", [b], [b], gemm_m=8, gemm_n=8, gemm_k=8)
    prog.compute(tile, core_id=1)
    graph = prog.build_graph(4)
    order = [e.event_id for e in graph.topological_order()]
    assert order.index(0) < order.index(1)
