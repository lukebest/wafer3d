"""End-to-end smoke tests."""

from voxelsim.chip.config import (
    ChipConfig,
    ComputationParadigm,
    TensorToBankMapping,
)
from voxelsim.models.llm import LLAMA2_13B
from voxelsim.models.paradigms import build_program_for_paradigm
from voxelsim.sim.engine import SimulationEngine


def test_e2e_smoke_small():
    cfg = ChipConfig(num_cores=16, batch_size=4, sequence_length=128)
    prog = build_program_for_paradigm(
        cfg, LLAMA2_13B, seq_len=128, batch=4, stage="prefill"
    )
    graph = prog.build_graph(cfg.num_cores)
    stats = SimulationEngine(cfg).run(graph)
    assert stats.total_cycles > 0
    assert stats.num_events > 0


def test_compute_shift_faster_than_spmd_trend():
    """Trend check: compute-shift should have lower or equal NoC overhead vs SPMD."""
    spmd = ChipConfig(
        num_cores=16,
        batch_size=4,
        sequence_length=64,
        computation_paradigm=ComputationParadigm.SPMD,
    )
    shift = ChipConfig(
        num_cores=16,
        batch_size=4,
        sequence_length=64,
        computation_paradigm=ComputationParadigm.COMPUTE_SHIFT,
    )
    spmd_prog = build_program_for_paradigm(spmd, LLAMA2_13B, seq_len=64, batch=4)
    shift_prog = build_program_for_paradigm(shift, LLAMA2_13B, seq_len=64, batch=4)
    spmd_stats = SimulationEngine(spmd).run(spmd_prog.build_graph(16))
    shift_stats = SimulationEngine(shift).run(shift_prog.build_graph(16))
    assert shift_stats.total_cycles <= spmd_stats.total_cycles * 1.5
    assert spmd_stats.noc_overhead_cycles >= shift_stats.noc_overhead_cycles


def test_software_aware_lowers_row_conflict():
    uniform = ChipConfig(
        num_cores=16,
        batch_size=4,
        sequence_length=64,
        tensor_to_bank_mapping=TensorToBankMapping.UNIFORM,
        computation_paradigm=ComputationParadigm.COMPUTE_SHIFT,
    )
    aware = ChipConfig(
        num_cores=16,
        batch_size=4,
        sequence_length=64,
        tensor_to_bank_mapping=TensorToBankMapping.SOFTWARE_AWARE,
        computation_paradigm=ComputationParadigm.COMPUTE_SHIFT,
    )
    u_prog = build_program_for_paradigm(uniform, LLAMA2_13B, seq_len=64, batch=4, stage="decode")
    a_prog = build_program_for_paradigm(aware, LLAMA2_13B, seq_len=64, batch=4, stage="decode")
    u_stats = SimulationEngine(uniform).run(u_prog.build_graph(16))
    a_stats = SimulationEngine(aware).run(a_prog.build_graph(16))
    assert u_stats.dram_access_cycles > 0
    assert a_stats.row_conflict_overhead_cycles <= u_stats.row_conflict_overhead_cycles


def test_core_group_reduces_row_conflict():
    g1 = ChipConfig(
        num_cores=16,
        batch_size=4,
        sequence_length=64,
        core_group_size=1,
        computation_paradigm=ComputationParadigm.COMPUTE_SHIFT,
    )
    g8 = ChipConfig(
        num_cores=16,
        batch_size=4,
        sequence_length=64,
        core_group_size=8,
        computation_paradigm=ComputationParadigm.COMPUTE_SHIFT,
    )
    prog = build_program_for_paradigm(g1, LLAMA2_13B, seq_len=64, batch=4, stage="decode")
    graph = prog.build_graph(16)
    s1 = SimulationEngine(g1).run(graph)
    s8 = SimulationEngine(g8).run(graph)
    assert s8.row_conflict_overhead_cycles <= s1.row_conflict_overhead_cycles
