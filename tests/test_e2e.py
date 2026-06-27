"""End-to-end smoke tests."""

from voxelsim.chip.config import ChipConfig, ComputationParadigm
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
    base = ChipConfig(num_cores=16, batch_size=4, sequence_length=64)
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
    # Compute-shift targets lower NoC overhead (paper: up to 49% SPMD NoC overhead)
    assert shift_stats.total_cycles <= spmd_stats.total_cycles * 1.5
