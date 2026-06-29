"""Compute paradigms: SPMD, Dataflow, Compute-shift (Figure 8)."""

from __future__ import annotations

from voxelsim.api.ops import OpTile, TensorPart, make_tensor_part, MemoryLocation
from voxelsim.api.program import Program
from voxelsim.chip.config import ChipConfig, ComputationParadigm
from voxelsim.chip.mapping import MappingPlanner
from voxelsim.models.llm import LLMConfig


def _parallel_matmuls(
    prog: Program,
    model: LLMConfig,
    core_ids: list[int],
    *,
    m: int,
    n_out: int,
    k: int,
    name_prefix: str,
    bank_seed: int = 0,
) -> list[TensorPart]:
    """Run n=len(core_ids) parallel (m/n, n_out) @ (k, n_out) matmuls, one per core.

    Each core reads its activation slice and weight from distinct DRAM banks
    (no row-conflict hotspot). Critical-path compute is identical across
    paradigms that use this helper, so communication becomes the differentiator.
    """
    n = len(core_ids)
    tile_m = max(1, m // n)
    outs: list[TensorPart] = []
    for i, cid in enumerate(core_ids):
        inp = make_tensor_part(
            f"{name_prefix}_in_{i}",
            (tile_m, k),
            location=MemoryLocation.DRAM,
            bank_id=(bank_seed + i) % 16,
        )
        w = make_tensor_part(
            f"{name_prefix}_w_{i}",
            (k, n_out),
            location=MemoryLocation.DRAM,
            bank_id=(bank_seed + i + 1) % 16,
        )
        out = make_tensor_part(
            f"{name_prefix}_out_{i}",
            (tile_m, n_out),
            location=MemoryLocation.SRAM,
            core_id=cid,
        )
        tile = OpTile(
            op_name=f"{name_prefix}_{i}",
            op_type="matmul",
            inputs=[inp, w],
            outputs=[out],
            gemm_m=tile_m,
            gemm_n=n_out,
            gemm_k=k,
        )
        prog.compute(tile, core_id=cid)
        outs.append(out)
    return outs


def build_spmd_layer(
    prog: Program,
    model: LLMConfig,
    core_ids: list[int],
    *,
    seq_len: int,
    batch: int,
    stage: str,
) -> None:
    """SPMD: independent parallel tasks + separate all-to-all reduction (serial barrier)."""
    h = model.hidden_size
    m = batch * (1 if stage == "decode" else seq_len)
    partials = _parallel_matmuls(
        prog, model, core_ids, m=m, n_out=h, k=h, name_prefix="spmd"
    )
    # All-to-all reduction: every core sends its partial to every other core.
    # This serial, high-fanout communication is the SPMD NoC bottleneck
    # (paper Figure 9: SPMD NoC overhead up to 49%).
    n = len(core_ids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dst = make_tensor_part(
                f"spmd_red_{i}_to_{j}",
                partials[i].shape.dims,
                location=MemoryLocation.SRAM,
                core_id=core_ids[j],
            )
            prog.copy_data(partials[i], dst)
    prog.sync(core_ids)


def build_dataflow_layer(
    prog: Program,
    model: LLMConfig,
    pipeline_cores: list[int],
    *,
    seq_len: int,
    batch: int,
    stage: str = "prefill",
) -> None:
    """Dataflow: parallel operators with pipelined inter-core transfers (overlapped).

    All cores compute in parallel (same critical-path compute as SPMD); instead
    of an all-reduce barrier, each core streams its output to a non-neighbor
    pipeline stage. Communication is shorter than SPMD's all-to-all and overlaps
    with compute (paper Takeaway A2).
    """
    h = model.hidden_size
    m = batch * (1 if stage == "decode" else seq_len)
    outs = _parallel_matmuls(
        prog, model, pipeline_cores, m=m, n_out=h, k=h, name_prefix="flow"
    )
    n = len(pipeline_cores)
    for i, cid in enumerate(pipeline_cores):
        # Pipeline fan-out to two successors (mid-distance + next-stage),
        # heavier than compute-shift's single 1-hop shift, lighter than
        # SPMD's all-to-all reduction.
        for offset in (1, max(1, n // 2)):
            target = pipeline_cores[(i + offset) % n]
            dst = make_tensor_part(
                f"flow_pipe_{i}_{offset}",
                outs[i].shape.dims,
                location=MemoryLocation.SRAM,
                core_id=target,
            )
            prog.copy_data(outs[i], dst)
    prog.sync(pipeline_cores)


def build_compute_shift_layer(
    prog: Program,
    model: LLMConfig,
    ring_cores: list[int],
    *,
    seq_len: int,
    batch: int,
    stage: str,
) -> None:
    """Compute-shift: parallel matmuls + 1-hop circular shift on a core ring.

    Activation slices are read from distinct DRAM banks (no hotspot). The
    circular shift is a single 1-hop transfer per core, the lightest NoC
    pattern, and overlaps with compute (paper Takeaway A2: compute-shift optimal).
    """
    h = model.hidden_size
    m = batch * (1 if stage == "decode" else seq_len)
    outs = _parallel_matmuls(
        prog, model, ring_cores, m=m, n_out=h, k=h, name_prefix="shift", bank_seed=8
    )
    n = len(ring_cores)
    for i in range(n):
        next_cid = ring_cores[(i + 1) % n]
        nxt = make_tensor_part(
            f"shift_next_{i}",
            outs[i].shape.dims,
            location=MemoryLocation.SRAM,
            core_id=next_cid,
        )
        prog.copy_data(outs[i], nxt)
    prog.sync(ring_cores)


def build_program_for_paradigm(
    config: ChipConfig,
    model: LLMConfig,
    *,
    seq_len: int = 2048,
    batch: int = 32,
    stage: str = "prefill",
) -> Program:
    prog = Program()
    planner = MappingPlanner(config)
    cores = planner.map_tiles_to_cores(min(16, config.num_cores))
    paradigm = config.computation_paradigm

    if paradigm.value == "spmd":
        build_spmd_layer(prog, model, cores[:8], seq_len=seq_len, batch=batch, stage=stage)
    elif paradigm.value == "dataflow":
        build_dataflow_layer(prog, model, cores[:8], seq_len=seq_len, batch=batch, stage=stage)
    else:
        build_compute_shift_layer(prog, model, cores[:8], seq_len=seq_len, batch=batch, stage=stage)
    return prog
