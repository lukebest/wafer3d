"""Compute paradigms: SPMD, Dataflow, Compute-shift (Figure 8)."""

from __future__ import annotations

from voxelsim.api.ops import OpTile, TensorPart, make_tensor_part, MemoryLocation
from voxelsim.api.program import Program
from voxelsim.api.collectives import all_reduce
from voxelsim.chip.config import ChipConfig, ComputationParadigm
from voxelsim.chip.mapping import MappingPlanner
from voxelsim.models.llm import LLMConfig


def build_spmd_layer(
    prog: Program,
    model: LLMConfig,
    core_ids: list[int],
    *,
    seq_len: int,
    batch: int,
    stage: str,
) -> None:
    """SPMD: independent tasks + separate reduction."""
    h = model.hidden_size
    m = batch * (1 if stage == "decode" else seq_len)
    partials = []
    for i, cid in enumerate(core_ids):
        tile_m = max(1, m // len(core_ids))
        inp = make_tensor_part(f"spmd_in_{i}", (tile_m, h), location=MemoryLocation.DRAM, bank_id=i)
        w = make_tensor_part(f"spmd_w_{i}", (h, h), location=MemoryLocation.DRAM, bank_id=i + 1)
        out = make_tensor_part(f"spmd_partial_{i}", (tile_m, h), location=MemoryLocation.SRAM, core_id=cid)
        tile = OpTile(
            op_name=f"spmd_matmul_{i}",
            op_type="matmul",
            inputs=[inp, w],
            outputs=[out],
            gemm_m=tile_m,
            gemm_n=h,
            gemm_k=h,
        )
        prog.compute(tile, core_id=cid)
        partials.append(out)

    reduced = make_tensor_part("spmd_reduced", (tile_m, h), location=MemoryLocation.SRAM, core_id=core_ids[0])
    all_reduce(prog, partials[0], reduced, core_ids)


def build_dataflow_layer(
    prog: Program,
    model: LLMConfig,
    pipeline_cores: list[int],
    *,
    microbatch_count: int = 1,
    seq_len: int,
    batch: int,
    stage: str = "prefill",
) -> None:
    """Dataflow: pipeline single operators across cores with microbatch overlap."""
    h = model.hidden_size
    ffn_n = model.ffn_size
    m = batch * (1 if stage == "decode" else seq_len)
    stages = min(2, len(pipeline_cores))
    cores = pipeline_cores[:stages]
    mb = max(1, m // max(1, len(cores)))

    for mb_i in range(microbatch_count):
        prev_out: TensorPart | None = None
        for stage_idx, cid in enumerate(cores):
            if stage_idx == 0:
                inp = make_tensor_part(
                    f"flow_in_{mb_i}",
                    (mb, h),
                    location=MemoryLocation.DRAM,
                    bank_id=mb_i % 16,
                )
                w = make_tensor_part(
                    f"flow_qkv_w_{mb_i}",
                    (h, h),
                    location=MemoryLocation.DRAM,
                    bank_id=(mb_i + 1) % 16,
                )
                out_n = h
                gemm_k = h
            elif stage_idx == 1:
                assert prev_out is not None
                inp = make_tensor_part(
                    f"flow_mid_{mb_i}",
                    prev_out.shape.dims,
                    location=MemoryLocation.SRAM,
                    core_id=cores[stage_idx - 1],
                )
                prog.copy_data(prev_out, inp)
                w = make_tensor_part(
                    f"flow_ffn1_w_{mb_i}",
                    (h, ffn_n // stages),
                    location=MemoryLocation.DRAM,
                    bank_id=(mb_i + 2) % 16,
                )
                out_n = ffn_n // stages
                gemm_k = h
            else:
                assert prev_out is not None
                inp = make_tensor_part(
                    f"flow_ffn_mid_{mb_i}",
                    prev_out.shape.dims,
                    location=MemoryLocation.SRAM,
                    core_id=cores[stage_idx - 1],
                )
                prog.copy_data(prev_out, inp)
                w = make_tensor_part(
                    f"flow_ffn2_w_{mb_i}",
                    (ffn_n // stages, h),
                    location=MemoryLocation.DRAM,
                    bank_id=(mb_i + 3) % 16,
                )
                out_n = h
                gemm_k = ffn_n // stages

            out = make_tensor_part(
                f"flow_out_{mb_i}_{stage_idx}",
                (mb, out_n),
                location=MemoryLocation.SRAM,
                core_id=cid,
            )
            tile = OpTile(
                op_name=f"flow_{stage_idx}_{mb_i}",
                op_type="matmul",
                inputs=[inp, w],
                outputs=[out],
                gemm_m=mb,
                gemm_n=out_n,
                gemm_k=gemm_k,
            )
            prog.compute(tile, core_id=cid)
            prev_out = out

    prog.sync(cores)


def build_compute_shift_layer(
    prog: Program,
    model: LLMConfig,
    ring_cores: list[int],
    *,
    seq_len: int,
    batch: int,
    stage: str,
) -> None:
    """Compute-shift: circular shift of shared tensor on a core ring."""
    h = model.hidden_size
    m = batch * (1 if stage == "decode" else seq_len)
    shared = make_tensor_part(
        "shared_tensor",
        (m, h),
        location=MemoryLocation.DRAM,
        bank_id=0,
    )
    n = len(ring_cores)
    for i, cid in enumerate(ring_cores):
        local = make_tensor_part(
            f"shift_local_{i}",
            (max(1, m // n), h),
            location=MemoryLocation.SRAM,
            core_id=cid,
        )
        prog.copy_data(shared, local)
        w = make_tensor_part(f"shift_w_{i}", (h, h), location=MemoryLocation.DRAM, bank_id=i)
        out = make_tensor_part(f"shift_out_{i}", local.shape.dims, location=MemoryLocation.SRAM, core_id=cid)
        tile = OpTile(
            op_name=f"shift_matmul_{i}",
            op_type="matmul",
            inputs=[local, w],
            outputs=[out],
            gemm_m=local.shape.dims[0],
            gemm_n=h,
            gemm_k=h,
        )
        prog.compute(tile, core_id=cid)
        # Circular shift to next core
        next_cid = ring_cores[(i + 1) % n]
        nxt = make_tensor_part(
            f"shift_next_{i}",
            local.shape.dims,
            location=MemoryLocation.SRAM,
            core_id=next_cid,
        )
        prog.copy_data(out, nxt)
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
        build_dataflow_layer(
            prog, model, cores[:4], seq_len=seq_len, batch=batch, stage=stage
        )
    else:
        build_compute_shift_layer(prog, model, cores[:8], seq_len=seq_len, batch=batch, stage=stage)
    return prog
