"""Collective communication helpers built from copy_data + compute."""

from __future__ import annotations

from voxelsim.api.ops import OpTile, TensorPart, make_tensor_part, MemoryLocation
from voxelsim.api.program import Program


def all_reduce(
    prog: Program,
    partial: TensorPart,
    output: TensorPart,
    core_ids: list[int],
    reduce_op: str = "sum",
) -> None:
    """AllReduce via ring of copy_data + local compute on each core."""
    n = len(core_ids)
    for i, cid in enumerate(core_ids):
        src_core = core_ids[(i - 1) % n]
        src = make_tensor_part(
            f"{partial.name}_from_{src_core}",
            partial.shape.dims,
            dtype=partial.dtype,
            location=MemoryLocation.SRAM,
            core_id=src_core,
        )
        dst = make_tensor_part(
            f"{partial.name}_core_{cid}",
            partial.shape.dims,
            dtype=partial.dtype,
            location=MemoryLocation.SRAM,
            core_id=cid,
        )
        prog.copy_data(src, dst)
        tile = OpTile(
            op_name=f"reduce_{reduce_op}",
            op_type="elementwise",
            inputs=[dst],
            outputs=[output],
        )
        prog.compute(tile, core_id=cid)
    prog.sync(core_ids)


def reduce_scatter(
    prog: Program,
    partials: list[TensorPart],
    outputs: list[TensorPart],
    core_ids: list[int],
) -> None:
    """Reduce-scatter: each partial goes to designated core."""
    for i, (partial, out) in enumerate(zip(partials, outputs)):
        dst_core = core_ids[i % len(core_ids)]
        dst = make_tensor_part(
            out.name,
            out.shape.dims,
            dtype=out.dtype,
            location=MemoryLocation.SRAM,
            core_id=dst_core,
        )
        prog.copy_data(partial, dst)
        tile = OpTile(
            op_name="reduce_scatter",
            op_type="elementwise",
            inputs=[dst],
            outputs=[out],
        )
        prog.compute(tile, core_id=dst_core)
    prog.sync(core_ids)


def all_gather(
    prog: Program,
    inputs: list[TensorPart],
    output: TensorPart,
    core_ids: list[int],
) -> None:
    """AllGather via copy_data to consecutive output tiles."""
    for i, inp in enumerate(inputs):
        dst_core = core_ids[i % len(core_ids)]
        dst = make_tensor_part(
            f"{output.name}_slice_{i}",
            inp.shape.dims,
            dtype=inp.dtype,
            location=MemoryLocation.SRAM,
            core_id=dst_core,
        )
        prog.copy_data(inp, dst)
    prog.sync(core_ids)


def broadcast(
    prog: Program,
    src: TensorPart,
    replicas: list[TensorPart],
    root_core: int,
) -> None:
    for rep in replicas:
        prog.copy_data(src, rep)
    prog.sync([root_core] + [r.core_id for r in replicas if r.core_id is not None])
