"""LLM model definitions and operator graph builders."""

from __future__ import annotations

from dataclasses import dataclass

from voxelsim.api.ops import OpTile, make_tensor_part, MemoryLocation
from voxelsim.api.program import Program


@dataclass
class LLMConfig:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    ffn_size: int
    vocab_size: int = 32000


# Paper evaluation models (§9.1)
LLAMA2_13B = LLMConfig("Llama2-13B", 40, 5120, 40, 13824)
GEMMA2_27B = LLMConfig("Gemma2-27B", 46, 4608, 32, 36864)
OPT_30B = LLMConfig("Opt-30B", 48, 7168, 56, 28672)
LLAMA3_70B = LLMConfig("Llama3-70B", 80, 8192, 64, 28672)
DIT_XL = LLMConfig("DiT-XL", 28, 1152, 16, 4608)

MODELS = {
    "llama2-13b": LLAMA2_13B,
    "gemma2-27b": GEMMA2_27B,
    "opt-30b": OPT_30B,
    "llama3-70b": LLAMA3_70B,
    "dit-xl": DIT_XL,
}


def build_transformer_layer_program(
    prog: Program,
    model: LLMConfig,
    *,
    seq_len: int,
    batch: int,
    stage: str = "decode",
    core_id: int = 0,
    num_tiles: int = 4,
) -> None:
    """Build one transformer block as matmul tiles (prefill or decode)."""
    h = model.hidden_size
    tokens = 1 if stage == "decode" else seq_len
    m = batch * tokens
    k = h
    n = h

    tile_m = max(1, m // num_tiles)
    for t in range(num_tiles):
        cid = (core_id + t) % max(1, num_tiles)
        inp = make_tensor_part(
            f"qkv_in_{t}",
            (tile_m, k),
            location=MemoryLocation.DRAM,
            bank_id=t,
        )
        out = make_tensor_part(
            f"qkv_out_{t}",
            (tile_m, n),
            location=MemoryLocation.SRAM,
            core_id=cid,
        )
        weight = make_tensor_part(
            f"qkv_w_{t}",
            (k, n),
            location=MemoryLocation.DRAM,
            bank_id=(t + 1) % 16,
        )
        tile = OpTile(
            op_name=f"qkv_{stage}_{t}",
            op_type="matmul",
            inputs=[inp, weight],
            outputs=[out],
            gemm_m=tile_m,
            gemm_n=n,
            gemm_k=k,
        )
        prog.compute(tile, core_id=cid)

    # FFN
    ffn_n = model.ffn_size
    for t in range(num_tiles):
        cid = (core_id + t) % max(1, num_tiles)
        inp = make_tensor_part(f"ffn_in_{t}", (tile_m, h), location=MemoryLocation.SRAM, core_id=cid)
        w = make_tensor_part(f"ffn_w_{t}", (h, ffn_n // num_tiles), location=MemoryLocation.DRAM, bank_id=t + 8)
        out = make_tensor_part(f"ffn_out_{t}", (tile_m, ffn_n // num_tiles), location=MemoryLocation.SRAM, core_id=cid)
        tile = OpTile(
            op_name=f"ffn_{stage}_{t}",
            op_type="matmul",
            inputs=[inp, w],
            outputs=[out],
            gemm_m=tile_m,
            gemm_n=ffn_n // num_tiles,
            gemm_k=h,
        )
        prog.compute(tile, core_id=cid)

    prog.sync()


def build_full_model_program(
    model: LLMConfig,
    *,
    seq_len: int = 2048,
    batch: int = 32,
    stage: str = "prefill",
    layers: int | None = None,
    num_cores: int = 256,
) -> Program:
    prog = Program()
    n_layers = layers if layers is not None else min(4, model.num_layers)  # sample for speed
    num_tiles = min(16, num_cores)
    for layer in range(n_layers):
        build_transformer_layer_program(
            prog,
            model,
            seq_len=seq_len,
            batch=batch,
            stage=stage,
            core_id=layer % num_cores,
            num_tiles=num_tiles,
        )
    return prog
