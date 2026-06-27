"""Tensor and operator tile data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryLocation(str, Enum):
    DRAM = "dram"
    SRAM = "sram"


@dataclass(frozen=True)
class TensorShape:
    dims: tuple[int, ...]

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class TensorPart:
    """A shard of a tensor at a specific memory location."""

    name: str
    dtype: str
    shape: TensorShape
    location: MemoryLocation
    bank_id: int | None = None
    core_id: int | None = None
    base_addr: int = 0

    def byte_size(self, elem_bytes: int = 2) -> int:
        return self.shape.num_elements * elem_bytes


@dataclass
class OpTile:
    """Partitioned tile of a tensor operator."""

    op_name: str
    op_type: str  # matmul, elementwise, fused, etc.
    inputs: list[TensorPart]
    outputs: list[TensorPart]
    gemm_m: int = 0
    gemm_n: int = 0
    gemm_k: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def gemm_dims(self) -> tuple[int, int, int]:
        if self.gemm_m and self.gemm_n and self.gemm_k:
            return self.gemm_m, self.gemm_n, self.gemm_k
        if self.op_type == "matmul" and len(self.inputs) >= 2:
            a, b = self.inputs[0], self.inputs[1]
            m = a.shape.dims[0] if a.shape.dims else 1
            k = a.shape.dims[1] if len(a.shape.dims) > 1 else 1
            n = b.shape.dims[1] if len(b.shape.dims) > 1 else k
            return m, n, k
        return 1, 1, 1


def make_tensor_part(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: str = "bf16",
    location: MemoryLocation = MemoryLocation.DRAM,
    bank_id: int | None = None,
    core_id: int | None = None,
    base_addr: int = 0,
) -> TensorPart:
    return TensorPart(
        name=name,
        dtype=dtype,
        shape=TensorShape(shape),
        location=location,
        bank_id=bank_id,
        core_id=core_id,
        base_addr=base_addr,
    )
