"""Chip configuration dataclasses (Table 2/3/4 defaults from Voxel paper)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ComputationParadigm(str, Enum):
    SPMD = "spmd"
    DATAFLOW = "dataflow"
    COMPUTE_SHIFT = "compute_shift"


class TileToCoreMapping(str, Enum):
    SEQUENTIAL = "sequential"
    DIMENSION_ORDERED = "dimension_ordered"


class TensorToBankMapping(str, Enum):
    UNIFORM = "uniform"
    INTERLEAVE_SIZE = "interleave_size"
    SOFTWARE_AWARE = "software_aware"


class NoCTopology(str, Enum):
    MESH_2D = "mesh_2d"
    TORUS_2D = "torus_2d"
    ALL_TO_ALL = "all_to_all"


class NoCBackend(str, Enum):
    ANALYTIC = "analytic"
    BOOKSIM = "booksim"


class Precision(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass
class DramTiming:
    tCL: int = 14
    tRCD: int = 14
    tRP: int = 14
    tRAS: int = 34


@dataclass
class DramConfig:
    layer_count: int = 8
    banks_per_layer: int = 16
    capacity_gb: float = 192.0
    interface_bytes: int = 128
    timing: DramTiming = field(default_factory=DramTiming)
    burst_length: str = "auto"

    @property
    def total_banks(self) -> int:
        return self.layer_count * self.banks_per_layer


@dataclass
class AreaConfig:
    systolic_arrays_mm2: float = 260.0
    srams_mm2: float = 433.0
    tsvs_mm2: float = 18.4
    other_mm2: float = 91.2
    per_die_limit_mm2: float = 850.0

    @property
    def total_mm2(self) -> float:
        return (
            self.systolic_arrays_mm2
            + self.srams_mm2
            + self.tsvs_mm2
            + self.other_mm2
        )


@dataclass
class NoCConfig:
    backend: NoCBackend = NoCBackend.ANALYTIC
    routing: str = "dimension_order"
    link_bandwidth_bytes_per_cycle: int = 32


@dataclass
class BackendPaths:
    ramulator_bin: Path = Path("third_party/ramulator2/build/ramulator2")
    booksim_bin: Path = Path("third_party/booksim2/src/booksim")
    dsent_bin: Path = Path("third_party/dsent/dsent")


@dataclass
class ChipConfig:
    """Full configurable design space for the 3D AI chip simulator."""

    # Software (Table 2)
    computation_paradigm: ComputationParadigm = ComputationParadigm.COMPUTE_SHIFT
    tile_to_core_mapping: TileToCoreMapping = TileToCoreMapping.DIMENSION_ORDERED
    tensor_to_bank_mapping: TensorToBankMapping = TensorToBankMapping.SOFTWARE_AWARE

    # Hardware (Table 2)
    noc_topology: NoCTopology = NoCTopology.MESH_2D
    dram_bandwidth_tbps: float = 12.0
    num_cores: int = 256
    systolic_array_size: int = 32
    core_group_size: int = 8
    per_core_sram_kb: int = 2048

    # System (Table 3)
    frequency_ghz: float = 1.6
    power_density_limit_w_per_mm2: float = 0.7
    batch_size: int = 32
    sequence_length: int = 2048
    precision: Precision = Precision.BF16

    dram: DramConfig = field(default_factory=DramConfig)
    noc: NoCConfig = field(default_factory=NoCConfig)
    area: AreaConfig = field(default_factory=AreaConfig)
    backends: BackendPaths = field(default_factory=BackendPaths)

    @property
    def frequency_hz(self) -> float:
        return self.frequency_ghz * 1e9

    @property
    def per_core_sram_bytes(self) -> int:
        return self.per_core_sram_kb * 1024

    @property
    def bytes_per_element(self) -> int:
        if self.precision in (Precision.BF16, Precision.FP16):
            return 2
        return 4

    @property
    def grid_side(self) -> int:
        import math

        side = int(math.isqrt(self.num_cores))
        if side * side != self.num_cores:
            raise ValueError(f"num_cores={self.num_cores} must be a perfect square")
        return side

    @property
    def dram_bandwidth_bytes_per_cycle(self) -> float:
        return (self.dram_bandwidth_tbps * 1e12) / self.frequency_hz

    @property
    def peak_flops_per_core(self) -> float:
        sa = self.systolic_array_size
        return 2.0 * sa * sa * self.frequency_hz

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChipConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ChipConfig":
        sw = raw.get("software", {})
        hw = raw.get("hardware", {})
        dram_raw = raw.get("dram", {})
        timing_raw = dram_raw.get("timing", {})
        noc_raw = raw.get("noc", {})
        area_raw = raw.get("area_mm2", {})
        sys_raw = raw.get("system", {})
        be_raw = raw.get("backends", {})

        dram = DramConfig(
            layer_count=dram_raw.get("layer_count", 8),
            banks_per_layer=dram_raw.get("banks_per_layer", 16),
            capacity_gb=dram_raw.get("capacity_gb", 192),
            interface_bytes=dram_raw.get("interface_bytes", 128),
            timing=DramTiming(
                tCL=timing_raw.get("tCL", 14),
                tRCD=timing_raw.get("tRCD", 14),
                tRP=timing_raw.get("tRP", 14),
                tRAS=timing_raw.get("tRAS", 34),
            ),
            burst_length=dram_raw.get("burst_length", "auto"),
        )

        noc = NoCConfig(
            backend=NoCBackend(noc_raw.get("backend", "analytic")),
            routing=noc_raw.get("routing", "dimension_order"),
            link_bandwidth_bytes_per_cycle=hw.get(
                "noc_link_bandwidth_bytes_per_cycle",
                noc_raw.get("link_bandwidth_bytes_per_cycle", 32),
            ),
        )

        area = AreaConfig(
            systolic_arrays_mm2=area_raw.get("systolic_arrays", 260.0),
            srams_mm2=area_raw.get("srams", 433.0),
            tsvs_mm2=area_raw.get("tsvs", 18.4),
            other_mm2=area_raw.get("other", 91.2),
            per_die_limit_mm2=area_raw.get("per_die_limit", 850.0),
        )

        backends = BackendPaths(
            ramulator_bin=Path(be_raw.get("ramulator_bin", BackendPaths.ramulator_bin)),
            booksim_bin=Path(be_raw.get("booksim_bin", BackendPaths.booksim_bin)),
            dsent_bin=Path(be_raw.get("dsent_bin", BackendPaths.dsent_bin)),
        )

        return cls(
            computation_paradigm=ComputationParadigm(
                sw.get("computation_paradigm", "compute_shift")
            ),
            tile_to_core_mapping=TileToCoreMapping(
                sw.get("tile_to_core_mapping", "dimension_ordered")
            ),
            tensor_to_bank_mapping=TensorToBankMapping(
                sw.get("tensor_to_bank_mapping", "software_aware")
            ),
            noc_topology=NoCTopology(hw.get("noc_topology", "mesh_2d")),
            dram_bandwidth_tbps=hw.get("dram_bandwidth_tbps", 12.0),
            num_cores=hw.get("num_cores", 256),
            systolic_array_size=hw.get("systolic_array_size", 32),
            core_group_size=hw.get("core_group_size", 8),
            per_core_sram_kb=hw.get("per_core_sram_kb", 2048),
            frequency_ghz=sys_raw.get("frequency_ghz", 1.6),
            power_density_limit_w_per_mm2=sys_raw.get(
                "power_density_limit_w_per_mm2", 0.7
            ),
            batch_size=sys_raw.get("batch_size", 32),
            sequence_length=sys_raw.get("sequence_length", 2048),
            precision=Precision(sys_raw.get("precision", "bf16")),
            dram=dram,
            noc=noc,
            area=area,
            backends=backends,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "software": {
                "computation_paradigm": self.computation_paradigm.value,
                "tile_to_core_mapping": self.tile_to_core_mapping.value,
                "tensor_to_bank_mapping": self.tensor_to_bank_mapping.value,
            },
            "hardware": {
                "noc_topology": self.noc_topology.value,
                "dram_bandwidth_tbps": self.dram_bandwidth_tbps,
                "num_cores": self.num_cores,
                "systolic_array_size": self.systolic_array_size,
                "core_group_size": self.core_group_size,
                "noc_link_bandwidth_bytes_per_cycle": self.noc.link_bandwidth_bytes_per_cycle,
                "per_core_sram_kb": self.per_core_sram_kb,
            },
            "dram": {
                "layer_count": self.dram.layer_count,
                "banks_per_layer": self.dram.banks_per_layer,
                "capacity_gb": self.dram.capacity_gb,
                "interface_bytes": self.dram.interface_bytes,
                "timing": {
                    "tCL": self.dram.timing.tCL,
                    "tRCD": self.dram.timing.tRCD,
                    "tRP": self.dram.timing.tRP,
                    "tRAS": self.dram.timing.tRAS,
                },
            },
            "system": {
                "frequency_ghz": self.frequency_ghz,
                "power_density_limit_w_per_mm2": self.power_density_limit_w_per_mm2,
                "batch_size": self.batch_size,
                "sequence_length": self.sequence_length,
                "precision": self.precision.value,
            },
            "noc": {
                "backend": self.noc.backend.value,
                "routing": self.noc.routing,
            },
        }


def default_config() -> ChipConfig:
    repo = Path(__file__).resolve().parents[2]
    cfg_path = repo / "configs" / "default.yaml"
    if cfg_path.exists():
        return ChipConfig.from_yaml(cfg_path)
    return ChipConfig()
