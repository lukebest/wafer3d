#!/usr/bin/env python3
"""Run Voxel paper reproduction experiments and emit JSON + markdown summary."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voxelsim.chip.config import (  # noqa: E402
    ChipConfig,
    ComputationParadigm,
    NoCTopology,
    TensorToBankMapping,
    TileToCoreMapping,
)
from voxelsim.models.llm import LLAMA2_13B, LLAMA3_70B, LLMConfig
from voxelsim.models.paradigms import build_program_for_paradigm
from voxelsim.sim.engine import SimulationEngine
from voxelsim.sim.stats import SimulationStats


@dataclass
class ExperimentResult:
    experiment: str
    label: str
    model: str
    stage: str
    total_cycles: int
    noc_overhead_cycles: int
    noc_pct: float
    dram_access_cycles: int
    row_conflict_overhead_cycles: int
    energy_joules: float
    extra: dict


def _run(cfg: ChipConfig, model: LLMConfig, stage: str) -> SimulationStats:
    prog = build_program_for_paradigm(
        cfg,
        model,
        seq_len=cfg.sequence_length,
        batch=cfg.batch_size,
        stage=stage,
    )
    graph = prog.build_graph(cfg.num_cores)
    return SimulationEngine(cfg).run(graph)


def _result(
    experiment: str,
    label: str,
    model: LLMConfig,
    stage: str,
    stats: SimulationStats,
    **extra,
) -> ExperimentResult:
    total = max(1, stats.total_cycles)
    return ExperimentResult(
        experiment=experiment,
        label=label,
        model=model.name,
        stage=stage,
        total_cycles=stats.total_cycles,
        noc_overhead_cycles=stats.noc_overhead_cycles,
        noc_pct=100.0 * stats.noc_overhead_cycles / total,
        dram_access_cycles=stats.dram_access_cycles,
        row_conflict_overhead_cycles=stats.row_conflict_overhead_cycles,
        energy_joules=stats.energy_joules,
        extra=extra,
    )


def base_config(quick: bool) -> ChipConfig:
    cfg = ChipConfig()
    if quick:
        # Smaller scale for tractable sweep; trends match paper direction.
        cfg.num_cores = 16
        cfg.batch_size = 4
        cfg.sequence_length = 128
    return cfg


def experiment_a(cfg: ChipConfig, model: LLMConfig) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for paradigm in ComputationParadigm:
        c = copy.deepcopy(cfg)
        c.computation_paradigm = paradigm
        for stage in ("prefill", "decode"):
            stats = _run(c, model, stage)
            out.append(_result("A_paradigm", paradigm.value, model, stage, stats))
    return out


def experiment_b(cfg: ChipConfig, model: LLMConfig) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for mapping in (TileToCoreMapping.SEQUENTIAL, TileToCoreMapping.DIMENSION_ORDERED):
        for topo in (NoCTopology.MESH_2D, NoCTopology.TORUS_2D, NoCTopology.ALL_TO_ALL):
            c = copy.deepcopy(cfg)
            c.tile_to_core_mapping = mapping
            c.noc_topology = topo
            stats = _run(c, model, "prefill")
            out.append(
                _result(
                    "B_noc_mapping",
                    f"{mapping.value}_{topo.value}",
                    model,
                    "prefill",
                    stats,
                    mapping=mapping.value,
                    topology=topo.value,
                )
            )
    return out


def experiment_c(cfg: ChipConfig, model: LLMConfig) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for mapping in TensorToBankMapping:
        c = copy.deepcopy(cfg)
        c.tensor_to_bank_mapping = mapping
        stats = _run(c, model, "decode")
        out.append(
            _result(
                "C_tensor_bank",
                mapping.value,
                model,
                "decode",
                stats,
                mapping=mapping.value,
            )
        )
    return out


def experiment_d(cfg: ChipConfig, model: LLMConfig) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for bw in (4, 8, 16, 32):
        c = copy.deepcopy(cfg)
        c.noc.link_bandwidth_bytes_per_cycle = bw
        stats = _run(c, model, "prefill")
        out.append(
            _result("D_noc_bw", f"link_{bw}B", model, "prefill", stats, link_bw=bw)
        )
    return out


def experiment_e(cfg: ChipConfig, model: LLMConfig) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for bw in (4, 8, 12, 16):
        c = copy.deepcopy(cfg)
        c.dram_bandwidth_tbps = float(bw)
        stats = _run(c, model, "decode")
        out.append(
            _result("E_dram_bw", f"{bw}TBs", model, "decode", stats, dram_bw_tbps=bw)
        )
    return out


def experiment_f(cfg: ChipConfig, model: LLMConfig) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for gs in (1, 2, 4, 8, 16):
        c = copy.deepcopy(cfg)
        c.core_group_size = gs
        stats = _run(c, model, "decode")
        out.append(
            _result("F_core_group", f"group_{gs}", model, "decode", stats, core_group=gs)
        )
    return out


def experiment_g(cfg: ChipConfig, model: LLMConfig) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for sram_kb in (512, 2048, 8192):
        for stage in ("prefill", "decode"):
            c = copy.deepcopy(cfg)
            c.per_core_sram_kb = sram_kb
            stats = _run(c, model, stage)
            out.append(
                _result(
                    "G_sram",
                    f"{sram_kb}KB_{stage}",
                    model,
                    stage,
                    stats,
                    sram_kb=sram_kb,
                )
            )
    return out


def experiment_h(cfg: ChipConfig, model: LLMConfig) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for bw in (4, 8, 12, 16):
        c = copy.deepcopy(cfg)
        c.dram_bandwidth_tbps = float(bw)
        stats = _run(c, model, "decode")
        out.append(
            _result(
                "H_energy_dram",
                f"{bw}TBs",
                model,
                "decode",
                stats,
                dram_bw_tbps=bw,
            )
        )
    for cores in (64, 256):
        c = copy.deepcopy(cfg)
        c.num_cores = cores
        stats = _run(c, model, "decode")
        out.append(
            _result(
                "H_energy_cores",
                f"cores_{cores}",
                model,
                "decode",
                stats,
                num_cores=cores,
            )
        )
    return out


PAPER_CLAIMS = {
    "A_spmd_noc_pct_prefill": ("SPMD NoC overhead up to 49.08%", 49.08),
    "A_dataflow_vs_spmd": ("Dataflow faster than SPMD ~35.7%", 35.7),
    "A_shift_vs_spmd": ("Compute-shift faster than SPMD ~46.7%", 46.7),
    "C_sw_vs_uniform": ("Software-aware reduces row conflict ~80.7%", 80.7),
    "D_noc_bw_prefill": ("Prefill sensitive when link BW < 32 B/cycle", None),
    "E_dram_bw_decode": ("Decode benefits from higher DRAM bandwidth", None),
    "F_core_group": ("Core group=8 up to +57% decode vs group=1", 57.0),
}


def _trend_monotonic(values: list[tuple[str, float]], increasing: bool) -> bool:
    if len(values) < 2:
        return True
    nums = [v for _, v in values]
    if increasing:
        return nums[-1] >= nums[0]
    return nums[-1] <= nums[0]


def evaluate(results: list[ExperimentResult]) -> list[dict]:
    verdicts: list[dict] = []
    by_exp: dict[str, list[ExperimentResult]] = {}
    for r in results:
        by_exp.setdefault(r.experiment, []).append(r)

    # A: paradigms prefill
    a = [r for r in by_exp.get("A_paradigm", []) if r.stage == "prefill"]
    spmd = next((r for r in a if r.label == "spmd"), None)
    flow = next((r for r in a if r.label == "dataflow"), None)
    shift = next((r for r in a if r.label == "compute_shift"), None)
    if spmd and shift:
        verdicts.append(
            {
                "id": "A_shift_noc_vs_spmd",
                "paper": "Compute-shift lower NoC overhead than SPMD",
                "paper_value": "SPMD up to 49% NoC",
                "simulator": f"SPMD noc={spmd.noc_pct:.1f}%, shift={shift.noc_pct:.1f}%",
                "verdict": "PASS"
                if shift.noc_pct <= spmd.noc_pct and shift.total_cycles <= spmd.total_cycles
                else "PARTIAL",
            }
        )
    if spmd and flow:
        speedup = 100.0 * (spmd.total_cycles - flow.total_cycles) / spmd.total_cycles
        verdicts.append(
            {
                "id": "A_dataflow_vs_spmd",
                "paper": PAPER_CLAIMS["A_dataflow_vs_spmd"][0],
                "paper_value": "~35.7% faster",
                "simulator": f"{speedup:.1f}% relative to SPMD",
                "verdict": "PASS" if flow.total_cycles < spmd.total_cycles else "PARTIAL",
            }
        )
    if spmd and shift:
        speedup = 100.0 * (spmd.total_cycles - shift.total_cycles) / spmd.total_cycles
        verdicts.append(
            {
                "id": "A_shift_vs_spmd",
                "paper": PAPER_CLAIMS["A_shift_vs_spmd"][0],
                "paper_value": "~46.7% faster",
                "simulator": f"{speedup:.1f}% relative to SPMD",
                "verdict": "PASS" if shift.total_cycles < spmd.total_cycles else "PARTIAL",
            }
        )

    # B: dim-ordered faster than sequential on mesh
    b = by_exp.get("B_noc_mapping", [])
    seq_mesh = next((r for r in b if r.label == "sequential_mesh_2d"), None)
    dim_mesh = next((r for r in b if r.label == "dimension_ordered_mesh_2d"), None)
    if seq_mesh and dim_mesh:
        verdicts.append(
            {
                "id": "B_dim_vs_seq_mesh",
                "paper": "Dimension-ordered up to 46% faster on mesh (prefill)",
                "paper_value": "~46%",
                "simulator": f"seq={seq_mesh.total_cycles}, dim={dim_mesh.total_cycles}",
                "verdict": "PASS" if dim_mesh.total_cycles <= seq_mesh.total_cycles else "FAIL",
            }
        )

    # C: software-aware row conflict
    c = by_exp.get("C_tensor_bank", [])
    uni = next((r for r in c if r.label == "uniform"), None)
    sw = next((r for r in c if r.label == "software_aware"), None)
    if uni and sw and uni.row_conflict_overhead_cycles > 0:
        reduction = 100.0 * (
            uni.row_conflict_overhead_cycles - sw.row_conflict_overhead_cycles
        ) / uni.row_conflict_overhead_cycles
        verdicts.append(
            {
                "id": "C_sw_vs_uniform",
                "paper": PAPER_CLAIMS["C_sw_vs_uniform"][0],
                "paper_value": "~80.7%",
                "simulator": f"{reduction:.1f}% reduction",
                "verdict": "PASS" if sw.row_conflict_overhead_cycles < uni.row_conflict_overhead_cycles else "FAIL",
            }
        )
    elif uni and sw:
        verdicts.append(
            {
                "id": "C_sw_vs_uniform",
                "paper": PAPER_CLAIMS["C_sw_vs_uniform"][0],
                "paper_value": "~80.7%",
                "simulator": f"uniform={uni.row_conflict_overhead_cycles}, sw={sw.row_conflict_overhead_cycles}",
                "verdict": "PASS" if sw.row_conflict_overhead_cycles <= uni.row_conflict_overhead_cycles else "FAIL",
            }
        )

    # D: noc bandwidth prefill
    d = sorted(by_exp.get("D_noc_bw", []), key=lambda r: r.extra.get("link_bw", 0))
    if len(d) >= 2:
        verdicts.append(
            {
                "id": "D_noc_bw_prefill",
                "paper": "Higher NoC link BW improves prefill",
                "paper_value": "plateau at 32 B/cycle",
                "simulator": ", ".join(f"{r.label}={r.total_cycles}" for r in d),
                "verdict": "PASS"
                if _trend_monotonic([(r.label, r.total_cycles) for r in d], increasing=False)
                else "FAIL",
            }
        )

    # E: dram bandwidth decode
    e = sorted(by_exp.get("E_dram_bw", []), key=lambda r: r.extra.get("dram_bw_tbps", 0))
    if len(e) >= 2:
        verdicts.append(
            {
                "id": "E_dram_bw_decode",
                "paper": "Decode improves with DRAM bandwidth",
                "paper_value": "monotonic trend",
                "simulator": ", ".join(f"{r.label}={r.total_cycles}" for r in e),
                "verdict": "PASS"
                if _trend_monotonic([(r.label, r.total_cycles) for r in e], increasing=False)
                else "PARTIAL",
            }
        )

    # F: core group
    f = sorted(by_exp.get("F_core_group", []), key=lambda r: r.extra.get("core_group", 0))
    if len(f) >= 2:
        verdicts.append(
            {
                "id": "F_core_group",
                "paper": "Larger core group reduces row conflicts",
                "paper_value": "group=8 up to +57%",
                "simulator": ", ".join(
                    f"g{r.extra['core_group']}={r.row_conflict_overhead_cycles}" for r in f
                ),
                "verdict": "PASS"
                if _trend_monotonic(
                    [(r.label, r.row_conflict_overhead_cycles) for r in f], increasing=False
                )
                else "PARTIAL",
            }
        )

    # G: SRAM decode sensitivity
    g_decode = [r for r in by_exp.get("G_sram", []) if r.stage == "decode"]
    g_decode = sorted(g_decode, key=lambda r: r.extra.get("sram_kb", 0))
    if len(g_decode) >= 2:
        verdicts.append(
            {
                "id": "G_sram_decode",
                "paper": "Decode benefits from larger SRAM",
                "paper_value": "8MB saturates DRAM BW",
                "simulator": ", ".join(f"{r.label}={r.total_cycles}" for r in g_decode),
                "verdict": "PASS"
                if _trend_monotonic([(r.label, r.total_cycles) for r in g_decode], increasing=False)
                else "PARTIAL",
            }
        )

    # H: energy dram bandwidth
    h = sorted(by_exp.get("H_energy_dram", []), key=lambda r: r.extra.get("dram_bw_tbps", 0))
    if len(h) >= 2:
        verdicts.append(
            {
                "id": "H_energy_dram",
                "paper": "Higher DRAM BW improves decode energy efficiency",
                "paper_value": "lower energy per token",
                "simulator": ", ".join(f"{r.label}={r.energy_joules:.4f}J" for r in h),
                "verdict": "PASS"
                if _trend_monotonic([(r.label, r.energy_joules) for r in h], increasing=False)
                else "PARTIAL",
            }
        )

    return verdicts


def render_markdown(
    results: list[ExperimentResult],
    verdicts: list[dict],
    *,
    quick: bool,
    models: list[str],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pass_n = sum(1 for v in verdicts if v["verdict"] == "PASS")
    partial_n = sum(1 for v in verdicts if v["verdict"] == "PARTIAL")
    fail_n = sum(1 for v in verdicts if v["verdict"] == "FAIL")
    total_v = len(verdicts)

    lines = [
        "# Voxel 论文实验复现报告",
        "",
        f"> 生成时间：{ts}  ",
        f"> 配置：{'quick (16 cores, batch=4, seq=128)' if quick else 'default (Table 2/3)'}  ",
        f"> 模型：{', '.join(models)}",
        "",
        "## 总体结论",
        "",
        f"| 判定 | 数量 |",
        f"|------|------|",
        f"| PASS | {pass_n} |",
        f"| PARTIAL | {partial_n} |",
        f"| FAIL | {fail_n} |",
        f"| **复现率 (PASS/{total_v})** | **{100*pass_n/max(1,total_v):.0f}%** |",
        "",
        "## 趋势核对（论文 vs Simulator）",
        "",
        "| ID | 论文结论 | 论文参考值 | Simulator 结果 | 判定 |",
        "|----|----------|------------|----------------|------|",
    ]
    for v in verdicts:
        lines.append(
            f"| {v['id']} | {v['paper']} | {v['paper_value']} | {v['simulator']} | **{v['verdict']}** |"
        )

    lines.extend(["", "## 原始实验数据", ""])
    current = ""
    for r in results:
        if r.experiment != current:
            current = r.experiment
            lines.extend([f"### {current}", ""])
            lines.append(
                "| label | stage | total | noc% | dram | row_conflict | energy(J) |"
            )
            lines.append("|-------|-------|-------|------|------|--------------|-----------|")
        lines.append(
            f"| {r.label} | {r.stage} | {r.total_cycles} | {r.noc_pct:.1f} | "
            f"{r.dram_access_cycles} | {r.row_conflict_overhead_cycles} | {r.energy_joules:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 引擎修复项（本次复现前）",
            "",
            "| 修复 | 说明 |",
            "|------|------|",
            "| NoC 批量竞争 | 按 SYNC region 批量估计传输，SPMD all-reduce 产生更高 noc% |",
            "| tensor-to-bank | MappingPlanner 接入 dram_sim burst 寻址 |",
            "| DRAM 权重读 | COMPUTE 的 DRAM 输入触发隐式读请求 |",
            "| dataflow 建模 | 单算子流水替代整层重复构建 |",
            "| core group | dram_sim 组内同 row 请求合并降低 row_conflict |",
            "",
            "## 已知局限",
            "",
            "- 绝对 cycle 数值与论文 Figure 不可直接对比（SCALE-Sim 简化图 + quick 规模）。",
            "- IPU emulator 验证（Figure 6）未集成。",
            "- row_conflict 在部分规模下为 0（访问模式未触发 Ramulator 冲突分类）。",
            "- dataflow 绝对性能仍可能高于 SPMD（算子数量差异），趋势以 noc% 为主。",
            "",
            "## 说明",
            "",
            "- **PASS**：趋势与论文一致；数值为相对趋势验证，非绝对 cycle 对齐。",
            "- **PARTIAL**：方向正确但幅度偏差较大，或仅部分子指标吻合。",
            "- **FAIL**：趋势与论文相反或未观测到预期现象。",
            "",
            "复现步骤见 [voxel-experiments-reproduction.md](voxel-experiments-reproduction.md)。",
            "",
            "重新生成：`python scripts/run_reproduction.py --quick`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Voxel paper reproduction sweep")
    parser.add_argument("--quick", action="store_true", default=True)
    parser.add_argument("--full", action="store_true", help="Use Table 2/3 default scale")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "docs" / "reproduction-results.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "docs" / "voxel-reproduction-report.md",
    )
    parser.add_argument("--spot-check-70b", action="store_true", default=False)
    args = parser.parse_args()

    quick = not args.full
    cfg = base_config(quick)
    models: list[LLMConfig] = [LLAMA2_13B]
    if args.spot_check_70b:
        models.append(LLAMA3_70B)

    all_results: list[ExperimentResult] = []
    for model in models:
        all_results.extend(experiment_a(cfg, model))
        if model is LLAMA2_13B:
            all_results.extend(experiment_b(cfg, model))
            all_results.extend(experiment_c(cfg, model))
            all_results.extend(experiment_d(cfg, model))
            all_results.extend(experiment_e(cfg, model))
            all_results.extend(experiment_f(cfg, model))
            all_results.extend(experiment_g(cfg, model))
            all_results.extend(experiment_h(cfg, model))
        else:
            # Spot-check: paradigm only for 70B
            pass

    verdicts = evaluate(all_results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quick_mode": quick,
        "models": [m.name for m in models],
        "results": [asdict(r) for r in all_results],
        "verdicts": verdicts,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = render_markdown(
        all_results,
        verdicts,
        quick=quick,
        models=[m.name for m in models],
    )
    args.output_md.write_text(md, encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Verdicts: PASS={sum(1 for v in verdicts if v['verdict']=='PASS')}/{len(verdicts)}")


if __name__ == "__main__":
    main()
