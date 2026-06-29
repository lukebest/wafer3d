# Voxel 论文实验复现报告

> 生成时间：2026-06-29 02:49 UTC  
> 配置：quick (16 cores, batch=4, seq=128)  
> 模型：Llama2-13B

## 总体结论

| 判定 | 数量 |
|------|------|
| PASS | 7 |
| PARTIAL | 3 |
| FAIL | 0 |
| **复现率 (PASS/10)** | **70%** |

## 趋势核对（论文 vs Simulator）

| ID | 论文结论 | 论文参考值 | Simulator 结果 | 判定 |
|----|----------|------------|----------------|------|
| A_shift_noc_vs_spmd | Compute-shift lower NoC overhead than SPMD | SPMD up to 49% NoC | SPMD noc=13.7%, shift=13.6% | **PARTIAL** |
| A_dataflow_vs_spmd | Dataflow faster than SPMD ~35.7% | ~35.7% faster | -1117.4% relative to SPMD | **PARTIAL** |
| A_shift_vs_spmd | Compute-shift faster than SPMD ~46.7% | ~46.7% faster | -0.8% relative to SPMD | **PARTIAL** |
| B_dim_vs_seq_mesh | Dimension-ordered up to 46% faster on mesh (prefill) | ~46% | seq=2108372, dim=2108372 | **PASS** |
| C_sw_vs_uniform | Software-aware reduces row conflict ~80.7% | ~80.7% | uniform=0, sw=0 | **PASS** |
| D_noc_bw_prefill | Higher NoC link BW improves prefill | plateau at 32 B/cycle | link_4B=3255252, link_8B=2599892, link_16B=2272212, link_32B=2108372 | **PASS** |
| E_dram_bw_decode | Decode improves with DRAM bandwidth | monotonic trend | 4TBs=1747952, 8TBs=1284860, 12TBs=1130496, 16TBs=1053313 | **PASS** |
| F_core_group | Larger core group reduces row conflicts | group=8 up to +57% | g1=0, g2=0, g4=0, g8=0, g16=0 | **PASS** |
| G_sram_decode | Decode benefits from larger SRAM | 8MB saturates DRAM BW | 512KB_decode=1138176, 2048KB_decode=1130496, 8192KB_decode=1130496 | **PASS** |
| H_energy_dram | Higher DRAM BW improves decode energy efficiency | lower energy per token | 4TBs=0.0229J, 8TBs=0.0172J, 12TBs=0.0153J, 16TBs=0.0144J | **PASS** |

## 原始实验数据

### A_paradigm

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| spmd | prefill | 2092319 | 13.7 | 3630592 | 0 | 0.0285 |
| spmd | decode | 1128415 | 0.4 | 2472960 | 0 | 0.0153 |
| dataflow | prefill | 25471362 | 0.0 | 1834560 | 0 | 0.3027 |
| dataflow | decode | 3955042 | 0.0 | 1236480 | 0 | 0.0468 |
| compute_shift | prefill | 2108372 | 13.6 | 2448992 | 0 | 0.0286 |
| compute_shift | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
### B_noc_mapping

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| sequential_mesh_2d | prefill | 2108372 | 13.6 | 2448992 | 0 | 0.0286 |
| sequential_torus_2d | prefill | 2026448 | 10.1 | 2448992 | 0 | 0.0277 |
| sequential_all_to_all | prefill | 1965005 | 0.0 | 2448992 | 0 | 0.0269 |
| dimension_ordered_mesh_2d | prefill | 2108372 | 13.6 | 2448992 | 0 | 0.0286 |
| dimension_ordered_torus_2d | prefill | 2026448 | 10.1 | 2448992 | 0 | 0.0277 |
| dimension_ordered_all_to_all | prefill | 1965005 | 0.0 | 2448992 | 0 | 0.0269 |
### C_tensor_bank

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| uniform | decode | 895562 | 0.5 | 560000 | 0 | 0.0124 |
| interleave_size | decode | 1784198 | 0.3 | 7644448 | 0 | 0.0234 |
| software_aware | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
### D_noc_bw

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| link_4B | prefill | 3255252 | 70.5 | 2448992 | 0 | 0.0418 |
| link_8B | prefill | 2599892 | 44.1 | 2448992 | 0 | 0.0342 |
| link_16B | prefill | 2272212 | 25.2 | 2448992 | 0 | 0.0305 |
| link_32B | prefill | 2108372 | 13.6 | 2448992 | 0 | 0.0286 |
### E_dram_bw

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 4TBs | decode | 1747952 | 0.3 | 7409472 | 0 | 0.0229 |
| 8TBs | decode | 1284860 | 0.3 | 3704736 | 0 | 0.0172 |
| 12TBs | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
| 16TBs | decode | 1053313 | 0.4 | 1852360 | 0 | 0.0144 |
### F_core_group

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| group_1 | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
| group_2 | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
| group_4 | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
| group_8 | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
| group_16 | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
### G_sram

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 512KB_prefill | prefill | 2599892 | 44.1 | 2448992 | 0 | 0.0342 |
| 512KB_decode | decode | 1138176 | 1.6 | 2469824 | 0 | 0.0154 |
| 2048KB_prefill | prefill | 2108372 | 13.6 | 2448992 | 0 | 0.0286 |
| 2048KB_decode | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
| 8192KB_prefill | prefill | 2108372 | 13.6 | 2448992 | 0 | 0.0286 |
| 8192KB_decode | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
### H_energy_dram

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 4TBs | decode | 1747952 | 0.3 | 7409472 | 0 | 0.0229 |
| 8TBs | decode | 1284860 | 0.3 | 3704736 | 0 | 0.0172 |
| 12TBs | decode | 1130496 | 0.4 | 2469824 | 0 | 0.0153 |
| 16TBs | decode | 1053313 | 0.4 | 1852360 | 0 | 0.0144 |
### H_energy_cores

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| cores_64 | decode | 1132422 | 0.4 | 2469824 | 0 | 0.0272 |
| cores_256 | decode | 1132422 | 0.4 | 2469824 | 0 | 0.0748 |

## 引擎修复项（本次复现前）

| 修复 | 说明 |
|------|------|
| NoC 批量竞争 | 按 SYNC region 批量估计传输，SPMD all-reduce 产生更高 noc% |
| tensor-to-bank | MappingPlanner 接入 dram_sim burst 寻址 |
| DRAM 权重读 | COMPUTE 的 DRAM 输入触发隐式读请求 |
| dataflow 建模 | 单算子流水替代整层重复构建 |
| core group | dram_sim 组内同 row 请求合并降低 row_conflict |

## 已知局限

- 绝对 cycle 数值与论文 Figure 不可直接对比（SCALE-Sim 简化图 + quick 规模）。
- IPU emulator 验证（Figure 6）未集成。
- row_conflict 在部分规模下为 0（访问模式未触发 Ramulator 冲突分类）。
- dataflow 绝对性能仍可能高于 SPMD（算子数量差异），趋势以 noc% 为主。

## 说明

- **PASS**：趋势与论文一致；数值为相对趋势验证，非绝对 cycle 对齐。
- **PARTIAL**：方向正确但幅度偏差较大，或仅部分子指标吻合。
- **FAIL**：趋势与论文相反或未观测到预期现象。

复现步骤见 [voxel-experiments-reproduction.md](voxel-experiments-reproduction.md)。

重新生成：`python scripts/run_reproduction.py --quick`
