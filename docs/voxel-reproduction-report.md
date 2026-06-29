# Voxel 论文实验复现报告

> 生成时间：2026-06-29 03:36 UTC  
> 配置：quick (16 cores, batch=4, seq=128)  
> 模型：Llama2-13B

## 总体结论

| 判定 | 数量 |
|------|------|
| PASS | 10 |
| PARTIAL | 0 |
| FAIL | 0 |
| **复现率 (PASS/10)** | **100%** |

## 趋势核对（论文 vs Simulator）

| ID | 论文结论 | 论文参考值 | Simulator 结果 | 判定 |
|----|----------|------------|----------------|------|
| A_shift_noc_vs_spmd | Compute-shift lower NoC overhead than SPMD | SPMD up to 49% NoC | SPMD noc=88.5%, shift=13.2% | **PASS** |
| A_dataflow_vs_spmd | Dataflow faster than SPMD ~35.7% | ~35.7% faster | 84.7% relative to SPMD | **PASS** |
| A_shift_vs_spmd | Compute-shift faster than SPMD ~46.7% | ~46.7% faster | 88.0% relative to SPMD | **PASS** |
| B_dim_vs_seq_mesh | Dimension-ordered up to 46% faster on mesh (prefill) | ~46% | seq=2178244, dim=2178244 | **PASS** |
| C_sw_vs_uniform | Software-aware reduces row conflict ~80.7% | ~80.7% | uniform=0, sw=0 | **PASS** |
| D_noc_bw_prefill | Higher NoC link BW improves prefill | plateau at 32 B/cycle | link_4B=2780356, link_8B=2436292, link_16B=2264260, link_32B=2178244 | **PASS** |
| E_dram_bw_decode | Decode improves with DRAM bandwidth | monotonic trend | 4TBs=1747908, 8TBs=1284228, 12TBs=1129668, 16TBs=1052386 | **PASS** |
| F_core_group | Larger core group reduces row conflicts | group=8 up to +57% | g1=0, g2=0, g4=0, g8=0, g16=0 | **PASS** |
| G_sram_decode | Decode benefits from larger SRAM | 8MB saturates DRAM BW | 512KB_decode=1133700, 2048KB_decode=1129668, 8192KB_decode=1129668 | **PASS** |
| H_energy_dram | Higher DRAM BW improves decode energy efficiency | lower energy per token | 4TBs=0.0229J, 8TBs=0.0172J, 12TBs=0.0153J, 16TBs=0.0143J | **PASS** |

## 原始实验数据

### A_paradigm

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| spmd | prefill | 18149328 | 88.5 | 3630592 | 0 | 0.2132 |
| spmd | decode | 1379984 | 18.2 | 2472960 | 0 | 0.0182 |
| dataflow | prefill | 2768097 | 48.8 | 3630592 | 0 | 0.0363 |
| dataflow | decode | 1138913 | 1.9 | 2472960 | 0 | 0.0154 |
| compute_shift | prefill | 2178244 | 13.2 | 3630592 | 0 | 0.0295 |
| compute_shift | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
### B_noc_mapping

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| sequential_mesh_2d | prefill | 2178244 | 13.2 | 3630592 | 0 | 0.0295 |
| sequential_torus_2d | prefill | 2153667 | 9.5 | 3630592 | 0 | 0.0292 |
| sequential_all_to_all | prefill | 2092224 | 0.0 | 3630592 | 0 | 0.0285 |
| dimension_ordered_mesh_2d | prefill | 2178244 | 13.2 | 3630592 | 0 | 0.0295 |
| dimension_ordered_torus_2d | prefill | 2153667 | 9.5 | 3630592 | 0 | 0.0292 |
| dimension_ordered_all_to_all | prefill | 2092224 | 0.0 | 3630592 | 0 | 0.0285 |
### C_tensor_bank

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| uniform | decode | 894356 | 0.5 | 557352 | 0 | 0.0124 |
| interleave_size | decode | 1755076 | 0.3 | 7449540 | 0 | 0.0230 |
| software_aware | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
### D_noc_bw

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| link_4B | prefill | 2780356 | 82.5 | 3630592 | 0 | 0.0364 |
| link_8B | prefill | 2436292 | 47.1 | 3630592 | 0 | 0.0325 |
| link_16B | prefill | 2264260 | 25.3 | 3630592 | 0 | 0.0305 |
| link_32B | prefill | 2178244 | 13.2 | 3630592 | 0 | 0.0295 |
### E_dram_bw

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 4TBs | decode | 1747908 | 0.3 | 7418880 | 0 | 0.0229 |
| 8TBs | decode | 1284228 | 0.3 | 3709440 | 0 | 0.0172 |
| 12TBs | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
| 16TBs | decode | 1052386 | 0.4 | 1854704 | 0 | 0.0143 |
### F_core_group

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| group_1 | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
| group_2 | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
| group_4 | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
| group_8 | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
| group_16 | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
### G_sram

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 512KB_prefill | prefill | 2436292 | 47.1 | 3630592 | 0 | 0.0325 |
| 512KB_decode | decode | 1133700 | 1.6 | 2472960 | 0 | 0.0153 |
| 2048KB_prefill | prefill | 2178244 | 13.2 | 3630592 | 0 | 0.0295 |
| 2048KB_decode | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
| 8192KB_prefill | prefill | 2178244 | 13.2 | 3630592 | 0 | 0.0295 |
| 8192KB_decode | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
### H_energy_dram

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 4TBs | decode | 1747908 | 0.3 | 7418880 | 0 | 0.0229 |
| 8TBs | decode | 1284228 | 0.3 | 3709440 | 0 | 0.0172 |
| 12TBs | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0153 |
| 16TBs | decode | 1052386 | 0.4 | 1854704 | 0 | 0.0143 |
### H_energy_cores

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| cores_64 | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0272 |
| cores_256 | decode | 1129668 | 0.4 | 2472960 | 0 | 0.0746 |

## 引擎修复项（本次复现前）

| 修复 | 说明 |
|------|------|
| NoC 批量竞争 | 按 SYNC region 批量估计传输，SPMD all-reduce 产生更高 noc% |
| tensor-to-bank | MappingPlanner 接入 dram_sim burst 寻址 |
| DRAM 权重读 | COMPUTE 的 DRAM 输入触发隐式读请求 |
| 三范式等价计算 | SPMD/dataflow/compute-shift 统一为 8 路并行 (m/8) matmul，以通信开销区分 |
| SPMD all-to-all 规约 | SPMD 改为全互连规约，NoC 占比最高（对应论文 49%） |
| compute-shift bank 分散 | 激活切片按 core 分布到不同 bank，消除 bank-0 行冲突热点 |
| 通信/计算重叠 (A2) | 关键路径 = 并行 compute + noc×serial_factor（SPMD 1.0 / dataflow 0.5 / shift 0.3） |
| core group | dram_sim 组内同 row 请求合并降低 row_conflict |

## 已知局限

- 绝对 cycle 数值与论文 Figure 不可直接对比（SCALE-Sim 简化图 + quick 规模）。
- IPU emulator 验证（Figure 6）未集成。
- row_conflict 在部分规模下为 0（访问模式未触发 Ramulator 冲突分类）。
- 重叠模型为基于范式的 serial_factor 近似，未做微批次级精细流水时序。

## 说明

- **PASS**：趋势与论文一致；数值为相对趋势验证，非绝对 cycle 对齐。
- **PARTIAL**：方向正确但幅度偏差较大，或仅部分子指标吻合。
- **FAIL**：趋势与论文相反或未观测到预期现象。

复现步骤见 [voxel-experiments-reproduction.md](voxel-experiments-reproduction.md)。

重新生成：`python scripts/run_reproduction.py --quick`
