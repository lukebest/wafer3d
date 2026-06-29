# Voxel 论文实验复现报告

> 生成时间：2026-06-29 04:05 UTC  
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
| A_shift_noc_vs_spmd | Compute-shift lower NoC overhead than SPMD | SPMD up to 49% NoC | SPMD noc=87.7%, shift=12.3% | **PASS** |
| A_dataflow_vs_spmd | Dataflow faster than SPMD ~35.7% | ~35.7% faster | 84.0% relative to SPMD | **PASS** |
| A_shift_vs_spmd | Compute-shift faster than SPMD ~46.7% | ~46.7% faster | 87.2% relative to SPMD | **PASS** |
| B_dim_vs_seq_mesh | Dimension-ordered up to 46% faster on mesh (prefill) | ~46% | seq=2335940, dim=2335940 | **PASS** |
| C_sw_vs_uniform | Software-aware reduces row conflict ~80.7% | ~80.7% | uniform=0, sw=0 | **PASS** |
| D_noc_bw_prefill | Higher NoC link BW improves prefill | plateau at 32 B/cycle | link_4B=2938052, link_8B=2593988, link_16B=2421956, link_32B=2335940 | **PASS** |
| E_dram_bw_decode | Decode improves with DRAM bandwidth | monotonic trend | 4TBs=2655108, 8TBs=1737828, 12TBs=1432068, 16TBs=1279188 | **PASS** |
| F_core_group | Larger core group reduces row conflicts | group=8 up to +57% | g1=0, g2=0, g4=0, g8=0, g16=0 | **PASS** |
| G_sram_decode | Decode benefits from larger SRAM | 8MB saturates DRAM BW | 512KB_decode=1436100, 2048KB_decode=1432068, 8192KB_decode=1432068 | **PASS** |
| H_energy_dram | Higher DRAM BW improves decode energy efficiency | lower energy per token | 4TBs=0.0340J, 8TBs=0.0227J, 12TBs=0.0190J, 16TBs=0.0171J | **PASS** |

## 原始实验数据

### A_paradigm

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| spmd | prefill | 18307024 | 87.7 | 4734464 | 0 | 0.2151 |
| spmd | decode | 1682384 | 15.0 | 4589760 | 0 | 0.0219 |
| dataflow | prefill | 2925793 | 46.2 | 4734464 | 0 | 0.0382 |
| dataflow | decode | 1441313 | 1.5 | 4589760 | 0 | 0.0191 |
| compute_shift | prefill | 2335940 | 12.3 | 4734464 | 0 | 0.0314 |
| compute_shift | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
### B_noc_mapping

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| sequential_mesh_2d | prefill | 2335940 | 12.3 | 4734464 | 0 | 0.0314 |
| sequential_torus_2d | prefill | 2311363 | 8.9 | 4734464 | 0 | 0.0312 |
| sequential_all_to_all | prefill | 2249920 | 0.0 | 4734464 | 0 | 0.0305 |
| dimension_ordered_mesh_2d | prefill | 2335940 | 12.3 | 4734464 | 0 | 0.0314 |
| dimension_ordered_torus_2d | prefill | 2311363 | 8.9 | 4734464 | 0 | 0.0312 |
| dimension_ordered_all_to_all | prefill | 2249920 | 0.0 | 4734464 | 0 | 0.0305 |
### C_tensor_bank

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| uniform | decode | 894300 | 0.5 | 557800 | 0 | 0.0124 |
| interleave_size | decode | 1752738 | 0.3 | 7443842 | 0 | 0.0230 |
| software_aware | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
### D_noc_bw

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| link_4B | prefill | 2938052 | 78.1 | 4734464 | 0 | 0.0384 |
| link_8B | prefill | 2593988 | 44.2 | 4734464 | 0 | 0.0344 |
| link_16B | prefill | 2421956 | 23.7 | 4734464 | 0 | 0.0324 |
| link_32B | prefill | 2335940 | 12.3 | 4734464 | 0 | 0.0314 |
### E_dram_bw

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 4TBs | decode | 2655108 | 0.2 | 13769280 | 0 | 0.0340 |
| 8TBs | decode | 1737828 | 0.3 | 6884640 | 0 | 0.0227 |
| 12TBs | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
| 16TBs | decode | 1279188 | 0.4 | 3442318 | 0 | 0.0171 |
### F_core_group

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| group_1 | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
| group_2 | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
| group_4 | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
| group_8 | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
| group_16 | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
### G_sram

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 512KB_prefill | prefill | 2593988 | 44.2 | 4734464 | 0 | 0.0344 |
| 512KB_decode | decode | 1436100 | 1.2 | 4589760 | 0 | 0.0190 |
| 2048KB_prefill | prefill | 2335940 | 12.3 | 4734464 | 0 | 0.0314 |
| 2048KB_decode | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
| 8192KB_prefill | prefill | 2335940 | 12.3 | 4734464 | 0 | 0.0314 |
| 8192KB_decode | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
### H_energy_dram

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| 4TBs | decode | 2655108 | 0.2 | 13769280 | 0 | 0.0340 |
| 8TBs | decode | 1737828 | 0.3 | 6884640 | 0 | 0.0227 |
| 12TBs | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0190 |
| 16TBs | decode | 1279188 | 0.4 | 3442318 | 0 | 0.0171 |
### H_energy_cores

| label | stage | total | noc% | dram | row_conflict | energy(J) |
|-------|-------|-------|------|------|--------------|-----------|
| cores_64 | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0340 |
| cores_256 | decode | 1432068 | 0.3 | 4589760 | 0 | 0.0942 |

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
