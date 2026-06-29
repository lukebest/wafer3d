---
name: voxel reproduction report
overview: Fix the four engine modeling gaps that currently flatten paper trends (NoC contention, tensor-to-bank wiring, DRAM weight reads, dataflow cost, core-group tracker), then run a quick reproduction sweep on Llama2-13B (+Llama3-70B spot checks) and generate docs/voxel-reproduction-report.md comparing simulator trends against the paper's quantitative claims with PASS/PARTIAL/FAIL verdicts.
todos:
  - id: fix-noc
    content: 重构 engine.run() 按 SYNC region 批量估计 NoC 传输，使 noc_overhead_cycles 反映竞争（SPMD > compute-shift）
    status: completed
  - id: fix-bank
    content: 在引擎接入 MappingPlanner，将 tensor-to-bank 策略落到 dram_sim 的 burst 寻址，使 uniform/software_aware 产生不同 row_conflict
    status: completed
  - id: fix-dram-read
    content: 为 COMPUTE 的 DRAM 输入张量生成隐式 DRAM 读请求，使 decode 出现非零 DRAM/row-conflict 且随带宽变化
    status: completed
  - id: fix-dataflow
    content: 修正 build_dataflow_layer 为单算子流水重叠，消除 ~37x 异常开销
    status: completed
  - id: fix-coregroup
    content: 在 dram_sim 增加 core group request tracker：组内同 row 请求合并，组越大冲突越少
    status: completed
  - id: regress
    content: 运行 pytest tests/ -q 并按新行为更新趋势断言，确保全绿
    status: completed
  - id: runner
    content: 新增 scripts/run_reproduction.py 运行 A-H 实验并输出 JSON + markdown 表
    status: completed
  - id: report
    content: 生成 docs/voxel-reproduction-report.md（论文 vs simulator 逐项 PASS/PARTIAL/FAIL + 原因 + 复现率小结），并在 voxel-experiments-reproduction.md 末尾加链接
    status: completed
isProject: false
---

# Voxel 实验复现与报告

目标：先修复 simulator 关键建模缺口，使论文核心趋势可复现，再运行快速实验（1-2 模型）并生成对照报告。

## 背景：实测发现的缺口

运行当前 simulator 后确认：
- `noc_overhead_cycles` 恒为 0 —— [`engine.py`](voxelsim/sim/engine.py) 的 `_simulate_copy` 对每个传输单独调用 `estimate_transfers([t])`，[`noc_sim.py`](voxelsim/sim/noc_sim.py) 的 contention 模型永远看到 1 个传输。
- tensor-to-bank 三策略结果完全相同 —— [`mapping.py`](voxelsim/chip/mapping.py) 的 `assign_banks_for_event` 从未被引擎调用，bank_id 由 model builder 硬编码。
- decode 下 `dram_access_cycles=0` —— [`dram_sim.py`](voxelsim/sim/dram_sim.py) 只在 `COPY_DATA` 且 dest 为 DRAM 时建模；权重的 DRAM 读取（compute 的 DRAM 输入）从未建模。
- dataflow 比其它范式慢 ~37× —— [`paradigms.py`](voxelsim/models/paradigms.py) 的 `build_dataflow_layer` 每 stage×microbatch 重建整层。
- `core_group_size` 仅存在于配置，[`dram_sim.py`](voxelsim/sim/dram_sim.py) 无 request tracker。

## 修复方案

### 1. NoC 批量竞争 (Fig 9/10)
重构 [`engine.py`](voxelsim/sim/engine.py) `run()`：以 `SYNC` 事件为界把执行图切成 region；region 内所有 `src.location==SRAM` 的 COPY_DATA 收集为一批，单次 `noc_sim.estimate_transfers(batch)`，使 `AnalyticNoCModel` 的 link 竞争生效。region 的 NoC 贡献取批内 max 延迟（并发），`noc_overhead_cycles` 累加批内 overhead。预期：SPMD（all-reduce 多对多）NoC 占比 > compute-shift（环形 1-hop）。

### 2. tensor-to-bank 接入 DRAM (Fig 11/12)
引擎初始化时用 `MappingPlanner.detect_concurrent_tensors` + `assign_banks_for_event` 生成 `tensor_name -> [banks]`。改 [`dram_sim.py`](voxelsim/sim/dram_sim.py) `enqueue_copy`：按映射的 bank 列表对 burst 做 round-robin 寻址（uniform→并发张量 bank 重叠→冲突高；software_aware→不相交→冲突低）。

### 3. DRAM 权重读取建模 (decode DRAM 趋势)
在 [`engine.py`](voxelsim/sim/engine.py) 处理 `COMPUTE` 时，为 `location==DRAM` 的输入张量生成隐式 DRAM 读请求，计入 `dram_sim`，使 decode 出现非零 DRAM/row-conflict，并随 DRAM 带宽变化。

### 4. dataflow 建模修正 (Fig 9)
改 [`paradigms.py`](voxelsim/models/paradigms.py) `build_dataflow_layer`：每 stage 只做单算子（不重建整层），microbatch 流水重叠，使总量介于 SPMD 与 compute-shift 之间。

### 5. core group request tracker (Fig 16)
[`dram_sim.py`](voxelsim/sim/dram_sim.py) 增加按 `core_group_size` 对同 row 请求做合并的简单模型：组内同 row 访问只计一次冲突，组越大冲突越少。

### 6. 回归
`pytest tests/ -q` 全绿；必要时更新 [`test_e2e.py`](tests/test_e2e.py)、[`test_noc.py`](tests/test_noc.py) 的趋势断言为「compute-shift NoC 占比 < SPMD」。

## 实验与报告（quick：Llama2-13B 为主，Llama3-70B 抽查）

新增 [`scripts/run_reproduction.py`](scripts/run_reproduction.py)：批量运行并输出 JSON + markdown 表。覆盖：
- A 计算范式 spmd/dataflow/compute_shift（prefill+decode）→ total、noc%
- B 映射×拓扑 seq/dim × mesh/torus/all-to-all（prefill）→ noc%
- C tensor-to-bank uniform/interleave/software_aware（decode）→ row_conflict
- D NoC link 带宽 4/8/16/32（prefill）
- E DRAM 带宽 4/8/12/16（decode）
- F core group 1/2/4/8/16（decode）
- G per-core SRAM 512/2048/8192（decode/prefill）
- H 能耗：DRAM 带宽 / core 数扫掠 → energy + breakdown

生成 [`docs/voxel-reproduction-report.md`](docs/voxel-reproduction-report.md)：每个实验一节，列「论文结论 / 论文数值 / 本 simulator 数值 / 趋势是否一致 (PASS/PARTIAL/FAIL) / 偏差原因」，并附 §12 速查表的逐项核对与总体复现率小结。更新 [`voxel-experiments-reproduction.md`](docs/voxel-experiments-reproduction.md) 末尾链接到报告。

## 验收标准
- A：compute-shift 的 noc% 最低、SPMD 最高（趋势一致）
- C：software_aware 的 row_conflict < uniform
- D：link 带宽↑ → prefill total↓
- E：DRAM 带宽↑ → decode total↓（plateau 可见）
- F：core group↑ → decode row_conflict↓
- 报告含每实验 PASS/PARTIAL/FAIL 与原因
- `pytest tests/ -q` 通过