# 华数杯 C 题支撑附件（公开版）

本附件用于补充论文正式结果所对应的源代码、输入数据和关键结果文件。附件中的 Q1--Q4 正式源程序均从原项目直接复制，未改写算法；本次打包未重新训练、预测、调度或优化，也未执行全规模复跑。

## 1. 目录结构

```text
README.md
FILE_MANIFEST_SHA256.csv
code/
  requirements.txt
  q1/
  q2/
  q3/
  q4/
data/
results/
  q1/
  q2/
  q3/
  q4/
```

- `code/`：论文正式结果对应的完整主程序、必要核心模块和运行入口。
- `data/`：正式程序读取的官方附件 Excel，以及 Q1 已生成的正式聚合输入。
- `results/`：与论文表格、数值和最终方案直接对应的关键结果，不包含整个中间实验目录。
- `FILE_MANIFEST_SHA256.csv`：附件中每个文件的相对路径、字节数和 SHA-256，可用于完整性核验。

## 2. Python 环境

- Python 3.12.13
- numpy 2.3.5
- pandas 3.0.1
- scipy 1.18.0
- lightgbm 4.7.0
- openpyxl 3.1.5
- matplotlib 3.11.1

完整依赖版本见 `code/requirements.txt`。

## 3. Q1：LightGBM 预测与基础调度

### 正式程序

- `code/q1/q1_peak_patch_experiment.py`：滚动窗口特征、LightGBM 点预测与分位数预测、高负荷修正和候选结果核验。
- `code/q1/q1_model.py`：数据读取、预测指标、任务候选生成、确定性基础调度、硬约束验证与结果输出。

### 主要输入

- `data/q1_hourly_gpu_demand.csv`
- `data/workload_trace.xlsx`
- `data/GPU_information.xlsx`
- `data/network_latency.xlsx`

### 关键结果

- `results/q1/prediction_comparison.csv`
- `results/q1/schedule_assignments_base.csv`
- `results/q1/schedule_results.csv`
- `results/q1/frozen_numbers.json`

### 基本顺序

先执行 `q1_peak_patch_experiment.py` 生成正式预测，再由 `q1_model.py` 完成基础调度及硬约束检查。两者均可能读取原项目目录结构中的既有处理数据和结果目录。

## 4. Q2：任务调度与 ε 约束

### 正式程序

- `code/q2/run_all.py`：Q2 运行入口。
- `code/q2/q2_model.py`：最早可行调度、成本局部改进、任务交换、服务质量指标和分层 ε 约束。
- `code/q2/q2_core.py`：正式主程序实际调用的已验证调度核心原样副本；仅在附件中使用中性文件名。

### 主要输入

- `data/workload_trace.xlsx`
- `data/GPU_information.xlsx`
- `data/network_latency.xlsx`
- Q1 已生成的正式预测与基础调度结果。

### 关键结果

- `results/q2/baseline_assignments.csv`
- `results/q2/epsilon_3_assignments.csv`
- `results/q2/q2_final_comparison.csv`
- `results/q2/q2_service_quality_metrics.csv`
- `results/q2/frozen_numbers.json`

### 基本顺序

恢复原项目目录结构后执行 `python code/q2/run_all.py`。`q2_model.py` 的原始路径约定会从模型工作区加载已验证核心；附件中的 `q2_core.py` 用于说明和恢复该依赖，不对算法作任何改写。Q2 全量任务调度可能运行较久。

## 5. Q3：储能/SOC 与 HiGHS 分层优化

### 正式程序

- `code/q3/run_all.py`：Q3 运行入口。
- `code/q3/q3_model.py`：储能充放电、SOC、峰值、波动和终端状态约束，以及基于 HiGHS 的成本--碳排--峰值--波动分层优化。
- `code/q3/q3_core.py`：正式主程序实际调用的已验证能源核心原样副本。

### 主要输入

- `data/region_time_data.xlsx`
- `data/storage_information.xlsx`
- `data/power_mapping.xlsx`
- Q2 正式任务分配结果。

### 关键结果

- `results/q3/q3_final_comparison.csv`
- `results/q3/q3_regional_peak_comparison.csv`
- `results/q3/q3_typical_24h_window.csv`
- `results/q3/frozen_numbers.json`

### 基本顺序

Q2 正式任务分配可用后，恢复原项目目录结构并执行 `python code/q3/run_all.py`。HiGHS 分层优化以及必要时的稀疏 MILP 求解可能运行较久。

## 6. Q4：任务--能源交替优化与候选筛选

### 正式程序

- `code/q4/run_all.py`：Q4 联合模型运行入口。
- `code/q4/q4_model.py`：来源分账能源模型、任务侧与能源侧交替改进、联合方案输出与硬约束核验。
- `code/q4/q4_pareto_refine.py`：迁移预算候选生成、指标整理和非支配筛选。
- `code/q4/q4_selection_rule_verify.py`：达到最大额外改善 85% 后按最小迁移预算选择候选的独立核验。

### 主要输入

- `data/` 中全部官方附件。
- `results/q2/baseline_assignments.csv`
- `results/q2/epsilon_3_assignments.csv`
- Q3 正式能源参数与调度结果。

### 关键结果

- `results/q4/q4_candidate_metrics.csv`
- `results/q4/q4_task_assignments_joint_m0500.csv`
- `results/q4/q4_energy_dispatch_joint_m0500.csv`
- `results/q4/q4_final_comparison.csv`
- `results/q4/q4_selection_rule_verification.json`
- `results/q4/frozen_numbers.json`

### 基本顺序

1. 恢复原项目目录结构后执行 `python code/q4/run_all.py`。
2. 执行 `q4_pareto_refine.py` 生成和筛选迁移预算候选。
3. 执行 `q4_selection_rule_verify.py` 核验最终候选选择规则。

Q4 联合候选流程通常是本附件中耗时最长的部分。

## 7. 环境与路径说明

正式源程序按原项目相对路径编写，部分程序会读取 `02_数据处理/`、`03_模型与推导/` 和 `05_结果与图表/`。若仅解压本附件，需要将 `data/`、`results/` 和核心模块恢复到程序约定的原项目目录结构后运行。为保证论文正式结果不被覆盖，本附件未把路径改写成新的独立运行版本。

## 8. 排除内容与隐私检查

附件未包含 `.git`、`.env`、`.vscode`、`__pycache__`、测试缓存、LaTeX 构建目录、probe、debug、绘图脚本、临时实验或废弃模型。快速检查未发现个人绝对路径、个人账号、API key、token 或密码。
