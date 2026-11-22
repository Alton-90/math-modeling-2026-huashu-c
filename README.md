# 2026 华数杯 C 题 · 全国一等奖

论文《多区域数据中心 GPU 需求预测与算—储—电多目标联合优化》研究六个区域的数据中心怎样分配算力、调度储能并与电网交互。前两问先估计需求、安排任务，后两问把同一批任务接入用电与储能模型，再讨论算力与能源同时调度的方案。 [参赛论文 PDF](paper/paper.pdf)

## 四问的做法

| 问题 | 做法 | 对应材料 |
| --- | --- | --- |
| Q1 需求预测与基础调度 | 用滚动窗口构造特征，以 LightGBM 做点预测和分位数预测，并对高负荷时段单独核查。预测结果进入基础任务调度。 | [源码](reproduction/code/q1) · [预测结果](reproduction/results/q1/prediction_comparison.csv) |
| Q2 多目标任务调度 | 在满足任务约束的前提下，通过局部改进和任务交换搜索方案，以 ε 约束比较成本、碳排和服务质量。 | [源码](reproduction/code/q2) · [方案对比](reproduction/results/q2/q2_final_comparison.csv) |
| Q3 储能与电网调度 | 固定 Q2 的任务分配，以 HiGHS 分层求解成本、碳排、购电峰值和波动，核对储能状态与终端约束。 | [源码](reproduction/code/q3) · [方案对比](reproduction/results/q3/q3_final_comparison.csv) |
| Q4 算力与能源联合调度 | 在任务分配与能源调度之间交替改进，比较迁移预算不同的候选方案，再按约定的收益门槛选择方案。 | [源码](reproduction/code/q4) · [候选指标](reproduction/results/q4/q4_candidate_metrics.csv) |

在 Q1 的留出时段，LightGBM 的汇总 WAPE 为 62.29%，历史均值基线为 65.65%。在同一时段，历史均值的 RMSE 为 53.79，LightGBM 为 57.22。Q4 留下了按任务侧和能源侧分别优化的对照，最终选中的联合方案迁移了 500 项任务，占任务总数的 1%。 [预测对比](reproduction/results/q1/prediction_comparison.csv) · [联合方案对比](reproduction/results/q4/q4_final_comparison.csv)

## 查阅与复现

[支撑材料](reproduction)包含四问源码、输入数据、论文所用结果和文件校验清单。建议先浏览结果表，再按 Q1 到 Q4 的依赖顺序查阅实现。Q2 至 Q4 使用前一问的正式输出，部分程序还依赖比赛工作区的目录约定。运行前请按[复现说明](reproduction/README.md)恢复相对路径，并参照[依赖清单](reproduction/code/requirements.txt)准备 Python 环境。公开归档核对了材料和摘要值，全量优化以比赛原有运行记录为准。

论文公开副本移除了封面参赛编号。协作规则与 28 项 Skill 收录在 [华数杯初版 Skill](https://github.com/Alton-90/huashu-modeling-skills)。

## 许可

本仓库的原创源码采用 [MIT 许可证](LICENSE)。参赛论文、赛题附件、发布方数据和求解结果作为项目资料供查阅，其权利以相应作者或来源的说明为准。
