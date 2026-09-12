# `paper/` 目录说明

**提交用的成稿是 [`main.md`](main.md)**（七章 + 摘要 + 参考文献 + 8 张配图引用）。
其余文件是成稿的原料或历史版本，保留是因为它们含 `main.md` 为篇幅所限省略的推导细节，
最新运行版本及本轮独立结果以 `q3_200_optimization.md`、`q4_410_optimization.md` 与根目录 `README.md`「当前成绩」节为准；`main.md` 保留论文原稿，不能将其中历史指标视为新版结果。

## 文件定位

| 文件 | 行数 | 定位 |
|---|---|---|
| **`main.md`** | 468 | **提交成稿。** 唯一需要通读的文件 |
| `q3_200_optimization.md` | — | Q3v2，200秒目标、160局独立配对与交付说明 |
| `q3_200_development.md` | — | 本轮全部开发试验与否决原因 |
| `q4_410_optimization.md` | — | Q4v3，410秒目标及上一轮独立配对结果 |
| `q4_450_optimization.md` | — | 2026-09-12 问题四新策略、独立配对结果、450 秒目标与交付说明 |
| `q3_220_optimization.md` | — | 2026-09-12 问题三新策略、100 局独立配对评估与分源数压力测试、220 秒目标与交付说明 |
| `q3_220_development.md` | — | 问题三冻结前的方案取舍；开发数据与最终独立证据分别记录 |
| `00_symbols.md` | 28 | 符号表，已并入 `main.md` §1.2 |
| `q1_diameter_covering.md` | 209 | Q1 完整推导（含 Jung 定理紧上界、覆盖裕度、反例构造）|
| `q2_second_point.md` | 190 | Q2 完整推导（含一阶误差传播、三约束、候选区界定）|
| `q3q4_coverage_design.md` | 111 | 覆盖站位设计论证（bisector 几何、最坏间隙）|
| `q3q4_path_planning.md` | 244 | 路径规划（TSP 巡回、移动耗时归因）|
| `b_optimization_section.md` | 696 | **优化历程全记录**，含全部被证伪方案的实测数据 |
| `q3q4_official_results.md` | 125 | 官方测试结果表（表 1 格式模板）|
| `01_problem1.md` | 134 | ⚠ Q1 早期精简稿，已被 `q1_diameter_covering.md` 取代 |
| `02_problem2.md` | 163 | ⚠ Q2 早期精简稿，已被 `q2_second_point.md` 取代 |
| `03_problem34.md` | 152 | ⚠ Q3/Q4 早期稿：把 TSP 列为「待做，上限 −28%」，环参数表止于 1250 m |
| `STATUS_FOR_WRITER.md` | 202 | ⚠ 一次性交底材料，数字已过时（文首有存档横幅）|

⚠ 标记的四份**不要引用其数字**：`01`/`02` 是 `q1_*`/`q2_*` 的早期精简版（后者多出
Jung 定理、误差传播等内容），`03_problem34.md` 写于 TSP 落地之前（文中仍把它列为待做事项、环参数止于 1250 m），
`STATUS_FOR_WRITER.md` 停留在「204 项测试」「基线 n=9」的状态。

## 为什么不删早期稿

竞赛要求提交物可追溯。这些文件记录了推导与结论的演进（例如 Q1 从「直径判据」
到「最小外接圆 + Jung 上界」的修正过程），删掉会让「为什么改」失去凭证。
`b_optimization_section.md` 同理——它保存了预算扩张、批量归航、survey_first 等
被数据否决方案的完整实测，`main.md` 第 6 章只是其摘要。

## 配图

8 张 PNG 在 `../figures/`，全部被 `main.md` 引用；生成脚本见 `../src/b_q1_figures.py`、
`b_q2_figures.py`、`b_coverage_figures.py`、`b_budget_pareto.py`。
