# Q3 200秒优化：开发取舍记录

所有本页数字来自开发案例，不能当作独立验证。对照为当前Q3v1；部分早期试验使用旧评测器，但显式指定生产Q3Fast作为对照，没有使用其指向DogStrategy的baseline别名。

## 最终选择

- 自适应/1150米六站环按预测巡回与扫频成本二选一，结合安全清除落点及扫频中的16源上界重查。
- 最终24局开发：合计平均236.4529→233.6648秒；逐局均值243.7394→240.1380秒；两臂309/309全清。4个开发案例的生产重放在时间、移动、动作计数、配置、覆盖访问和事件上精确一致。
- 1150米环的解析最大接收距离为988.511420米，余量11.488580米。只有本频道实际七点沉默才能发出连续覆盖证书。

## 否决与保留

- 连续站位移动和停车点替换未有稳定收益。
- 更紧方格沉默证明（含联合覆盖细分）8局均值247.27→248.47秒，未采用。
- 单示向度直接估计大幅变慢；更长基线/更快推进仅微小变化。
- 250米提前清除阈值在第二组反而变慢，保留150米；搭载补测距离0/100/400/600均未稳定优于250米，保留250米。
- 仅固定粗格保证1264米六站在第二组变慢；连续认证1150米环表现更稳，Hybrid在两局保留自适应而避免固定环退化。
- 1130米环虽满足连续覆盖，组合第二组池化略退化；选择有更大覆盖裕度且开发组合更好的1150米。
- 最初连续站位试验有12条TypeError错误（误把格点索引当坐标），全部保留；修正后重跑18条完整配对，未拿错误或残缺样本作性能结论。

## 全部开发记录

下表按文件/实验臂逐项列出，供复核；不同文件种子不同，不能跨文件直接排序。文件名中的baseline在新版b_q3_200_eval中是精确Q3v1快照。

| 记录 | 实验臂 | 完成/计划 | 错误 | 全清源/总源 | 局均/s | 池化/s |
|---|---|---:|---:|---:|---:|---:|
| [q3_200_analytic_pilot](../results/q3_200_analytic_pilot.json) | baseline | 32/32 | 0 | 98/98 | 248.131 | 242.571 |
| [q3_200_analytic_pilot](../results/q3_200_analytic_pilot.json) | src.b_q3_200_analytic_experiments:Continuous1130 | 32/32 | 0 | 98/98 | 244.504 | 238.658 |
| [q3_200_analytic_pilot](../results/q3_200_analytic_pilot.json) | src.b_q3_200_analytic_experiments:Continuous1150 | 32/32 | 0 | 98/98 | 244.565 | 239.101 |
| [q3_200_analytic_pilot](../results/q3_200_analytic_pilot.json) | src.b_q3_200_analytic_experiments:Continuous1200 | 32/32 | 0 | 98/98 | 248.867 | 243.750 |
| [q3_200_cells_pilot](../results/q3_200_cells_pilot.json) | baseline | 24/24 | 0 | 98/98 | 247.272 | 244.662 |
| [q3_200_cells_pilot](../results/q3_200_cells_pilot.json) | src.b_q3_200_cells_experiments:ExactSquares | 24/24 | 0 | 98/98 | 248.470 | 245.716 |
| [q3_200_cells_pilot](../results/q3_200_cells_pilot.json) | src.b_q3_200_cells_experiments:UnionSquares | 24/24 | 0 | 98/98 | 248.470 | 245.716 |
| [q3_200_certified_pilot](../results/q3_200_certified_pilot.json) | baseline | 18/18 | 0 | 78/78 | 253.572 | 252.565 |
| [q3_200_certified_pilot](../results/q3_200_certified_pilot.json) | src.b_q3_200_certified_ring:Certified6 | 18/18 | 0 | 78/78 | 246.856 | 246.664 |
| [q3_200_certified_pilot](../results/q3_200_certified_pilot.json) | src.b_q3_200_certified_ring:Certified7 | 18/18 | 0 | 78/78 | 256.878 | 255.746 |
| [q3_200_clear_pilot](../results/q3_200_clear_pilot.json) | src.b_q3_200_clear_experiments:Early1000 | 36/36 | 0 | 77/77 | 253.013 | 247.408 |
| [q3_200_clear_pilot](../results/q3_200_clear_pilot.json) | src.b_q3_200_clear_experiments:Early250 | 36/36 | 0 | 77/77 | 250.739 | 245.466 |
| [q3_200_clear_pilot](../results/q3_200_clear_pilot.json) | src.b_q3_200_clear_experiments:Early400 | 36/36 | 0 | 77/77 | 252.653 | 247.123 |
| [q3_200_clear_pilot](../results/q3_200_clear_pilot.json) | src.b_q3_200_clear_experiments:RouteClear | 36/36 | 0 | 77/77 | 250.627 | 245.321 |
| [q3_200_clear_pilot](../results/q3_200_clear_pilot.json) | src.b_q3_200_clear_experiments:SoftCover | 36/36 | 0 | 77/77 | 250.503 | 245.161 |
| [q3_200_clear_pilot](../results/q3_200_clear_pilot.json) | src.b_q3_200_clear_experiments:ThinBaseline | 36/36 | 0 | 77/77 | 253.614 | 248.367 |
| q3_200_continuous_pilot | src.b_q3_200_search_experiments:ContinuousLocalized | 18/18 | 6 | 0/0 | — | — |
| q3_200_continuous_pilot | src.b_q3_200_search_experiments:ContinuousStations | 18/18 | 6 | 0/0 | — | — |
| [q3_200_continuous_pilot](../results/q3_200_continuous_pilot.json) | src.b_sim.q3_fast:Q3FastStrategy | 18/18 | 0 | 78/78 | 253.572 | 252.565 |
| [q3_200_continuous_pilot_fixed](../results/q3_200_continuous_pilot_fixed.json) | src.b_q3_200_search_experiments:ContinuousLocalized | 18/18 | 0 | 78/78 | 254.693 | 253.561 |
| [q3_200_continuous_pilot_fixed](../results/q3_200_continuous_pilot_fixed.json) | src.b_q3_200_search_experiments:ContinuousStations | 18/18 | 0 | 78/78 | 254.926 | 253.577 |
| [q3_200_continuous_pilot_fixed](../results/q3_200_continuous_pilot_fixed.json) | src.b_sim.q3_fast:Q3FastStrategy | 18/18 | 0 | 78/78 | 253.572 | 252.565 |
| [q3_200_fixed6_confirmation](../results/q3_200_fixed6_confirmation.json) | baseline | 24/24 | 0 | 166/166 | 231.434 | 225.547 |
| [q3_200_fixed6_confirmation](../results/q3_200_fixed6_confirmation.json) | src.b_q3_200_fixed_search:Fixed6 | 24/24 | 0 | 166/166 | 234.476 | 229.716 |
| [q3_200_fixed_pilot](../results/q3_200_fixed_pilot.json) | baseline | 24/24 | 0 | 78/78 | 253.572 | 252.565 |
| [q3_200_fixed_pilot](../results/q3_200_fixed_pilot.json) | src.b_q3_200_fixed_search:Fixed6 | 24/24 | 0 | 78/78 | 247.519 | 247.032 |
| [q3_200_fixed_pilot](../results/q3_200_fixed_pilot.json) | src.b_q3_200_fixed_search:Fixed7 | 24/24 | 0 | 78/78 | 255.557 | 254.674 |
| [q3_200_fixed_pilot](../results/q3_200_fixed_pilot.json) | src.b_q3_200_fixed_search:Fixed8 | 24/24 | 0 | 78/78 | 266.133 | 265.306 |
| [q3_200_homing_confirmation](../results/q3_200_homing_confirmation.json) | baseline | 36/36 | 0 | 155/155 | 239.113 | 234.833 |
| [q3_200_homing_confirmation](../results/q3_200_homing_confirmation.json) | src.b_q3_200_clear_experiments:RouteClear | 36/36 | 0 | 155/155 | 238.708 | 234.404 |
| [q3_200_homing_confirmation](../results/q3_200_homing_confirmation.json) | src.b_q3_200_homing_combo_experiments:RouteEarly250 | 36/36 | 0 | 155/155 | 240.356 | 236.020 |
| [q3_200_homing_pilot](../results/q3_200_homing_pilot.json) | src.b_q3_200_homing_experiments:LongBaseline | 30/30 | 0 | 77/77 | 249.917 | 244.881 |
| [q3_200_homing_pilot](../results/q3_200_homing_pilot.json) | src.b_q3_200_homing_experiments:RapidAdvance | 30/30 | 0 | 77/77 | 251.259 | 245.977 |
| [q3_200_homing_pilot](../results/q3_200_homing_pilot.json) | src.b_q3_200_homing_experiments:SilentHull | 30/30 | 0 | 77/77 | 250.893 | 245.605 |
| [q3_200_homing_pilot](../results/q3_200_homing_pilot.json) | src.b_q3_200_homing_experiments:SingleStraight | 30/30 | 0 | 77/77 | 307.678 | 300.355 |
| [q3_200_homing_pilot](../results/q3_200_homing_pilot.json) | src.b_sim.q3_fast:Q3FastStrategy | 30/30 | 0 | 77/77 | 250.893 | 245.605 |
| [q3_200_hybrid_development](../results/q3_200_hybrid_development.json) | baseline | 36/36 | 0 | 166/166 | 231.434 | 225.547 |
| [q3_200_hybrid_development](../results/q3_200_hybrid_development.json) | src.b_q3_200_analytic_experiments:Continuous1150 | 36/36 | 0 | 166/166 | 227.582 | 223.052 |
| [q3_200_hybrid_development](../results/q3_200_hybrid_development.json) | src.b_q3_200_hybrid_search:Hybrid | 36/36 | 0 | 166/166 | 226.189 | 221.519 |
| [q3_200_piggyback_confirmation](../results/q3_200_piggyback_confirmation.json) | baseline | 24/24 | 0 | 145/145 | 261.987 | 258.034 |
| [q3_200_piggyback_confirmation](../results/q3_200_piggyback_confirmation.json) | src.b_q3_200_piggyback_experiments:Piggyback400 | 24/24 | 0 | 145/145 | 261.860 | 258.096 |
| [q3_200_piggyback_pilot](../results/q3_200_piggyback_pilot.json) | baseline | 30/30 | 0 | 73/73 | 257.668 | 255.259 |
| [q3_200_piggyback_pilot](../results/q3_200_piggyback_pilot.json) | src.b_q3_200_piggyback_experiments:Piggyback100 | 30/30 | 0 | 73/73 | 258.536 | 256.098 |
| [q3_200_piggyback_pilot](../results/q3_200_piggyback_pilot.json) | src.b_q3_200_piggyback_experiments:Piggyback400 | 30/30 | 0 | 73/73 | 255.672 | 252.774 |
| [q3_200_piggyback_pilot](../results/q3_200_piggyback_pilot.json) | src.b_q3_200_piggyback_experiments:Piggyback600 | 30/30 | 0 | 73/73 | 263.845 | 261.271 |
| [q3_200_piggyback_pilot](../results/q3_200_piggyback_pilot.json) | src.b_q3_200_piggyback_experiments:PiggybackZero | 30/30 | 0 | 73/73 | 266.602 | 263.795 |
| [q3_200_replacement_pilot](../results/q3_200_replacement_pilot.json) | baseline | 18/18 | 0 | 78/78 | 253.572 | 252.565 |
| [q3_200_replacement_pilot](../results/q3_200_replacement_pilot.json) | src.b_q3_200_replace_search:Replacement80 | 18/18 | 0 | 78/78 | 255.219 | 253.920 |
| [q3_200_replacement_pilot](../results/q3_200_replacement_pilot.json) | src.b_q3_200_replace_search:ReplacementScan | 18/18 | 0 | 78/78 | 255.219 | 253.920 |
| [q3_200_ring_clear_combo_confirmation](../results/q3_200_ring_clear_combo_confirmation.json) | baseline | 36/36 | 0 | 143/143 | 256.045 | 249.113 |
| [q3_200_ring_clear_combo_confirmation](../results/q3_200_ring_clear_combo_confirmation.json) | src.b_q3_200_ring_clear_combo_experiments:Continuous1130RouteClear | 36/36 | 0 | 143/143 | 255.582 | 249.295 |
| [q3_200_ring_clear_combo_confirmation](../results/q3_200_ring_clear_combo_confirmation.json) | src.b_q3_200_ring_clear_combo_experiments:Continuous1150RouteClear | 36/36 | 0 | 143/143 | 253.906 | 247.525 |
| [q3_200_selected_development](../results/q3_200_selected_development.json) | baseline | 48/48 | 0 | 309/309 | 243.739 | 236.453 |
| [q3_200_selected_development](../results/q3_200_selected_development.json) | src.b_q3_200_selected_experiment:Selected | 48/48 | 0 | 309/309 | 240.138 | 233.665 |

开发源文件在生产修改前已完整冻结至 `results/q3_200_development_sources.zip`。实验类多数继承当时的Q3Fast，因此复现历史候选应解压该快照到独立目录，不能直接让当前Q3v2继承链运行后再声称复现旧候选。最终holdout则直接以当前生产Q3v2与精确Q3v1快照配对。
