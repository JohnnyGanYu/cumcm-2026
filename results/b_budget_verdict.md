# 检测预算交换比：结论是「不该扩张，应该收缩」

## 问题与前提检验

原始假设：虚拟时间只用了预算的 5%，所以扩张检测预算去多清源，应能改善评分指标。

**该假设的前提不成立。** 评分指标 `mean_clear_time_s = 各源清除时刻的平均`，
其改善依赖「分母（清除数）还有增长空间」。离线实测（含 `nearby_revive` 的当前 HEAD 工作树）：

| 场景 | 局数 | 清除率 | 全清局数 |
|---|---|---|---|
| Q3 全向 | 300 (seed 9000+) | **3872/3872 = 100%** | 300/300 |
| Q4 5定向 | 300 (seed 9000+) | 3867/3872 = 99.87% | 295/300 |
| Q4 10定向 | 200 (seed 7000+) | 2589/2599 = 99.6% | 190/200 |

Q3 分母**已完全饱和**，无源可换；Q4 也仅剩 0.13% 缺口。

## 盈亏平衡的解析条件

设当前总时间 T、清除数 N，指标 M = T/N。多花 ΔT 多清 1 个源后指标改善的条件：

```
(T + ΔT)/(N + 1) < T/N   ⟺   ΔT < T/N = M
```

**边际预算的盈亏平衡点恰等于当前指标值本身。** 实测：

- Q3：M ≈ 446 s ⇒ 多清 1 个源最多可花 446 s
- Q4：M ≈ 1184 s ⇒ 多清 1 个源最多可花 1184 s

虚拟时间余量确实巨大（用掉 1.5%–4.4%，余 34 万秒以上），
但**约束根本不是虚拟时间，而是「已经没有源可清」**。

## Pareto 扫描：扩张全部劣化指标

Q4 按「扫描站位总数」定义投入倍数（基线 = 原点 + 内环12 = 13 站位），40 局同种子：

| 配置 | 指标 s | Δ% | 清除率 | 检测次数 |
|---|---|---|---|---|
| x1 基线 | 1281.4 | +0.00% | 509/509 | 525.6 |
| x3 inner18+outer20 | 1344.5 | **+4.93%** | 509/509 | 594.2 |
| x2 inner24 | 1383.2 | **+7.95%** | 509/509 | 660.0 |
| x5 inner24+outer40 | 1383.2 | **+7.95%** | 509/509 | 660.0 |
| x5 三环 | 1440.7 | **+12.43%** | 501/509 | 693.9 |
| x5 inner64 | 1665.2 | **+30.0%** | 509/509 | 1121.4 |

清除率不动（已饱和），时间单调上升 ⇒ 指标单调劣化。5x 投入劣化 30%。
三环方案甚至**掉了 8 个源**（更长的空驶挤掉了归航时间）。

尾部收尾旋钮（`belief_search_rounds` 400、`blind_clear_max_targets` 150、
`ray_clear_max_attempts` 800）实测**逐字节零影响** —— 这些上限在基线下从未被触及，
是无效旋钮。

## 最优方向是反向：收缩内环

三个独立种子块一致复现：

**Q3**（`probe_ring_stations` 12 → 9）

| 配置 | seed 9000+ (300局) | seed 7000+ (200局) |
|---|---|---|
| 基线 n=12 | 452.5 s / 100% | 461.2 s / 100% |
| **n=9** | **426.1 s (-5.8%) / 3872/3872** | — |
| n=10 | 432.2 s (-4.5%) / 100% | 442.2 s (-4.1%) / 100% |
| n=8 | 420.3 s (-7.1%) / 3871/3872（掉1） | 426.2 s (-7.6%) / 2598/2599 |

n=8 指标更低但开始掉源；**n=9 是保持零漏失的拐点**。

**Q4**（内环收缩 + 定向环适度加密）

| 配置 | 指标 s | Δ% | 清除率 |
|---|---|---|---|
| 基线 (inner12, dirring18) | 1211.7 | +0.00% | 3867/3872 |
| **inner8 + dirring30** | **1170.2** | **-3.42%** | 3867/3872 |
| inner8 + dirring26 | 1168.2 | -3.59% | 3865/3872（掉2） |
| dirring30 单独 | 1201.2 | -0.86% | 3867/3872 |

注意 `dirring30` 在 120 局 seed 7000+ 上达到 **1558/1558 全清**且指标持平基线
（1241.0 vs 1241.4）——这是唯一「扩张换到源且不亏」的点，但收益仅 2 个源、
在 300 局上未复现全清。

## 推荐参数

```python
# Q3 全向
StrategyConfig(probe_ring_stations=9)                    # 452.5 -> 426.1 s (-5.8%)，零漏失

# Q4 含定向
StrategyConfig(probe_ring_stations=8,
               directional_ring_stations=30)             # 1211.7 -> 1170.2 s (-3.4%)
```

保守取向：Q4 若优先保清除率而非指标，用 `probe_ring_stations=10,
directional_ring_stations=30`（1178.0 s，-2.8%）。

## 一个代码事实

`StrategyConfig.max_measures_per_channel` 是**死参数**——定义后从未被读取
（`grep` 全仓仅出现在定义行与历史日志）。想限制单频道检测量应改
`max_homing_measures` / `narrow_sector_extra_measures`。

另：Q4 下 `outer_ring_stations` 也不生效——`strategy.py:284` 在
`directional_probe=True` 时改用 `directional_ring_stations`。
本次扫描中 `x2_inner12_outer14` / `x5_inner24_outer40` 因此与其内环同配置逐字节相同。

## 复现

```bash
python -m src.b_budget_sweep --plan multiplier --episodes 40 --directional 5 \
    --out results/b_budget_mult_q4.json
python -m src.b_budget_sweep --plan lean --episodes 40 --directional 0 \
    --out results/b_budget_lean_q3.json
python -m src.b_budget_pareto          # -> figures/b_budget_pareto.png
```

数据：`results/b_budget_{base,mult,lean,holdout,expand,final}_*.json`、
`results/b_budget_pareto_summary.json`。33/33 单测通过。
