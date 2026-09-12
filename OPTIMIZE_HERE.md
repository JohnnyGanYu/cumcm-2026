# 两小时优化指引

> **最新问题三版本为 `q3-fast-v2`（200秒目标本轮更新）**。100局独立随机配对，合计平均时间241.24→237.47秒（降低1.56%），两版均1265/1265全清；尚未达到200秒。新增自适应/六站环选择、逐频道连续覆盖证书、安全清除落点与扫频中16源上界重查。见 [200秒优化报告](paper/q3_200_optimization.md) 和 [全部开发取舍](paper/q3_200_development.md)。
>
> 再次优化Q3应以当前v2为基线；`b_q3_200_eval` 的baseline固定为本轮修改前v1，`b_q3_220_eval` 的baseline更早、实际是DogStrategy。不要混用。当前Q4仍为v3，以下Q4结果与更早官方日志保留原口径。

> **最新生产版本为 `q4-fast-v3`（2026-09-12 第二轮更新）**。160 局独立配对中，合计平均时间从 499.80 降至 454.60 秒，两版均清除 2082/2082 个源，仍未达到 410 秒。最新实现、分层结果和复现命令见 [410 秒优化报告](paper/q4_410_optimization.md)。
>
> **外环半径 1870 m 不要动。** 它是几何硬约束 `R >= 1800/cos(pi/12) = 1863.5 m` 的下界加余量：贴边且径向朝外的定向源，其 180° 扇区只含更外侧的点，站位必须落在过源的切线之外。曾据「最坏覆盖间隙 + 巡回长度 + 随机场景 240 局配对」缩到 1800，随机样本里两档都满清、看着降 1.6–3.0%，**但按几何筛出的 60 局边界压力样本里 1800 每局漏 1 个源，且 T̄ 反而高 24.84 秒**（漏清频道会耗尽射线盲扫与复活兜底预算）。已回退。改这个常量前必须跑：
>
> ```bash
> python3 -m src.b_outer_beam_guarantee        # 几何必盲率，秒级
> python3 -m src.b_outer_radius_boundary --scan 6000 --episodes 60 --out results/xxx.json
> ```
>
> 教训可推广：**改动只影响稀有子群体时，随机场景的均值配对检验没有检出能力**，必须按机制构造针对该子群体的压力样本。纯距离指标（最坏覆盖间隙）对定向源不可靠——它衡量"能否走到附近"，定向可见性要求的是"站在扇区内"。
>
> 本轮已加入多起点巡回、定位区域最多 48 点就地清除、射线必失败点过滤、16 个已见频道上界停止，并调整覆盖布局。再次比较必须固定当前 v3；本次 `b_q4_410_eval` 的 baseline 特指备份 v2。下文的官方成绩与失败清除诊断保留为本轮修改前的历史依据。

看到这个文件就对了。当前成绩、还剩什么可打、怎么验证，都在下面。
不用读别的文档就能开工。

## 环境

```bash
cd <仓库根目录>
python3 -m pytest src/b_sim/test_b_sim.py -q    # 快速回归，应 172 passed（约 32 s）
```

交付的策略代码纯标准库，但**测试与实验脚本需要 pytest / hypothesis / numpy / scipy**
（见 `requirements.txt`）。本机没有裸 `python`，用 `python3` 或
`~/.venvs/cumcm-2026/bin/python3`（pytest 装在这个 venv 里）。

仓库根目录有 `pytest.ini` 后，`pytest src` 可直接全量跑，**354 项约 8 分钟**：

```bash
~/.venvs/cumcm-2026/bin/python3 -m pytest src -q
```

删掉 `pytest.ini` 会退回收集阶段挂死（20 分钟无输出）—— `src/` 混有非 B 题脚本在模块级
执行重活，如 `src/c_q1.py` 第 55 行直接 `pd.read_excel` 并往下算。下面显式列文件的写法
仍然有效，机器上没有 `pytest.ini` 时用它：

```bash
~/.venvs/cumcm-2026/bin/python3 -m pytest \
    src/b_sim/test_b_sim.py src/b_sim/test_b_pbt.py \
    src/test_q3_fast.py src/test_q4_fast.py src/test_q4_outer_radius.py \
    src/test_q4_410_geometry.py src/test_q4_410_route.py src/test_q4_near_field.py \
    src/test_q3_200_search.py src/test_q3_200_homing.py \
    src/test_q3_200_homing_candidates.py src/test_b_optimization_benchmark.py \
    src/b_eval/test_provenance.py -q
```

本机只有 4 核，`--workers 4` 是上限，开多了会互相抢 CPU 反而更慢。

## 当前成绩（官方模拟器实测，2026-09-12）

| | 池化评分指标 |
|---|---|
| Q3 | 209.0 s |
| Q4 | 486.9 s |

评分指标 = `总虚拟时间 / 清除数`，越小越好。清除率已接近满分，
**所以提速是唯一有效方向，但清除数掉一个就白干**。

## 改哪个文件

```
src/b_sim/q3_fast.py    <- 问题 3 的生产策略
src/b_sim/q4_fast.py    <- 问题 4 的生产策略
```

**`src/b_sim/strategy.py` 里的 `DogStrategy` 已不上场**，它只是 `--legacy-*` 回退路径。
在它上面调参不会改变成绩 —— 这是最容易白干两小时的坑。

## 最值得打的一处：Q4 的失败清除

真值插桩实测（Fast 版，各 30 局，数据见 `results/b_clearfail_audit_postpr7.json`）：

| | Q3 | Q4 d5 |
|---|---|---|
| `/clear` 调用 | 446 次 | 720 次 |
| 失败率 | 17.3% | **46.7%** |
| 失败时离真源多远（中位） | 27.9 m | **91.0 m** |
| 其中 >100 m 的 | 0 次 | **160 次** |
| 其中只有 1 条示向度的 | 0 次 | 91 次 |

清除半径是 20 m。**Q4 有 160 次清除打在 100 m 开外，纯属空跑。**
失败本身只花 3 s，但把机器狗送到那个错位置要约 58 s。

Q3 那 17.3% 不用管 —— 失败也都在 28 m 内，属必要微调，没有攻击面。

### 一个具体思路（未验证，交给你们）

成功清除时定位区域直径中位约 49 m，失败时约 93 m。
所以「直径还很大就先补一次测向、而不是直接扑过去」可能有收益：
一次测向 5 s 加切频 1 s，比白跑一趟 58 s 便宜十倍。

**但要小心**：单示向度时的射线盲扫是清除率支柱 —— 消融实验里关掉它，
60 局漏失从 0 涨到 9（p=0.0056），官方六局里它救回过 6 个源。
任何门控都必须先保清除率，再谈提速。

## 怎么验证（这步别省）

单局跑出来的数字没有意义 —— Q4 单局方差能到 ±25%，
我们自己曾三次用小样本得出错误结论。用配对实验：

```bash
# Q4：你的改动 vs 当前生产策略，40 局配对
python -m src.b_miss_rescue --plan paired \
    --variants prod_q4,你的变体名 \
    --episodes 40 --seed-start 71000 --directional 5 \
    --workers 4 --out results/my_test.json

# Q3：注意 --directional 0 必须显式写，该参数默认是 10（不写就跑成 Q4 场景了）
python -m src.b_miss_rescue --plan paired \
    --variants prod_q3,你的变体名 \
    --episodes 40 --seed-start 72000 --directional 0 \
    --workers 4 --out results/my_q3.json
```

`prod_q3` / `prod_q4` 就是当前生产策略，务必用它们当基线，且**要写在 `--variants` 的第一位** —— 参照臂取的是第一个。
变体注册在 `src/b_miss_rescue.py` 的 `VARIANTS` 字典里，照着加一行。

40 局约 5 分钟（4 核）。输出直接给出配对差、清除数对比和逐局胜负。

### 判断标准

1. **清除数不能降。** 输出里 `cleared=` 和 `d_src=` 任何退化直接否决，不管快多少。
2. **看配对差不看绝对值。** `d=+21.49%` 这种是相对基线的百分比。
3. **40 局不够就跑 120 局。** 我们踩过的坑：48 局看着 −26.9%，
   120 局才确认是 −29.5%；另一次 40 局的「−14.5%」在复现时消失了。

## 别再试这些（都试过了，有数据）

| 想法 | 结果 |
|---|---|
| 加大检测预算（2x/3x/5x） | 单调劣化最多 +30%，清除率已饱和 |
| 距离够近就跳过交会 | 1049 个决策点从未触发，判据无法证明 |
| 批量归航 | 越攒越慢 |
| 先普扫再逐个清除 | +34.6% |
| 远源延后处理 | 移动距离 +17.1% |
| 共址补测 | 与巡回排序叠加为负收益 |
| 提前判定频道为空 | 31 站位全沉默仍有 0.65% 误判率 |

详细数据见 `paper/main.md` 第 6 章「被数据否决的假设」。

## 改完之后

```bash
python3 -m pytest src/b_sim/test_b_sim.py -q     # 回归必须全绿（172 项）
python3 -m src.pack_dist_windows --check         # 同步到交付副本
python3 -m src.build_windows_zip --check         # 交付 zip 也要同步
```

后两条各挡一层漂移：`pack_dist_windows` 管 `src/b_sim/` -> `dist_windows/`，
`build_windows_zip` 管 `dist_windows/` -> 根目录交付 zip。交付包是从 `dist_windows/`
打的，只改 `src/` 不同步的话正式测试跑的还是旧代码。这两个坑都踩过。

`dist_windows/` 用 CRLF 行尾，手工同步时要用 `newline=''` 写入，别让整个文件
变成一行 diff。

## 时间不够时的优先级

1. 只动 `q4_fast.py`。Q4 的 486.9 s 是 Q3 的两倍多，同样的百分比改进收益更大。
2. 先跑 20 局看方向，有戏再补到 40 局，别一上来就跑 120 局。
3. 拿不准就别合。当前版本已官方实测确认，退化的风险比收益大。
