# CUMCM 2026 — B 题：无线电干扰源的快速定位与清除

私有仓库，队内只读协作。**本队选定 B 题**，仓库只保留 B 题相关内容。

## 这个仓库怎么用

按用途分四类，各自只需读一份文档：

| 我要做什么 | 读哪里 |
|---|---|
| **最终打包与提交（赛日收尾）** | [`RACEDAY.md`](RACEDAY.md) —— 自检、脱敏、构建、MD5 冻结 |
| **提交前逐项核对** | [`state/submission_checklist.md`](state/submission_checklist.md) |
| **读论文 / 改论文** | [`paper/README.md`](paper/README.md) —— 成稿是 `paper/main.md` |

一条命令跑完全部提交前检查：

```
python scripts/raceday_preflight.py           # 五闸门，约 8 分钟
python scripts/raceday_preflight.py --quick   # 跳过全量回归，约 1.5 秒
```

五个闸门分别挡：交付包未同步（跑的是旧策略）、zip 未重打（MD5 与文档不符）、
论文含身份信息或附录漏源码、文档与代码参数脱节、回归失败。

## 目录

```
problem/B题/     题目原文与附件（只读）
src/b_sim/       离线仿真器 + 机器狗策略（交付的就是这些）
src/             辅助脚本：诊断、时间预算分解、脱敏、论文与表格构建
scripts/         赛日总闸、提交物哈希冻结
dist_windows/    Windows 交付包（跑官方模拟器用）
official_runs/   官方模拟器实测记录（仅脱敏版）
results/         机器可读结果（论文每个数字都能在此复算）
figures/         论文配图（13 张：8 张正文图 + 5 张问题分析流程图 图 2-1~2-5；与策略参数联动，改参数须重生成）
paper/           论文源与渲染，成稿 = paper/main.md
supporting/      支撑材料清单与脱敏说明
state/           决策日志、AI 台账、主张-证据台账、提交检查单
```

## 题目要点

机器狗在半径 1800 m 的圆形区域内搜索并清除未知数量的无线电干扰源。每个频道
（1–20）至多一个干扰源，有效接收半径不低于 1000 m。检测得到示向度（含 ±1° 误差），
清除半径 20 m。通过 HTTP+JSON 接口对接官方模拟器（`127.0.0.1:2026`）。

时间成本：检测 5 s、清除 3 s、移动按距离折算；虚拟时间上限 360000 s，程序运行
时间上限 1200 s。**评分指标 = 虚拟时间 / 清除个数**（平均定位清除时间，越小越好）。

问题 1、2 为解析与算法设计；问题 3（全向源）、问题 4（含定向源）需实机测试。

---

## 赛制约束

> **正式测试已完成（六局全清）；最终打包与提交前请先读 [`RACEDAY.md`](RACEDAY.md)** ——
> 自检、成绩判断、脱敏、MD5 冻结流程。基于六局官方演练与六局正式测试实测编写。


- 演练测试不限次数，结束会告示本局干扰源总数。
- 问题 3、问题 4 各只有 **3 次**正式测试（已用完，全部回填入稿）。
- **第一阶段 MD5 提交 09-13 20:00 前**；第二阶段上传 09-13 20:30–09-14 14:00。

---

## 当前成绩

**2026-09-13 Q4 外环半径回退 1800 → 1870（交付版本 v1.0.3 起）**。
曾据「最坏覆盖间隙 + 巡回长度 + 随机场景 240 局配对」把外环缩到 1800 m（即区域半径），
三项指标同向、两档满清零退化，看着是稳的改进 —— **实际是错的**。
1870 是几何硬约束的下界加余量：贴边且辐射轴径向朝外的定向源，其 180° 扇区只含更外侧的点，
站位必须落在过源的切线之外，n = 12 时需 `R·cos(π/12) ≥ 1800`，即 `R ≥ 1863.5 m`。

按几何条件筛出的边界压力样本（60 局配对，只改该常量）：

| | 局数 | 清除 | 满清局 | 平均 T̄ |
|---|---:|---:|---:|---:|
| 外环 1800 | 60 | 899/960 | 0 | 417.53 s |
| 外环 1870 | 60 | **960/960** | **60** | **392.69 s** |

耗时 d = −24.84 s CI95[−33.46, −16.21]；清除数 d = +1.017 CI95[+0.984, +1.049]。
1800 在 60/60 局全部漏源，**且耗时也更差** —— 漏清频道会耗尽射线盲扫与就近复活的兜底预算，
所以这不是「速度换清除率」的权衡，1800 在两个维度上都更差。

教训有普适性：**改动只影响稀有子群体时，随机场景的均值配对检验没有检出能力**，
必须按机制构造针对该子群体的压力样本；纯距离指标（最坏覆盖间隙）衡量的是「能否走到附近」，
而定向可见性要求的是「站位落在扇区内」，两者对贴边朝外的源结论相反。
详见 [PR #9](https://github.com/JohnnyGanYu/cumcm-2026/pull/9) 与论文 7.13 节。

**2026-09-12 问题四再次更新**：`--directional` 默认已接入 `q4-fast-v3`。
160 局独立配对中，合计平均时间 **499.80 → 454.60 s（降低 9.04%）**，两版均清除 **2082/2082** 个源。**410 秒目标尚未达成**；新版 48/160 局在 410 秒以内。
新版为离线验证，尚无官方上机成绩。实现、分层结果和复现命令见 [问题四 410 秒优化报告](paper/q4_410_optimization.md)，此前 v2 结果保留在 [历史报告](paper/q4_450_optimization.md)。
**2026-09-12 问题三再次更新**：默认接入 `q3-fast-v2`。独立100个随机案例的合计平均时间 **241.24 → 237.47 s（降低1.56%）**，逐局均值 **245.86 → 241.85 s**，两版均 **1265/1265全清**；**200秒目标尚未达到**。
本轮为离线结果，尚无新版官方成绩。见 [200秒优化报告](paper/q3_200_optimization.md) 和 [开发取舍](paper/q3_200_development.md)。`--legacy-q3` 回退到更早的DogStrategy；精确Q3v1备份见 `results/q3_200_baseline/`。
以下原有成绩表保留为历史基线，不代表两个新版策略的官方成绩。

**历史：问题三第一轮独立验证（q3-fast-v1）**：100 个未参与调参的随机源数案例中，逐局平均时间的算术均值从 **254.08 s** 降到 **246.04 s**（降低 **3.17%**），两臂均清除 **1263/1263** 个源。新版在随机组与 10/13/16 源三组压力测试合计 **160/160 局全清、2043/2043 个源清除**。**综合均值仍高于 220 s，差 26.04 s**；16 源组均值为 **192.41 s**。均为离线结果，本次未连接官方模拟器。

**历史对照配置**：覆盖环 `probe_ring_radius=1100.0` / `probe_ring_stations=8`（含原点共 9 站位），
TSP 巡回排序（最近邻 + 2-opt + Or-opt），单示向度时射线盲扫兜底。

### 官方模拟器演练（2026-09-11，六局；脱敏件见 `official_runs/*_20260911_*.sanitized.json`）

| 问题 | 源数配置 | 清除 | T̄ | 检测 | 程序运行 |
|---|---|---|---|---|---|
| Q3 | 14 全向 | 14/14 | 233.8 s | 97 | 7 s |
| Q3 | 12 全向 | 12/12 | 287.7 s | 132 | 7 s |
| Q3 | 10 全向 | 10/10 | 298.3 s | 117 | 7 s |
| Q4 | 3 全向 + 12 定向 | 15/15 | 672.5 s | 290 | 85 s |
| Q4 | 4 全向 + 7 定向 | 11/11 | 900.5 s | 404 | 70 s |
| Q4 | 6 全向 + 5 定向 | 11/11 | 794.1 s | 350 | 72 s |

**六局合计 73/73 零漏失。** 池化 T̄：Q3 269.7 s、Q4 776.4 s。程序运行时间上限 1200 s，余量充足。

Q4 三局里射线盲扫共救回 6 个源（日志中 `cleared by ray blind sweep`）—— 该分支若缺失这 6 个源会漏掉。

### 官方模拟器演练（2026-09-12，Fast 策略首次上机；脱敏件见 `official_runs/*_20260912_*.sanitized.json`）

| 问题 | 清除/总数 | T̄ | 虚拟时间 | 检测 | 清除调用 |
|---|---|---|---|---|---|
| Q3 | 11/11 | 286.5 s | 3151 s | 131 | 11 |
| Q3 | 15/15 | **168.1 s** | 2521 s | 95 | 16 |
| Q3 | 15/15 | 193.1 s | 2897 s | 102 | 19 |
| Q4 | 15/15 | **446.5 s** | 6698 s | 231 | 29 |
| Q4 | 10/10 | 608.6 s | 6086 s | 303 | 16 |
| Q4 | 13/13 | 439.8 s | 5717 s | 237 | 24 |

**六局 79/79 全清。** 池化 T̄（`总虚拟时间 / 总清除数`，与 `arena.py` 评分口径一致）：**Q3 209.0 s、Q4 486.9 s**。

与 9-11 那六局旧策略同环境对比：**Q3 269.7 → 209.0 s（−22.5%）、Q4 776.4 → 486.9 s（−37.3%）**。
这是官方环境对 PR #7 的实测确认 —— 离线 120 局配对测出的 Q4 −29.5% 在真实模拟器上更高。
Q4 清除调用从旧策略的约 70 次降到 16–29 次，与「射线盲扫沿射线密铺」被消掉相符。

七局中另有一局因队号误填 `2025…` 开头被服务端拒绝（`/enter` 返回 `accepted: false`），
归档为 `q3_practice_20260912_r1_enterfail.sanitized.json`。**与代码无关**，
但它说明日志兜底有效：崩溃时仍完整落盘了 `crash.type` 与协议记录。

每局日志的 `strategy_class` 字段确认七局全部跑的是 `Q3FastStrategy`/`Q4FastStrategy`，非回退版。

### 离线留出集

| 场景 | 局数 | T̄ | 清除率 |
|---|---|---|---|
| Q3 全向 | 200（seed 5200+） | 247.1 s | 2640/2640 |
| Q4 5 定向 | 40（seed 7041+） | 680.5 s | 512/512 |

**边界场景专项**（交付策略实测，源数与定向占比取两端，各 40 局，seed 21000）：六档合计
3120/3120 零漏失，T̄ 区间 **196.2 s**（16 源全向）– **642.5 s**（10 源全定向），跨度 3.3 倍。
该区间由案例参数决定，非策略差异，当日判断成绩时须按本局源数比对（见 `RACEDAY.md`）。
数据源 `results/b_boundary_matrix.json`，重跑 `python -m src.b_boundary_matrix`。
旧策略同档为 209.7 – 1168.4 s，每档都更慢，最差档差 45% —— 引用旧区间会把退化当成正常。

---

## 交付包

当前版本 **[v1.0.5](https://github.com/JohnnyGanYu/cumcm-2026/releases/tag/v1.0.5)**，MD5 `20d5a2ff97ae8201bedf415b8f3e8a40`，14 个文件，纯 Python 标准库。
含 `q3-fast-v2` / `q4-fast-v3`，Q4 外环 1870 m。
v1.0.2 已弃用：那版外环 1800 m 会漏清贴边朝外的定向源（见 PR #9）。

v1.0.1（MD5 `bd90df7b406a66921afe8cfaec556f03`）为上一代基线，其官方六局实测成绩见上文「官方模拟器演练（2026-09-12）」，仅在需要对照时使用。

```
python run_official.py --robot-id 你的参赛队号                 # 问题 3
python run_official.py --robot-id 你的参赛队号 --directional    # 问题 4
```

出意外要回退到优化前策略时加 `--legacy-q3` / `--legacy-q4`。日志的 `strategy_class`
字段记录了这一局实际跑的策略类，赛后凭它归因成绩。

根目录那份同名 zip 是构建产物，由 `python -m src.build_windows_zip` 生成、
`--check` 校验它与 `dist_windows/` 一致。**它存在的唯一目的是防版本漂移**：
此前它曾停留在旧版本，当天若直接解压会拿到慢约 24% 的策略（见 `RACEDAY.md`）。
正式分发时以对应 Release 的策略版本及 MD5 为准；当前分支构建包的信息见上方。
## 怎么跑

离线演练（本机，无需官方模拟器）：

```bash
python -m src.b_sim.rehearse --episodes 60 --seed-start 5000                  # 问题 3
python -m src.b_sim.rehearse --episodes 40 --seed-start 7001 --directional 5   # 问题 4
python -m pytest src/b_sim/test_b_sim.py -q                                    # 172 项单元/回归
python -m pytest src/b_sim/test_b_pbt.py -q                                    # 33 项性质测试
```

优化诊断与调参：

```bash
python -m src.b_sim.timeline    --episodes 20                    # 四项耗时分解 + 绕路倍数
python -m src.b_sim.phase_probe --episodes 20                    # 按阶段归因移动距离
python -m src.b_sim.coverage_audit --episodes 12                 # 顺路覆盖审计
python -m src.b_sim.tune --episodes 200                          # 单点评估（并行）
python -m src.b_sim.tune --episodes 200 --ablation               # 消融实验
python -m src.b_sim.tune --episodes 200 --sweep dogleg_offset_deg=8,13,18,24
```

生成论文配图：

```bash
python -m src.b_q1_figures          # Q1 定位区域直径 + 外接圆覆盖/反例
python src/b_q2_figures.py          # Q2 候选区域 + 交会角 + 精度-耗时权衡
python -m src.b_sim.coverage_design # 覆盖站位数值依据
python -m src.b_coverage_figures    # 覆盖站位散点 + 最坏间隙条形图
```

对接官方模拟器（Windows）：先同步交付包，再见 `dist_windows/README_WINDOWS.md`。

```bash
python -m src.pack_dist_windows           # 同步 src/b_sim -> dist_windows/b_sim
python -m src.pack_dist_windows --check   # 提交前自检是否已同步
```

测试依赖（`pytest` + `hypothesis`，仅离线测试/彩排用；提交给官方模拟器的机器狗程序本身只用标准库）：

```bash
python -m pip install -r requirements-dev.txt
```

---

## 策略要点

### 新增模块

| 模块 | 作用 |
|---|---|
| `src/b_sim/route.py` | 开放 TSP 排序：最近邻 + 2-opt + Or-opt + 廉价插入 |
| `src/b_sim/feasible.py` | 可行域集合约束 +（含定向模式的）贪心最大覆盖站位规划 |
| `src/b_sim/timeline.py` | 四项耗时分解 + 最近邻巡回基准 + 绕路倍数 + 检测有效率 |
| `src/b_sim/phase_probe.py` | 按策略阶段归因移动距离 |
| `src/b_sim/coverage_audit.py` | 顺路覆盖审计：停车点已经免费覆盖了多少目标区域 |
| `src/b_sim/tune.py` | 并行参数扫描与消融实验 |
| `src/pack_dist_windows.py` | 把运行时同步到 Windows 交付包（含 `--check` 自检） |
> 顺带修掉一个此前的隐患：`dist_windows/`（真正连官方模拟器的那份代码）是手抄的，已与
> `src/b_sim/` 漂移——`client.py` 少了发送前坐标越界拦截与 400/409 处理，`geometry.py` 少了三分之二。
> 现在由 `src/pack_dist_windows.py` 脚本化同步，避免"离线调好的策略"与"正式测试跑的策略"不是同一份。

### 测试

| 套件 | 数量 |
|---|---:|
| `src/b_sim/test_b_sim.py` 单元 / 回归 | **172** |
| `src/b_sim/test_b_pbt.py` 面向性质测试（P1–P8，`hypothesis`） | **33** |
| `src/test_q4_410_geometry.py` Q4 定位几何 | **17** |
| `src/test_q3_fast.py` Q3 生产策略 | **15** |
| `src/test_q4_410_route.py` 路径规划 | **12** |
| `src/test_q3_200_search.py` Q3 搜索与连续覆盖判空 | **12** |
| `src/test_q3_200_homing.py` Q3 归航 | **12** |
| `src/test_q4_fast.py` Q4 生产策略 | **10** |
| `src/test_b_optimization_benchmark.py` 配对评测框架自测 | **10** |
| `src/b_eval/test_provenance.py` 评测溯源与 fail-closed | **10** |
| `src/test_q3_200_homing_candidates.py` Q3 归航候选 | **5** |
| `src/test_q4_outer_radius.py` Q4 外环半径几何下界 | **4** |
| `src/test_q4_near_field.py` 近场定向约束 | **2** |
| `src/test_run_official_inputs.py` 正式入口输入护栏 | **2** |
| `src/test_figures_match_strategy.py` 配图与策略参数一致性 | **2** |
| `src/test_docs_match_code.py` 文档声称参数与代码一致性 | **4** |
| `src/test_build_table1.py` 表 1 生成器（池化口径/作废局过滤） | **5** |
| `src/test_sanitize_official_run.py` 脱敏完整性与复核独立性 | **6** |
| `src/test_run_official_crash_safety.py` 崩溃仍落盘与 None 指标护栏 | **5** |
| `src/test_zip_reproducible.py` 交付 zip 字节可复现 | **3** |
| `src/test_boundary_matrix.py` 边界矩阵池化口径与表-数据一致 | **6** |
| `src/test_paper_answers_all_subquestions.py` 题目子问覆盖 | **3** |
| `src/test_abstract_traceable.py` 摘要数字可溯源 | **4** |
| `src/test_paper_citations.py` 论文引用文件存在与成绩可复算 | **4** |
| 合计 | **358，全绿（487.4 s）** |

`test_q4_outer_radius.py` 是把 `outer_radius >= 1800/cos(π/12)` 这条几何下界写成断言，
防止再有人凭随机场景的均值指标把外环缩回 1800（详见 PR #9 与论文 7.13 节）。

**提交前一条命令跑完全部闸门**：

```bash
python scripts/raceday_preflight.py           # 五闸门，含 8 min 全量回归
python scripts/raceday_preflight.py --quick   # 跳过回归，约 1.5 s
```

它串起交付包同步、zip 同步、论文匿名与附录覆盖、四组一致性断言、全量回归，
任一不过即非零退出，并说明该闸门挡的是什么错。

**跑法**：`pytest src` 现在可直接全量跑（358 项、约 8 分钟），前提是仓库根目录的 `pytest.ini` 在位。

删除该文件会退回旧行为：收集阶段挂死，20 分钟无输出。根源曾是 `src/` 混有非 B 题脚本在
模块级执行重活（无 `if __name__` 保护）；这些脚本已清理（2026-09-13），但 `pytest.ini`
仍保留，作为收集根锚定，避免回归。
完整命令见 [`OPTIMIZE_HERE.md`](OPTIMIZE_HERE.md) 的「环境」一节。测试依赖 `pytest` /
`hypothesis` / `numpy` / `scipy`（`requirements.txt`），交付的策略代码本身仍是纯标准库。

---

---

## 优化过程（归因与消融）

本章是开发期方法记录（基线画像、五项被否决策略、消融实验、时间预算分解、TSP 收益上限），
赛日不需要读；结论已固化进交付版策略与论文第 7、8 章，完整实测数据见
`paper/b_optimization_section.md`（696 行全记录）与 `results/b_*.json`。

## 项目状态

**代码与交付包已冻结，正式测试六局已完成并入稿，论文已成稿定版（等待最终打包）。**

已经拿到的：

- **评分指标有硬提升且可复现**：Q3 T̄ −44.9%、Q4 −44.0%，清除率不降。调参集与
  留出集降幅一致，排除对种子过拟合；独立留出集 260 局 3347 源全清（7.14 节）。
- **优化是归因驱动的**：先用 `timeline.py` / `phase_probe.py` 定位「移动占 80%、
  一进一出的乒乓占绕路大头」，再逐条动刀，每条机制都有消融数据。
- **结论边界诚实**：明确写出「完全认证在问题 4 不可行」（47 站位 / 21.5 km 仍剩
  120 格），不掩盖 99.81% 而非 100%；被数据否决的方案单独成章（第 8 章模型检验）。
- **问题 1 / 2 的理论真正用进了策略**：最小外接圆半径 ≤ 20 m 是可证明的直接清除
  判据；交会角与基线的关系决定 dog-leg 长度按距离比例取。
- **提交链路全脚本化**：论文构建、表 1 生成、脱敏、哈希冻结各有脚本与回归，
  358 项测试里三分之一是防「文档与代码脱节」的断言。

正式测试（2026-09-13）：Q3/Q4 各 3 次共六局全部清除
（Q3 229.50 / 270.71 / 242.79 s，Q4 438.85 / 346.73 / 523.80 s），
已在 7.16 节表 7-14 回填、官方 `.jlog` 与客户端 `.json` 均脱敏入库
（`official_runs/formal-p*.sanitized.jlog`、`official_runs/*_formal_*.sanitized.json`）。

已知局限（写进论文 9.2 节，不隐藏）：

1. **`near` 分支未经真实验证。** 逻辑正确（收到 `near` 立即清除，全仓单一
   `_measure` 调用点），但官方演练 2415 次检测里 `near` 出现 **0 次**，
   只在离线 mock 走到过。
2. **沉默覆盖认证（PR #6）未采纳。** 自建蒙特卡洛显示定向档判空误判率偏高
   （12 站位全沉默时 11.43%），故未纳入交付版；`silence_coverage.py` 不在
   `src/b_sim/` 内。若能把误判率压到 1% 以下则有实质收益 —— 实测空频道
   占检测总数 62%–79%。

## 支撑材料文件清单（提交件逐项对应）

以下全部路径相对仓库根目录；**提交物 = 论文 PDF + 附件说明表（附录 A）所列文件**，
打包时按下表逐项核对存在性。

| # | 材料 | 路径 |
|---|---|---|
| 1 | **论文成稿 PDF**（最终提交件，含摘要页首） | `paper/build/submission.pdf` |
| 2 | 论文源（可复现构建） | `paper/main.md` + `src/build_paper.py` |
| 3 | **AI 工具使用详情**（文件名按规定钉死，勿改名） | `paper/build/AI工具使用详情.pdf`（源 `state/ai_usage_log.csv` + `src/build_ai_usage_pdf.py`） |
| 4 | 正式测试官方日志（Q3/Q4 各 3 次，脱敏） | `official_runs/formal-p3-{1,2,3}-{编码}.sanitized.jlog`、`official_runs/formal-p4-{1,2,3}-{编码}.sanitized.jlog` |
| 5 | 正式测试客户端协议日志（脱敏） | `official_runs/q3_formal_20260913_{090030,133127,134515}.sanitized.json`、`official_runs/q4_formal_20260913_{090154,133414,134657}.sanitized.json` |
| 6 | 正式测试控制台输出（脱敏） | `official_runs/*_console.sanitized.txt` |
| 7 | 离线彩排与实验聚合结果（正文数字可复算） | `results/b_*.json`（240+ 个） |
| 8 | 论文配图（13 张，含 5 张流程图） | `figures/*.png`（生成脚本 `src/b_problem_analysis_flow.py`、`b_q1_figures.py` 等） |
| 9 | 提交给官方模拟器的完整程序（纯标准库） | `dist_windows/`（`run_official.py` + `b_sim/`） |
| 10 | 交付 zip（与 `dist_windows/` 字节一致，`build_windows_zip --check` 验证） | `CUMCM2026B_robot_windows.zip`（MD5 `20d5a2ff97ae8201bedf415b8f3e8a40`） |

**正式测试六局对应关系**（表 7-14 / 表 1）：Q3 `XXGQ-5ASG-XBGT-4UJT` →
`formal-p3-1`，`JR95-ZFS4-E7GX-CMM2` → `formal-p3-2`，`UJWN-YJCB-4EKD-5535` →
`formal-p3-3`；Q4 `2TAA-DXZ6-TYVZ-25YX` → `formal-p4-1`，`7GMK-C5YS-AP87-8858` →
`formal-p4-2`，`5ZFU-A2JM-4RFW-89SN` → `formal-p4-3`。

打包检查（逐项 `test -f` 或见 `scripts/raceday_preflight.py` 五闸门之一）。

## 匿名性

提交物全程不得出现学校、姓名、队号、指导教师。官方产出的 `.jlog` / `.json`
内嵌真实队号与 `server_ticket`，控制台输出含 Windows 用户名，**只有脱敏件可入库**：

```
python src/sanitize_official_run.py <文件...> --team-no <真实队号>
```

脚本替换队号与 Windows 用户名、把 7 个服务端密钥字段置零，并用**独立清单**
复核（不复用替换清单，否则摘掉一个字段就自动跳过检查）。原始件由 `.gitignore`
白名单式排除。论文与 PDF 的匿名由 `build_paper.py` 自检，含 DOCX 元数据里的
作者字段。详见 [`supporting/README.md`](supporting/README.md)。
