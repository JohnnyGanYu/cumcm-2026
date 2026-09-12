# 归档：已废弃的实验分支残留

评审 PR 时用 `git worktree` 在 `/tmp` 下检出过几个分支。清理这些临时目录前，
把其中**没有任何 commit 收着**的改动存到这里；已在分支或 main 上有副本的内容不重复归档。

## pr5_bounded_homing_experiment.patch

来源：`/tmp/pr5wt` 工作区的未提交改动（分支 `pr5`，PR #5 已合并，这些改动在合并之后）。

内容是把 PR #6 的 `bounded_homing` / `certified_clear` 接到 `src/b_sim/strategy.py`
上的集成实验：用示向度楔形的保守外接多边形替代最小二乘交会，若可行域直径 <= 2x18 m，
则存在一点其 18 m 圆盘覆盖整个可行域，走到该点清除必中。

**为什么不捡回来：**

1. 它改的是 `strategy.py` 里的 `DogStrategy`，而该类已不上场 —— 生产走
   `q3_fast.py` / `q4_fast.py`，`strategy.py` 只是 `--legacy-*` 回退路径。
2. 同一判据的生产实现已经在用：`q3_fast.py:139` 与 `q4_fast.py:230` 都按
   `cover_radius = max(dist(estimate, 多边形顶点))` 决定何时停止逼近，
   `q4_fast.py` 另有 `region_cover_radius_m: 19.9` 的就地清除落点。
   即 PR #6 关闭说明里那句「有价值的部分已摘入 main」是准确的。
3. 它 import 的 `src/b_sim/homing.py` 只存在于 `pr6` 分支，main 里没有。

## pr5_homing.py

`/tmp/pr5wt/src/b_sim/homing.py`（91 行，未跟踪文件，`git diff` 抓不到所以单独存）。

与 `pr6` 分支上的同名文件 MD5 完全一致（`f1dcb041...`），所以它在分支上本来就有备份，
这里只是为了让上面那份 patch 能独立读懂。

## 相关分支（保留，未删）

- `pr6` —— PR #6 主动关闭（非合并），含 `homing.py` / `silence_coverage.py`
  两套机制。关闭理由见 PR #6 讨论：Q3 120 局配对 −1.62% CI95[−6.7,−2.3] 是真收益
  且已摘入 main；`silence_coverage.py` 的判空在定向档误判率过高
  （12 站位全沉默时 11.43%、31 站位仍 0.65%），故 `empty_verdict_stations`
  设为 999 等于关闭，整个 PR 不合并。

若要重启这个方向，起点是 `pr6` 分支的 `silence_coverage.py`，
需要先解决定向档判空误判率，而不是这份接在 `DogStrategy` 上的 patch。
