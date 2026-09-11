# Windows 端操作说明

## 一次性准备

1. 解压 `Jammers-simulator-win64.7z`，得到 `Jammers-simulator/jammers-simulator.exe`
   （若报缺运行环境，改用 `Jammers-simulator-full-win64.7z`）
2. 双击运行，联网登录：填**参赛队号 + 队员1姓名 + 队员1手机号**，设置密码
   - 密码 12-64 字符，含字母和数字，不能包含完整队号或手机号连续 6 位
3. 确认已装 Python 3.10+（`python --version`）。本程序只用标准库，无需 pip 安装

## 跑演练测试（不限次数，放心练）

1. 模拟器界面选 **"问题3演练测试"** 或 **"问题4演练测试"**
2. 点开始 → 等数据准备 → **看到 5 秒倒计时结束**、接口开放
3. 立刻在解压出的本目录下运行：

```
python run_official.py --robot-id 你的参赛队号
```

问题 4 加 `--directional`：

```
python run_official.py --robot-id 你的参赛队号 --directional
```

`--robot-id` 必须和模拟器登录的参赛队号**逐字节一致**，否则返回 accepted=false。

## 重要注意

- **不要在正式测试窗口里用 `--dry-run`**。它会真的调一次 `/enter`，
  而一次测试只允许 enter 一次；探测后必须重开一次测试才能正式跑。
  `--dry-run` 只用来确认"模拟器装好了、端口对了"。
- 程序跑完会在当前目录生成 `official_log_<时间戳>.json`，
  含客户端完整指令与响应序列。请与模拟器界面成绩及可导出的官方日志一同保存，作为复核依据。
- 正式测试问题3、问题4**各只有 3 次机会**，且 **9/13 17:30 后不能开新测试**，
  建议 15:30 前跑完。
- 同一账号不能双设备登录；原设备崩溃后服务器最多约 120 秒释放会话。

## 看什么指标

输出里的 **平均定位清除时间** 就是评分指标，越小越好。同时确认：
- 清除个数与同一局结束后告示的干扰源总数核对（应该 10-16 个）
- 程序运行时间远低于 1200 s
- 虚拟时间远低于 360000 s

## 本交付包的策略与复验证据

本包保留 main 的 TSP 巡回；Q3 默认增加全部示向的有界交会与实际沉默覆盖认证，Q4 因实验候选新增漏清而默认关闭这两个开关。覆盖增益均改用等价整数位集计算。完整每档 300 对评测、Q3 300 秒/Q4 500 秒目标差距、误判审计见仓库根目录 `项目优化.md`，原始数据见 `results/b_tsp_optimization_final.json`。这些是离线验证，官方模拟器成绩以实际运行结果为准。

运行日志分别记录 `certified_empty_channels`、`abandoned_channels` 和 `unresolved_channels`。放弃搜索不等于确认空频道；只有已清 16 个或其余频道均有覆盖认证时，程序才给出相应的全清依据。未知总数时，不能把清除数本身当成总数。

本包由 `python -m src.pack_dist_windows` 从源码同步，包含 `MANIFEST.sha256`。在完整仓库可运行 `python -m src.pack_dist_windows --check` 校验源码、入口、说明和 ZIP 一致性。

## 记录干扰源总数（题目要求的比例统计值）

干扰源总数**官方接口不返回**（附件2 明确说明），只在演练测试结束时由模拟器**界面**告示。
机器狗程序拿不到这个数，必须人工抄录，否则算不出题目要求的
"被清除干扰源个数的比例"。

演练测试（先跑，结束后从界面抄下总数）：

```
python run_official.py --robot-id 你的队号
```

看到界面告示总数（例如 14）后，保留原始日志，并将**同一局**的总数、案例编码和成绩记录到复核表。
若补录到日志，需同时核对 `reported_total_jammers`、比例及论文表格字段；只改一个字段不会自动重算其他字段。
`--total-jammers` 只用于运行前已经可靠知道本局总数的情况，绝不能把上一局的总数传给下一局。

正式测试（务必记录案例编码，论文表1需要该列）：

```
python run_official.py --robot-id 你的队号 --test-kind formal --case-code 界面上的案例编码
python run_official.py --robot-id 你的队号 --test-kind formal --case-code 编码 --directional   # 问题4
```

日志文件名会自动区分，例如 `official_log_q3_formal_20260913_101530.json`，
其中 `paper_table1_row` 字段可直接填进论文表 1。

正式测试未告示总数时，`--total-jammers` 留空；检查 `full_clear_assessment` 的完成依据及
`unresolved_channels`，无法认证的频道应保留为未解决，不能据程序退出认定全清。
