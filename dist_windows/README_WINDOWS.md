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

2026-09-12 更新：问题四默认使用 `q4-fast-v2`，采用 950 米内环 6 站、1900 米外环 14 站的统一巡回，
并修正旧观测点方位角被套用到当前位置的问题。日志 `summary.strategy_profile` 记录实际策略版本。
最新离线结果见交付包内的 `Q4_OPTIMIZATION_REPORT.md`；下方早期成绩表仅为历史记录。
如需与原版对照，可加 `--legacy-q4`：

```
python run_official.py --robot-id 你的参赛队号 --directional --legacy-q4
```

`--robot-id` 必须和模拟器登录的参赛队号**逐字节一致**，否则返回 accepted=false。

## 重要注意

- **不要在正式测试窗口里用 `--dry-run`**。它会真的调一次 `/enter`，
  而一次测试只允许 enter 一次；探测后必须重开一次测试才能正式跑。
  `--dry-run` 只用来确认"模拟器装好了、端口对了"。
- 程序跑完会在当前目录生成 `official_log_<时间戳>.json`，
  含完整指令与响应序列。**模拟器自己不提供这个功能，支撑材料需要它。**
- 正式测试问题3、问题4**各只有 3 次机会**，且 **9/13 17:30 后不能开新测试**，
  建议 15:30 前跑完。
- 同一账号不能双设备登录；原设备崩溃后服务器最多约 120 秒释放会话。

## 看什么指标

输出里的 **平均定位清除时间** 就是评分指标，越小越好。同时确认：
- 清除个数 = 本局干扰源总数（应该 10-16 个）
- 程序运行时间远低于 1200 s
- 虚拟时间远低于 360000 s

## 离线仿真器上的当前成绩（供对照）

| 场景 | 局数 | 全清 | 清除率 | 平均定位清除时间 |
|---|---|---|---|---|
| 问题3 全向 | 60 | 60/60 | 100% | 477 s |
| 问题4 含5定向 | 40 | 40/40 | 100% | 1275 s |
| 问题4 换种子交叉验证 | 30 | 30/30 | 100% | 1259 s |

**若官方模拟器结果与此差距明显，把 `official_log_*.json` 回传**，
我据此校准离线仿真器再重新调参。官方行为是最终判据。

## 记录干扰源总数（题目要求的比例统计值）

干扰源总数**官方接口不返回**（附件2 明确说明），只在演练测试结束时由模拟器**界面**告示。
机器狗程序拿不到这个数，必须人工抄录，否则算不出题目要求的
"被清除干扰源个数的比例"。

演练测试（先跑，结束后从界面抄下总数）：

```
python run_official.py --robot-id 你的队号
```

看到界面告示总数（例如 14）后，日志里的 `reported_total_jammers` 字段直接改成 14，
或下次运行时带上参数：

```
python run_official.py --robot-id 你的队号 --total-jammers 14
```

正式测试（务必记录案例编码，论文表1需要该列）：

```
python run_official.py --robot-id 你的队号 --test-kind formal --case-code 界面上的案例编码
python run_official.py --robot-id 你的队号 --test-kind formal --case-code 编码 --directional   # 问题4
```

日志文件名会自动区分，例如 `official_log_q3_formal_20260913_101530.json`，
其中 `paper_table1_row` 字段可直接填进论文表 1。

正式测试不告示总数，`--total-jammers` 留空即可；此时用 `full_clear_assessment`
里的 `suspect_missed_channels` 判断是否可能漏失。
