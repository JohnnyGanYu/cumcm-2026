# Q4 v2 证据与提交核验

最终结论见 [优化报告](../paper/q4_450_optimization.md)。逐局算术均值为 **465.64 s**，
710/710 源、48/48 局全清；整体仍未达到 450 s。以下数据均来自离线仿真。

## 最终评估与提交复核

| 文件 | 用途 |
|---|---|
| `q4_450_final_plan.json` | 最终 16 个种子 × 3 档场景的运行前登记；共 48 对 |
| `q4_450_final_validation.json` | 原始全部 96 条逐局结果，保留事件、配置、失败与源码检查字段 |
| `q4_450_final_summary.json` | 两种均值、清除率、尾部指标及按种子簇 bootstrap 区间 |
| `q4_450_pr_source_audit.json` | 提交阶段补记运行依赖哈希；不冒充运行前登记 |
| `q4_450_pr_candidate_replay.json` | 在基于远端 main 的 PR 工作区重放全部 48 条候选 |
| `q4_450_pr_replay_comparison.json` | 重放与原始候选的逐项比较，以及提交运行依赖哈希核验 |
| `q4_450_pr_base_regression.txt` | 提交工作区基础回归结果 |
| `q4_450_pr_targeted_tests.txt` | Q4 10 项、近场 2 项及来源/配置 10 项测试 |
| `q4_450_delivery_check.txt`、`q4_450_zip_check.txt` | 源码、交付目录及 ZIP 一致性检查 |
| `q4_450_stdlib_smoke.txt` | 无第三方依赖的 Windows 启动帮助检查 |

运行环境为 CPython 3.12.13、numpy 2.5.3、pytest 9.1.1、hypothesis 6.168.0。
交付运行时仅用标准库；标准库烟测使用 CPython 3.14.6。

`InProcessClient` 从原本地开发框架提取为独立模块，其类 AST 与 `ROBOT_ID`、
`REAL_BUDGET_S` 常量均与实测版本相同；无需引入未上线的 Q3/Q4 控制器框架。
两臂均使用固定 1200 s 真实剩余预算，避免主机负载改变虚拟时间比较；本次数据不验证官方墙钟限制。
HTTP 等价性仅单独检查 seed=7007、5 定向源的一局。

对照臂是**同样包含近场修复的 DogStrategy**，不是远端 main 的完整原始快照。
原始运行期间哈希检查仅覆盖结果中列出的 5 个源码文件，完整依赖清单在提交阶段补记。
`.gitattributes` 保留这 5 个文件的实测字节，尤其是使用 CRLF 的实验模块，以便原哈希可以复核。

在仓库根目录复核（以下 `python` 指已安装测试依赖的解释器）：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest src/b_sim/test_b_sim.py src/b_sim/test_b_pbt.py -q
python -m pytest src/test_q4_fast.py src/test_q4_near_field.py src/b_eval/test_provenance.py -q
python -m src.b_q4_450_report
python -m src.pack_dist_windows --check
python -m src.build_windows_zip --check
```

完整 96 条评估的命令见优化报告。`q4_450_base_regression.txt`、
`q4_450_regression_final.txt`、`q4_450_provenance_tests.txt` 保留原工作区的验证记录。

## 开发与废弃试验

`q4_450_*development.json`、`q4_450_geometry.json`、`q4_450_freeze_equivalence.json`
及 `q4_450_v2_diagnostic.json` 属于调参与诊断，不能当作未见样本最终成绩。
开发文件按一行一局保存，省略重复 `events` 流，但保留全部逐局指标、配置、错误和审计字段；
顶层 `storage_note` 记录省略范围与原本地文件哈希。最终 96 条证据不作此裁剪。

`q4_450_holdout.json` 是**废弃的第一批留出试验**：最后一个候选案例混用了 v2，
文件内已标记 `mixed_source_detected`，不得将其候选总体指标作为 v1 成绩。
`q4_450_v1_replay_audit.json` 保存旧版重放核验；v1 的漏清促成了随后重新登记的最终评估。

`q4_450_station_design.json` 是保守几何网格审计，存在未认证区域，不能视为固定双环必然全覆盖的证明。
