# 竞赛支撑材料清单（B 题：无线电干扰源快速定位与清除）

本目录汇总提交给竞赛的**支撑材料清单**及其去标识（脱敏）说明。支撑材料本身
（正式测试导出日志）在比赛当天产生后按下述流程脱敏并归档；本清单先行说明"放什么、
怎么脱敏、匿名如何保证"，便于比赛日按图索骥、避免遗漏。

> **匿名硬约束**：提交物全程不得出现学校、姓名、队号、指导教师。见根目录 `README.md`
> 「匿名性」与 `official_runs/README.md`「匿名与安全」。

---

## 1. 支撑材料构成

| # | 材料 | 来源 / 生成方式 | 去标识方式 | 归档位置 |
|---|---|---|---|---|
| 1 | **Q3 / Q4 正式测试导出日志（各 3 次）** | 比赛当天官方 Windows 客户端产出的 `.jlog` / `.json`（含协议序列、结果指标） | `src/sanitize_official_run.py` 脱敏（队号→占位、票据→置零、用户名→占位） | `official_runs/*.sanitized.{jlog,json}`（比赛日追加） |
| 2 | **正式测试控制台输出（各 3 次）** | 官方客户端 stdout（含队号、结果摘要、日志落盘路径含 Windows 用户名） | 同上脚本（`robot_id`→`TTT…`、`C:\Users\<user>`→`C:\Users\UUU…`） | `official_runs/*_console.sanitized.txt` |
| 3 | **客户端自记协议日志（`client.dump_log`，R5.9）** | `src/b_sim/client.py` 的 `RobotClient.dump_log()`：把完整、有序的 `(request, http_status, response)` 序列（每条含 `request_id` 与时间戳）写为 JSON | 由本队程序生成、不含服务端机密；仍按同一脚本复核队号/用户名 | 与官方日志同目录（`*.sanitized.json`） |
| 4 | **离线彩排结果** | `python -m src.b_sim.rehearse`（Q3 ≥60 局、Q4 ≥40 局）聚合的评分项（`T̄`、清除率、运行时长） | 纯仿真数据，无 PII | `results/*.json` |
| 5 | **论文配图** | `src/b_q1_figures.py` / `b_q2_*` / `coverage_design` 等生成 | 纯算法配图，无 PII | `figures/*.png` |

### 关于 `client.dump_log`（R5.9）——为什么必须自记

官方模拟器**不提供指令回放**。要在赛后复盘每一步 `(请求, 状态码, 响应)`、核对时间
记账与协议健壮性（400/409 不重试、坐标越界拦截、重试语义），只能靠客户端自己持久化
记录。`RobotClient` 在每次动作后（无论成功或失败）都向 `self.log` 追加一条完整记录，
`dump_log(path)` 落盘为可读 JSON。该日志由本队程序生成、**不含 `server_ticket` /
`wrapped_dek` 等服务端机密**，是官方 `.jlog` 之外的独立、可复核凭证。

---

## 2. 脱敏流程（比赛日必做）

官方原始 `.jlog` / `.json` 内嵌**真实参赛队号**与 `server_ticket_b64`；`.jlog` 信封
还含 RSA-OAEP 包裹的数据加密密钥 `wrapped_dek_b64`；控制台日志含本机 **Windows 用户名**。
**只有脱敏版本（`*.sanitized.*`）入库**，原始件由 `.gitignore` 白名单式排除，仅留本地。

**精确命令：**

```bash
python src/sanitize_official_run.py <文件...> --team-no <真实队号>
```

例如：

```bash
python src/sanitize_official_run.py \
  official_runs/q3_official_run1.jlog \
  official_runs/q3_official_run1_console.txt \
  official_runs/q4_official_run1.json \
  --team-no <真实队号>
# 生成同名 *.sanitized.* 并打印命中项；脚本末尾复核队号/用户名是否残留
```

脚本 `sanitize_bytes()` 做的替换（文本层正则，不重写二进制长度字段）：

- **队号** → 等长 `T…` 占位（如 `TTTTTTTTTTTT`），保持字节偏移；
- **Windows 用户目录** `C:\Users\<name>` → `C:\Users\UUU…`（等长占位）；
- **敏感 JSON 字段置零**（`REDACT_KEYS`）：`server_ticket_b64`、`authorization_nonce_b64`、
  `package_signing_public_key_b64`、`device_digest`、`signature_b64`、
  `payload_signature_b64`、`wrapped_dek_b64` → 同长 `0…` 填充；
- 运行结束**自动复核**：若队号或非占位 Windows 用户名仍残留，脚本以非零码退出并告警。

---

## 3. 匿名检查清单（入库前逐项确认）

- [ ] 无学校名称
- [ ] 无个人姓名 / 指导教师
- [ ] 无真实队号（`robot_id` / `--robot-id` / 日志内均为 `T…` 占位）
- [ ] 无 `server_ticket_b64` / `authorization_nonce_b64` 明文（均为 `0…`）
- [ ] 无 `wrapped_dek_b64` 等密钥材料明文（均为 `0…`）
- [ ] 无 `C:\Users\<真实用户名>`（均为 `C:\Users\UUU…`）
- [ ] 仓库仅提交 `*.sanitized.*` 与 `README.md`，原始件未入库（`.gitignore` 已排除）
- [ ] 论文正文、图表、文件名均不含上述任一标识

> **快速复核命令**（应无输出即为通过）：
> ```bash
> # 残留真实 Windows 用户名（非 U 占位）
> grep -oaE 'C:\\Users\\[A-Za-z0-9._-]+' official_runs/*.sanitized.* | grep -avE 'C:\\Users\\U+$'
> # 未置零的敏感字段值
> grep -oaE '"(server_ticket_b64|wrapped_dek_b64|authorization_nonce_b64)"\s*:\s*"[^"]*"' \
>   official_runs/*.sanitized.* | grep -avE ':"0*"'
> ```

---

## 4. R6.5 原则：官方实测为最终判据

离线仿真（`src/b_sim/`）只用于无限次调参与策略迭代。**任何离线结果与官方实测在任一
指标上不一致，一律以官方实测为最终判据**，并据此回头重新校准仿真参数（`arena.py` /
`server.py` / 策略参数）。因此支撑材料以**官方脱敏日志**为主证据，离线彩排结果与配图
为佐证与推导过程，二者偏差已知（Q3 ≈ 5.4%、Q4 ≈ 9.6%，见 `official_runs/README.md`），
做正式测试时间预期时按官方偏大值取保守余量。
