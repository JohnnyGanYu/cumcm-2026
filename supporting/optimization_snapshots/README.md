# 实验冻结源码归档

`bounded_candidate.zip` 保存配对评估器实际运行的候选模块，来自运行期间的 `cumcm-paired-*` 临时快照，未用后来修改的工作区文件重新拼装。ZIP 根目录对应 `src/b_sim/`，包含 17 个非测试 Python 源码文件，不含缓存、结果或官方日志。

归档源码摘要为 `acb34f11480be8e62ad0b98a7c25e4f7b9e01acc2c21c356e3b1af93f3bae72c`。它同时对应以下两份报告；源码相同，消融开关不同：

| 报告 | bounded_homing | certified_silence | 阶段 |
| --- | --- | --- | --- |
| `results/b_tsp_bounded_300.json` | true | false | validation |
| `results/b_tsp_silence_development.json` | false | true | development |

第二份报告已完成，其原始源码确实保留在本归档中。第一份报告的完成状态应直接读取报告；保存源码不代表实验已经完成。两个实验使用的基线为 `d560cbae6675314541391cc8c7e0bd628a1cf50b`，基线源码摘要及配置记录在对应 manifest 中。种子、分层、计分、逐局结果和完整配置以原报告为准。

## 组合候选的失败证据

`combined_candidate.zip` 另行保存 `results/b_tsp_combined_rejected.json` 对应的原始组合源码，摘要为 `c8e842723a3073f2b77f693bc225a90375f91ea8972e28931a8b7fe5aac3b3e8`，文件清单与字节摘要见 `combined_candidate.manifest.json`。该归档也经过重复构建一致性和解包源码摘要校验。此实验在 600 对检查点因新增漏清停止，报告保留完整原始行和终止信息，`complete=false`、`acceptance.passed=false`；不能用未完成的报告声称最终验收通过。

这轮组合候选在 Q4 出现新增漏清，不能作为默认 Q4 发布版本的通过证据。`candidate_regression_evidence.json` 保留当时已出现的两组原始配对记录及其版本信息：

- bounded-only：`seed=962270, directional=10`，比基线新增漏清频道 18。
- combined：`seed=963141, directional=5`，比基线新增漏清频道 9。

该文件是原报告已出现失败行的副本，不表示完整实验结果，也没有改写原始行。它保留归档时的历史报告文件名：旧组合报告当时名为 `results/b_tsp_optimization_final.json`，之后移存到 `results/b_tsp_combined_rejected.json`；manifest 已更新现存路径并记录旧路径。新的 `results/b_tsp_optimization_final.json` 使用收窄后的发布政策与新种子块，属于另一轮实验，不能替代或覆盖这里的失败证据。

## 与后续工作区的差异

归档时用于对照的工作区核心摘要是 `c8e842723a3073f2b77f693bc225a90375f91ea8972e28931a8b7fe5aac3b3e8`。这里只描述该次对照，不将它自动视为最终发布版本。除以下两个文件外，17 个核心文件规范化换行后的内容相同：

- `strategy.py`：后续版本把两个优化开关默认开启，调整相应注释；严格区分 `direction` 和 `no_signal`，对缺少示向度或未知测量结果抛出协议错误。归档实验通过显式覆盖开关运行，不依赖后续默认值。
- `server.py`：后续版本为拒绝且请求体尚未读完的 HTTP 请求增加有字节数和超时上限的读取与连接关闭处理，避免 Windows TCP 重置遮蔽错误响应。

这些改动使最终源码摘要不同。因此本归档消融结果只能直接归属于这里保存的源码和对应开关；最终组合版本的结果应引用其自身报告，不能把两者的 SHA 当作相同。

## 校验与复现

`bounded_candidate.manifest.json` 包含每个文件的原始字节 SHA-256、ZIP 摘要、源码摘要算法及上述两文件的对照摘要。`certificate_only_development.manifest.json` 记录已完成的证书单独启用实验，并明确复用同一 ZIP。

ZIP 按相对路径排序，时间戳固定为 1980-01-01 00:00:00、权限为 0644，使用 deflate level 9。已在归档环境验证两次构建字节相同，以及从 ZIP 重新读取计算出的源码摘要与两份报告一致。

在仓库根目录可用以下只读 Python 代码核验归档（只依赖标准库）：

```python
import hashlib
import json
from pathlib import Path
import zipfile

folder = Path("supporting/optimization_snapshots")
manifest = json.loads((folder / "bounded_candidate.manifest.json").read_text(encoding="utf-8"))
archive = folder / manifest["archive"]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
digest = hashlib.sha256()
with zipfile.ZipFile(archive) as bundle:
    assert sorted(bundle.namelist()) == [row["path"] for row in manifest["files"]]
    for row in manifest["files"]:
        name = row["path"]
        data = bundle.read(name)
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        digest.update(name.encode("utf-8") + b"\0" + data.replace(b"\r\n", b"\n") + b"\0")
assert digest.hexdigest() == manifest["candidate_source_sha256"]
print(digest.hexdigest())
```

重跑时应在独立临时 checkout 中，把 ZIP 的文件恢复到 `src/b_sim/`，沿用基线提交与原报告的全部配置、种子和分层运行评估器；不要直接用当前工作区核心替代归档源码。恢复后应先计算源码摘要，确认与报告匹配。本目录没有更改或重写实验报告。
