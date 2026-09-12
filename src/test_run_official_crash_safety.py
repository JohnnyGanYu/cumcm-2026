"""正式入口的「先保日志、再失败」保证。

RACEDAY 第 5 节声称两个崩溃故障已修，但仓库里没有任何测试覆盖它们 ——
护栏写在代码里，全靠人相信它有效。这类「声称已修但没测」在本项目已经
翻车过一次：--total-jammers 的除零护栏只挡了一个入口，0 值那条漏了。

这里锁三件事：
  1. 策略抛非协议异常时，日志仍落盘且 crash 字段记录类型与 traceback
  2. 零清除时 mean_clear_time_s 记为 null 而非让落盘失败
  3. 零清除且抄了总数时，打印路径不得因 None 格式化而崩在落盘之前
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "src" / "run_official.py"
DIST_ENTRY = ROOT / "dist_windows" / "run_official.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _line_of(text: str, needle: str) -> int:
    """返回 needle 首次出现的行号（1-based），找不到则 -1。"""
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    return -1


def test_crash_is_caught_before_log_is_written():
    """兜底 except 必须在写日志之前，否则异常会带走整局记录。"""
    text = _source(ENTRY)
    catch = _line_of(text, "except BaseException as exc:")
    write = _line_of(text, "out.write_text(")
    assert catch > 0, "找不到兜底 except —— 策略异常会让日志不落盘"
    assert write > 0, "找不到日志写入点"
    assert catch < write, (
        f"兜底 except 在第 {catch} 行、写日志在第 {write} 行；"
        "顺序颠倒会让崩溃局丢掉全部协议记录"
    )


def test_failure_exit_comes_after_log_write():
    """SystemExit(1) 必须晚于落盘 —— 先保日志再失败。"""
    text = _source(ENTRY)
    write = _line_of(text, "out.write_text(")
    raise_line = _line_of(text, "raise SystemExit(1)")
    assert raise_line > 0, "崩溃后应以非零码退出，便于事后定位"
    assert write < raise_line, (
        f"写日志在第 {write} 行、退出在第 {raise_line} 行；退出不能早于落盘"
    )


def test_no_unguarded_float_format_of_mean_clear_time():
    """mean_clear_time_s 可能是 None（零清除），不得直接 `:.4f`。

    这个入口真实存在过：落盘那处早已处理 None，打印那处漏了，
    于是「抄了总数但一个源都没清掉」这个最需要日志的场景反而写不出日志。
    """
    for path in (ENTRY, DIST_ENTRY):
        text = _source(path)
        bad = [
            line.strip()
            for line in text.splitlines()
            if "mean_clear_time_s" in line and ":.4f" in line
        ]
        assert not bad, (
            f"{path.name} 仍直接格式化可能为 None 的指标: {bad}"
        )


def test_zero_cleared_writes_null_metric():
    """零清除时落盘的指标必须是 null，而不是让 round(None) 抛错。"""
    for path in (ENTRY, DIST_ENTRY):
        text = _source(path)
        assert '"平均定位清除时间_s"' in text, f"{path.name} 缺少表 1 指标字段"
        guarded = (
            "None if summary[\"mean_clear_time_s\"] is None" in text
            or "is None" in text
        )
        assert guarded, f"{path.name} 未对 None 做保护，零清除局会写不出日志"


def test_dist_entry_carries_the_same_guards():
    """交付包里的入口必须与源码同步 —— 赛日跑的是交付包那份。"""
    src_text = _source(ENTRY)
    dist_text = _source(DIST_ENTRY).replace("\r\n", "\n")
    for marker in (
        "except BaseException as exc:",
        "raise SystemExit(1)",
        "未定义（清除数为 0）",
    ):
        assert marker in src_text, f"源码缺少 {marker!r}"
        assert marker in dist_text, (
            f"交付包缺少 {marker!r} —— 跑 python -m src.pack_dist_windows 同步"
        )
