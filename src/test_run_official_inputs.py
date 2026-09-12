"""正式测试入口的输入健壮性回归。

--total-jammers 0 曾使 run_official 在日志落盘之前抛 ZeroDivisionError，
赛日手误即丢掉整局协议记录与一次测试机会。此处把该入口钉住。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_total_jammers_zero_is_rejected_before_division():
    """0 应被判非法并降级处理，而不是抛异常。"""
    src = (ROOT / "src" / "run_official.py").read_text(encoding="utf-8")
    guard = "args.total_jammers is not None and args.total_jammers <= 0"
    assert guard in src, "缺少非正数护栏，0 会触发除零并丢日志"
    # 护栏必须出现在除法之前，否则不起作用。
    assert src.index(guard) < src.index("len(cleared) / args.total_jammers")


def test_dist_package_carries_the_same_guard():
    """交付包是实际上机的那份，护栏必须同步过去。"""
    dist = (ROOT / "dist_windows" / "run_official.py").read_text(encoding="utf-8")
    assert "args.total_jammers <= 0" in dist, "交付包未同步护栏"
