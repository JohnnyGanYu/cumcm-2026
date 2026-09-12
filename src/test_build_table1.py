"""表 1 生成器的回归。

赛日要靠它把六局数字填进论文，算错或漏行都会直接写进提交稿，
所以池化口径、作废局过滤、骨架回退三件事都要钉住。
"""
from __future__ import annotations

from src.build_table1 import load, pooled, table


def test_pooled_is_total_virtual_over_total_cleared():
    """池化不是均值的均值：两局清除数不同时二者必然不等。"""
    rows = [
        {"virtual": 3000.0, "cleared": 10, "tbar": 300.0},
        {"virtual": 3000.0, "cleared": 15, "tbar": 200.0},
    ]
    assert pooled(rows) == f"{6000.0 / 25:.2f}"
    mean_of_means = (300.0 + 200.0) / 2
    assert abs(float(pooled(rows)) - mean_of_means) > 1.0


def test_pooled_handles_zero_cleared():
    assert pooled([{"virtual": 0.0, "cleared": 0}]) == "—"
    assert pooled([]) == "—"


def test_voided_run_is_excluded_from_practice_rows():
    """9-12 演练第一局 /enter 被拒（队号年份打错），清除与虚拟时间都是 0。

    它必须不进表，否则均值被拉低。
    """
    groups = load("practice")
    files = [r["file"] for r in groups["Q3"]]
    assert not any("enterfail" in name for name in files), files
    for row in groups["Q3"] + groups["Q4"]:
        assert row["cleared"] or row["virtual"], row["file"]


def test_practice_pooled_matches_documented_values():
    """9-12 演练池化值应复现文档记载的 Q3 209.0 / Q4 486.9 s。"""
    groups = load("practice")
    assert abs(float(pooled(groups["Q3"])) - 209.0) < 0.1
    assert abs(float(pooled(groups["Q4"])) - 486.9) < 0.1


def test_empty_group_falls_back_to_skeleton():
    lines = table("表 1-Q3", [])
    body = "\n".join(lines)
    assert body.count("待填") == 3
    assert "三次均值" in body
