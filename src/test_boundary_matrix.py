"""边界场景矩阵的回归。

b_boundary_matrix.py 产出的区间是 RACEDAY 第 3 节判断「要不要重打」的依据，
数字错了会让赛日误判。这里锁三件事：

1. 池化口径必须是「总虚拟时间 / 总清除数」，不是各局 T̄ 的平均；
2. 六档配置必须覆盖源数与定向占比的两端（题面范围 10-16）；
3. 落盘 JSON 里的 profile 必须是当前交付策略，否则表与代码脱节。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.b_boundary_matrix import CELLS, aggregate
from src.b_sim.arena import SOURCE_COUNT_MAX, SOURCE_COUNT_MIN
from src.b_sim.q3_fast import PROFILE as Q3_PROFILE
from src.b_sim.q4_fast import PROFILE as Q4_PROFILE

MATRIX = Path("results/b_boundary_matrix.json")


def test_pooled_is_total_time_over_total_cleared():
    """池化不是均值的均值 —— 各局源数不同时两者必然不等。"""
    rows = [
        {"cleared": 16, "virtual_time_s": 3200.0, "total_jammers": 16, "measure_calls": 90},
        {"cleared": 10, "virtual_time_s": 2900.0, "total_jammers": 10, "measure_calls": 110},
    ]
    agg = aggregate(rows)
    assert agg["pooled_mean_clear_time_s"] == pytest.approx(6100.0 / 26, abs=0.01)
    naive = (3200.0 / 16 + 2900.0 / 10) / 2
    assert abs(agg["pooled_mean_clear_time_s"] - naive) > 10.0, (
        "池化与均值的均值必须可区分，否则口径断言形同虚设"
    )
    assert agg["clear_ratio"] == pytest.approx(1.0)


def test_zero_cleared_gives_none_not_zero_division():
    """一局都没清掉时池化值无定义，必须是 None 而不是崩溃。"""
    agg = aggregate([{"cleared": 0, "virtual_time_s": 500.0,
                      "total_jammers": 12, "measure_calls": 40}])
    assert agg["pooled_mean_clear_time_s"] is None
    assert agg["cleared"] == 0


def test_cells_span_both_ends_of_source_count():
    """六档必须覆盖题面源数两端与定向占比两端。"""
    totals = {c[0] for c in CELLS}
    assert SOURCE_COUNT_MIN in totals and SOURCE_COUNT_MAX in totals, (
        f"源数两端 {SOURCE_COUNT_MIN}/{SOURCE_COUNT_MAX} 必须都在档位里"
    )
    for total, directional in CELLS:
        assert 0 <= directional <= total
    assert any(d == 0 for _, d in CELLS), "缺全向档（问题 3）"
    assert any(d == t for t, d in CELLS), "缺全定向档（最难档）"


def test_stored_matrix_matches_current_strategy_profiles():
    """落盘结果的 profile 必须是当前交付策略，否则 RACEDAY 的表已过期。"""
    if not MATRIX.exists():
        pytest.skip("尚未跑过 python -m src.b_boundary_matrix")
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    cells = data["cells"]
    assert len(cells) == len(CELLS)
    for cell in cells:
        summary = cell["summary"]
        expected = Q4_PROFILE if summary["directional"] > 0 else Q3_PROFILE
        assert summary["strategy_profile"] == expected, (
            f'{summary["total_jammers"]} 源/{summary["directional"]} 定向档的 profile 是 '
            f'{summary["strategy_profile"]}，当前交付是 {expected} —— '
            "重跑 python -m src.b_boundary_matrix 并更新 RACEDAY 第 3 节"
        )


def test_stored_matrix_has_no_missed_source():
    """六档零漏失是 RACEDAY 的明文声称，破了必须立刻发现。"""
    if not MATRIX.exists():
        pytest.skip("尚未跑过 python -m src.b_boundary_matrix")
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        summary = cell["summary"]
        assert summary["clear_ratio"] == 1.0, (
            f'{summary["total_jammers"]} 源/{summary["directional"]} 定向档清除率 '
            f'{summary["clear_ratio"]}，RACEDAY 声称六档零漏失'
        )


def test_raceday_interval_matches_stored_matrix():
    """RACEDAY 写的区间上下限必须等于落盘结果的实际极值。"""
    if not MATRIX.exists():
        pytest.skip("尚未跑过 python -m src.b_boundary_matrix")
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    pooled = [c["summary"]["pooled_mean_clear_time_s"] for c in data["cells"]]
    text = Path("RACEDAY.md").read_text(encoding="utf-8")
    for value in (min(pooled), max(pooled)):
        assert f"{value:.1f}" in text, (
            f"RACEDAY 缺区间端点 {value:.1f} s —— 表与 results/ 脱节"
        )
