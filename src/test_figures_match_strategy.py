"""配图与生产策略参数的一致性回归。

图 6 / 图 7 曾长期画着 1250 m x 12 的初始基线：布局改过两轮而图未重生成，
论文正文说交付是 1150 x 6，图里却是 13 站位。评审看图会得出错误结论。
这里把「图脚本的环参数 == 策略类的环参数」钉成断言。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from src.b_coverage_figures import (  # noqa: E402
    RING_RADIUS_M,
    RING_STATIONS,
    _chosen_stations,
)
from src.b_sim.arena import RECEPTION_MIN_M  # noqa: E402
from src.b_sim.coverage_design import worst_coverage_gap  # noqa: E402
from src.b_sim.q3_fast import Q3FastStrategy  # noqa: E402


def test_figure_ring_radius_matches_production_strategy():
    assert RING_RADIUS_M == Q3FastStrategy.ring_radius, (
        "配图环半径与 Q3FastStrategy.ring_radius 不一致，"
        "图会展示非交付布局；改策略后请重跑 python -m src.b_coverage_figures"
    )


def test_figure_layout_still_guarantees_detection():
    stations = _chosen_stations()
    assert len(stations) == RING_STATIONS + 1
    gap, _ = worst_coverage_gap(stations)
    assert gap < RECEPTION_MIN_M, (
        f"最坏覆盖间隙 {gap:.2f} m 已不低于接收半径下限 {RECEPTION_MIN_M} m"
    )
