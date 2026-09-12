"""锁住 Q4 外环半径的几何下界。

曾据「最坏覆盖间隙 / 巡回长度 / 随机场景 240 局配对」把 outer_radius 从 1870
改成 1800，随后在按几何筛选的边界压力样本（60 局）上确认它稳定漏源且耗时更差。
这里把下界写成断言，避免同样的缩环改动再次被均值指标"证明"为改进。

依据：贴边（|s| -> ARENA_RADIUS）且辐射轴径向朝外的定向源，其 180 度扇区只含
比源更外侧的点，站位必须落在过源的切线之外。n 站均布环的最坏方位在相邻两站
正中间，径向投影为 R*cos(pi/n)，故需 R*cos(pi/n) >= ARENA_RADIUS。
"""

from __future__ import annotations

import math

from src.b_sim.arena import ARENA_RADIUS, RECEPTION_MIN_M
from src.b_sim.directional import in_beam
from src.b_sim.q4_fast import Q4FastStrategy, ring_points


def required_outer_radius(count: int) -> float:
    return ARENA_RADIUS / math.cos(math.pi / count)


def test_outer_ring_satisfies_boundary_beam_bound():
    """外环必须够到贴边朝外源的切线之外，否则这类源在保守接收半径下必盲。"""
    bound = required_outer_radius(Q4FastStrategy.outer_count)
    assert bound == 1800.0 / math.cos(math.pi / 12)
    assert Q4FastStrategy.outer_radius >= bound, (
        f"outer_radius={Q4FastStrategy.outer_radius} 低于几何下界 {bound:.1f} m；"
        "贴边径向朝外的定向源会必盲，见 src/b_outer_beam_guarantee.py 与"
        " results/b_outer_radius_boundary.json"
    )


def test_shrinking_outer_ring_to_arena_radius_creates_blind_sources():
    """反向确认：把外环缩到区域半径会在最外环带产生必盲源。

    构造一个贴边、径向朝外的源，检查两档布局下的"保证可见站位数"。
    """
    stations_by_radius = {}
    for radius in (ARENA_RADIUS, Q4FastStrategy.outer_radius):
        points = [(0.0, 0.0)]
        points += ring_points(Q4FastStrategy.inner_radius, Q4FastStrategy.inner_count)
        points += ring_points(radius, Q4FastStrategy.outer_count,
                              Q4FastStrategy.outer_phase_deg)
        stations_by_radius[radius] = points

    # 最坏方位：外环相邻两站正中间。外环相位 15 度、12 站间隔 30 度,
    # 故站位在 15,45,... 度，中点在 0,30,... 度。
    worst_bearing_deg = 0.0
    angle = math.radians(worst_bearing_deg)
    source = (1799.0 * math.cos(angle), 1799.0 * math.sin(angle))
    heading = worst_bearing_deg  # 径向朝外

    def guaranteed(stations):
        return sum(
            1 for px, py in stations
            if math.dist((px, py), source) <= RECEPTION_MIN_M
            and in_beam(source[0], source[1], heading, px, py)
        )

    assert guaranteed(stations_by_radius[ARENA_RADIUS]) == 0
    assert guaranteed(stations_by_radius[Q4FastStrategy.outer_radius]) >= 1


def stations_reaching_within(margin_m: float, ring_radius: float) -> int:
    """在半径 ring_radius 的环上，覆盖到距边界 margin_m 的源所需的最少站位数。

    最坏方位在相邻两站正中间，径向投影 ring_radius*cos(pi/n) 必须不小于源半径
    ARENA_RADIUS - margin_m。
    """
    target = ARENA_RADIUS - margin_m
    if ring_radius >= target:
        ratio = target / ring_radius
        if ratio <= 0.0:
            return 1
        return max(1, math.ceil(math.pi / math.acos(min(1.0, ratio))))
    raise ValueError("环半径小于目标源半径，任何站数都不够")


def test_angular_density_is_possible_but_far_more_expensive():
    """把外环放在区域半径上并角向加密**能**覆盖边界，但站位数不划算。

    这里纠正一个直觉误区：R = ARENA_RADIUS 的环并非"任何站数都必盲"。
    只要 R*cos(pi/n) >= |s|，最坏方位也能收到。真正的问题是所需站数发散：
    |s| -> ARENA_RADIUS 时 n -> 无穷。所以取径向余量（1870 m / 12 站）
    比角向加密便宜得多，这是成本论证，不是可行性论证。
    """
    # 1870 m / 12 站能覆盖到边界本身。
    assert Q4FastStrategy.outer_radius * math.cos(
        math.pi / Q4FastStrategy.outer_count) >= ARENA_RADIUS

    # 放在 1800 m 上时，站位数随余量收缩迅速发散。
    needed = {m: stations_reaching_within(m, ARENA_RADIUS) for m in (50.0, 10.0, 1.0)}
    assert needed[50.0] > Q4FastStrategy.outer_count
    assert needed[10.0] > needed[50.0]
    assert needed[1.0] > needed[10.0]
    assert needed[1.0] >= 90, f"预期约 95 站，实得 {needed[1.0]}"


def test_worst_bearing_blind_at_arena_radius_for_production_ring_count():
    """在生产站数（12）下，把外环缩到区域半径确实在最坏方位必盲。"""
    worst_deg = 180.0 / Q4FastStrategy.outer_count
    angle = math.radians(worst_deg)
    source = (1799.0 * math.cos(angle), 1799.0 * math.sin(angle))
    stations = ring_points(ARENA_RADIUS, Q4FastStrategy.outer_count)
    seen = sum(
        1 for px, py in stations
        if math.dist((px, py), source) <= RECEPTION_MIN_M
        and in_beam(source[0], source[1], worst_deg, px, py)
    )
    assert seen == 0
