"""b_sim 面向性质测试（Property-Based Testing）基座与共享生成器。

Feature: b-radio-interference-locate-clear

本文件是设计文档 §7.2「Correctness Properties」P1–P8 的 hypothesis 测试基座：
它提供**可复用的 hypothesis 策略**（源、检测点、示向度观测）以及复用离线仿真核心
`arena.bearing_error_deg` 的确定性 ±1° 误差场，供后续各性质测试（任务 2.3/2.4/2.5、
3.4、5.4/5.5、6.5、7.4/7.5）直接 import。

设计约束（_Requirements: 6.1_）：
- 源位置在半径 1800 m 圆域内均匀分布；有效接收半径在 [1000, 1500] m 内取值；
  定向源朝向在 [0, 360) 内取值（可选）。
- 复用 `arena.bearing_error_deg`（确定性误差场）使同一检测点重复检测返回完全相同的
  示向度读数——这是把「同一地点电磁环境干扰固定」这一物理假设注入性质测试的方式。
- 所有性质测试统一配置 `max_examples >= 100`（见 `PBT_SETTINGS`）。

运行（从仓库根目录）：
    python -m pytest src/b_sim/test_b_pbt.py -q
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.b_sim.arena import (
    ARENA_RADIUS,
    ERROR_ABS_MAX_DEG,
    RECEPTION_MAX_M,
    RECEPTION_MIN_M,
    Jammer,
    bearing_error_deg,
    norm_deg,
)
from src.b_sim.geometry import Bearing

# ---------------------------------------------------------------------------
# 全局 PBT 配置：所有性质测试统一使用 max_examples >= 100。
# 各性质测试可用 @PBT_SETTINGS 装饰（等价 @settings(...)）。
# ---------------------------------------------------------------------------
PBT_MAX_EXAMPLES = 100

PBT_SETTINGS = settings(
    max_examples=PBT_MAX_EXAMPLES,
    deadline=None,  # 几何/仿真单例开销略大，关闭单例超时以避免误报
    suppress_health_check=[HealthCheck.filter_too_much],
)

# 误差绝对界（度）——复用 arena 的物理常量，避免魔法数字漂移。
ERROR_DEG = ERROR_ABS_MAX_DEG

# 默认误差场盐值：与 arena 案例的 error_salt 口径一致，可按需覆盖。
DEFAULT_ERROR_SALT = "pbt"


# ---------------------------------------------------------------------------
# 共享 hypothesis 策略
# ---------------------------------------------------------------------------
@st.composite
def points_in_arena(draw, *, radius: float = ARENA_RADIUS, margin: float = 0.0):
    """半径 `radius` 圆域内均匀分布的一个检测点 (x, y)。

    用「半径按 sqrt 采样、角度均匀采样」保证面积均匀（与 `arena.generate_case` 一致）。
    `margin` 用于把点约束在略小的圆内（例如避免恰好落在竞技场边界上）。
    """
    r_max = max(0.0, radius - margin)
    u = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    theta = draw(st.floats(min_value=0.0, max_value=2.0 * math.pi, allow_nan=False, allow_infinity=False))
    r = r_max * math.sqrt(u)
    return (r * math.cos(theta), r * math.sin(theta))


@st.composite
def sources_in_arena(draw, *, directional: bool | None = False):
    """半径 1800 m 圆域内均匀分布的一个干扰源 `Jammer`。

    - 位置：`points_in_arena()`（面积均匀）。
    - 有效接收半径：在 [1000, 1500] m 内均匀取值（`arena.RECEPTION_MIN_M/MAX_M`）。
    - 朝向：`directional` 控制——
        * `False`（默认）：全向源（`directional_deg=None`）。
        * `True`：定向源，朝向在 [0, 360) 内均匀取值。
        * `None`：随机决定全向 / 定向（各半）。
    频道号取 1（几何性质与频道无关；需要多源无重复频道的性质测试可自行组装）。
    """
    x, y = draw(points_in_arena())
    reception = draw(
        st.floats(
            min_value=RECEPTION_MIN_M,
            max_value=RECEPTION_MAX_M,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    if directional is None:
        directional = draw(st.booleans())
    heading = None
    if directional:
        heading = draw(
            st.floats(min_value=0.0, max_value=360.0, exclude_max=True, allow_nan=False, allow_infinity=False)
        )
    return Jammer(
        channel=1,
        x=x,
        y=y,
        reception_radius_m=reception,
        directional_deg=heading,
    )


def deterministic_error_deg(px: float, py: float, salt: str = DEFAULT_ERROR_SALT) -> float:
    """复用 `arena.bearing_error_deg` 的确定性 ±1° 误差场。

    同一 (px, py) 恒返回同一误差（同点重复检测读数一致），不同点误差近似独立均匀。
    """
    return bearing_error_deg(px, py, salt)


def observed_bearing(source: Jammer, px: float, py: float, salt: str = DEFAULT_ERROR_SALT) -> float:
    """在检测点 (px, py) 对 `source` 的确定性含误差示向度读数（度，∈ [0, 360)）。

    真实方位叠加确定性 ±1° 误差场，与 `arena.Arena.measure` 的 direction 分支口径一致。
    """
    truth = source.true_bearing_from(px, py)
    return norm_deg(truth + deterministic_error_deg(px, py, salt))


def make_bearing(source: Jammer, px: float, py: float, salt: str = DEFAULT_ERROR_SALT) -> Bearing:
    """把一次含误差观测封装为 `geometry.Bearing`（供 `localization_polygon` 等使用）。"""
    return Bearing(px, py, observed_bearing(source, px, py, salt))


def is_observable(source: Jammer, px: float, py: float) -> bool:
    """检测点能否观测到该源（在有效接收半径内且落在辐射扇区内）。

    与 `arena.Arena.measure` 的判定一致：全向源仅受接收半径约束，定向源额外要求
    检测点落在 ±90° 扇区内。
    """
    return source.covers(px, py) and source.distance_to(px, py) <= source.reception_radius_m


# ---------------------------------------------------------------------------
# 冒烟测试：确保基座与共享策略可被 pytest 收集且行为符合约束。
# 后续 P1–P8 的具体性质测试在各自任务中新增，形如：
#   # Feature: b-radio-interference-locate-clear, Property N: ...
# ---------------------------------------------------------------------------
# Feature: b-radio-interference-locate-clear, Property 0 (smoke):
# 共享生成器产出的源满足设计 §3.1 / R6.1 的圆域、接收半径与朝向约束。
@PBT_SETTINGS
@given(source=sources_in_arena(directional=None))
def test_smoke_source_within_arena_constraints(source: Jammer):
    assert math.hypot(source.x, source.y) <= ARENA_RADIUS + 1e-6
    assert RECEPTION_MIN_M <= source.reception_radius_m <= RECEPTION_MAX_M
    if source.directional_deg is not None:
        assert 0.0 <= source.directional_deg < 360.0


# Feature: b-radio-interference-locate-clear, Property 0 (smoke):
# 检测点生成器落在竞技场圆域内。
@PBT_SETTINGS
@given(point=points_in_arena())
def test_smoke_point_within_arena(point):
    px, py = point
    assert math.hypot(px, py) <= ARENA_RADIUS + 1e-6


# Feature: b-radio-interference-locate-clear, Property 0 (smoke):
# 复用的 ±1° 误差场是确定性的（同点重复检测读数一致）且在界内。
@PBT_SETTINGS
@given(source=sources_in_arena(), point=points_in_arena())
def test_smoke_error_field_is_deterministic_and_bounded(source: Jammer, point):
    px, py = point
    err1 = deterministic_error_deg(px, py)
    err2 = deterministic_error_deg(px, py)
    assert err1 == err2  # 同点重复检测误差不变
    assert -ERROR_DEG <= err1 <= ERROR_DEG

    # observed_bearing 亦确定性，且落在 [0, 360)
    svd1 = observed_bearing(source, px, py)
    svd2 = observed_bearing(source, px, py)
    assert svd1 == svd2
    assert 0.0 <= svd1 < 360.0

    # make_bearing 复用同一读数，可直接喂给 geometry 层
    bearing = make_bearing(source, px, py)
    assert isinstance(bearing, Bearing)
    assert bearing.x == px and bearing.y == py
    assert bearing.svd_deg == svd1



# ===========================================================================
# 任务 2.3 / 2.4 / 2.5：Q1 几何内核性质测试（设计 §7.2 P1 / P2 / P3）
# ===========================================================================
from hypothesis import assume  # noqa: E402  (集中放在性质测试段首)
from hypothesis.strategies import lists  # noqa: E402

from src.b_sim.geometry import (  # noqa: E402
    circle_on_diameter_covers,
    localization_polygon,
    polygon_diameter,
)


# ---------------------------------------------------------------------------
# 几何辅助：凸多边形点包含判定 / 凸包 / 面积（性质测试内部使用，不属于被测代码）。
# ---------------------------------------------------------------------------
def point_in_convex_polygon(px: float, py: float, polygon, tol: float = 1e-6) -> bool:
    """判断点 (px, py) 是否落在凸多边形（顶点有序，可能顺/逆时针）内部或边界上。

    对每条有向边 (v_i → v_{i+1}) 计算叉积符号；凸多边形所有叉积同号（含 0）时点在内。
    容差 ``tol`` 吸收楔形裁剪的浮点误差。
    """
    n = len(polygon)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > tol:
            cur = 1
        elif cross < -tol:
            cur = -1
        else:
            cur = 0
        if cur == 0:
            continue
        if sign == 0:
            sign = cur
        elif cur != sign:
            return False
    return True


def convex_hull(points):
    """Andrew monotone chain 凸包，返回逆时针顶点列表（去重、去共线端点）。

    退化（<3 唯一点或全共线）时返回去重后的极端点（可能 1–2 个点）。
    """
    pts = sorted(set((round(px, 9), round(py, 9)) for px, py in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull


# ---------------------------------------------------------------------------
# Feature: b-radio-interference-locate-clear, Property 1: 定位区域含真值（误差界内）
# 设计 §7.2 P1 / _Requirements: 1.3_
# 对任意全向源与任意可观测该源的 ≥2 个检测点，用确定性 ±1° 误差场构造 Bearing，
# `localization_polygon(bearings, error_deg=1.0)` 生成非空凸区域且**包含真实源位置**。
# ---------------------------------------------------------------------------
# **Validates: Requirements 1.3**
@PBT_SETTINGS
@given(
    source=sources_in_arena(directional=False),
    points=lists(points_in_arena(), min_size=2, max_size=6),
)
def test_property1_localization_region_contains_truth(source: Jammer, points):
    # 仅保留能观测到该源（在接收半径内、全向源恒在扇区内）的检测点。
    observable = [(px, py) for px, py in points if is_observable(source, px, py)]
    assume(len(observable) >= 2)

    bearings = [make_bearing(source, px, py) for px, py in observable]
    # 至少需要两条不近似共线的示向线，交会区域才有界非空；否则 assume 跳过。
    poly = localization_polygon(bearings, error_deg=1.0)
    assume(poly)  # 非空定位区域（面积 > 1e-9）

    # 真实源位置必落在各 ±1° 楔形之交内（真方位在 [svd−1°, svd+1°] 内）。
    assert point_in_convex_polygon(source.x, source.y, poly), (
        f"真值 ({source.x:.3f}, {source.y:.3f}) 不在定位区域内: {poly}"
    )


# ---------------------------------------------------------------------------
# Feature: b-radio-interference-locate-clear, Property 2: 直径在顶点对取到
# 设计 §7.2 P2 / _Requirements: 1.5_
# 对任意随机凸多边形（随机点取凸包），`polygon_diameter` 返回的 D 等于顶点对最大距离，
# 且 D ≥ 该多边形任意两个内部采样点之间的距离（直径是全体点对距离上确界）。
# ---------------------------------------------------------------------------
# **Validates: Requirements 1.5**
@PBT_SETTINGS
@given(
    raw_points=lists(points_in_arena(), min_size=3, max_size=10),
    weights=lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=6,
    ),
)
def test_property2_diameter_attained_on_vertex_pair(raw_points, weights):
    poly = convex_hull(raw_points)
    assume(len(poly) >= 3)  # 需要非退化凸多边形

    diameter, a, b = polygon_diameter(poly)

    # (a) 直径确等于顶点对最大距离，且端点 A、B 就是取得该最大距离的顶点对。
    brute = max(
        math.hypot(poly[i][0] - poly[j][0], poly[i][1] - poly[j][1])
        for i in range(len(poly))
        for j in range(i + 1, len(poly))
    )
    assert diameter == pytest.approx(brute, abs=1e-6)
    assert math.hypot(a[0] - b[0], a[1] - b[1]) == pytest.approx(diameter, abs=1e-6)

    # (b) 直径 ≥ 任意两个内部采样点之间的距离。
    # 用凸组合（重心坐标）在多边形内部/边界上取样点：Σλ_k·v_k 为凸多边形内的点。
    def interior_sample(w):
        w1, w2, w3 = w
        s = w1 + w2 + w3
        if s <= 1e-9:
            return poly[0]
        # 取多边形前三个顶点的凸组合（凸多边形任意顶点凸组合都在区域内）。
        v0, v1, v2 = poly[0], poly[1 % len(poly)], poly[2 % len(poly)]
        return (
            (w1 * v0[0] + w2 * v1[0] + w3 * v2[0]) / s,
            (w1 * v0[1] + w2 * v1[1] + w3 * v2[1]) / s,
        )

    samples = [interior_sample(w) for w in weights]
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            d = math.hypot(samples[i][0] - samples[j][0], samples[i][1] - samples[j][1])
            assert d <= diameter + 1e-6, (
                f"内部采样点对距离 {d:.6f} 超过直径 {diameter:.6f}"
            )


# ---------------------------------------------------------------------------
# Feature: b-radio-interference-locate-clear, Property 3: 外接圆覆盖判定正确（含正三角形反例）
# 设计 §7.2 P3 / _Requirements: 1.7, 1.8_
# 对任意非退化凸多边形，`circle_on_diameter_covers(poly)` 的 covers 布尔值等于
# `max‖v−M‖ ≤ D/2 + 1e-7`（M 为直径中点）；并显式断言正三角形反例返回 covers=False，
# 其覆盖裕度 margin = −(√3/2 − 1/2)·L（Jung 平面上界 D/√3 非平凡）。
# ---------------------------------------------------------------------------
# **Validates: Requirements 1.7, 1.8**
COVER_TOL_M = 1e-7


@PBT_SETTINGS
@given(raw_points=lists(points_in_arena(), min_size=3, max_size=10))
def test_property3_circle_cover_decision_matches_definition(raw_points):
    poly = convex_hull(raw_points)
    assume(len(poly) >= 3)

    diameter, a, b = polygon_diameter(poly)
    assume(diameter > 1e-7)  # 非退化（非单点）

    covers, margin = circle_on_diameter_covers(poly)

    # 按定义重算：中点 M、最远顶点距离 worst。
    mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    worst = max(math.hypot(x - mx, y - my) for x, y in poly)
    radius = diameter / 2.0

    # 判据一致：covers ⟺ worst ≤ D/2 + 1e-7。
    assert covers == (worst <= radius + COVER_TOL_M)
    # 裕度定义一致：margin = D/2 − max‖v−M‖。
    assert margin == pytest.approx(radius - worst, abs=1e-6)


def test_property3_equilateral_triangle_counterexample_not_covered():
    """正三角形反例：以任一边为直径的圆不覆盖第三顶点（Jung 上界 D/√3 非平凡）。

    边长 L 的正三角形，直径 D = L；直径圆半径 L/2，圆心在某边中点，第三顶点到该中点
    距离 = (√3/2)·L > L/2，落在圆外 → covers=False，
    margin = D/2 − (√3/2)L = −(√3/2 − 1/2)·L。
    """
    L = 100.0
    # 顶点：(0,0)、(L,0)、(L/2, √3/2·L)。直径必在某对顶点间取到（三边等长 = L）。
    poly = [(0.0, 0.0), (L, 0.0), (L / 2.0, math.sqrt(3.0) / 2.0 * L)]

    diameter, _, _ = polygon_diameter(poly)
    assert diameter == pytest.approx(L, abs=1e-6)

    covers, margin = circle_on_diameter_covers(poly)
    assert covers is False, "正三角形反例应判定为『不覆盖』"

    expected_margin = -(math.sqrt(3.0) / 2.0 - 0.5) * L  # = −(√3/2 − 1/2)·L
    assert margin == pytest.approx(expected_margin, abs=1e-6)
    assert margin < -COVER_TOL_M  # 明确落在容差之外（真·不覆盖）



# ===========================================================================
# 任务 3.4：Q2 逼近误差单调收缩性质测试（设计 §7.2 P8）
# ===========================================================================
from src.b_sim.strategy import lateral_error_at  # noqa: E402


# ---------------------------------------------------------------------------
# Feature: b-radio-interference-locate-clear, Property 8: 逼近误差单调收缩
# 设计 §7.2 P8 / _Requirements: 2.8_
# `lateral_error_at(r)` 关于 `r` 严格单调递增，验证"越近越准"，为逐步逼近提供依据。
# 对任意 `r2 > r1 ≥ 0`，`lateral_error_at(r2) > lateral_error_at(r1)`；
# 并核对其闭式定义 `e(r) = r·tan(1°)`（R2.8）。
# ---------------------------------------------------------------------------
# **Validates: Requirements 2.8**
@PBT_SETTINGS
@given(
    r1=st.floats(min_value=0.0, max_value=3000.0, allow_nan=False, allow_infinity=False),
    r2=st.floats(min_value=0.0, max_value=3000.0, allow_nan=False, allow_infinity=False),
)
def test_property8_lateral_error_strictly_monotonic(r1: float, r2: float):
    # 规范化为 0 ≤ r1 < r2；两者相等时跳过（严格单调只对 r2 > r1 成立）。
    lo, hi = min(r1, r2), max(r1, r2)
    assume(hi > lo)

    # (a) 严格单调递增：更远的距离横向误差更大 → 逼近使误差单调收缩。
    assert lateral_error_at(hi) > lateral_error_at(lo), (
        f"lateral_error_at({hi:.6f})={lateral_error_at(hi):.9f} 未严格大于 "
        f"lateral_error_at({lo:.6f})={lateral_error_at(lo):.9f}"
    )

    # (b) 闭式定义一致：e(r) = r·tan(1°)（1° 为示向度误差绝对上界）。
    tan1 = math.tan(math.radians(1.0))
    assert lateral_error_at(lo) == pytest.approx(lo * tan1, abs=1e-9)
    assert lateral_error_at(hi) == pytest.approx(hi * tan1, abs=1e-9)



# ===========================================================================
# 任务 5.4：时间记账匹配协议表性质测试（设计 §5 / §7.2 P4）
# ===========================================================================
import threading  # noqa: E402

from src.b_sim.arena import (  # noqa: E402
    CHANNEL_MAX,
    CHANNEL_MIN,
    CHANNEL_SWITCH_SECONDS,
    CLEAR_FAIL_SECONDS,
    CLEAR_SUCCESS_SECONDS,
    MEASURE_SECONDS,
    SPEED_MPS,
    Arena,
    Case,
    Jammer,
)
from src.b_sim.client import RobotClient  # noqa: E402
from src.b_sim.server import Session, make_server  # noqa: E402

PBT_ROBOT_ID = "99999"


def _make_pbt_case() -> Case:
    """构造一局固定案例，覆盖 1..20 全部频道，使 /clear 既能成功也能失败。

    源分布在竞技场内不同位置（接收半径足够大以便被清除成功命中判定复用），频道 1..20
    各一个源；因每个频道都存在源，故 `/clear` 命中与否只取决于清除点是否落在源 20 m 内，
    与本性质无关（本性质只关心时间记账口径，清除成功/失败均从 ClearResult 读回）。
    """
    jammers = [
        Jammer(
            channel=ch,
            x=120.0 * ((ch % 5) - 2),
            y=120.0 * ((ch // 5) - 2),
            reception_radius_m=1200.0,
            directional_deg=None,
        )
        for ch in range(CHANNEL_MIN, CHANNEL_MAX + 1)
    ]
    return Case(jammers=jammers, error_salt="pbt-p4", seed=0)


# 一次动作的抽象描述：类型（measure/clear）+ 目标坐标 + 频道。
_actions_strategy = st.lists(
    st.fixed_dictionaries(
        {
            "kind": st.sampled_from(["measure", "clear"]),
            "point": points_in_arena(margin=1.0),
            "channel": st.integers(min_value=CHANNEL_MIN, max_value=CHANNEL_MAX),
        }
    ),
    min_size=1,
    max_size=8,
)


# ---------------------------------------------------------------------------
# Feature: b-radio-interference-locate-clear, Property 4: 时间记账匹配协议表
# 设计 §5 时间记账表 / §7.2 P4 / _Requirements: 5.6_
# 对任意动作序列（每个动作 = 竞技场内随机点 + 随机频道 1..20 + measure/clear），
# 经 mock server + arena 回放后 `RobotClient.virtual_time_s` 等于按 §5 表逐项累加值：
#   移动 ‖Δ‖/5 s；切频（measure 且 channel≠当前测量频道）+1 s；/measure +5 s；
#   /clear 成功 +5 s / 失败 +3 s（/clear 不切换测量频道、无切频费）。
# 差值 ≤ 1e-6 s。
# ---------------------------------------------------------------------------
# **Validates: Requirements 5.6**
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(actions=_actions_strategy)
def test_property4_time_accounting_matches_protocol_table(actions):
    # 每个抽样案例启动一个全新的 mock server + arena（hypothesis 与 pytest fixture
    # 不能混用，故在测试体内自建线程化 server），回放动作后与独立记账对比。
    session = Session(PBT_ROBOT_ID, Arena(_make_pbt_case()))
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client = RobotClient(PBT_ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        client.enter()  # /enter 不推进虚拟时钟

        # 独立按协议表**逐项累加**期望虚拟时间。§5 明确"以微秒整数累加"，故每个动作先
        # 把 (移动 + 动作) 秒数按微秒四舍五入为整数微秒再累加（与 Arena._commit_move 的
        # int(round(seconds * 1e6)) 口径一致），最后换算回秒。这样"逐项累加值"是协议规定
        # 的记账语义，而非把原始浮点秒直接求和（后者在多次大位移下会偏离微秒累加值）。
        # 状态：当前机器狗坐标（从原点出发）、当前"测量频道"（初始 1）。
        cur_x, cur_y = 0.0, 0.0
        cur_channel = 1  # RobotState.channel 初值
        expected_us = 0  # 期望虚拟时间（整数微秒）

        def to_us(seconds: float) -> int:
            return int(round(seconds * 1_000_000))

        for act in actions:
            tx, ty = act["point"]
            ch = act["channel"]
            move_cost = math.hypot(tx - cur_x, ty - cur_y) / SPEED_MPS

            if act["kind"] == "measure":
                cost = move_cost + MEASURE_SECONDS
                if ch != cur_channel:
                    cost += CHANNEL_SWITCH_SECONDS  # 切频（换测量频道）+1 s
                expected_us += to_us(cost)
                cur_channel = ch  # measure 提交后更新当前测量频道
                client.measure(tx, ty, ch)
            else:  # clear：无切频费、不改变测量频道；成功 +5 / 失败 +3
                result = client.clear(tx, ty, ch)
                action_cost = CLEAR_SUCCESS_SECONDS if result.ok else CLEAR_FAIL_SECONDS
                expected_us += to_us(move_cost + action_cost)
                # 注意：/clear 不切换 cur_channel

            cur_x, cur_y = tx, ty  # 移动落点更新机器狗坐标

        expected = expected_us / 1_000_000.0
        assert abs(client.virtual_time_s - expected) <= 1e-6, (
            f"客户端虚拟时刻 {client.virtual_time_s:.9f} s 与协议表逐项累加期望 "
            f"{expected:.9f} s 偏差 {abs(client.virtual_time_s - expected):.3e} s 超过 1e-6"
        )
        # 交叉核对：服务端 arena 的虚拟时刻与客户端一致（同一真值来源）。
        assert client.virtual_time_s == pytest.approx(
            session.arena.robot.virtual_time_s, abs=1e-9
        )
    finally:
        httpd.shutdown()
        httpd.server_close()



# ===========================================================================
# 任务 5.5：清除幂等性质测试（设计 §5 / §7.2 P5）
# ===========================================================================
from src.b_sim.arena import CLEAR_RADIUS_M  # noqa: E402


def _single_source_case(source: Jammer) -> Case:
    """构造仅含一个源的固定案例（复用 P4 的确定性误差场口径）。

    该源占频道 1；除该源外竞技场内无其它源，故对频道 1 的 `/clear` 命中与否只取决于
    清除点是否落在源 20 m 内。清除成功一次后，源被标记 `cleared`，其后对同一源再次
    `/clear` 必返回 `no_target_in_range`（幂等）。
    """
    return Case(jammers=[source], error_salt="pbt-p5", seed=0)


# ---------------------------------------------------------------------------
# Feature: b-radio-interference-locate-clear, Property 5: 清除幂等
# 设计 §5 时间记账表 / §7.2 P5 / _Requirements: 5.7_
# 对任意（竞技场内、可被清除的）源与频道，成功清除一次后再次 `/clear` 同一点：
#   (a) 竞技场已清除源计数保持不变（不重复计入）；
#   (b) 第二次 ClearResult 非 ok（返回 `no_target_in_range`）；
#   (c) 第二次 `/clear` 的虚拟时间增量 = 失败清除耗时 +3 s（同点无移动费，移动费=0）。
# ---------------------------------------------------------------------------
# **Validates: Requirements 5.7**
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(source_point=points_in_arena(margin=1.0), channel=st.integers(min_value=CHANNEL_MIN, max_value=CHANNEL_MAX))
def test_property5_clear_is_idempotent(source_point, channel):
    sx, sy = source_point
    # 该源占据抽样频道；接收半径取固定值（本性质与接收半径无关，只需能被 20 m 内清除命中）。
    source = Jammer(channel=channel, x=sx, y=sy, reception_radius_m=1200.0, directional_deg=None)

    session = Session(PBT_ROBOT_ID, Arena(_single_source_case(source)))
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client = RobotClient(PBT_ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        client.enter()

        # 清除点：与源位置重合（落在源 20 m 清除半径内，保证第一次成功命中）。
        cx, cy = sx, sy
        assert math.hypot(cx - source.x, cy - source.y) <= CLEAR_RADIUS_M

        # 第一次 /clear：成功清除该源。
        first = client.clear(cx, cy, channel)
        assert first.ok, "第一次对源内点的 /clear 应成功命中"
        cleared_after_first = session.arena.case.cleared_count
        assert cleared_after_first == 1
        vt_after_first = client.virtual_time_s

        # 第二次 /clear：同一点（移动费=0），源已清除 → 幂等失败。
        second = client.clear(cx, cy, channel)
        vt_after_second = client.virtual_time_s

        # (a) 已清除源计数保持不变（不重复计入）。
        assert session.arena.case.cleared_count == cleared_after_first, (
            f"再次 /clear 后已清除计数从 {cleared_after_first} 变为 "
            f"{session.arena.case.cleared_count}，违反幂等"
        )
        # (b) 第二次 ClearResult 非 ok（no_target_in_range）。
        assert not second.ok, "对已清除源再次 /clear 应返回失败（no_target_in_range）"
        assert second.result == "no_target_in_range"

        # (c) 第二次虚拟时间增量 = 失败清除耗时 +3 s（同点移动费=0）。
        delta = vt_after_second - vt_after_first
        assert delta == pytest.approx(CLEAR_FAIL_SECONDS, abs=1e-6), (
            f"再次 /clear 虚拟时间增量 {delta:.9f} s 应等于失败清除耗时 "
            f"{CLEAR_FAIL_SECONDS:.1f} s（同点无移动费）"
        )
    finally:
        httpd.shutdown()
        httpd.server_close()



# ===========================================================================
# 任务 7.4：候选消除不删真值性质测试（设计 §3.4 / §7.2 P6）
# ===========================================================================
from src.b_sim.directional import (  # noqa: E402
    GRID_STEP_M,
    HEADING_STEP_DEG,
    DirectionalBelief,
    in_beam,
)


def _circular_heading_gap_deg(a: float, b: float) -> float:
    """两个朝向角（度）的最小圆周角差，落在 [0, 180]。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------
# Feature: b-radio-interference-locate-clear, Property 6: 候选消除不删真值
# 设计 §3.4 / §7.2 P6 / _Requirements: 4.4_
#
# P6 的核心语义是「消除不删真值」——候选消除必须是**可靠的（sound）**：约束更新只
# 删除**确实与观测矛盾**的候选，绝不删除与观测一致的候选。真实源（连同其确定性 ±1°
# 示向度）对任何一致观测永不矛盾，故只要消除是可靠的，真值信息就永不被误删。
#
# 【关于 R4.4 字面「≤60 m ∧ ≤15° 的近邻格点始终保留」为何不作为普适断言】
# R4.4 字面要求"位置偏差 ≤ 60 m（半个位置步长）且朝向偏差 ≤ 15° 的至少一个格点始终
# 保留"。经 hypothesis 反例核实，该字面界在两类边界情形下**并不普适成立**，且均**非**
# apply_silence/apply_bearing 的缺陷，而是「保守接收下限 1000 m」×「120 m 网格离散化」
# 的固有产物：
#   1. 接收半径边界壳：真值源距某沉默点恰在 1000 m 之外（如 1005.3 m，合法 no_signal），
#      但其半步近邻格点（距真值仅 59.1 m）却落在该沉默点 1000 m 之内（如 981.2 m）且
#      朝其辐射 → 该格点**确实与沉默观测矛盾**，被 apply_silence 正确删除。此时真值半步
#      近邻格点被合法消除。
#   2. 竞技场外边界：网格是无限点阵在半径 1800 m 圆域上的截断，真值贴边（外半格）时理想
#      对角近邻格点越界不在网格中，最近在网格格点可 >60 m（如真值 (972.5,1514.6) 最近
#      格点 (960,1440) 欧氏 75.7 m）。
# 因此本性质断言 P6 的**可证语义**——消除可靠性（每个被删候选都确与某次观测矛盾；每个
# 存活候选与所有观测一致）——它普适成立且直接刻画「消除不删真值」，不依赖网格足够密、
# 无需人为门控输入。（设计 §7.2 已附 R4.4 边界壳适用范围说明。）
# ---------------------------------------------------------------------------
# **Validates: Requirements 4.4**
POS_TOL_M = GRID_STEP_M / 2.0  # 60 m（半个位置步长）
HEADING_TOL_DEG = HEADING_STEP_DEG / 2.0  # 15°（半个朝向步长）

# apply_bearing 的角差容差：示向度误差 ±1° + 网格离散化 6°（与 directional.py 一致）。
_BEARING_TOL_DEG = 1.0 + 6.0


def _silence_contradicts(belief: DirectionalBelief, node, px: float, py: float) -> bool:
    """沉默约束下，候选 node=(pi,hi,tag) 是否与在 (px,py) 的 no_signal 观测矛盾。

    矛盾 ⟺ dist(node位置, p) ≤ 1000 m 且 p 落在该候选朝向的 ±90° 扇区内
    （与 directional.apply_silence 的删除条件逐字一致）。
    """
    pi, hi, _ = node
    sx, sy = belief.positions[pi]
    return math.hypot(px - sx, py - sy) <= RECEPTION_MIN_M and in_beam(
        sx, sy, belief.headings[hi], px, py
    )


def _bearing_consistent(belief: DirectionalBelief, node, px: float, py: float, svd_deg: float) -> bool:
    """示向度约束下，候选 node 是否与在 (px,py) 的 svd 观测一致（应被保留）。

    一致 ⟺ 候选位置相对 p 的方位与 svd 的圆周角差 ≤ 7° 且 p 落在候选朝向扇区内
    （与 directional.apply_bearing 的保留条件逐字一致）。
    """
    pi, hi, _ = node
    sx, sy = belief.positions[pi]
    bearing = math.degrees(math.atan2(sy - py, sx - px))
    if abs((bearing - svd_deg + 180.0) % 360.0 - 180.0) > _BEARING_TOL_DEG:
        return False
    return in_beam(sx, sy, belief.headings[hi], px, py)


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    source=sources_in_arena(directional=True),
    points=lists(points_in_arena(), min_size=1, max_size=12),
)
def test_property6_candidate_elimination_is_sound(source: Jammer, points):
    # 只对定向源成立（φ 必须存在，才能谈"朝向"与扇区）。
    assume(source.directional_deg is not None)

    belief = DirectionalBelief.fresh()

    # 记录每一步观测（类型 + 坐标 [+ svd]），逐点施加与真值一致的约束：
    #   可观测（接收半径内且落在真值扇区内）→ apply_bearing(确定性 ±1° 示向度)；
    #   不可观测（no_signal）→ apply_silence。
    #
    # 退化剔除：检测点与真值源重合（或几乎重合，≤5 m）时，源→点方位角无定义
    # （atan2(0,0)），且真实竞技场此时返回 `near` 而非 `direction`——喂 apply_bearing
    # 一个无意义方位本身即不一致观测，故与 arena 的 `near` 阈值一致地跳过这类点。
    NEAR_THRESHOLD_M = 5.0
    observations: list[tuple] = []  # ("silence", px, py) 或 ("bearing", px, py, svd)
    prev = set(belief.candidates)
    for px, py in points:
        if is_observable(source, px, py):
            if source.distance_to(px, py) <= NEAR_THRESHOLD_M:
                continue  # `near`：非示向度观测，退化跳过
            svd = observed_bearing(source, px, py)
            observations.append(("bearing", px, py, svd))
            belief.apply_bearing(px, py, svd)
            # 单步可靠性：该步删除的候选都与本次示向度观测不一致；存活的都一致。
            removed = prev - belief.candidates
            for node in removed:
                assert not _bearing_consistent(belief, node, px, py, svd), (
                    f"apply_bearing 误删了与观测一致的候选 {node} @({px:.1f},{py:.1f}) svd={svd:.3f}"
                )
        else:
            observations.append(("silence", px, py))
            belief.apply_silence(px, py)
            # 单步可靠性：该步删除的候选都确与本次沉默观测矛盾。
            removed = prev - belief.candidates
            for node in removed:
                assert _silence_contradicts(belief, node, px, py), (
                    f"apply_silence 误删了与沉默观测不矛盾的候选 {node} @({px:.1f},{py:.1f})"
                )
        prev = set(belief.candidates)

    # 端到端可靠性：每个**存活**候选必与**所有**观测一致（没有"本应被删却残留"，
    # 也没有"本不该删却被删"——后者由上面的单步断言覆盖）。这直接刻画「消除不删真值」：
    # 真实源（及其确定性示向度）与所有一致观测兼容，故其对应信息永不被消除。
    for node in belief.candidates:
        for obs in observations:
            if obs[0] == "silence":
                _, px, py = obs
                assert not _silence_contradicts(belief, node, px, py), (
                    f"存活候选 {node} 与沉默观测 @({px:.1f},{py:.1f}) 矛盾却未被删除"
                )
            else:
                _, px, py, svd = obs
                assert _bearing_consistent(belief, node, px, py, svd), (
                    f"存活候选 {node} 与示向度观测 @({px:.1f},{py:.1f}) svd={svd:.3f} 不一致却残留"
                )
