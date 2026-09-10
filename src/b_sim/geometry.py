"""B 题几何工具：楔形交会定位区域、区域直径、覆盖点集。

对应问题 1（定位区域直径算法与外接圆覆盖判定）与问题 3/4 的定位内核。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .arena import ARENA_RADIUS, ERROR_ABS_MAX_DEG, norm_deg

EPS = 1e-9


@dataclass(frozen=True)
class Bearing:
    """一次示向度观测：检测点 (x, y) 与读数 svd_deg。"""

    x: float
    y: float
    svd_deg: float


def _half_plane_from_ray(px: float, py: float, angle_deg: float, keep_left: bool) -> tuple[float, float, float]:
    """把过 (px,py)、方向 angle_deg 的射线转成半平面 a*x + b*y <= c。

    keep_left=True 保留射线左侧（逆时针一侧）。
    """
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    # 左侧法向为 (-dy, dx)；左侧满足 (-dy)(x-px) + dx(y-py) >= 0
    a, b = -(-dy), -dx
    c = a * px + b * py
    if keep_left:
        return a, b, c
    return -a, -b, -c


def wedge_half_planes(bearing: Bearing, error_deg: float = ERROR_ABS_MAX_DEG) -> list[tuple[float, float, float]]:
    """一次观测对应的 2*error 楔形，等于两个半平面的交。"""
    lower = bearing.svd_deg - error_deg
    upper = bearing.svd_deg + error_deg
    return [
        _half_plane_from_ray(bearing.x, bearing.y, lower, keep_left=True),
        _half_plane_from_ray(bearing.x, bearing.y, upper, keep_left=False),
    ]


def clip_polygon(polygon: list[tuple[float, float]], plane: tuple[float, float, float]) -> list[tuple[float, float]]:
    """用半平面 a*x + b*y <= c 裁剪凸多边形（Sutherland-Hodgman）。"""
    a, b, c = plane
    out: list[tuple[float, float]] = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        d1, d2 = a * x1 + b * y1 - c, a * x2 + b * y2 - c
        if d1 <= EPS:
            out.append((x1, y1))
        if (d1 > EPS) != (d2 > EPS):
            t = d1 / (d1 - d2)
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return out


# 定位区域退化面积阈值（平方米）：交会面积不超过该值即视为空/退化交会。
MIN_REGION_AREA = 1e-9


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    """凸多边形（顶点有序）的面积，鞋带公式，返回非负值。"""
    n = len(polygon)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2.0


@dataclass(frozen=True)
class LocalizationRegion:
    """定位区域结果：顶点列表 + 退化/空交会指示。

    - ``vertices``: 楔形交会所得凸多边形顶点（逆时针）；空/退化时为 ``[]``。
    - ``empty``: 交会为空或面积 ≤ ``MIN_REGION_AREA``（观测互相矛盾/退化）时为 ``True``。
    - ``reason``: 供人读的说明；``empty`` 为 ``True`` 时取 ``"交会为空"``。
    """

    vertices: list[tuple[float, float]]
    empty: bool
    reason: str = ""

    def __bool__(self) -> bool:  # 兼容 ``if region:`` 用法
        return not self.empty


def localization_region(
    bearings: list[Bearing],
    *,
    error_deg: float = ERROR_ABS_MAX_DEG,
    bound: float = 4.0 * ARENA_RADIUS,
) -> LocalizationRegion:
    """多个检测点的楔形交会得到的凸定位区域，附带"交会为空"指示。

    R1.2：观测数 ``N < 2`` 时拒绝输入（抛 ``ValueError``），不产生定位区域。
    R1.4：各楔形之交为空或面积 ≤ ``MIN_REGION_AREA``（矛盾/退化）时返回空区域并标记
    ``empty=True``、``reason="交会为空"``。
    """
    if len(bearings) < 2:
        raise ValueError(
            f"交会定位至少需要 2 个检测点，收到 {len(bearings)} 个（R1.2）"
        )
    polygon = [(-bound, -bound), (bound, -bound), (bound, bound), (-bound, bound)]
    for bearing in bearings:
        for plane in wedge_half_planes(bearing, error_deg):
            polygon = clip_polygon(polygon, plane)
            if not polygon:
                return LocalizationRegion([], empty=True, reason="交会为空")
    if _polygon_area(polygon) <= MIN_REGION_AREA:
        return LocalizationRegion([], empty=True, reason="交会为空")
    return LocalizationRegion(polygon, empty=False)


def localization_polygon(
    bearings: list[Bearing],
    *,
    error_deg: float = ERROR_ABS_MAX_DEG,
    bound: float = 4.0 * ARENA_RADIUS,
) -> list[tuple[float, float]]:
    """多个检测点的楔形交会得到的凸定位区域顶点（逆时针）。

    以一个足够大的正方形为初始区域，逐个半平面裁剪。

    - R1.2：观测数 ``N < 2`` 时抛 ``ValueError``（此前静默返回 ``[]``）。
    - R1.4：交会为空或面积 ≤ ``MIN_REGION_AREA`` 时返回 ``[]``（"交会为空"，可用
      :func:`localization_region` 取得显式指示）。
    """
    return localization_region(bearings, error_deg=error_deg, bound=bound).vertices


def polygon_diameter(polygon: list[tuple[float, float]]) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """凸多边形直径：任意两点最大距离，必在顶点对上取到。返回 (直径, 端点A, 端点B)。

    R1.5：存在多个取得最大距离的顶点对时，确定性地返回**按顶点索引 (i, j) 升序的首个**
    （外层 ``i`` 升序、内层 ``j`` 升序，配合严格大于比较，只有更大的距离才会替换当前最优，
    故首个达到最大值的顶点对被保留）。
    """
    if len(polygon) < 2:
        return 0.0, (0.0, 0.0), (0.0, 0.0)
    best = -1.0
    pair = (polygon[0], polygon[0])
    for i in range(len(polygon)):
        for j in range(i + 1, len(polygon)):
            d = math.hypot(polygon[i][0] - polygon[j][0], polygon[i][1] - polygon[j][1])
            if d > best:  # 严格大于 → 保留首个（索引升序）达到最大距离的顶点对
                best, pair = d, (polygon[i], polygon[j])
    return best, pair[0], pair[1]


# 外接圆覆盖判定的数值容差（米）：|worst − D/2| ≤ COVER_TOL 视为落在圆上。
COVER_TOL_M = 1e-7


def circle_on_diameter_covers(polygon: list[tuple[float, float]]) -> tuple[bool, float]:
    """以区域直径为直径的圆（圆心取直径中点）是否覆盖整个定位区域。

    返回 ``(covers, margin_m)``（R1.7 / R1.8）：

    - ``margin = D/2 − max_{v ∈ vertices} ‖v − M‖``（单位：米，``M`` 为直径中点、``D``
      为区域直径）。``margin`` 为正表示所有顶点在圆内、有裕度；为负表示存在顶点落在圆外
      （如正三角形型区域，最远顶点距中点 ``≈0.866L > L/2``，Jung 平面上界 ``D/√3``）。
    - ``covers ⟺ max‖v − M‖ ≤ D/2 + COVER_TOL_M``，等价于 ``margin ≥ −COVER_TOL_M``。

    退化几何（R1.6）：
    - 单点退化（所有顶点两两距离 ``≤ 1e-7`` 米，``D = 0``）标记"覆盖"，返回 ``margin = 0.0``。
    - 线段退化（顶点共线，``D`` = 两端点距离）：所有顶点落在以直径为直径的圆上（即区间内），
      故 ``max‖v − M‖ ≤ D/2``，判定"覆盖"，``margin ≥ 0``。
    """
    if not polygon:
        # 空区域无顶点：无外接圆可言，按"覆盖"处理并给出 0 裕度（不供正常路径使用）。
        return True, 0.0
    diameter, pa, pb = polygon_diameter(polygon)
    radius = diameter / 2.0
    if diameter <= 1e-7:
        # 单点退化：所有顶点两两 ≤ 1e-7，D = 0，圆退化为一点仍"覆盖"该点集。
        return True, 0.0
    cx, cy = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
    worst = max(math.hypot(x - cx, y - cy) for x, y in polygon)
    margin = radius - worst
    return worst <= radius + COVER_TOL_M, margin


def centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    """凸多边形面积质心；退化时回退到顶点平均。"""
    n = len(polygon)
    if n == 0:
        raise ValueError("空多边形")
    if n < 3:
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    area2 = 0.0
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < EPS:
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    return (cx / (3.0 * area2), cy / (3.0 * area2))


def least_squares_intersection(bearings: list[Bearing]) -> tuple[float, float] | None:
    """多条示向线的最小二乘交点（最小化到各直线的垂距平方和）。"""
    if len(bearings) < 2:
        return None
    a11 = a12 = a22 = b1 = b2 = 0.0
    for bearing in bearings:
        rad = math.radians(bearing.svd_deg)
        nx, ny = -math.sin(rad), math.cos(rad)  # 直线法向
        c = nx * bearing.x + ny * bearing.y
        a11 += nx * nx
        a12 += nx * ny
        a22 += ny * ny
        b1 += nx * c
        b2 += ny * c
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return None
    return ((b1 * a22 - b2 * a12) / det, (b2 * a11 - b1 * a12) / det)


def estimate_position(bearings: list[Bearing]) -> tuple[float, float] | None:
    """优先用楔形交会区域的质心，退化时用最小二乘交点。"""
    if len(bearings) < 2:
        return None
    polygon = localization_polygon(bearings)
    if polygon:
        cx, cy = centroid(polygon)
        if math.isfinite(cx) and math.isfinite(cy):
            return cx, cy
    return least_squares_intersection(bearings)


def bearing_spread_deg(bearings: list[Bearing]) -> float:
    """观测集合中任意两条示向线的最大夹角，用来判断交会几何是否良态。"""
    if len(bearings) < 2:
        return 0.0
    best = 0.0
    for i in range(len(bearings)):
        for j in range(i + 1, len(bearings)):
            diff = abs(norm_deg(bearings[i].svd_deg) - norm_deg(bearings[j].svd_deg))
            diff = min(diff, 360.0 - diff)
            best = max(best, min(diff, 180.0 - diff) if diff > 90 else diff)
    return best


def second_point_candidates(
    p1: tuple[float, float],
    theta1: float,
    *,
    offsets: tuple[float, ...] = (32.0, 16.0, 0.0),
    length: float = 380.0,
    recep_lower: float = 1000.0,
) -> list[tuple[float, float]]:
    """Q2：第一检测点 ``p1`` 已获示向度 ``theta1`` 后，生成第二检测点候选序列。

    每个候选沿示向度方向前进 ``length`` 米并叠加小角度偏移 ``δ``：

        ``p2 = p1 + length · (cos(theta1 + s·δ), sin(theta1 + s·δ))``

    角度约定与 ``arena``/本模块一致：正东 = 0°、逆时针为正，读数以度计
    （``atan2(dy, dx)`` 口径），故 ``(cos, sin)`` 即为沿方位角前进的单位向量。

    候选序列按偏移角 ``δ`` **降序**排列（大 → 小逐级退让），对每个 ``δ > 0`` 生成
    ``s = +1`` 与 ``s = -1`` 一对候选（先 ``+`` 后 ``-``）；``δ = 0°`` 只生成一个
    沿示向度直行的候选。此顺序保证使用方按"先大偏移博取良态交会角、失败再退让、
    最终退化为沿示向度直行"的策略遍历（R2.1、R2.4）。

    仅实现 R2.1（有序候选生成）与 R2.4（沿示向度前进 + ``|δ| ≤ 32°`` 小偏移；
    前进分量 ``length·cos δ`` 恒为正，绝不产生纯横向平移或反向候选）。本函数**不**
    对候选做有效性/交会角/接收半径/越界判定，也不实现退让/失败语义或候选区域界定。

    扩展点（后续任务）：
      * R2.2/R2.3/R2.5/R2.6（有效性判定与退让/显式失败）——任务 3.2。
      * R2.7（``θ1±32°`` 前向楔形内、``α∈[60,120]``、``d2≤1000``、``‖p2‖≤1800``
        的环状扇形候选区域界定）——任务 3.3。
    ``recep_lower``（有效接收半径保证下限，默认 1000 m）为上述扩展预留，本函数不使用。

    参数中偏移角必须满足 ``|δ| ≤ 32°``（R2.4 上界），否则抛 ``ValueError``。
    """
    x1, y1 = p1
    max_offset = 32.0
    candidates: list[tuple[float, float]] = []
    # 按 δ 降序遍历，保证候选序列由大偏移到沿示向度直行。
    for delta in sorted(offsets, reverse=True):
        if abs(delta) > max_offset + EPS:
            raise ValueError(
                f"偏移角 |δ|={abs(delta)}° 超过 32° 上界（R2.4），不生成非法横向候选"
            )
        signs = (1.0, -1.0) if abs(delta) > EPS else (1.0,)
        for sign in signs:
            phi = math.radians(theta1 + sign * delta)
            x2 = x1 + length * math.cos(phi)
            y2 = y1 + length * math.sin(phi)
            candidates.append((x2, y2))
    return candidates


def coverage_points(radius: float, arena_radius: float = ARENA_RADIUS) -> list[tuple[float, float]]:
    """构造覆盖点集：使目标区域内任意一点到某个覆盖点的距离不超过 radius。

    用同心环布点。返回的点全部位于目标区域内（含边界）。
    """
    if radius <= 0:
        raise ValueError("radius 必须为正")
    points = [(0.0, 0.0)]
    ring_step = radius * 1.35  # 环间距，留裕量
    ring_radius = ring_step
    while ring_radius <= arena_radius + ring_step:
        r = min(ring_radius, arena_radius)
        # 该环上相邻点弦长不超过 radius
        count = max(3, int(math.ceil(2.0 * math.pi / (2.0 * math.asin(min(1.0, radius / (2.0 * r)))))))
        for k in range(count):
            theta = 2.0 * math.pi * k / count
            points.append((r * math.cos(theta), r * math.sin(theta)))
        if r >= arena_radius:
            break
        ring_radius += ring_step
    return points


def verify_coverage(points: list[tuple[float, float]], radius: float,
                    arena_radius: float = ARENA_RADIUS, samples: int = 20000) -> float:
    """数值校验覆盖率：在目标区域内采样，返回被覆盖的比例。"""
    covered = 0
    total = 0
    steps = int(math.sqrt(samples))
    for i in range(steps):
        for j in range(steps):
            x = -arena_radius + 2.0 * arena_radius * (i + 0.5) / steps
            y = -arena_radius + 2.0 * arena_radius * (j + 0.5) / steps
            if math.hypot(x, y) > arena_radius:
                continue
            total += 1
            if any(math.hypot(x - px, y - py) <= radius for px, py in points):
                covered += 1
    return covered / total if total else 1.0
