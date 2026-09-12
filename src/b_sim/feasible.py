"""频道可行域：把每一次检测结果当成集合约束，序贯收缩"这个频道的源还可能是什么样"。

## 一个模型同时覆盖问题 3 与问题 4

候选状态是**(位置, 辐射模式)** 二元组：

    模式 0        全向源（有效覆盖角度 360°）
    模式 1..H     定向源，定向方向取 (m−1)·360/H（H = `headings`，默认 12 → 30° 一档）

问题 3 已知全部为全向源，取 `headings=0`，状态空间退化成纯位置格点；问题 4 取
`headings=12`，同一套约束更新与站位规划代码原封不动地复用。这样"是否含定向源"从
两套策略变成了**一个参数**。

## 三类检测结果都是集合约束

设检测点 p、源位置 s、有效接收半径 R_s ∈ [1000, 1500]（附录 2(2)，具体值未知）、
定向源有效覆盖角度为定向方向两侧各 90°（附录 1(3)）。记 `in_beam(s, φ, p)` 为"p 落在
以 φ 为轴、半角 90° 的扇区内"。

| 结果 | 蕴含 | 本模块的保守实现 |
|---|---|---|
| `no_signal` | dist(p,s) > R_s **或** ¬in_beam | 仅当 dist ≤ 1000−h 时才删，且只删 in_beam 有裕度的模式 |
| `direction` | dist ≤ R_s ≤ 1500 **且** in_beam **且** 方位在 ±1° 内 | 只保留同时满足三者（含离散化裕度）的状态 |
| `near` | dist ≤ 5 且 in_beam | 只保留该邻域内的状态 |

**沉默对全向源是硬约束**（dist > R_s ≥ 1000），原策略只在问题 4 用它，问题 3 完全靠
固定站位盲扫。事实上原点普扫一轮就排除了 π·1000²/π·1800² ≈ 31% 的面积。

## 用途

可行域**不做最终定位**（那由 `geometry.py` 的交会定位区域负责，精度到米级），而是回答
三个搜索决策问题：

1. **这个频道还可能有源吗？** 状态集合为空 ⟹ 该频道确实没有未清除的源，可立即停止
   搜索。原实现只能靠"检测预算耗尽"被动放弃，为不存在的源反复跑站位。
2. **下一个补测站位站哪？** 见 `cover_stations`：贪心最大覆盖选出**有覆盖保证的最小
   站位集**。注意"站在 p 沉默能排除的状态"与"站在 p 能保证发现的状态"是同一个集合，
   所以最大覆盖同时就是**最大信息增益**探测点选择，不需要另写一套。
3. **盲清除该打哪些格？** 见 `alive_cells`：`/clear` 只看距离、与辐射方向无关
   （附录 2(8)），所以候选位置压缩到少数格点后，逐格盲清除比继续找信号便宜。

## 离散化与可靠性

位置离散为边长 `CELL_STEP_M` 的方格中心，记半对角 h = (√2/2)·step；朝向离散为 H 档，
记半档宽 Δ/2 = 180/H 度。所有约束都按**保守方向**取舍，保证"真值最近的格点状态永不被
误删"：

* 位置裕度 h：删除只在 dist(p, 格心) ≤ 1000 − h 时发生（此时整格都在保证接收范围内）；
* 朝向裕度 Δ/2：真定向方向 φ 与最近档 φ̂ 相差 ≤ Δ/2，故"删除"要求角差有 Δ/2 的余量，
  "保留"则放宽 Δ/2；
* 方位裕度 arctan(h/(d−h))：格心方位与格内真值方位之差的上界。

因此可行域是真实可行集的**外包近似（超集）**：不会把真源排除掉，代价是略保守。

## 数据结构

`alive` 用 `dict[格点下标, 模式位掩码]`：一次约束更新只需遍历"还活着的格点"（≤1009 个），
每格几次三角函数即可批量处理它的全部模式（掩码上的位运算）。相比"把状态展开成集合"
（问题 4 有 1009×13 ≈ 1.3 万个状态）快一个量级，这在 20 分钟现实时间预算里很关键。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from functools import lru_cache

from .arena import (
    ARENA_RADIUS,
    CHANNEL_SWITCH_SECONDS,
    DIRECTIONAL_HALF_ANGLE_DEG,
    ERROR_ABS_MAX_DEG,
    MEASURE_SECONDS,
    NEAR_THRESHOLD_M,
    RECEPTION_MAX_M,
    RECEPTION_MIN_M,
    SPEED_MPS,
)
from .route import Point, order_route, path_length

# 格点边长。100 m 时圆域内约 1009 个格点：一次约束更新是千次量级的浮点运算（微秒级）。
# 100 m 分辨率对"下一站站哪"这个决策足够——站位覆盖半径本身是 1000 m 量级。
CELL_STEP_M = 100.0
# 定向方向的离散档数（问题 4）。12 档 = 30° 一档，与扇区半角 90° 相比足够细，
# 而保守裕度 Δ/2 = 15° 只让"沉默可删除的扇区"从 180° 收窄到 150°，代价可接受。
HEADING_COUNT = 12
MODE_OMNI = 0                     # 模式 0 恒为全向


def half_diagonal(step: float) -> float:
    """格点半对角：格心到格内最远点的距离，所有保守裕度都以它为单位。"""
    return step * math.sqrt(2.0) / 2.0


def guaranteed_reach(step: float = CELL_STEP_M,
                     reception_min_m: float = RECEPTION_MIN_M) -> float:
    """站在某点能"保证"覆盖到的格心距离上限（米）。"""
    return reception_min_m - half_diagonal(step)


# 格点半径上限。**关键**：干扰源都在半径 1800 m 的目标区域内（题面），但机器狗可以走到
# 目标区域之外（附件 2 §1.1：「机器狗可以向模拟器提交目标区域外的位置；模拟器仍会按提交的
# 位置计算移动耗时和检测、清除结果」）。这一点对问题 4 至关重要：一个贴近边界、朝**区域外**
# 辐射的定向源，其可见扇区几乎全部落在目标区域之外，站在区域内任何地方都收不到它的信号。
# 因此候选站位必须允许伸出边界 1000 m（超过接收半径下限就没有意义了）。
STATION_RADIUS_M = ARENA_RADIUS + RECEPTION_MIN_M


@lru_cache(maxsize=8)
def _lattice(step: float) -> tuple[tuple[Point, ...], dict[tuple[int, int], int]]:
    """格心（按 (x, y) 升序）与 (i, j) -> 下标 的索引。

    格心覆盖到 `STATION_RADIUS_M`（含目标区域外的候选站位）；其中"可能是源"的格心由
    `_source_cells` 另行给出（只含目标区域内的）。两者共用一套正规格，"距离 ≤ reach"的邻域
    才能继续用平移不变模板表示。
    """
    n = int(math.ceil(STATION_RADIUS_M / step))
    keys = [
        (i, j)
        for i in range(-n, n + 1)
        for j in range(-n, n + 1)
        if math.hypot(i * step, j * step) <= STATION_RADIUS_M
    ]
    keys.sort(key=lambda k: (k[0] * step, k[1] * step))
    points = tuple((i * step, j * step) for i, j in keys)
    index = {key: pos for pos, key in enumerate(keys)}
    return points, index


def _cells(step: float) -> tuple[Point, ...]:
    return _lattice(step)[0]


@lru_cache(maxsize=8)
def _source_cells(step: float) -> tuple[int, ...]:
    """可能是干扰源位置的格心下标：只有目标区域（半径 1800 m）内的格心。"""
    points = _cells(step)
    return tuple(i for i, (x, y) in enumerate(points)
                 if math.hypot(x, y) <= ARENA_RADIUS)


@lru_cache(maxsize=8)
def _coverage_table(step: float, reach: float) -> tuple[tuple[int, ...], ...]:
    """格心 -> 站在该格心能保证覆盖（按距离）的格点下标。

    **候选站位就取格心本身**，这一点很关键。早先版本用一套独立的 200 m 站位网格，结果
    与"归航途中在任意位置产生的沉默圆盘"不对齐，两者边缘之间留下大量新月形碎片；每块
    碎片都要一个独立站位、每个站位增益只有个位数格点，把自适应补测拖成上百轮抖动。改用
    格心当候选后，任何碎片都能被"站到碎片里"一次吞掉（覆盖半径 929 m ≈ 271 格）。

    另一个好处：格心构成正规格，"距离 ≤ reach"的邻域是**平移不变的模板（stencil）**，
    只需算一次偏移集合，全部格点复用。
    """
    points, index = _lattice(step)
    reach_sq = max(0.0, reach) ** 2
    span = int(math.floor(reach / step))
    stencil = [
        (di, dj)
        for di in range(-span, span + 1)
        for dj in range(-span, span + 1)
        if (di * step) ** 2 + (dj * step) ** 2 <= reach_sq
    ]
    table: list[tuple[int, ...]] = []
    for x, y in points:
        i, j = int(round(x / step)), int(round(y / step))
        covered = [
            target for di, dj in stencil
            if (target := index.get((i + di, j + dj))) is not None
        ]
        table.append(tuple(covered))
    return tuple(table)


@lru_cache(maxsize=8)
def _beam_stencil(step: float, reach: float, headings: int
                  ) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    """定向情形的平移不变模板：`((di, dj, 掩码桶号), ...)` 与桶号 -> 模式掩码。

    对格心站位 s 与格心源 x，"站在 s 能保证发现 (x, 模式 m)"只依赖偏移 o = s − x：
    距离条件 |o| ≤ reach，扇区条件 |方位(o) − 定向方向_m| ≤ 90° − 裕度。两者都是 o 的函数，
    因此可以像全向情形的 stencil 一样预计算一次、全格点复用。

    掩码只有几十种取值（12 个旋转 × 几档宽度 + 全掩码），故按桶号索引即可。
    """
    span = int(math.floor(reach / step))
    half = half_diagonal(step)
    _, _, half_band, _ = _mode_geometry(headings)
    masks: list[int] = []
    bucket: dict[int, int] = {}
    offsets: list[tuple[int, int, int]] = []
    for di in range(-span, span + 1):
        for dj in range(-span, span + 1):
            d = math.hypot(di * step, dj * step)
            if d > reach:
                continue
            if d <= half:
                # 与 `FeasibleSet._visible_mask(conservative_drop=True)` 一致：
                # 近场内定向模式不能被判定为必然可见，只有全向位可以。
                mask = 1
            else:
                slack = half_band + math.degrees(math.atan2(half, d - half))
                mask = 1 | _arc_mask(headings, math.degrees(math.atan2(dj, di)),
                                     DIRECTIONAL_HALF_ANGLE_DEG - slack)
            index = bucket.get(mask)
            if index is None:
                index = bucket[mask] = len(masks)
                masks.append(mask)
            offsets.append((di, dj, index))
    return tuple(offsets), tuple(masks)


def _angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@lru_cache(maxsize=16)
def _mode_geometry(headings: int) -> tuple[int, tuple[float, ...], float, int]:
    """(模式总数, 各模式的定向方向, 朝向半档宽, 全模式掩码)。"""
    modes = 1 + headings
    step_deg = 360.0 / headings if headings else 0.0
    directions = tuple((0.0,) + tuple(k * step_deg for k in range(headings)))
    return modes, directions, step_deg / 2.0, (1 << modes) - 1


def _arc_mask(headings: int, centre_deg: float, half_width_deg: float) -> int:
    """朝向落在 [centre−half, centre+half] 弧内的模式位掩码（不含全向位）。"""
    if headings <= 0 or half_width_deg <= 0.0:
        return 0
    _, directions, _, _ = _mode_geometry(headings)
    mask = 0
    for mode in range(1, headings + 1):
        if _angle_diff_deg(directions[mode], centre_deg) <= half_width_deg:
            mask |= 1 << mode
    return mask


class FeasibleSet:
    """一个频道上"源可能在哪、朝向哪"的保守外包状态集合。"""

    __slots__ = ("step", "headings", "_half", "_modes", "_all_mask", "_half_band",
                 "alive", "observations", "silence_count")

    def __init__(self, step: float = CELL_STEP_M, headings: int = 0) -> None:
        self.step = step
        self.headings = headings
        self._half = half_diagonal(step)
        self._modes, _, self._half_band, self._all_mask = _mode_geometry(headings)
        # alive: 格点下标 -> 该格点上仍可能的模式位掩码（只初始化目标区域内的格心）
        self.alive: dict[int, int] = {i: self._all_mask for i in _source_cells(step)}
        self.observations = 0
        self.silence_count = 0

    # ---- 查询 ----

    @property
    def cells(self) -> tuple[Point, ...]:
        return _cells(self.step)

    @property
    def possible(self) -> bool:
        """状态集合非空 = 该频道仍可能存在未清除的源。"""
        return bool(self.alive)

    @property
    def state_count(self) -> int:
        """剩余候选状态数（位置 × 模式）。"""
        return sum(mask.bit_count() for mask in self.alive.values())

    def __len__(self) -> int:
        """剩余候选位置数（对模式做边缘化）。"""
        return len(self.alive)

    def alive_cells(self) -> list[int]:
        return sorted(self.alive)

    def alive_points(self) -> list[Point]:
        cells = _cells(self.step)
        return [cells[i] for i in sorted(self.alive)]

    def weighted_points(self) -> list[tuple[Point, int]]:
        """候选位置及其剩余模式数，用于盲清除按可能性排序。"""
        cells = _cells(self.step)
        return sorted(((cells[i], mask.bit_count()) for i, mask in self.alive.items()),
                      key=lambda item: (-item[1], item[0]))

    def centroid(self) -> Point | None:
        points = self.alive_points()
        if not points:
            return None
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))

    def spread_m(self) -> float:
        """候选位置包围盒对角线，用于判断"是否已收敛到可以直接走过去清除"。"""
        points = self.alive_points()
        if len(points) <= 1:
            return 0.0
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def copy(self) -> FeasibleSet:
        clone = FeasibleSet(self.step, self.headings)
        clone.alive = dict(self.alive)
        clone.observations = self.observations
        clone.silence_count = self.silence_count
        return clone

    # ---- 内部：某格点在观测点 p 处"可见"的模式掩码 ----

    def _visible_mask(self, cell: Point, px: float, py: float, *,
                      conservative_drop: bool) -> int:
        """站在 p 时，格点 cell 上哪些模式会让信号"落在有效覆盖角度内"。

        `conservative_drop=True`（用于删除，如沉默约束与覆盖保证）时把扇区**收窄**
        Δ/2 + 方位裕度；`False`（用于保留，如示向度约束）时把扇区**放宽**同样的量。
        两个方向都保证真值所在的最近档不被误删。
        """
        if self.headings <= 0:
            return 1                       # 只有全向模式
        dx, dy = px - cell[0], py - cell[1]
        d = math.hypot(dx, dy)
        if d <= self._half:
            # 紧贴格点：源到观测点的方位完全不受格心约束，因此**没有**定向模式可以被
            # 判定为"必然覆盖"（删除方向必须只保留全向位，否则会误删真值）；
            # 保留方向仍然认为任何朝向都可能覆盖它。
            return 1 if conservative_drop else self._all_mask
        slack = self._half_band + math.degrees(math.atan2(self._half, d - self._half))
        half_width = DIRECTIONAL_HALF_ANGLE_DEG + (-slack if conservative_drop else slack)
        bearing = math.degrees(math.atan2(dy, dx))     # 源 -> 观测点 的方位
        return 1 | _arc_mask(self.headings, bearing, half_width)

    # ---- 约束更新 ----

    def apply_silence(self, px: float, py: float,
                      reception_min_m: float = RECEPTION_MIN_M) -> int:
        """在 (px,py) 检测无信号：删除"该点必然应当收到信号"的状态，返回删除的状态数。

        必然应当收到信号 ⟺ 整格都在保证接收范围内（dist ≤ 1000 − h）**且**该模式的扇区
        有裕度地包含 p。全向模式恒满足后者，因此问题 3 里这一步就是"挖掉一个圆盘"。
        """
        cells = _cells(self.step)
        limit = reception_min_m - self._half
        self.observations += 1
        self.silence_count += 1
        if limit <= 0.0:
            return 0
        limit_sq = limit * limit
        removed = 0
        survivors: dict[int, int] = {}
        for i, mask in self.alive.items():
            cell = cells[i]
            if (cell[0] - px) ** 2 + (cell[1] - py) ** 2 > limit_sq:
                survivors[i] = mask
                continue
            keep = mask & ~self._visible_mask(cell, px, py, conservative_drop=True)
            removed += (mask & ~keep).bit_count()
            if keep:
                survivors[i] = keep
        self.alive = survivors
        return removed

    def apply_bearing(self, px: float, py: float, svd_deg: float,
                      error_deg: float = ERROR_ABS_MAX_DEG,
                      reception_max_m: float = RECEPTION_MAX_M) -> int:
        """在 (px,py) 得到示向度：只保留楔形内、接收半径上限内、且扇区含 p 的状态。"""
        cells = _cells(self.step)
        half = self._half
        range_limit = reception_max_m + half
        removed = 0
        survivors: dict[int, int] = {}
        for i, mask in self.alive.items():
            cx, cy = cells[i]
            dx, dy = cx - px, cy - py
            d = math.hypot(dx, dy)
            if d > range_limit:
                removed += mask.bit_count()
                continue
            if d > half:
                slack = math.degrees(math.atan2(half, d - half))
                if _angle_diff_deg(math.degrees(math.atan2(dy, dx)), svd_deg) > error_deg + slack:
                    removed += mask.bit_count()
                    continue
            keep = mask & self._visible_mask(cells[i], px, py, conservative_drop=False)
            removed += (mask & ~keep).bit_count()
            if keep:
                survivors[i] = keep
        self.alive = survivors
        self.observations += 1
        return removed

    def apply_near(self, px: float, py: float,
                   threshold_m: float = NEAR_THRESHOLD_M) -> int:
        """检测返回 near：源距该点 ≤ threshold_m 且 p 在有效覆盖角度内（信号过强）。"""
        cells = _cells(self.step)
        limit = threshold_m + self._half
        limit_sq = limit * limit
        removed = 0
        survivors: dict[int, int] = {}
        for i, mask in self.alive.items():
            cell = cells[i]
            if (cell[0] - px) ** 2 + (cell[1] - py) ** 2 > limit_sq:
                removed += mask.bit_count()
                continue
            keep = mask & self._visible_mask(cell, px, py, conservative_drop=False)
            removed += (mask & ~keep).bit_count()
            if keep:
                survivors[i] = keep
        self.alive = survivors
        self.observations += 1
        return removed

    # ---- 覆盖 / 信息增益 ----

    def coverage_gain(self, px: float, py: float,
                      reception_min_m: float = RECEPTION_MIN_M) -> int:
        """站在 (px,py) 能**保证发现**的剩余状态数。

        这同时是"若该点沉默可排除的状态数"，即信息增益：一次检测要么发现目标，要么把
        这批状态全部排除。所以最大覆盖 = 最大信息增益，两个目标在本题里是同一个。
        """
        cells = _cells(self.step)
        limit = max(0.0, reception_min_m - self._half)
        limit_sq = limit * limit
        total = 0
        for i, mask in self.alive.items():
            cell = cells[i]
            if (cell[0] - px) ** 2 + (cell[1] - py) ** 2 > limit_sq:
                continue
            total += (mask & self._visible_mask(cell, px, py,
                                                conservative_drop=True)).bit_count()
        return total

    def detection_possible(self, px: float, py: float,
                           reception_max_m: float = RECEPTION_MAX_M) -> bool:
        """站在 (px,py) 是否**有可能**收到该频道的信号。

        与 `coverage_gain`（"保证"发现，用下界 1000 m）对偶：这里用接收半径**上界** 1500 m。
        若所有候选位置都在 1500 m 之外，那么在此检测既不可能发现目标（超出任何可能的接收
        半径），沉默也排除不了任何候选（沉默只能排除 1000 m 内的）——这次检测的信息量与
        发现概率**同时为零**，是纯粹浪费的 5-6 s，可以安全跳过。

        这个门禁专治"补测站位是为别的频道选的"这种情形：一站要扫 7-9 个待搜索频道，其中
        往往有几个的剩余可行域离该站位很远。
        """
        if not self.alive:
            return False
        cells = _cells(self.step)
        limit = reception_max_m + self._half
        limit_sq = limit * limit
        return any((cells[i][0] - px) ** 2 + (cells[i][1] - py) ** 2 <= limit_sq
                   for i in self.alive)

    def uncovered_by(self, stations: Iterable[Point],
                     reception_min_m: float = RECEPTION_MIN_M) -> set[int]:
        """给定站位集合后仍未被保证覆盖的候选位置（对模式做边缘化）。"""
        cells = _cells(self.step)
        limit = max(0.0, reception_min_m - self._half)
        limit_sq = limit * limit
        remaining = set(self.alive)
        for sx, sy in stations:
            remaining = {
                i for i in remaining
                if (cells[i][0] - sx) ** 2 + (cells[i][1] - sy) ** 2 > limit_sq
            }
        return remaining


def cover_stations(sets: Sequence[FeasibleSet], current: Point, *,
                   max_stations: int = 8, cell_step: float = CELL_STEP_M,
                   reception_min_m: float = RECEPTION_MIN_M,
                   min_gain: int = 1, travel_weight_m: float = 20.0,
                   gain_tolerance: float = 0.9,
                   anchors: Sequence[Point] = ()) -> list[Point]:
    """贪心最大覆盖：选出一个"有覆盖保证"的最小站位集（未排序）。

    把所有未定位频道的剩余候选位置并成待覆盖多重集合（同一格点在不同频道上算不同元素），
    每步选净收益最大的候选站位。贪心集合覆盖有 ln n + 1 近似保证，对本题（待覆盖集合是
    圆盘上的几块连通区域）实际非常接近最优。

        净收益 = 覆盖格点数 − 到最近锚点的距离 / travel_weight_m

    `anchors` 传入"机器狗接下来反正要去的地方"（当前位置 + 待清除目标的估计位置），让站位
    倾向落在既有巡回附近。实测 `travel_weight_m` 宜取大（弱折价）：折价过强会让贪心选近处
    的低增益站位，站位数变多、总路程反而更长（0.5 m/格 → T̄ 365 s；20 m/格 → 262 s）。
    站位次序另由 2-opt 巡回负责，所以"选站位"环节应以覆盖效率为主、移动为辅。

    覆盖判定按**位置**（距离 ≤ 1000 − h）而非逐模式判定。对定向源这只是必要条件——站得够近
    但落在扇区外仍收不到信号，这正是"盲目加密站位有天花板"的根源（见
    `coverage_design.py`）；问题 4 因此以候选格点盲清除收尾（`/clear` 与辐射方向无关）。

    `min_gain` 是"值不值得专门跑一趟"的闸门。实现上维护"格心 -> 当前增益"，每步只对受影响
    格心做增量修正，避免重扫全表。
    """
    if not sets:
        return []
    if any(fs.headings for fs in sets):
        return _cover_stations_directional(
            sets, current, max_stations=max_stations, cell_step=cell_step,
            reception_min_m=reception_min_m, min_gain=min_gain,
            travel_weight_m=travel_weight_m, gain_tolerance=gain_tolerance,
            anchors=anchors)
    points = _cells(cell_step)
    table = _coverage_table(cell_step, guaranteed_reach(cell_step, reception_min_m))

    # need[格点] = 还有多少个频道需要覆盖该格点
    need: dict[int, int] = {}
    for feasible in sets:
        for index in feasible.alive:
            need[index] = need.get(index, 0) + 1
    if not need:
        return []

    # "距离 ≤ reach" 是对称关系，故 table 同时充当正向覆盖表与逆向索引。
    gain: dict[int, int] = {}
    for cell, count in need.items():
        for si in table[cell]:
            gain[si] = gain.get(si, 0) + count

    chosen: list[Point] = []
    live_anchors = [current, *anchors]
    scale = max(travel_weight_m, 1e-6)
    for _ in range(max_stations):
        # 只在"增益够大、值得专门跑一趟"的候选里比净收益。若把 min_gain 检查放在按净收益
        # 选出的最优者之后，移动折价会先把一个近处的低增益站位顶上来，于是刚跑两站就误判
        # "没得选了"而收工——即便远处仍有成百上千个状态没排除。
        usable = [si for si, g in gain.items() if g >= min_gain]
        if not usable:
            break

        def net(si: int) -> float:
            reach_cost = min(math.dist(a, points[si]) for a in live_anchors)
            return gain[si] - reach_cost / scale

        best = _pick_station(usable, gain, points, live_anchors, net,
                             gain_tolerance=gain_tolerance)
        chosen.append(points[best])
        live_anchors = [points[best], *anchors]
        for cell in table[best]:
            count = need.pop(cell, 0)
            if not count:
                continue
            for si in table[cell]:
                remaining = gain.get(si, 0) - count
                if remaining > 0:
                    gain[si] = remaining
                else:
                    gain.pop(si, None)
        if not need:
            break
    return _prune_redundant(chosen, sets, reception_min_m=reception_min_m)


def _cover_stations_directional(sets: Sequence[FeasibleSet], current: Point, *,
                                max_stations: int, cell_step: float,
                                reception_min_m: float, min_gain: int,
                                travel_weight_m: float, gain_tolerance: float,
                                anchors: Sequence[Point]) -> list[Point]:
    """问题 4 的站位规划：按**状态**（位置 × 朝向）而非仅按位置做贪心最大覆盖。

    为什么必须按状态：定向源"站得够近"并不等于"能收到信号"，还得落在它的扇区内。若只按
    位置算增益，贪心会反复选那些"位置早已覆盖、但只能从同一个方向看"的站位，增益虚高而
    真实信息增量为零——实测会退化成上百个站位、几十公里空驶。按状态计增益后，同一位置的
    第二次"同方向"访问增益自然为 0，贪心被迫去找**新的观测方向**，这就自动实现了
    "角度多样性"，不需要另外设计交错环。

    两处让它跑得快的工程细节：

    1. 各待搜索频道的观测点完全相同（每到一站就把所有待搜索频道扫一遍），因此它们的可行域
       恒等；取状态数最多的一个作代表即可，增益只差一个公共倍数，不影响贪心排序。
    2. 候选站位先按 2 倍格距粗筛，再在最优者的 8 邻域细化，把每步代价从 |全部格心|×|模板|
       降到约 1/4，而结果几乎不变。
    """
    points, index = _lattice(cell_step)
    reach = guaranteed_reach(cell_step, reception_min_m)
    offsets, masks = _beam_stencil(cell_step, reach, max(fs.headings for fs in sets))
    representative = max(sets, key=lambda fs: fs.state_count)
    work = dict(representative.alive)
    if not work:
        return []
    ij = {si: (int(round(x / cell_step)), int(round(y / cell_step)))
          for si, (x, y) in enumerate(points)}
    coarse = [si for si, (i, j) in ij.items() if i % 2 == 0 and j % 2 == 0]
    scale = max(travel_weight_m, 1e-6)

    def gain_at(si: int) -> int:
        i, j = ij[si]
        total = 0
        for di, dj, bucket in offsets:
            mask = work.get(index.get((i - di, j - dj), -1), 0)
            if mask:
                total += (mask & masks[bucket]).bit_count()
        return total

    chosen: list[Point] = []
    live_anchors = [current, *anchors]
    for _ in range(max_stations):
        if not work:
            break
        # 同全向情形：min_gain 必须先筛、再比净收益，否则移动折价会把近处的低增益站位顶上来，
        # 触发"没得选了"的误判而提前收工。
        scored = [(si, gain_at(si)) for si in coarse]
        usable = [si for si, g in scored if g >= min_gain]
        if not usable:
            break

        def net(si: int) -> float:
            return gain_at(si) - min(math.dist(a, points[si]) for a in live_anchors) / scale

        gains = dict(scored)
        best = _pick_station(usable, gains, points, live_anchors, net,
                             gain_tolerance=gain_tolerance)
        # 在粗筛最优者的邻域细化（粗筛格距 2·step 可能错过局部最优）
        bi, bj = ij[best]
        neighbourhood = [index[(bi + a, bj + b)]
                         for a in (-1, 0, 1) for b in (-1, 0, 1)
                         if (bi + a, bj + b) in index]
        refined = min(neighbourhood, key=lambda si: (-round(net(si), 6), points[si]))
        if gain_at(refined) >= min_gain:
            best = refined
        station = points[best]
        chosen.append(station)
        live_anchors = [station, *anchors]
        # 模拟"站在该站位沉默"，把被它保证覆盖的状态从工作副本里删掉
        bi, bj = ij[best]
        for di, dj, bucket in offsets:
            cell = index.get((bi - di, bj - dj))
            if cell is None:
                continue
            mask = work.get(cell)
            if not mask:
                continue
            remaining = mask & ~masks[bucket]
            if remaining:
                work[cell] = remaining
            else:
                del work[cell]
    return chosen


def _ring_radii(count: int, reach: float, arena_radius: float = ARENA_RADIUS,
                margins: Sequence[float] = (0.0, 40.0, 120.0)) -> list[float]:
    """`原点 + 半径 ρ 的正 count 边形`能覆盖整个目标区域的**最小** ρ，以及几档余量。

    约束由外边界最坏点给出：该点在相邻两站位的角平分线上、半径 `arena_radius` 处，θ = π/count，

        arena² + ρ² − 2·arena·ρ·cosθ ≤ reach²
        ⟹ ρ ≥ arena·cosθ − √(reach² − arena²·sin²θ)      （需 reach ≥ arena·sinθ）

    ρ 越小巡回越短（环长 ≈ 2πρ），所以取下界。这里同时说明了一个反直觉的结论：**站位多、
    环半径小**反而更省——站位数 n 增大使 θ 变小、ρ 下界随之降低，环长 2πρ 的下降快过多一个
    站位带来的检测开销。实测最优在 n = 8-9、ρ ≈ 1000-1050 m 附近，而不是"尽量少站位"的
    n = 6、ρ ≈ 1330 m。

    另外注意 ρ 的下界直接依赖 `reach = 1000 − h`，而 h 是格点半对角。所以**加密可行域网格
    会直接缩短认证巡回**（h 从 85 m 降到 28 m 可让 ρ 下界降约 60 m），这是 `feasible_cell_step_m`
    值得调的原因。
    """
    theta = math.pi / count
    limit = arena_radius * math.sin(theta)
    if reach <= limit:
        return []
    base = arena_radius * math.cos(theta) - math.sqrt(reach * reach - limit * limit)
    return [base + extra for extra in margins if base + extra > 0.0]


@lru_cache(maxsize=4)
def _ring_layouts(step: float, reach: float) -> tuple[tuple[int, ...], ...]:
    """解析式覆盖布局候选：`原点 + 半径 ρ 的正 n 边形`，格心下标形式。

    ## 为什么要有它

    贪心最大覆盖是**逐步**决策的：第一步挑全局增益最大的点（多半在区域中部），剩下的残余
    形状就变得别扭，最终往往比理论最优多用 1-2 个站位（实测平均 7.7 个非原点站位）。而本题的
    覆盖问题有已知的解析解：用半径 r 的圆覆盖半径 R 的圆，R/r = 1.937 时**恰好需要 7 个圆**
    （6 个圆只能覆盖到 1.798r，5 个只能到 1.641r），且最优构型是"1 个在圆心 + 6 个在中间半径
    的正六边形上"。机器狗的原点普扫正好免费提供了圆心那一个，于是只需再走 **6** 个站位。

    半径由 `_ring_radii` 按外边界最坏点解析给出（并附几档余量），布局最终是否合格仍由
    `_layout_plan` 用覆盖表逐格复核，不依赖解析推导的正确性。

    ## 相位是免费的自由度

    正 n 边形整体旋转任意角度都同样覆盖，所以**相位可以拿来对齐清除巡回**：同样是 6 个站位，
    选一个让它们尽量落在"反正要去的目标"附近的相位，能省下可观空驶。这里枚举若干相位，最终由
    `cover_stations` 按巡回长度择优。
    """
    _, index = _lattice(step)
    layouts: list[tuple[int, ...]] = []
    for count in (6, 7, 8, 9, 10, 11, 12):
        for radius in _ring_radii(count, reach):
            for phase_step in range(12):
                phase = 2.0 * math.pi * phase_step / (12 * count)
                stations: list[int] = []
                for k in range(count):
                    theta = 2.0 * math.pi * k / count + phase
                    x, y = radius * math.cos(theta), radius * math.sin(theta)
                    key = (int(round(x / step)), int(round(y / step)))
                    si = index.get(key)
                    if si is not None and si not in stations:
                        stations.append(si)
                if len(stations) == count:
                    layouts.append(tuple(stations))
    return tuple(dict.fromkeys(layouts))


def _layout_plan(layout: Sequence[int], need: Sequence[int], table: Sequence[tuple[int, ...]],
                 points: Sequence[Point], current: Point) -> list[Point] | None:
    """若 layout 能覆盖全部待覆盖格点，返回**去掉无用站位后**的最短巡回；否则返回 None。"""
    pending = set(need)
    if not pending:
        return []
    kept: list[int] = []
    for si in layout:
        covered = pending & set(table[si])
        if covered:
            kept.append(si)
            pending -= covered
    if pending:
        return None                      # 覆盖不全，该布局不合格
    return order_route(current, [points[si] for si in kept])


def _pick_station(usable: Sequence[int], gain: dict[int, int], points: Sequence[Point],
                  anchors: Sequence[Point], net, *, gain_tolerance: float) -> int:
    """在候选里选下一个站位：先按覆盖效率筛出"并列最优"，再在其中选**最省路**的。

    为什么不用"增益 − 距离/权重"这种加性折价（虽然也保留了）：覆盖增益和移动距离量纲差得
    很远，加性折价的权重极难标定——权重偏小会退化成"就近乱走"（实测 T̄ 从 262 s 恶化到
    365 s），偏大则等于完全忽略移动。而本题有一个特殊结构可以利用：**覆盖某块残余可行域的
    站位位置自由度极大**（任何距该块 ≤ 929 m 的点都行），所以"最优覆盖"往往有一大片并列解。
    于是改成字典序两级决策：

        1. 保留增益 ≥ `gain_tolerance` × 最大增益的候选（默认 0.9，即只让出 10% 覆盖效率）；
        2. 其中选到"接下来反正要去的地方"（当前位置 + 待清除目标估计位置）最近的那个。

    这样覆盖效率几乎不损失，却能把补测站位主动挪到清除巡回旁边，顺路吃掉。
    """
    if gain_tolerance <= 0.0:
        return min(usable, key=lambda si: (-round(net(si), 6), points[si]))
    best_gain = max(gain[si] for si in usable)
    threshold = best_gain * gain_tolerance
    tied = [si for si in usable if gain[si] >= threshold]
    return min(tied, key=lambda si: (
        round(min(math.dist(a, points[si]) for a in anchors), 3), -gain[si], points[si]))


def _prune_redundant(chosen: Sequence[Point], sets: Sequence[FeasibleSet], *,
                     reception_min_m: float) -> list[Point]:
    """删掉"去掉它覆盖也不变差"的站位。

    贪心集合覆盖会留下冗余：早期选中的站位，其覆盖可能被后续站位完全吃掉。逐个试删
    （按离原点由远到近，优先删贵的）并复检覆盖，是标准的冗余消除后处理，只会让站位数与
    巡回长度更小，不会破坏覆盖保证。
    """
    kept = list(chosen)
    baseline = sum(len(fs.uncovered_by(kept, reception_min_m)) for fs in sets)
    for station in sorted(chosen, key=lambda p: -math.hypot(*p)):
        trial = [p for p in kept if p != station]
        if not trial:
            continue
        if sum(len(fs.uncovered_by(trial, reception_min_m)) for fs in sets) <= baseline:
            kept = trial
    return kept


def cover_route(sets: Sequence[FeasibleSet], current: Point, *,
                use_ring_layouts: bool = True, cell_step: float = CELL_STEP_M,
                reception_min_m: float = RECEPTION_MIN_M,
                speed_mps: float = SPEED_MPS,
                station_probe_cost_s: float = MEASURE_SECONDS + CHANNEL_SWITCH_SECONDS,
                **kwargs) -> list[Point]:
    """规划补测站位巡回：**贪心解与解析式六边形布局同台竞争，按巡回长度择优**。

    "选哪些站"与"按什么顺序走"是两个正交子问题：前者决定信息覆盖，后者决定移动耗时
    （`route.order_route`：最近邻 + 2-opt + Or-opt）。这里再加一层：候选方案不只有贪心的
    那一个——解析最优的"原点 + 正 n 边形"布局（`_ring_layouts`）常常用更少的站位就覆盖完，
    而且它的**相位可以自由旋转**去贴合当前巡回。两类方案都先检查"是否覆盖全部残余可行域"，
    再按开放路径长度取短者，所以这一步只会更好、不会更差。

    全向情形（问题 3）才用解析布局；定向情形（问题 4）的覆盖要求是"每格从多个方向各看一眼"，
    正多边形布局不成立，直接走贪心。
    """
    stations = cover_stations(sets, current, cell_step=cell_step,
                              reception_min_m=reception_min_m, **kwargs)
    greedy = order_route(current, stations)
    if not use_ring_layouts or not greedy or any(fs.headings for fs in sets):
        return greedy

    # 方案代价必须把**每个站位的检测开销**算进去，不能只比巡回长度：一个站位要把所有待搜索
    # 频道各测一遍（每次 5 s 检测 + 1 s 切频）。只比路程会选出"12 个站位、半径 935 m"这种
    # 环——路是短了，检测次数却多出一半（实测 T̄ 反而从 255 s 涨到 274 s）。
    def total_cost(plan: Sequence[Point]) -> float:
        return (path_length(current, plan) / speed_mps
                + len(plan) * len(sets) * station_probe_cost_s)

    need: set[int] = set()
    for feasible in sets:
        need |= set(feasible.alive)
    table = _coverage_table(cell_step, guaranteed_reach(cell_step, reception_min_m))
    points = _cells(cell_step)
    best = greedy
    best_cost = total_cost(greedy)
    for layout in _ring_layouts(cell_step, guaranteed_reach(cell_step, reception_min_m)):
        plan = _layout_plan(layout, need, table, points, current)
        if plan is None:
            continue
        cost = total_cost(plan)
        if cost < best_cost - 1e-9:
            best, best_cost = plan, cost
    return best


__all__ = ["CELL_STEP_M", "HEADING_COUNT", "MODE_OMNI", "STATION_RADIUS_M", "FeasibleSet", "cover_route",
           "cover_stations", "guaranteed_reach", "half_diagonal"]
