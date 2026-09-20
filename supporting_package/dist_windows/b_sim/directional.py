"""问题 4：定向干扰源的候选区推理与主动补测点选择。

盲目加密站位有天花板（实测 37 站位仅 98.7% 可见率，成本翻倍）。有效做法是把
"检测无信号"当作约束信息使用：

在点 p 对频道 c 检测得到 no_signal，说明该频道的源 s（若存在）满足
    dist(p, s) > R_s        （超出有效接收半径）
  或 p 落在 s 的辐射扇区之外（|bearing(s->p) - phi_s| > 90 度）。

由于 R_s ∈ [1000, 1500] 未知，取保守下限 1000 m：若 dist(p, s) <= 1000，则必然
是"扇区外"这一支成立。于是每个沉默点在网格上排除的是
    { s : dist(p,s) <= 1000 且 p 在 s 的扇区内 }
——这与"s 的朝向"耦合，所以候选状态是三维的 (x, y, phi)。

本模块用离散网格维护候选集合 (x, y, phi)，据此：
1. 判断某频道是否还可能有源（候选非空）；
2. 选出信息增益最大的下一个检测点（最大化"能被排除的候选比例"，
   即经典的最大期望熵减近似）。

网格分辨率是精度与算力的折中：位置 120 m、朝向 30 度，全域约 700 x 12 个状态，
单次评分在毫秒级，可在 20 分钟现实时间预算内反复调用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .arena import ARENA_RADIUS, DIRECTIONAL_HALF_ANGLE_DEG, RECEPTION_MIN_M

GRID_STEP_M = 120.0
HEADING_STEP_DEG = 30.0


def _grid_positions(step: float = GRID_STEP_M) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    n = int(math.ceil(ARENA_RADIUS / step))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = i * step, j * step
            if math.hypot(x, y) <= ARENA_RADIUS:
                points.append((x, y))
    return points


def _headings(step: float = HEADING_STEP_DEG) -> list[float]:
    count = int(round(360.0 / step))
    return [k * step for k in range(count)]


def in_beam(source_x: float, source_y: float, heading_deg: float,
            px: float, py: float) -> bool:
    """观测点 p 是否落在以 heading 为轴、半角 DIRECTIONAL_HALF_ANGLE_DEG 的辐射扇区内。"""
    bearing = math.degrees(math.atan2(py - source_y, px - source_x))
    diff = abs((bearing - heading_deg + 180.0) % 360.0 - 180.0)
    return diff <= DIRECTIONAL_HALF_ANGLE_DEG + 1e-9


@dataclass
class DirectionalBelief:
    """一个频道上"定向源可能在哪、朝向哪"的候选集合。"""

    candidates: set[tuple[int, int, int]] = field(default_factory=set)
    positions: list[tuple[float, float]] = field(default_factory=list)
    headings: list[float] = field(default_factory=list)
    _pos_index: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def fresh(cls) -> DirectionalBelief:
        positions = _grid_positions()
        headings = _headings()
        belief = cls(positions=positions, headings=headings)
        belief.candidates = {
            (pi, hi, 0) for pi in range(len(positions)) for hi in range(len(headings))
        }
        return belief

    # ---- 约束更新 ----

    def apply_silence(self, px: float, py: float) -> None:
        """在 (px,py) 检测无信号：排除"距离 <= 1000 m 且该点在扇区内"的候选。"""
        survivors = set()
        for pi, hi, tag in self.candidates:
            sx, sy = self.positions[pi]
            if math.hypot(px - sx, py - sy) <= RECEPTION_MIN_M and in_beam(
                sx, sy, self.headings[hi], px, py
            ):
                continue  # 该候选与观测矛盾
            survivors.add((pi, hi, tag))
        self.candidates = survivors

    def apply_bearing(self, px: float, py: float, svd_deg: float,
                      tolerance_deg: float = 1.0) -> None:
        """在 (px,py) 得到示向度：源必在该楔形内，且该点必在扇区内。"""
        survivors = set()
        for pi, hi, tag in self.candidates:
            sx, sy = self.positions[pi]
            bearing = math.degrees(math.atan2(sy - py, sx - px))
            if abs((bearing - svd_deg + 180.0) % 360.0 - 180.0) > tolerance_deg + 6.0:
                continue  # 允许网格离散化误差
            if not in_beam(sx, sy, self.headings[hi], px, py):
                continue
            survivors.add((pi, hi, tag))
        self.candidates = survivors

    # ---- 查询 ----

    @property
    def alive(self) -> bool:
        return bool(self.candidates)

    def candidate_positions(self) -> list[tuple[float, float]]:
        """候选位置（对朝向做边缘化），用于画图和挑补测点。"""
        seen = {pi for pi, _, _ in self.candidates}
        return [self.positions[pi] for pi in seen]

    def _probe_candidates(self, limit: int) -> list[tuple[float, float]]:
        """生成候选观测点。

        关键（一次实测教训）：不能只用"候选源位置"本身当观测点。定向源若朝区域外
        辐射，能看见它的点位于源的**外侧**（沿其朝向前方），而那些点未必是候选源
        位置。因此对每个候选状态 (位置, 朝向)，沿朝向前方按若干距离生成观测点，
        并裁剪到目标区域内。
        """
        seen: set[tuple[int, int]] = set()
        probes: list[tuple[float, float]] = []
        step = GRID_STEP_M
        for pi, hi, _ in self.candidates:
            sx, sy = self.positions[pi]
            heading = math.radians(self.headings[hi])
            # 沿朝向前方取几个距离；近距离点信息量最集中，远点覆盖朝外辐射的情形
            for reach in (step * 2, step * 4, RECEPTION_MIN_M * 0.75):
                px = sx + reach * math.cos(heading)
                py = sy + reach * math.sin(heading)
                if math.hypot(px, py) > ARENA_RADIUS:
                    # 越界则沿区域边界回退到最近的可行点
                    norm = math.hypot(px, py)
                    px, py = px / norm * ARENA_RADIUS, py / norm * ARENA_RADIUS
                key = (int(px // step), int(py // step))
                if key in seen:
                    continue
                seen.add(key)
                probes.append((px, py))
                if len(probes) >= limit:
                    return probes
        return probes

    def best_probe_point(self, current: tuple[float, float],
                         limit: int = 260) -> tuple[float, float] | None:
        """挑一个检测点，使"若无信号能排除的候选比例"最大，并对移动距离折价。

        评分 = 期望排除比例 - 距离惩罚。距离惩罚按"移动 5 m/s、检测 5 s"折算：
        走 500 m 耗 100 s ≈ 20 次检测，因此远点必须有显著信息增益才值得去。
        """
        if not self.candidates:
            return None
        probes = self._probe_candidates(limit)
        if not probes:
            return None
        total = len(self.candidates)
        best_score = -1e18
        best_point = None
        for px, py in probes:
            eliminated = 0
            for pi, hi, _ in self.candidates:
                sx, sy = self.positions[pi]
                if math.hypot(px - sx, py - sy) <= RECEPTION_MIN_M and in_beam(
                    sx, sy, self.headings[hi], px, py
                ):
                    eliminated += 1
            gain = eliminated / total
            distance = math.hypot(px - current[0], py - current[1])
            score = gain - distance / 500.0 * 0.05
            if score > best_score:
                best_score, best_point = score, (px, py)
        return best_point

    def summary(self) -> dict:
        return {
            "candidate_states": len(self.candidates),
            "candidate_positions": len(self.candidate_positions()),
        }


__all__ = ["DirectionalBelief", "in_beam", "GRID_STEP_M", "HEADING_STEP_DEG",
           "DIRECTIONAL_HALF_ANGLE_DEG"]
