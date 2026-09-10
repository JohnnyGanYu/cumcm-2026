"""机器狗搜索—定位—清除策略（问题 3 全向 / 问题 4 含定向）。

核心思路：**逐步逼近式归航**，而不是远距离一次性交会。

为什么不能只靠远距离交会：示向度误差 ±1°，在距离 r 处引起的横向定位误差约
r·tan(1°) ≈ 0.0175 r。r = 1500 m 时约 26 m，已超过 20 m 清除半径；而 r = 100 m
时仅约 1.7 m，远小于清除半径。所以策略不追求"站在远处算准位置"，而是
"边靠近边重测"，让几何误差随距离自然收缩。

三个阶段：
1. 原点普扫：在 (0,0) 遍历 20 个频道。题面保证有效接收半径 >= 1000 m，因此凡是
   距原点 1000 m 以内的干扰源必被发现。成本仅 20 次检测 + 19 次切频。
2. 归航清除：对已有示向度的频道做 dog-leg 交会 + 逐步逼近，估计距离降到阈值内
   即发 /clear；失败则小半径螺旋补点。频道间按"就近优先"排序减少空驶。
3. 环形补扫：对仍然沉默的频道，到覆盖环站位补测（半径 1100 m、8 个站位与原点
   共同保证全域任意点到某站位不超过 1000 m），既覆盖远距离源，也为问题 4 的
   定向源提供方位多样性。

时间成本：移动 = 距离/5 s，切频 1 s，检测 5 s，清除成功 5 s / 未发现 3 s。
优化目标是"定位清除总时间 / 被清除个数"，所以要压缩长距离空驶和无效检测，
而不是单纯压缩指令条数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .arena import (
    ARENA_RADIUS,
    CHANNEL_MAX,
    CHANNEL_MIN,
    CLEAR_RADIUS_M,
    MAX_VIRTUAL_SECONDS,
    RECEPTION_MIN_M,
    norm_deg,
)
from .client import ClearResult, MeasureResult, ProtocolError, RobotClient, TestEnded
from .directional import DirectionalBelief
from .geometry import Bearing, estimate_position, localization_polygon, polygon_diameter


@dataclass
class ChannelState:
    """一个频道的认知状态。"""

    channel: int
    bearings: list[Bearing] = field(default_factory=list)
    cleared: bool = False
    exhausted: bool = False          # 本轮放弃（可能在补扫后复活）
    silent_points: list[tuple[float, float]] = field(default_factory=list)
    measure_count: int = 0          # 累计检测次数（含搜索阶段）
    homing_measures: int = 0        # 归航阶段检测次数，拿到首个示向度后才计
    clear_attempts: int = 0
    consecutive_silent: int = 0
    revived: bool = False            # 补扫后已给过第二次机会
    belief: DirectionalBelief | None = None   # 问题 4：定向源候选集合
    estimate: tuple[float, float] | None = None
    diameter: float = float("inf")

    def add_bearing(self, x: float, y: float, svd_deg: float) -> None:
        first = not self.bearings
        self.bearings.append(Bearing(x, y, svd_deg))
        self.consecutive_silent = 0
        if first:
            # 搜索阶段可能已烧掉大量检测（定向源尤甚）。拿到首个示向度说明已锁定
            # 目标，归航预算必须重新起算，否则会在"已能定位"时被判预算耗尽而放弃。
            self.homing_measures = 0
            self.exhausted = False
        self._refresh()

    def _refresh(self) -> None:
        if len(self.bearings) < 2:
            self.estimate = None
            self.diameter = float("inf")
            return
        recent = self.bearings[-4:]          # 只用最近几条，近距离观测精度更高
        polygon = localization_polygon(recent)
        self.diameter = polygon_diameter(polygon)[0] if polygon else float("inf")
        self.estimate = estimate_position(recent) or estimate_position(self.bearings)

    @property
    def active(self) -> bool:
        return bool(self.bearings) and not self.cleared and not self.exhausted


@dataclass
class StrategyConfig:
    # 阶段 1
    sweep_origin: bool = True
    # 阶段 3 覆盖环。设计依据（数值实测，见 results/b_coverage_design.json）：
    # 干扰源随机分布时约 51% 在原点收不到信号，故补扫是主路径而非兜底，
    # 环参数要按"最坏覆盖间隙 < 接收半径下限 1000 m"直接选，而不是靠惰性触发省成本。
    # 半径 1250 m / 12 站位 + 原点：最坏间隙 670 m，13 个站位即可全域覆盖；
    # 双环方案（950x8 + 1560x14）需 23 站位，最坏间隙 539 m，覆盖更好但成本高 77%。
    probe_ring_radius: float = 1250.0
    probe_ring_stations: int = 12
    outer_ring_radius: float = 1560.0
    outer_ring_stations: int = 14
    use_outer_ring: bool = False          # 全向情形单环足够；定向情形自动启用
    # 问题 4：定向源只在 180 度扇区内可见，方位采样必须更密
    directional_ring_stations: int = 18
    # 归航
    dogleg_offset_deg: float = 32.0      # 第二个观测点相对示向度的偏转角
    dogleg_length_m: float = 380.0       # dog-leg 基线长度
    approach_fraction: float = 0.62      # 每次前进占剩余距离的比例
    lateral_fraction: float = 0.16       # 前进时的横向偏移，保持交会角
    clear_distance_m: float = 55.0       # 估计距离降到该值以内即尝试清除
    spiral_radius_m: float = 15.0
    spiral_points: int = 6
    max_measures_per_channel: int = 40    # 搜索阶段总额度（定向源需要大量沉默观测）
    max_homing_measures: int = 14         # 归航阶段额度，拿到首个示向度后起算
    max_clear_attempts: int = 9
    # 预算
    real_time_reserve_s: float = 15.0
    virtual_time_budget_s: float = MAX_VIRTUAL_SECONDS * 0.9
    directional_probe: bool = False       # 问题 4 打开：沉默频道走完整覆盖环


class DogStrategy:
    def __init__(self, client: RobotClient, config: StrategyConfig | None = None) -> None:
        self.client = client
        self.config = config or StrategyConfig()
        self.channels: dict[int, ChannelState] = {
            ch: ChannelState(ch) for ch in range(CHANNEL_MIN, CHANNEL_MAX + 1)
        }
        self.pos = (0.0, 0.0)
        self.radio_channel = 1
        self.events: list[str] = []
        self._stopped = False

    # ---- 入口 ----

    def run(self) -> dict:
        self.client.enter()
        try:
            if self.config.directional_probe:
                for state in self.channels.values():
                    state.belief = DirectionalBelief.fresh()
            if self.config.sweep_origin:
                self._sweep_at((0.0, 0.0), range(CHANNEL_MIN, CHANNEL_MAX + 1))
            self._home_all()
            self._probe_ring()
            self._revive_exhausted()
            self._home_all()
            if self.config.directional_probe:
                self._belief_search()
                self._home_all()
                self._blind_clear_sweep()
        except (TestEnded, ProtocolError) as exc:
            self.events.append(f"aborted: {exc}")
        finally:
            try:
                self.client.exit()
            except (TestEnded, ProtocolError):
                pass
        return self.summary()

    def summary(self) -> dict:
        cleared = sorted(ch for ch, st in self.channels.items() if st.cleared)
        measures = sum(1 for e in self.client.log if e["path"] == "/measure")
        clears = sum(1 for e in self.client.log if e["path"] == "/clear")
        return {
            "cleared_channels": cleared,
            "cleared_count": len(cleared),
            "virtual_time_s": self.client.virtual_time_s,
            "measure_calls": measures,
            "clear_calls": clears,
            "mean_clear_time_s": self.client.virtual_time_s / len(cleared) if cleared else None,
            "events": self.events,
        }

    # ---- 阶段 1 / 3：扫描 ----

    def _sweep_at(self, point: tuple[float, float], channels) -> None:
        """在一个站位上按频道号顺序扫描，减少切频次数（切频与频道距离无关，仍保持有序）。"""
        for channel in channels:
            if self._stop(): return
            state = self.channels[channel]
            if state.cleared or state.bearings:
                continue
            self._probe_channel_at(point, state)

    def _probe_target(self, state: ChannelState) -> bool:
        """补扫对象：未清除，且尚无示向度或归航已失败（换站位可改善几何）。"""
        return not state.cleared and (not state.bearings or state.exhausted)

    def _needs_probe(self) -> bool:
        return any(self._probe_target(st) for st in self.channels.values())

    def _probe_channel_at(self, point: tuple[float, float], state: ChannelState) -> None:
        result = self._measure(point, state.channel)
        if result is None:
            return
        if result.result == "near":
            self._clear_at(point, state.channel)
        elif result.svd_deg is not None:
            state.add_bearing(point[0], point[1], result.svd_deg)
            if state.belief is not None:
                state.belief.apply_bearing(point[0], point[1], result.svd_deg)
        else:
            state.silent_points.append(point)
            state.consecutive_silent += 1
            if state.belief is not None:
                state.belief.apply_silence(point[0], point[1])

    def _probe_stations(self) -> list[list[tuple[float, float]]]:
        """双环覆盖站位。内环先扫（离原点近、代价低），外环兜住边界源。

        问题 4 打开 directional_probe 时外环加密到 directional_ring_stations 个方位，
        因为定向源仅在 180 度扇区内可见，方位采样必须比全向情形更密。
        """
        cfg = self.config
        layers = [(cfg.probe_ring_radius, cfg.probe_ring_stations, 0.0)]
        if cfg.directional_probe or cfg.use_outer_ring:
            outer_n = cfg.directional_ring_stations if cfg.directional_probe else cfg.outer_ring_stations
            layers.append((cfg.outer_ring_radius, outer_n, math.pi / max(1, outer_n)))
        rings: list[list[tuple[float, float]]] = []
        for radius, count, phase in layers:
            rings.append([
                (radius * math.cos(2.0 * math.pi * k / count + phase),
                 radius * math.sin(2.0 * math.pi * k / count + phase))
                for k in range(count)
            ])
        return rings

    def _probe_ring(self) -> None:
        """覆盖环补扫：处理远距离源，并为定向源提供方位多样性。"""
        cfg = self.config
        if not self._needs_probe():
            return
        stations = self._probe_stations()
        # 分层惰性：先走内环，内环解决不了才启用外环。外环代价高（半径 1560 m、
        # 方位密），无条件全扫会把平均清除时间抬高约 50%，实测 419 s -> 638 s。
        for ring in stations:
            if not self._needs_probe() or self._stop():
                return
            self._walk_ring(ring)

    def _walk_ring(self, ring: list[tuple[float, float]]) -> None:
        """就近串行访问一个环上的站位，每站扫完立刻归航清除，避免回头空驶。"""
        remaining = ring[:]
        while remaining and not self._stop():
            todo = [st for st in self.channels.values() if self._probe_target(st)]
            if not todo:
                return
            remaining.sort(key=lambda p: math.hypot(p[0] - self.pos[0], p[1] - self.pos[1]))
            station = remaining.pop(0)
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop(): return
                self._probe_channel_at(station, state)
            self._home_all()

    # ---- 阶段 2：归航 ----

    def _revive_exhausted(self) -> None:
        """补扫后给放弃过的频道第二次机会（新观测可能改善交会几何）。"""
        for state in self.channels.values():
            if state.exhausted and not state.cleared and not state.revived:
                state.exhausted = False
                state.revived = True
                state.consecutive_silent = 0

    def _belief_search(self, max_rounds: int = 60) -> None:
        """问题 4 收尾：对仍沉默的频道，用候选集合推理挑信息增益最大的点补测。

        盲目加密站位有天花板（实测 37 站位仅 98.7% 定向源可见率，成本翻倍），
        而"检测无信号"本身是约束：距离 <= 1000 m 时无信号即意味着观测点落在
        辐射扇区之外。把 (位置, 朝向) 离散成候选集合并逐次排除矛盾状态，
        再挑"若再无信号可排除最多候选"的点，用推理代替蛮力。
        """
        for _ in range(max_rounds):
            if self._stop():
                return
            todo = [
                st for st in self.channels.values()
                if not st.cleared and not st.bearings and st.belief is not None and st.belief.alive
            ]
            if not todo:
                return
            # 候选空间最小的频道最接近定位成功，优先处理
            todo.sort(key=lambda st: len(st.belief.candidates))  # type: ignore[union-attr]
            progressed = False
            for state in todo:
                if self._stop():
                    return
                assert state.belief is not None
                probe = state.belief.best_probe_point(self.pos)
                if probe is None:
                    continue
                before = len(state.bearings)
                self._probe_channel_at(probe, state)
                if state.cleared or len(state.bearings) > before:
                    progressed = True
                    break
                progressed = True
            if not progressed:
                return

    def _blind_clear_sweep(self, max_targets: int = 24) -> None:
        """收尾：对仍未清除的频道，直接到候选位置发 /clear，不要求先看见目标。

        依据（实测）：漏失的定向源全部"物理可解"——其可见扇区一直延伸到源身边
        （最近 4-10 m），但该扇区仅占目标区域 0.1%-19%，靠站位撞上的概率很低。
        而 /clear 的判定只看距离（20 m 内即成功），与辐射方向无关。因此当 belief
        已把候选压缩到少数格点时，逐个盲清除比继续找信号便宜得多。

        每个候选格点按 20 m 清除半径在其 120 m 网格内布点，保证覆盖该格。
        """
        for state in sorted(self.channels.values(), key=lambda s: s.channel):
            if self._stop():
                return
            if state.cleared or state.belief is None or not state.belief.alive:
                continue
            positions = state.belief.candidate_positions()
            if not positions:
                continue
            if len(positions) > max_targets:
                # 候选仍很散：只清除最可能的若干格点（按候选状态数加权），
                # 其余留给剩余时间预算，避免把时间全耗在低概率格点上。
                weight: dict[tuple[float, float], int] = {}
                for pi, _, _ in state.belief.candidates:
                    q = state.belief.positions[pi]
                    weight[q] = weight.get(q, 0) + 1
                positions = [q for q, _ in sorted(weight.items(), key=lambda kv: -kv[1])[:max_targets]]
            positions.sort(key=lambda q: math.hypot(q[0] - self.pos[0], q[1] - self.pos[1]))
            for cx, cy in positions:
                if self._stop() or state.cleared:
                    break
                for point in self._grid_cover_points(cx, cy):
                    if self._stop() or state.cleared:
                        break
                    if math.hypot(*point) > ARENA_RADIUS:
                        continue
                    self._clear_at(point, state.channel)

    @staticmethod
    def _grid_cover_points(cx: float, cy: float,
                           cell: float = 120.0, radius: float = CLEAR_RADIUS_M) -> list[tuple[float, float]]:
        """在以 (cx,cy) 为中心、边长 cell 的网格内布点，使格内任意点距某点 <= radius。"""
        half = cell / 2.0
        step = radius * 1.35
        points: list[tuple[float, float]] = []
        k = int(math.ceil(half / step))
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                points.append((cx + i * step, cy + j * step))
        points.sort(key=lambda q: math.hypot(q[0] - cx, q[1] - cy))
        return points

    def _home_all(self) -> None:
        while not self._stop():
            targets = [st for st in self.channels.values() if st.active]
            if not targets:
                return
            targets.sort(key=lambda st: self._range_to(st))
            if not self._home_channel(targets[0]):
                targets[0].exhausted = True

    def _home_channel(self, state: ChannelState) -> bool:
        """把一个频道推进到清除完成。返回 False 表示放弃。"""
        while not self._stop():
            budget = self.config.max_homing_measures + (6 if state.revived else 0)
            if state.homing_measures >= budget:
                self.events.append(f"ch{state.channel}: measure budget exhausted")
                return self._final_attempt(state)
            if state.clear_attempts >= self.config.max_clear_attempts:
                self.events.append(f"ch{state.channel}: clear budget exhausted")
                return False
            if state.consecutive_silent >= 3:
                self.events.append(f"ch{state.channel}: lost signal, defer to ring probe")
                return False

            if state.estimate is None:
                if not self._dogleg(state):
                    return False
                continue

            distance = math.hypot(state.estimate[0] - self.pos[0], state.estimate[1] - self.pos[1])
            if distance <= self.config.clear_distance_m:
                if self._clear_at(state.estimate, state.channel):
                    return True
                if not self._spiral(state):
                    return False
                return state.cleared
            if not self._step_closer(state, distance):
                return False
        return False

    def _dogleg(self, state: ChannelState) -> bool:
        """只有一条示向度时，沿示向度方向前进并带小角度偏移，构造交会基线。

        关键：必须"沿着"示向度前进而不是横向平移。横向平移可能走出有效接收半径
        （实测该错误导致远距离源大量丢失）；沿示向度前进则单调靠近目标，信号只会
        更强，同时用小角度偏移仍能获得足够的交会角。角度从大到小逐级退让，
        最后退化为纯示向度方向直行。
        """
        last = state.bearings[-1]
        cfg = self.config
        for offset in (cfg.dogleg_offset_deg, cfg.dogleg_offset_deg * 0.5, 0.0):
            for sign in (1.0, -1.0) if offset else (1.0,):
                angle = math.radians(last.svd_deg + sign * offset)
                candidate = (self.pos[0] + cfg.dogleg_length_m * math.cos(angle),
                             self.pos[1] + cfg.dogleg_length_m * math.sin(angle))
                if math.hypot(*candidate) > ARENA_RADIUS:
                    continue
                before = len(state.bearings)
                self._probe_channel_at(candidate, state)
                if state.cleared:
                    return True
                if len(state.bearings) > before:
                    return True
                if self._stop() or state.homing_measures >= cfg.max_homing_measures + (
                        6 if state.revived else 0):
                    return False
        return False

    def _step_closer(self, state: ChannelState, distance: float) -> bool:
        """朝估计位置前进一段并带横向偏移，在更近处重测以收缩误差。"""
        assert state.estimate is not None
        cfg = self.config
        ex, ey = state.estimate
        dx, dy = ex - self.pos[0], ey - self.pos[1]
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        advance = max(cfg.clear_distance_m * 0.8, distance * cfg.approach_fraction)
        advance = min(advance, distance - cfg.clear_distance_m * 0.35) if distance > cfg.clear_distance_m else advance
        lateral = min(distance * cfg.lateral_fraction, 220.0)
        for sign in (1.0, -1.0):
            candidate = (self.pos[0] + ux * advance - uy * lateral * sign,
                         self.pos[1] + uy * advance + ux * lateral * sign)
            if math.hypot(*candidate) > ARENA_RADIUS * 1.02:
                continue
            before = len(state.bearings)
            self._probe_channel_at(candidate, state)
            if state.cleared:
                return True
            if len(state.bearings) > before:
                return True
            # no_signal：可能已进入 5 m 盲区外的遮挡或估计偏差，直接试一次清除
            if state.consecutive_silent >= 2 and state.estimate is not None:
                return self._clear_at(state.estimate, state.channel) or self._spiral(state)
            return True
        return False

    def _spiral(self, state: ChannelState) -> bool:
        """清除失败后围绕估计位置补点。"""
        if state.estimate is None:
            return False
        cx, cy = state.estimate
        cfg = self.config
        for k in range(cfg.spiral_points):
            if self._stop() or state.clear_attempts >= cfg.max_clear_attempts:
                return state.cleared
            theta = 2.0 * math.pi * k / cfg.spiral_points
            point = (cx + cfg.spiral_radius_m * math.cos(theta),
                     cy + cfg.spiral_radius_m * math.sin(theta))
            if self._clear_at(point, state.channel):
                return True
        return False

    def _final_attempt(self, state: ChannelState) -> bool:
        if state.estimate is None:
            return False
        if self._clear_at(state.estimate, state.channel):
            return True
        return self._spiral(state)

    # ---- 底层动作 ----

    def _measure(self, point: tuple[float, float], channel: int) -> MeasureResult | None:
        try:
            result = self.client.measure(point[0], point[1], channel)
        except (TestEnded, ProtocolError) as exc:
            self.events.append(f"measure stopped: {exc}")
            self._stopped = True
            return None
        self.pos = point
        self.radio_channel = channel
        state = self.channels[channel]
        state.measure_count += 1
        if state.bearings:
            state.homing_measures += 1
        return result

    def _clear_at(self, point: tuple[float, float], channel: int) -> bool:
        state = self.channels[channel]
        state.clear_attempts += 1
        try:
            result: ClearResult = self.client.clear(point[0], point[1], channel)
        except (TestEnded, ProtocolError) as exc:
            self.events.append(f"clear stopped: {exc}")
            self._stopped = True
            return False
        self.pos = point
        if result.ok:
            state.cleared = True
        return result.ok

    def _range_to(self, state: ChannelState) -> float:
        if state.estimate is None:
            if state.bearings:
                last = state.bearings[-1]
                return math.hypot(last.x - self.pos[0], last.y - self.pos[1])
            return 1e9
        return math.hypot(state.estimate[0] - self.pos[0], state.estimate[1] - self.pos[1])

    def _stop(self) -> bool:
        if self._stopped:
            return True
        if self.client.real_budget_left_s() <= self.config.real_time_reserve_s:
            self.events.append("real time budget reached")
            self._stopped = True
            return True
        if self.client.virtual_time_s >= self.config.virtual_time_budget_s:
            self.events.append("virtual time budget reached")
            self._stopped = True
            return True
        return False


def safe_clear_threshold() -> float:
    """定位区域直径小于该值时，走到质心即可保证落在清除半径内。"""
    return CLEAR_RADIUS_M


def lateral_error_at(range_m: float, error_deg: float = 1.0) -> float:
    """示向度误差在给定距离上引起的横向定位误差，用于论证逼近的必要性。"""
    return range_m * math.tan(math.radians(error_deg))


__all__ = [
    "ChannelState",
    "DogStrategy",
    "StrategyConfig",
    "lateral_error_at",
    "safe_clear_threshold",
    "ARENA_RADIUS",
    "RECEPTION_MIN_M",
    "norm_deg",
]
