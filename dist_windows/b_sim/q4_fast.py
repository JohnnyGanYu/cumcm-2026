"""Q4 observation-anchored homing and bounded, jointly routed search.

Only RobotClient replies inform the policy. DogStrategy's budgets, retries and
blind-clear fallbacks remain. Bounded search is not a proof that every unobserved
channel is empty; always evaluate clear rate alongside time.
"""
# ┌─────────────────────────────────────────────────────────────────┐
# │ 这是 问题 4 的生产策略 —— 交付入口默认跑的就是本文件。          │
# │ 要优化就改这里。调参前请先读仓库根目录的 OPTIMIZE_HERE.md，      │
# │ 里面有当前成绩、剩余攻击面实测数据、配对验证命令，              │
# │ 以及已经试过没用的 7 个方向（省你两小时）。                      │
# └─────────────────────────────────────────────────────────────────┘

from __future__ import annotations
import math
from .arena import ARENA_RADIUS, CLEAR_RADIUS_M, RECEPTION_MAX_M, SOURCE_COUNT_MAX
from .geometry import Bearing, centroid, clip_polygon, polygon_diameter, wedge_half_planes
from .route import nearest_neighbour, or_opt, order_route, path_length, two_opt
from .strategy import ChannelState, DogStrategy, StrategyConfig

Point = tuple[float, float]
PROFILE = "q4-fast-v3"


def bounded_region(bearings: list[Bearing]) -> list[Point]:
    """Intersect all bearings and circumscribed arena/reception polygons.

    Tangent half-planes preserve legal source positions. The extra 0.005 degrees
    covers rounding to 0.01 degrees in protocol bearings. Weak intersections
    therefore cannot send the robot far beyond the legal source region.
    """
    r = ARENA_RADIUS
    polygon = [(-r, -r), (r, -r), (r, r), (-r, r)]
    for k in range(48):
        angle = 2 * math.pi * k / 48
        polygon = clip_polygon(polygon, (math.cos(angle), math.sin(angle), r))
    for bearing in bearings:
        for plane in wedge_half_planes(bearing, 1.005):
            polygon = clip_polygon(polygon, plane)
        for k in range(24):
            angle = 2 * math.pi * k / 24
            nx, ny = math.cos(angle), math.sin(angle)
            polygon = clip_polygon(polygon, (nx, ny, RECEPTION_MAX_M + nx * bearing.x + ny * bearing.y))
    return polygon


def ring_points(radius: float, count: int, phase_deg: float = 0.0) -> list[Point]:
    points = [(radius * math.cos(2 * math.pi * k / count),
               radius * math.sin(2 * math.pi * k / count)) for k in range(count)]
    if not phase_deg:
        return points
    angle = math.radians(phase_deg)
    return [(x * math.cos(angle) - y * math.sin(angle),
             x * math.sin(angle) + y * math.cos(angle)) for x, y in points]


def region_cover_points(polygon: list[Point], radius: float = 19.9) -> list[Point]:
    """Cover the diameter-aligned bounding rectangle with at most 48 disks.

    Every cell side is <= radius*sqrt(2), so its centre is within radius of
    every cell point. The rectangle contains the entire feasible polygon.
    Large regions return no plan and retain the original recovery path.
    """
    diameter, a, b = polygon_diameter(polygon)
    if diameter <= 1e-9:
        return polygon[:1]
    ux, uy = (b[0] - a[0]) / diameter, (b[1] - a[1]) / diameter
    uv = [(ux * p[0] + uy * p[1], -uy * p[0] + ux * p[1]) for p in polygon]
    lo0, hi0 = min(p[0] for p in uv), max(p[0] for p in uv)
    lo1, hi1 = min(p[1] for p in uv), max(p[1] for p in uv)
    n0 = max(1, math.ceil((hi0 - lo0) / (radius * math.sqrt(2))))
    n1 = max(1, math.ceil((hi1 - lo1) / (radius * math.sqrt(2))))
    if n0 * n1 > 48:
        return []
    points = []
    for i in range(n0):
        for j in range(n1):
            u = lo0 + (hi0 - lo0) * (i + .5) / n0
            v = lo1 + (hi1 - lo1) * (j + .5) / n1
            points.append((ux * u - uy * v, uy * u + ux * v))
    return points


class Q4FastStrategy(DogStrategy):
    """Candidate frozen on development cases before independent validation."""

    inner_radius = 1000.0
    inner_count = 7
    # 1800 = 区域半径。站位越出圆域后覆盖圆被边界截去一块（域外无源），
    # 有效覆盖反而变小，且每站多走 70 m。实测 240 局配对：
    # d5 −2.99% CI95[−19.8,−8.5]、d10 −1.60% CI95[−13.8,−1.4]，清除率零退化。
    # 详见 results/b_outer_radius_1800.json。原 PR #8 取 1870 无论证依据。
    outer_radius = 1800.0
    outer_count = 12
    outer_phase_deg = 15.0
    relocate_inner = False
    relocate_radius = 450.0

    def __init__(self, client, config: StrategyConfig | None = None) -> None:
        super().__init__(client, config or StrategyConfig(directional_probe=True))
        self._visited_fast_stations: set[Point] = set()

    def _can_replace_inner(self, point: Point) -> bool:
        return 650 <= math.hypot(*point) <= 1250

    def _cover_plan(self) -> list[Point]:
        pending = [st for st in self.channels.values() if self._search_target(st)]
        if not pending:
            return []
        remaining: list[Point] = []
        # Both rings enter the same open tour as source estimates. Stations are
        # consumed once, not regenerated after every successful clear.
        # An outer ring beyond the source arena reveals outward facing sources.
        for radius, count in ((self.inner_radius, self.inner_count),
                              (self.outer_radius, self.outer_count)):
            phase = self.outer_phase_deg if radius == self.outer_radius else 0.0
            for point in ring_points(radius, count, phase):
                if point in self._visited_fast_stations:
                    continue
                if not any(not self._skip_probe(st, point) for st in pending):
                    continue
                if self.relocate_inner and radius == self.inner_radius and any(
                    st.active and st.estimate is not None and self._can_replace_inner(st.estimate)
                    and math.dist(point, st.estimate) <= self.relocate_radius
                    for st in self.channels.values()
                ):
                    # Defer only: consume after a successful clear and a real
                    # colocated sweep. Failed homing restores the original stop.
                    continue
                remaining.append(point)
        return remaining

    def _route_order(self, waypoints: list[Point]) -> list[Point]:
        """Compare deterministic first stops before the final open-tour search.

        Q4 combines up to 20 channels with both rings. The old 24-point guard
        skipped all route improvement at precisely these larger early tours.
        """
        best = order_route(self.pos, waypoints, improve=self.config.route_improve,
                           max_points=64)
        if not self.config.route_improve or len(best) < 3:
            return best
        points = list(best)
        length = path_length(self.pos, best)
        for first in points:
            candidate = [first] + nearest_neighbour(first, [p for p in points if p != first])
            candidate = two_opt(self.pos, candidate)
            candidate_length = path_length(self.pos, candidate)
            if candidate_length < length - 1e-9:
                best, length = candidate, candidate_length
        return or_opt(self.pos, best)

    def _next_waypoint(self) -> bool:
        waypoints = []
        kinds = []
        for state in self.channels.values():
            if not state.active:
                continue
            if state.estimate is not None:
                waypoints.append(state.estimate)
                kinds.append(("target", state))
            else:
                point = self._seed_waypoint(state)
                if point is not None:
                    waypoints.append(point)
                    kinds.append(("seed", state))
        if not (self.config.cover_defer and waypoints):
            for station in self._cover_plan():
                waypoints.append(station)
                kinds.append(("station", None))
        if not waypoints:
            return False
        order = self._route_order(waypoints)
        head = order[0] if order else waypoints[0]
        pick = min(range(len(waypoints)), key=lambda i: math.dist(waypoints[i], head))
        kind, state = kinds[pick]
        self._next_hint = order[1] if len(order) > 1 else None
        if kind == "station":
            return self._visit_cover_station(waypoints[pick])
        assert state is not None
        if not self._home_channel(state):
            state.exhausted = True
            self._invalidate_cover_plan()
        elif kind == "target":
            self._piggyback(exclude=state.channel)
            self._invalidate_cover_plan()
        return True

    def _search_target(self, state: ChannelState) -> bool:
        # Every positive channel occupies one of the public maximum 16 slots.
        # Observed but unresolved channels retain the inherited retry logic.
        if not state.bearings and sum(bool(st.bearings) or st.cleared
                                      for st in self.channels.values()) >= SOURCE_COUNT_MAX:
            return False
        return super()._search_target(state)

    def _visit_cover_station(self, station: Point) -> bool:
        self._visited_fast_stations.add(station)
        return super()._visit_cover_station(station)

    def _home_channel(self, state: ChannelState) -> bool:
        result = super()._home_channel(state)
        if result and self.relocate_inner and self._can_replace_inner(self.pos):
            nearby = [p for p in ring_points(self.inner_radius, self.inner_count)
                      if p not in self._visited_fast_stations
                      and math.dist(p, self.pos) <= self.relocate_radius]
            if nearby:
                self._visited_fast_stations.add(min(nearby, key=lambda p: math.dist(p, self.pos)))
                self._visit_cover_station(self.pos)
        return result

    def _probe_channel_at(self, point: Point, state: ChannelState) -> None:
        super()._probe_channel_at(point, state)
        if state.cleared or len(state.bearings) < 2:
            return
        polygon = bounded_region(state.bearings)
        if polygon:
            state.estimate = centroid(polygon)
            state.diameter = polygon_diameter(polygon)[0]
            state.cover_radius = max(math.dist(state.estimate, p) for p in polygon)

    def _dogleg(self, state: ChannelState) -> bool:
        last = state.bearings[-1]
        if math.dist(self.pos, (last.x, last.y)) > 30:
            before = len(state.bearings)
            self._probe_channel_at(self.pos, state)
            if state.cleared or len(state.bearings) > before:
                return True
        last = state.bearings[-1]
        centre = state.feasible.centroid() if state.feasible else None
        expected = math.dist((last.x, last.y), centre) if centre else 600
        length = max(100, min(700, expected * 0.65))
        # A bearing belongs to its observer, not an unrelated current pose.
        for offset in (12.0, -12.0, 0.0):
            angle = math.radians(last.svd_deg + offset)
            candidate = (last.x + length * math.cos(angle), last.y + length * math.sin(angle))
            before = len(state.bearings)
            self._probe_channel_at(candidate, state)
            if state.cleared or len(state.bearings) > before:
                return True
            if self._stop():
                break
        return False

    def _step_closer(self, state: ChannelState, distance: float) -> bool:
        assert state.estimate is not None
        ex, ey = state.estimate
        ux, uy = ((ex - self.pos[0]) / max(distance, 1e-9),
                  (ey - self.pos[1]) / max(distance, 1e-9))
        advance = min(distance * 0.8, max(0, distance - 15))
        lateral = min(80, advance * 0.08)
        candidate = (self.pos[0] + ux * advance - uy * lateral,
                     self.pos[1] + uy * advance + ux * lateral)
        # Robot travel outside the source arena is legal, and can keep a
        # boundary source visible. RobotClient still checks protocol coordinates.
        self._probe_channel_at(candidate, state)
        return True

    def _clear_at(self, point: Point, channel: int) -> bool:
        # Lazy import avoids the module-level Q3 -> bounded_region cycle.
        # This existing helper computes the nearest point whose 19.9m disk
        # contains every feasible vertex; it does not inspect simulator truth.
        from .q3_fast import nearest_cover_point
        state = self.channels[channel]
        if len(state.bearings) >= 2:
            candidate = nearest_cover_point(bounded_region(state.bearings), self.pos)
            if candidate is not None:
                point = candidate
        return super()._clear_at(point, channel)

    def _spiral(self, state: ChannelState) -> bool:
        """Resolve a small silent region locally before the old spiral fallback.

        This has a separate cap of 48 clears, rather than the old nine attempts
        around one estimate. Long thin regions commonly extend well beyond
        that spiral. Every action still obeys the global time/stop budgets.
        """
        if len(state.bearings) >= 2:
            polygon = bounded_region(state.bearings)
            if polygon:
                for point in order_route(self.pos, region_cover_points(polygon)):
                    if self._stop():
                        return state.cleared
                    if self._clear_at(point, state.channel):
                        self.events.append(f"ch{state.channel}: cleared by bounded region sweep")
                        return True
        return super()._spiral(state)

    def _ray_blind_clear(self, state: ChannelState) -> bool:
        """Preserve the original ray, filtering only provably futile clear points."""
        if not state.bearings or state.ray_swept:
            return False
        state.ray_swept = True
        original_attempts = state.clear_attempts
        last = state.bearings[-1]
        angle = math.radians(last.svd_deg)
        ux, uy = math.cos(angle), math.sin(angle)
        planes = [plane for b in state.bearings for plane in wedge_half_planes(b, 1.005)]
        points = []
        r = self.config.ray_clear_step_m
        while r <= self.config.ray_clear_max_range_m:
            offsets = [0.0]
            if r * math.radians(1.0) > self.config.ray_clear_lateral_m:
                offsets = [0.0, self.config.ray_clear_lateral_m, -self.config.ray_clear_lateral_m]
            for off in offsets:
                point = (last.x + ux * r - uy * off, last.y + uy * r + ux * off)
                if math.hypot(*point) > ARENA_RADIUS + CLEAR_RADIUS_M:
                    continue
                # All normals are unit length. If a centre is >20m outside
                # even one half-plane containing the source, it cannot clear.
                if any(nx * point[0] + ny * point[1] - c > CLEAR_RADIUS_M + 1e-6
                       for nx, ny, c in planes):
                    continue
                points.append(point)
            r += self.config.ray_clear_step_m
        for point in points:
            if self._stop() or state.cleared:
                return state.cleared
            if state.clear_attempts - original_attempts >= self.config.ray_clear_max_attempts:
                self.events.append(f"ch{state.channel}: ray blind clear budget exhausted")
                return state.cleared
            if self._clear_at(point, state.channel):
                self.events.append(f"ch{state.channel}: cleared by bounded ray sweep")
                return True
        return state.cleared

    def summary(self) -> dict:
        result = super().summary()
        result["strategy_profile"] = PROFILE
        result["q4_parameters"] = {
            "route_multistart": True, "route_max_points": 64,
            "region_rescue_max_clears": 48, "region_cover_radius_m": 19.9,
            "ray_filter_bearing_error_deg": 1.005,
            "identified_source_count_max": SOURCE_COUNT_MAX,
        }
        result["search_layout"] = {
            "inner_radius_m": self.inner_radius, "inner_stations": self.inner_count,
            "outer_radius_m": self.outer_radius, "outer_stations": self.outer_count,
            "outer_phase_deg": self.outer_phase_deg,
            "inner_relocation_limit_m": self.relocate_radius,
            "inner_relocation_enabled": self.relocate_inner,
        }
        return result
