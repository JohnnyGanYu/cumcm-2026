"""Q4 observation-anchored homing and bounded, jointly routed search.

Only RobotClient replies inform the policy. DogStrategy's budgets, retries and
blind-clear fallbacks remain. Bounded search is not a proof that every unobserved
channel is empty; always evaluate clear rate alongside time.
"""
from __future__ import annotations
import math
from .arena import ARENA_RADIUS, RECEPTION_MAX_M
from .geometry import Bearing, centroid, clip_polygon, polygon_diameter, wedge_half_planes
from .strategy import ChannelState, DogStrategy, StrategyConfig

Point = tuple[float, float]
PROFILE = "q4-fast-v2"


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


def ring_points(radius: float, count: int) -> list[Point]:
    return [(radius * math.cos(2 * math.pi * k / count),
             radius * math.sin(2 * math.pi * k / count)) for k in range(count)]


class Q4FastStrategy(DogStrategy):
    """Candidate frozen on development cases before independent validation."""

    inner_radius = 950.0
    inner_count = 6
    outer_radius = 1900.0
    outer_count = 14
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
            for point in ring_points(radius, count):
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

    def summary(self) -> dict:
        result = super().summary()
        result["strategy_profile"] = PROFILE
        result["search_layout"] = {
            "inner_radius_m": self.inner_radius, "inner_stations": self.inner_count,
            "outer_radius_m": self.outer_radius, "outer_stations": self.outer_count,
            "inner_relocation_limit_m": self.relocate_radius,
            "inner_relocation_enabled": self.relocate_inner,
        }
        return result
