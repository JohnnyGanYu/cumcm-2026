"""Observation-anchored Q4 strategy; uses only public protocol observations."""
from __future__ import annotations

import math

from .b_sim.arena import ARENA_RADIUS, CLEAR_RADIUS_M, RECEPTION_MAX_M
from .b_sim.geometry import clip_polygon, wedge_half_planes, polygon_diameter, centroid
from .b_sim.strategy import DogStrategy
from .b_sim.route import order_route


class ExperimentalQ4Strategy(DogStrategy):
    """Keep the legacy search/fallbacks while correcting directional homing."""

    fast_search = True
    inner_radius = 950.0
    inner_count = 8
    outer_radius = 1900.0
    outer_count = 16
    joint_rings = False
    relocate_inner = False
    relocate_radius = 450.0
    beam_homing = False
    prune_gain = -1
    scan_after_clear = False
    survey_first = False
    survey_local = 0.0

    def __init__(self, client, config=None):
        super().__init__(client, config)
        self._visited_fast_stations = set()
        self._clear_scan_points = []

    def _cover_plan(self):
        if not self.fast_search:
            return super()._cover_plan()
        pending = [st for st in self.channels.values() if self._search_target(st)]
        if not pending:
            return []
        all_remaining = []
        for radius, count in ((self.inner_radius, self.inner_count), (self.outer_radius, self.outer_count)):
            points = [(radius * math.cos(2*math.pi*k/count), radius * math.sin(2*math.pi*k/count))
                      for k in range(count)]
            remaining = [p for p in points if p not in self._visited_fast_stations and
                         any(not self._skip_probe(st,p) for st in pending)]
            if self.prune_gain >= 0:
                remaining = [p for p in remaining if any(st.feasible.coverage_gain(*p) > self.prune_gain
                                                        for st in pending)]
            if self.relocate_inner and radius == self.inner_radius:
                remaining = [p for p in remaining if not any(
                    st.active and st.estimate is not None and 650 <= math.hypot(*st.estimate) <= 1250
                    and math.dist(p,st.estimate) <= self.relocate_radius for st in self.channels.values())]
            if remaining:
                if not self.joint_rings:
                    return remaining
                all_remaining.extend(remaining)
        return all_remaining

    def _visit_cover_station(self, station):
        self._visited_fast_stations.add(station)
        return super()._visit_cover_station(station)

    def _next_waypoint(self):
        if self.survey_first:
            stations = self._cover_plan()
            if stations:
                if self.survey_local:
                    nearby = [st for st in self.channels.values() if st.active and st.estimate is not None
                              and math.dist(st.estimate,self.pos)<self.survey_local]
                    if nearby:
                        st = min(nearby,key=lambda st:math.dist(self.pos,st.estimate))
                        if not self._home_channel(st):
                            st.exhausted=True
                        return True
                point = order_route(self.pos,stations)[0]
                self._visit_cover_station(point)
                for st in self.channels.values():
                    if st.active and st.estimate is not None and st.diameter>80 and st.feasible.detection_possible(*self.pos):
                        if min(math.dist(self.pos,(b.x,b.y)) for b in st.bearings)>250:
                            self._probe_channel_at(self.pos,st)
                return True
        return super()._next_waypoint()

    def _home_channel(self, state):
        result = super()._home_channel(state)
        if result and self.scan_after_clear and all(math.dist(self.pos,p)>300 for p in self._clear_scan_points):
            self._clear_scan_points.append(self.pos)
            todo = [st for st in self.channels.values() if self._search_target(st)]
            for other in self._scan_order(todo):
                if not self._stop() and not self._skip_probe(other,self.pos):
                    self._probe_channel_at(self.pos,other)
        if result and self.relocate_inner and 650 <= math.hypot(*self.pos) <= 1250:
            points = [(self.inner_radius * math.cos(2*math.pi*k/self.inner_count),
                       self.inner_radius * math.sin(2*math.pi*k/self.inner_count)) for k in range(self.inner_count)]
            nearby = [p for p in points if p not in self._visited_fast_stations
                      and math.dist(p,self.pos) <= self.relocate_radius]
            if nearby:
                self._visited_fast_stations.add(min(nearby,key=lambda p:math.dist(p,self.pos)))
                self._visit_cover_station(self.pos)
        return result

    def _probe_channel_at(self, point, state):
        super()._probe_channel_at(point, state)
        if not state.bearings or state.cleared:
            return
        # Circumscribed polygons preserve every legal source position, including
        # the +/- 0.005 degree rounding of protocol bearings.
        polygon = [(-ARENA_RADIUS, -ARENA_RADIUS), (ARENA_RADIUS, -ARENA_RADIUS),
                   (ARENA_RADIUS, ARENA_RADIUS), (-ARENA_RADIUS, ARENA_RADIUS)]
        for k in range(48):
            angle = 2 * math.pi * k / 48
            polygon = clip_polygon(polygon, (math.cos(angle), math.sin(angle), ARENA_RADIUS))
        for bearing in state.bearings:
            for plane in wedge_half_planes(bearing, 1.005):
                polygon = clip_polygon(polygon, plane)
            for k in range(24):
                angle = 2 * math.pi * k / 24
                nx, ny = math.cos(angle), math.sin(angle)
                polygon = clip_polygon(polygon, (nx, ny, RECEPTION_MAX_M + nx * bearing.x + ny * bearing.y))
        if not polygon:
            return
        centre = centroid(polygon)
        if len(state.bearings) >= 2:
            state.estimate = centre
            state.diameter = polygon_diameter(polygon)[0]
            state.cover_radius = max(math.dist(centre, p) for p in polygon)

    def _dogleg(self, state):
        last = state.bearings[-1]
        # The bearing belongs to its observer, not to the current robot pose.
        if math.dist(self.pos, (last.x, last.y)) > 30:
            before = len(state.bearings)
            self._probe_channel_at(self.pos, state)
            if state.cleared or len(state.bearings) > before:
                return True
        last = state.bearings[-1]
        centre = state.feasible.centroid() if state.feasible else None
        expected = math.dist((last.x, last.y), centre) if centre else 600
        length = max(100, min(700, expected * 0.65))
        offsets = [12.0, -12.0, 0.0]
        if self.beam_homing and state.feasible:
            def gain(offset):
                angle = math.radians(last.svd_deg+offset)
                p = (last.x+length*math.cos(angle),last.y+length*math.sin(angle))
                return state.feasible.coverage_gain(*p)
            offsets.sort(key=lambda off:(-gain(off),abs(off)==0))
        for offset in offsets:
            angle = math.radians(last.svd_deg + offset)
            candidate = (last.x + length * math.cos(angle), last.y + length * math.sin(angle))
            before = len(state.bearings)
            self._probe_channel_at(candidate, state)
            if state.cleared or len(state.bearings) > before:
                return True
            if self._stop():
                break
        return False

    def _step_closer(self, state, distance):
        ex, ey = state.estimate
        ux, uy = (ex - self.pos[0]) / max(distance, 1e-9), (ey - self.pos[1]) / max(distance, 1e-9)
        advance = min(distance * 0.8, max(0, distance - 15))
        lateral = min(80, advance * 0.08)
        candidates = [(self.pos[0] + ux * advance - uy * lateral*sign,
                       self.pos[1] + uy * advance + ux * lateral*sign) for sign in (1,-1)]
        candidate = candidates[0]
        if self.beam_homing and state.feasible:
            candidate = max(candidates,key=lambda p:state.feasible.coverage_gain(*p))
        self._probe_channel_at(candidate, state)
        return True
