"""Offline-only Q4 coverage/route candidates against the frozen v2 baseline.

These policies receive observations only. Case truth is solely evaluator data.
"""
from __future__ import annotations

import math

from .b_q4_410_eval import baseline_class
from .b_sim.arena import SOURCE_COUNT_MAX
from .b_sim.route import nearest_neighbour, or_opt, order_route, path_length, two_opt


BaselineQ4 = baseline_class()


class Q4Ring1850(BaselineQ4):
    outer_radius = 1850.0


class Q4Ring1875(BaselineQ4):
    outer_radius = 1875.0


class Q4RouteAll(BaselineQ4):
    """Apply existing monotone route improvements to initial >24-point tours."""

    route_max_points = 64

    def _route_order(self, waypoints):
        return order_route(self.pos, waypoints, improve=self.config.route_improve,
                           max_points=self.route_max_points)

    def _next_waypoint(self):
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


class Q4KnownCap(BaselineQ4):
    """Sixteen identified channels exhaust the public source-count upper bound."""

    def _search_target(self, state):
        if not state.bearings and sum(
            bool(st.bearings) or st.cleared for st in self.channels.values()
        ) >= SOURCE_COUNT_MAX:
            return False
        return super()._search_target(state)


class Q4RouteRing1850(Q4RouteAll):
    outer_radius = 1850.0


class Q4Relocate200(BaselineQ4):
    relocate_inner = True
    relocate_radius = 200.0


class Q4Relocate300(BaselineQ4):
    relocate_inner = True
    relocate_radius = 300.0


class Q4RouteMulti(Q4RouteAll):
    """Choose the shortest of multiple deterministic open-tour initializations."""

    def _route_order(self, waypoints):
        best = super()._route_order(waypoints)
        if not self.config.route_improve or len(best) < 3:
            return best
        points = list(best)
        best_length = path_length(self.pos, best)
        for first in points:
            candidate = [first] + nearest_neighbour(first, [p for p in points if p != first])
            candidate = two_opt(self.pos, candidate)
            length = path_length(self.pos, candidate)
            if length < best_length - 1e-9:
                best, best_length = candidate, length
        return or_opt(self.pos, best)
