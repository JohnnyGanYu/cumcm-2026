"""Analytic full-disk certificate from actual Q3 silence observations."""
from __future__ import annotations

import math

from src.b_sim.q3_fast import Q3FastStrategy
from src.b_sim.arena import ARENA_RADIUS, RECEPTION_MIN_M
from src.b_sim.route import order_route, path_length


class Certified6(Q3FastStrategy):
    ring_count = 6
    ring_radius = 1150.0

    def _make_ring(self):
        # For source radius r in [1000,1800], squared distance to the nearest
        # ring station is bounded by r*r+rho*rho-2*r*rho*cos(pi/n).
        # This convex quadratic attains its maximum at one of the interval
        # endpoints; sources inside 1000 are covered by the origin observation.
        reach = RECEPTION_MIN_M - 0.001
        assert max(math.sqrt(r*r + self.ring_radius**2
                             -2*r*self.ring_radius*math.cos(math.pi/self.ring_count))
                   for r in (reach,ARENA_RADIUS)) < reach
        anchors = [s.estimate or self._seed_waypoint(s)
                   for s in self.channels.values() if s.active]
        anchors = [p for p in anchors if p is not None]
        candidates = [[(self.ring_radius*math.cos(2*math.pi*(k+j/12)/self.ring_count),
                        self.ring_radius*math.sin(2*math.pi*(k+j/12)/self.ring_count))
                       for k in range(self.ring_count)] for j in range(12)]
        self._ring_certificate = min(candidates,key=lambda p:
            path_length(self.pos,order_route(self.pos,anchors+p)))
        self._remaining_ring = list(self._ring_certificate)

    def _search_target(self,state):
        if not state.bearings and not state.cleared and hasattr(self,'_ring_certificate'):
            required = [(0.,0.),*self._ring_certificate]
            if all(any(math.dist(p,observed)<1e-6 for observed in state.silent_points)
                   for p in required):
                if state.feasible is not None:
                    state.feasible.alive.clear()
                state.abandoned = True
                return False
        return super()._search_target(state)

    def _cover_plan(self):
        pending = [s for s in self.channels.values() if self._search_target(s)]
        if not pending:
            return []
        if not hasattr(self,'_remaining_ring'):
            self._make_ring()
        self._remaining_ring = [p for p in self._remaining_ring
            if any(s.feasible is not None and s.feasible.detection_possible(*p) for s in pending)]
        if self._remaining_ring:
            return list(self._remaining_ring)
        return super()._cover_plan()

    def _visit_cover_station(self,station):
        result = super()._visit_cover_station(station)
        if hasattr(self,'_remaining_ring'):
            self._remaining_ring = [p for p in self._remaining_ring if p!=station]
        return result


class Certified7(Certified6):
    ring_count = 7
    ring_radius = 1010.0

