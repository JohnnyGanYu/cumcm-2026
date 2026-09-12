"""Q3-only continuous ring certification, from actual per-channel silences."""
from __future__ import annotations

import math

from src.b_sim.q3_fast import Q3FastStrategy
from src.b_sim.route import order_route, path_length


class Continuous1130(Q3FastStrategy):
    ring_radius = 1130.

    def __init__(self, client, config=None):
        super().__init__(client, config)
        self._ring_plan = None
        self._ring_visited = set()

    def _search_target(self, state):
        if not state.bearings and self._ring_plan is not None:
            silent = set(state.silent_points)
            if (0., 0.) in silent and all(p in silent for p in self._ring_plan):
                return False
        return super()._search_target(state)

    def _cover_plan(self):
        pending = [s for s in self.channels.values() if self._search_target(s)]
        if not pending:
            return []
        if self._ring_plan is None:
            anchors = [s.estimate or self._seed_waypoint(s)
                       for s in self.channels.values() if s.active]
            anchors = [p for p in anchors if p is not None]
            choices = [[(self.ring_radius*math.cos(2*math.pi*(k+j/12)/6),
                         self.ring_radius*math.sin(2*math.pi*(k+j/12)/6))
                        for k in range(6)] for j in range(12)]
            self._ring_plan = min(choices, key=lambda p:
                path_length(self.pos, order_route(self.pos, anchors+p)))
        remaining = [p for p in self._ring_plan if p not in self._ring_visited]
        return remaining or super()._cover_plan()

    def _visit_cover_station(self, station):
        result = super()._visit_cover_station(station)
        self._ring_visited.add(station)
        return result


class Continuous1150(Continuous1130):
    ring_radius = 1150.


class Continuous1200(Continuous1130):
    ring_radius = 1200.

