"""Phase-aligned analytic coverage rings on the actual Q3Fast profile."""
from __future__ import annotations

import math

from src.b_sim.q3_fast import Q3FastStrategy
from src.b_sim.feasible import guaranteed_reach
from src.b_sim.route import order_route, path_length


class Fixed6(Q3FastStrategy):
    ring_count = 6
    radius_margin = 1.0

    def _cover_plan(self):
        pending = [s for s in self.channels.values() if self._search_target(s)]
        if not pending:
            return []
        if not hasattr(self, '_analytic_stations'):
            reach = guaranteed_reach(self.config.feasible_cell_step_m)
            angle = math.pi/self.ring_count
            radius = (1800*math.cos(angle)
                      -math.sqrt(reach**2-(1800*math.sin(angle))**2)
                      +self.radius_margin)
            anchors = [s.estimate or self._seed_waypoint(s)
                       for s in self.channels.values() if s.active]
            anchors = [p for p in anchors if p is not None]
            candidates = [[(radius*math.cos(2*math.pi*(k+j/12)/self.ring_count),
                            radius*math.sin(2*math.pi*(k+j/12)/self.ring_count))
                           for k in range(self.ring_count)] for j in range(12)]
            # Retain only candidates covering every currently feasible cell.
            # The origin sweep plus the analytic boundary condition is sufficient
            # for the whole circular arena; verify the actual discrete superset.
            valid = [plan for plan in candidates
                     if all(s.feasible is None or not s.feasible.uncovered_by(plan)
                            for s in pending)]
            if not valid:
                return super()._cover_plan()
            self._analytic_stations = min(valid,key=lambda p:
                path_length(self.pos,order_route(self.pos,anchors+p)))
        self._analytic_stations = [p for p in self._analytic_stations
            if any(s.feasible is not None and s.feasible.coverage_gain(*p)>0 for s in pending)]
        if self._analytic_stations:
            return list(self._analytic_stations)
        return super()._cover_plan()

    def _visit_cover_station(self, station):
        result = super()._visit_cover_station(station)
        if hasattr(self,'_analytic_stations'):
            self._analytic_stations = [p for p in self._analytic_stations if p != station]
        return result


class Fixed7(Fixed6):
    ring_count = 7


class Fixed8(Fixed6):
    ring_count = 8

