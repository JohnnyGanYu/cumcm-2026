"""Geometry-screened development layouts; preserve all inherited recovery."""
import math
from .b_q4_410_combo_experiments import RouteFull1850
from .b_sim.q4_fast import ring_points


class PhaseLayout(RouteFull1850):
    inner_radius=1000.0
    outer_phase_deg=15.0

    def _cover_plan(self):
        pending=[st for st in self.channels.values() if self._search_target(st)]
        if not pending:
            return []
        a=math.radians(self.outer_phase_deg)
        stations=ring_points(self.inner_radius,self.inner_count)
        stations += [(x*math.cos(a)-y*math.sin(a),x*math.sin(a)+y*math.cos(a))
                     for x,y in ring_points(self.outer_radius,self.outer_count)]
        return [p for p in stations if p not in self._visited_fast_stations
                and any(not self._skip_probe(st,p) for st in pending)]


class Layout19(PhaseLayout):
    outer_radius=1870.0
    outer_count=12


class Layout20(Layout19):
    inner_count=7


class Layout21(PhaseLayout):
    pass
