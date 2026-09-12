"""Freeze initial Q3 search mode by joint route and scan cost."""
from __future__ import annotations

from src.b_q3_200_analytic_experiments import Continuous1150
from src.b_sim.q3_fast import Q3FastStrategy
from src.b_sim.route import order_route, path_length


class Hybrid(Continuous1150):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self._hybrid_mode = None

    def _search_target(self,state):
        if self._hybrid_mode == 'adaptive':
            return Q3FastStrategy._search_target(self,state)
        return super()._search_target(state)

    def _cover_plan(self):
        if self._hybrid_mode == 'adaptive':
            return Q3FastStrategy._cover_plan(self)
        if self._hybrid_mode == 'ring':
            return super()._cover_plan()
        pending = [s for s in self.channels.values() if self._search_target(s)]
        if not pending:
            return []
        adaptive = Q3FastStrategy._cover_plan(self)
        ring = super()._cover_plan()
        anchors = [s.estimate or self._seed_waypoint(s)
                   for s in self.channels.values() if s.active]
        anchors = [p for p in anchors if p is not None]
        def cost(plan):
            return (path_length(self.pos,order_route(self.pos,anchors+plan))/5
                    +len(pending)*len(plan)*6)
        adaptive_cost, ring_cost = cost(adaptive), cost(ring)
        self._hybrid_mode = 'ring' if ring and ring_cost < adaptive_cost else 'adaptive'
        self.events.append(f'initial search mode {self._hybrid_mode}: '
                           f'adaptive={adaptive_cost:.3f}s ring={ring_cost:.3f}s')
        if self._hybrid_mode == 'ring':
            return ring
        self._ring_plan = None
        self._ring_visited.clear()
        return adaptive

