"""Frozen-component combinations for development, not deployment."""
import math
from .b_q4_410_rescue_experiments import FullRescueStrategy, RegionRayStrategy
from .b_q4_410_search_experiments import Q4RouteAll
from .b_sim.arena import SOURCE_COUNT_MAX
from .b_sim.route import order_route, nearest_neighbour, two_opt, or_opt, path_length


class RouteMixin:
    _next_waypoint = Q4RouteAll._next_waypoint

    def _route_order(self, waypoints):
        best = order_route(self.pos,waypoints,improve=self.config.route_improve,max_points=64)
        if not self.config.route_improve or len(best)<3:
            return best
        points=list(best)
        length=path_length(self.pos,best)
        for first in points:
            candidate=[first]+nearest_neighbour(first,[p for p in points if p!=first])
            candidate=two_opt(self.pos,candidate)
            candidate_length=path_length(self.pos,candidate)
            if candidate_length<length-1e-9:
                best,length=candidate,candidate_length
        return or_opt(self.pos,best)

    def _search_target(self,state):
        if not state.bearings and sum(bool(st.bearings) or st.cleared
                                      for st in self.channels.values())>=SOURCE_COUNT_MAX:
            return False
        return super()._search_target(state)


class RouteRegion(RouteMixin,RegionRayStrategy):
    pass


class RouteFull(RouteMixin,FullRescueStrategy):
    pass


class RouteFull1850(RouteFull):
    outer_radius=1850.0
