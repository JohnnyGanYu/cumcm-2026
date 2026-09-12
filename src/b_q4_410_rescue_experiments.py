"""Observation-only bounded-region rescue candidates for Q4 development."""
from __future__ import annotations
import math

from .b_q4_410_homing_experiments import RayBoundStrategy
from .b_sim.arena import CLEAR_RADIUS_M
from .b_sim.geometry import polygon_diameter, wedge_half_planes
from .b_sim.q4_fast import Q4FastStrategy, bounded_region
from .b_sim.q3_fast import nearest_cover_point
from .b_sim.route import order_route


def region_cover_points(polygon, radius=19.9):
    """Cover the diameter-aligned enclosing rectangle with <=radius disks."""
    diameter, a, b = polygon_diameter(polygon)
    if diameter <= 1e-9:
        return polygon[:1]
    ux,uy=(b[0]-a[0])/diameter,(b[1]-a[1])/diameter
    uv=[(ux*p[0]+uy*p[1],-uy*p[0]+ux*p[1]) for p in polygon]
    lo0,hi0=min(p[0] for p in uv),max(p[0] for p in uv)
    lo1,hi1=min(p[1] for p in uv),max(p[1] for p in uv)
    n0=max(1,math.ceil((hi0-lo0)/(radius*math.sqrt(2))))
    n1=max(1,math.ceil((hi1-lo1)/(radius*math.sqrt(2))))
    if n0*n1>48:
        return []
    points=[]
    for i in range(n0):
        for j in range(n1):
            u=lo0+(hi0-lo0)*(i+.5)/n0
            v=lo1+(hi1-lo1)*(j+.5)/n1
            points.append((ux*u-uy*v,uy*u+ux*v))
    return points


class RegionRescueStrategy(Q4FastStrategy):
    rescue_after_spiral = False

    def _spiral(self,state):
        if self.rescue_after_spiral and super()._spiral(state):
            return True
        if len(state.bearings)>=2:
            polygon=bounded_region(state.bearings)
            if polygon:
                points=region_cover_points(polygon)
                for point in order_route(self.pos,points):
                    if self._stop():
                        return state.cleared
                    if self._clear_at(point,state.channel):
                        return True
        return super()._spiral(state) if not self.rescue_after_spiral else state.cleared


class RegionAfterStrategy(RegionRescueStrategy):
    rescue_after_spiral=True


class NearestClearStrategy(Q4FastStrategy):
    def _clear_at(self,point,channel):
        state=self.channels[channel]
        if len(state.bearings)>=2:
            candidate=nearest_cover_point(bounded_region(state.bearings),self.pos)
            if candidate is not None:
                point=candidate
        return super()._clear_at(point,channel)


class RegionRayStrategy(RegionRescueStrategy,RayBoundStrategy):
    pass


class FullRescueStrategy(RegionRescueStrategy,RayBoundStrategy,NearestClearStrategy):
    pass
