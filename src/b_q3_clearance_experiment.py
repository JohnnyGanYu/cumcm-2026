"""Development-only minimum-travel clearance within a guaranteed feasible lens."""
from __future__ import annotations
import math
from src.b_sim.strategy import DogStrategy
from src.b_sim.q4_fast import bounded_region


def nearest_cover_point(vertices, origin, radius=19.9):
    """Closest point in the intersection of equal-radius vertex disks."""
    if not vertices:
        return None
    def feasible(p):
        return all(math.dist(p,v)<=radius+1e-8 for v in vertices)
    if feasible(origin):
        return origin
    candidates=[]
    for v in vertices:
        d=math.dist(v,origin)
        if d>radius:
            candidates.append((v[0]+radius*(origin[0]-v[0])/d,
                               v[1]+radius*(origin[1]-v[1])/d))
    for i,a in enumerate(vertices):
        for b in vertices[i+1:]:
            d=math.dist(a,b)
            if d<1e-8 or d>2*radius:
                continue
            mid=((a[0]+b[0])/2,(a[1]+b[1])/2)
            h=math.sqrt(max(0,radius*radius-d*d/4))
            nx,ny=-(b[1]-a[1])/d,(b[0]-a[0])/d
            candidates.extend([(mid[0]+h*nx,mid[1]+h*ny),(mid[0]-h*nx,mid[1]-h*ny)])
    valid=[p for p in candidates if feasible(p)]
    return min(valid,key=lambda p:math.dist(p,origin)) if valid else None


class ClearanceStrategy(DogStrategy):
    def _clear_at(self,point,channel):
        state=self.channels[channel]
        if len(state.bearings)>=2:
            target=nearest_cover_point(bounded_region(state.bearings),self.pos)
            if target is not None:
                point=target
        return super()._clear_at(point,channel)


class KnownCountStrategy(DogStrategy):
    def _search_target(self,state):
        if not state.bearings and sum(bool(st.bearings) or st.cleared for st in self.channels.values())>=16:
            return False
        return super()._search_target(state)


class ClearanceKnownStrategy(ClearanceStrategy,KnownCountStrategy):
    pass
