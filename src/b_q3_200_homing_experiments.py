"""Q3-only homing development; all constraints use public observations."""
from __future__ import annotations

from dataclasses import replace
import math

from src.b_sim.geometry import centroid, clip_polygon, polygon_diameter
from src.b_sim.q3_fast import Q3FastStrategy, nearest_cover_point
from src.b_sim.q4_fast import bounded_region
from src.b_sim.strategy import DogStrategy


def hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return points
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    upper=[]
    for p in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def outside_disk_hull(polygon, centre, radius=999.999):
    """Convex outer bound of polygon minus an open reception-minimum disk."""
    points=[p for p in polygon if math.dist(p,centre)>=radius]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        ax,ay=a[0]-centre[0],a[1]-centre[1]
        aa=dx*dx+dy*dy
        bb=2*(ax*dx+ay*dy)
        cc=ax*ax+ay*ay-radius*radius
        disc=bb*bb-4*aa*cc
        if aa<1e-15 or disc<0:
            continue
        for t in ((-bb-math.sqrt(disc))/(2*aa),(-bb+math.sqrt(disc))/(2*aa)):
            if 0<=t<=1:
                points.append((a[0]+t*dx,a[1]+t*dy))
    return hull(points)


class SharedSilence(Q3FastStrategy):
    silence_disk = False
    one_estimate = False

    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.polygons={}

    def _region(self,state):
        polygon=bounded_region(state.bearings)
        for px,py in state.silent_points:
            for bearing in state.bearings:
                qx,qy=bearing.x,bearing.y
                # Silent P and positive Q imply |S-P| >= |S-Q|.
                polygon=clip_polygon(polygon,(px-qx,py-qy,(px*px+py*py-qx*qx-qy*qy)/2+1e-6))
            if self.silence_disk:
                polygon=outside_disk_hull(polygon,(px,py))
        return polygon

    def _probe_channel_at(self,point,state):
        super()._probe_channel_at(point,state)
        if state.cleared or len(state.bearings)<(1 if self.one_estimate else 2):
            return
        polygon=self._region(state)
        if polygon:
            self.polygons[state.channel]=polygon
            state.estimate=centroid(polygon)
            state.diameter=polygon_diameter(polygon)[0]
            state.cover_radius=max(math.dist(state.estimate,p) for p in polygon)

    def _clear_at(self,point,channel):
        state=self.channels[channel]
        if len(state.bearings)>=2:
            target=nearest_cover_point(self._region(state),self.pos)
            if target is not None:
                point=target
        return DogStrategy._clear_at(self,point,channel)


class SilentHull(SharedSilence):
    silence_disk=True


class OneEstimate(SilentHull):
    one_estimate=True


class RapidAdvance(Q3FastStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config=replace(self.config,approach_fraction=.95,lateral_fraction=.015,
                            dogleg_range_fraction=.85,dogleg_max_length_m=1000.)


class LongBaseline(Q3FastStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config=replace(self.config,dogleg_range_fraction=.9,dogleg_max_length_m=1000.)


class SingleStraight(OneEstimate):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config=replace(self.config,approach_fraction=.95,lateral_fraction=.025)
