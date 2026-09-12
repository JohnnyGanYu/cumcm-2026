"""Additional Q3-only clearance candidates using observed feasible polygons."""
from dataclasses import replace
import math
from src.b_sim.geometry import centroid
from src.b_sim.q3_fast import Q3FastStrategy, nearest_cover_point
from src.b_sim.q4_fast import bounded_region
from src.b_sim.strategy import DogStrategy


class Early250(Q3FastStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config=replace(self.config,direct_clear_diameter_m=250.)


class Early400(Q3FastStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config=replace(self.config,direct_clear_diameter_m=400.)


class Early1000(Q3FastStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config=replace(self.config,direct_clear_diameter_m=1000.)


class SoftCover(Q3FastStrategy):
    scale=.6
    def _clear_at(self,point,channel):
        state=self.channels[channel]
        if len(state.bearings)>=2:
            polygon=bounded_region(state.bearings)
            target=nearest_cover_point(polygon,self.pos)
            if target is None and polygon and state.clear_attempts==0:
                centre=centroid(polygon)
                soft=[(centre[0]+self.scale*(x-centre[0]),centre[1]+self.scale*(y-centre[1])) for x,y in polygon]
                target=nearest_cover_point(soft,self.pos)
            if target is not None:
                point=target
        return DogStrategy._clear_at(self,point,channel)


class Soft80(SoftCover):
    scale=.8


class ThinBaseline(Q3FastStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config=replace(self.config,dogleg_offset_deg=6.,dogleg_range_fraction=.9,
                            dogleg_max_length_m=1000.,approach_fraction=.95,lateral_fraction=.015)


class RouteClear(Q3FastStrategy):
    """Choose a guaranteed clear location with the next route leg in view."""
    def _clear_at(self,point,channel):
        state=self.channels[channel]
        if len(state.bearings)>=2:
            polygon=bounded_region(state.bearings)
            target=nearest_cover_point(polygon,self.pos)
            hint=getattr(self,'_next_hint',None)
            if target is not None and hint is not None:
                endpoint=nearest_cover_point(polygon,hint)
                if endpoint is not None:
                    def at(t):
                        return (target[0]+t*(endpoint[0]-target[0]),target[1]+t*(endpoint[1]-target[1]))
                    def cost(t):
                        p=at(t)
                        return math.dist(self.pos,p)+math.dist(p,hint)
                    lo,hi=0.,1.
                    for _ in range(28):
                        l=(2*lo+hi)/3
                        r=(lo+2*hi)/3
                        if cost(l)<cost(r):
                            hi=r
                        else:
                            lo=l
                    target=at((lo+hi)/2)
            if target is not None:
                point=target
        return DogStrategy._clear_at(self,point,channel)
