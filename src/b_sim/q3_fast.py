"""Q3-only observation-anchored homing with conservative short clear moves.

The policy sees only protocol observations and the public source-count bound.
Search, retries, failure recovery and all time budgets remain in DogStrategy.
"""
from __future__ import annotations

from dataclasses import replace
import math

from .arena import SOURCE_COUNT_MAX
from .geometry import centroid, polygon_diameter
from .q4_fast import bounded_region
from .strategy import DogStrategy, StrategyConfig

PROFILE = 'q3-fast-v1'


def nearest_cover_point(vertices, origin, radius=19.9):
    """Closest point whose radius disk contains every feasible polygon vertex.

    A convex source polygon is covered iff all its vertices are covered. The
    admissible clear positions are therefore an intersection of vertex disks.
    Its nearest point is the origin, a radial boundary projection, or a pair
    of circle boundaries' intersection. A 0.1 m margin absorbs round-off.
    """
    if not vertices:
        return None
    def feasible(point):
        return all(math.dist(point,v)<=radius+1e-8 for v in vertices)
    if feasible(origin):
        return origin
    candidates=[]
    for vertex in vertices:
        distance=math.dist(vertex,origin)
        if distance>radius:
            candidates.append((vertex[0]+radius*(origin[0]-vertex[0])/distance,
                               vertex[1]+radius*(origin[1]-vertex[1])/distance))
    for i,a in enumerate(vertices):
        for b in vertices[i+1:]:
            distance=math.dist(a,b)
            if distance<1e-8 or distance>2*radius:
                continue
            midpoint=((a[0]+b[0])/2,(a[1]+b[1])/2)
            height=math.sqrt(max(0,radius*radius-distance*distance/4))
            nx,ny=-(b[1]-a[1])/distance,(b[0]-a[0])/distance
            candidates.extend([(midpoint[0]+height*nx,midpoint[1]+height*ny),
                               (midpoint[0]-height*nx,midpoint[1]-height*ny)])
    valid=[point for point in candidates if feasible(point)]
    return min(valid,key=lambda point:math.dist(point,origin)) if valid else None


class Q3FastStrategy(DogStrategy):
    """Frozen Q3 profile; Q4 continues to use its separate existing strategy."""

    def __init__(self,client,config=None):
        config=config or StrategyConfig()
        if config.directional_probe:
            raise ValueError('Q3FastStrategy requires omnidirectional Q3 configuration')
        config=replace(config,approach_fraction=.85,lateral_fraction=.04,
                       dogleg_offset_deg=12.,dogleg_range_fraction=.65,
                       dogleg_min_length_m=100.,dogleg_max_length_m=700.)
        super().__init__(client,config)

    def _probe_channel_at(self,point,state):
        super()._probe_channel_at(point,state)
        if state.cleared or len(state.bearings)<2:
            return
        polygon=bounded_region(state.bearings)
        if polygon:
            state.estimate=centroid(polygon)
            state.diameter=polygon_diameter(polygon)[0]
            state.cover_radius=max(math.dist(state.estimate,p) for p in polygon)

    def _dogleg(self,state):
        last=state.bearings[-1]
        if math.dist(self.pos,(last.x,last.y))>30:
            before=len(state.bearings)
            self._probe_channel_at(self.pos,state)
            if state.cleared or len(state.bearings)>before:
                return True
        last=state.bearings[-1]
        centre=state.feasible.centroid() if state.feasible else None
        expected=math.dist((last.x,last.y),centre) if centre else 600
        cfg=self.config
        length=max(cfg.dogleg_min_length_m,
                   min(cfg.dogleg_max_length_m,expected*cfg.dogleg_range_fraction))
        for offset in (cfg.dogleg_offset_deg,-cfg.dogleg_offset_deg,0.):
            angle=math.radians(last.svd_deg+offset)
            point=(last.x+length*math.cos(angle),last.y+length*math.sin(angle))
            before=len(state.bearings)
            self._probe_channel_at(point,state)
            if state.cleared or len(state.bearings)>before:
                return True
            if self._stop():
                break
        return False

    def _step_closer(self,state,distance):
        ex,ey=state.estimate
        ux=(ex-self.pos[0])/max(distance,1e-9)
        uy=(ey-self.pos[1])/max(distance,1e-9)
        advance=min(distance*self.config.approach_fraction,max(0,distance-15))
        lateral=min(80,advance*self.config.lateral_fraction)
        self._probe_channel_at((self.pos[0]+ux*advance-uy*lateral,
                                self.pos[1]+uy*advance+ux*lateral),state)
        return True

    def _clear_at(self,point,channel):
        state=self.channels[channel]
        if len(state.bearings)>=2:
            target=nearest_cover_point(bounded_region(state.bearings),self.pos)
            if target is not None:
                point=target
        return super()._clear_at(point,channel)

    def _search_target(self,state):
        # Distinct positive channels count toward the public maximum even
        # before homing finishes. Observed-but-unresolved sources still get
        # the inherited retry/search treatment.
        if not state.bearings and sum(bool(st.bearings) or st.cleared
                                      for st in self.channels.values())>=SOURCE_COUNT_MAX:
            return False
        return super()._search_target(state)

    def summary(self):
        result=super().summary()
        result['strategy_profile']=PROFILE
        result['q3_parameters']={
            'approach_fraction':self.config.approach_fraction,
            'lateral_fraction':self.config.lateral_fraction,
            'dogleg_angle_deg':self.config.dogleg_offset_deg,
            'dogleg_range_fraction':self.config.dogleg_range_fraction,
            'clearance_guarantee_radius_m':19.9,
            'public_source_count_max':SOURCE_COUNT_MAX,
            'all_historical_bearings':True,
        }
        return result
