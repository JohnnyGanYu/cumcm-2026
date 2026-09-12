"""Q3-only hybrid coverage and observation-anchored homing.

The policy sees only protocol observations and the public source-count bound.
Retries, failure recovery and all time budgets remain in DogStrategy.
"""
# ┌─────────────────────────────────────────────────────────────────┐
# │ 这是 问题 3 的生产策略 —— 交付入口默认跑的就是本文件。          │
# │ 要优化就改这里。调参前请先读仓库根目录的 OPTIMIZE_HERE.md，      │
# │ 里面有当前成绩、剩余攻击面实测数据、配对验证命令，              │
# │ 以及已经试过没用的 7 个方向（省你两小时）。                      │
# └─────────────────────────────────────────────────────────────────┘

from __future__ import annotations

from dataclasses import replace
import math

from .arena import SOURCE_COUNT_MAX
from .geometry import centroid, polygon_diameter
from .q4_fast import bounded_region
from .route import order_route, path_length
from .strategy import DogStrategy, StrategyConfig

PROFILE = 'q3-fast-v2'


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

    ring_radius = 1150.

    def __init__(self,client,config=None):
        config=config or StrategyConfig()
        if config.directional_probe:
            raise ValueError('Q3FastStrategy requires omnidirectional Q3 configuration')
        config=replace(config,approach_fraction=.85,lateral_fraction=.04,
                       dogleg_offset_deg=12.,dogleg_range_fraction=.65,
                       dogleg_min_length_m=100.,dogleg_max_length_m=700.)
        super().__init__(client,config)
        self._ring_plan = None
        self._ring_visited = set()
        self._hybrid_mode = None

    def _ring_cover_plan(self):
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

    def _cover_plan(self):
        if not self.config.adaptive_cover or self._hybrid_mode == 'adaptive':
            return super()._cover_plan()
        if self._hybrid_mode == 'ring':
            return self._ring_cover_plan()
        pending = [s for s in self.channels.values() if self._search_target(s)]
        if not pending:
            return []
        adaptive = super()._cover_plan()
        ring = self._ring_cover_plan()
        anchors = [s.estimate or self._seed_waypoint(s)
                   for s in self.channels.values() if s.active]
        anchors = [p for p in anchors if p is not None]
        def cost(plan):
            # Public speed and measurement/switch costs, in virtual seconds.
            return (path_length(self.pos, order_route(self.pos, anchors+plan))/5
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

    def _visit_cover_station(self, station):
        result = super()._visit_cover_station(station)
        self._ring_visited.add(station)
        return result

    def _probe_channel_at(self,point,state):
        # The precomputed scan list can become stale after finding source 16.
        if (not state.bearings and not state.cleared
                and sum(bool(s.bearings) or s.cleared for s in self.channels.values())
                >= SOURCE_COUNT_MAX):
            return
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
            polygon=bounded_region(state.bearings)
            target=nearest_cover_point(polygon,self.pos)
            hint=self._next_hint
            if target is not None and hint is not None:
                endpoint=nearest_cover_point(polygon,hint)
                if endpoint is not None:
                    # Feasible clear poses are a convex intersection of disks.
                    # Every interpolation of these two safe endpoints is safe.
                    def at(t):
                        return (target[0]+t*(endpoint[0]-target[0]),
                                target[1]+t*(endpoint[1]-target[1]))
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
        return super()._clear_at(point,channel)

    def _search_target(self,state):
        if (not state.bearings and self._hybrid_mode != 'adaptive'
                and self._ring_plan is not None):
            # Actual per-channel observations certify the continuous arena.
            # The origin covers radius <=1000. For r in [1000,1800], the
            # nearest ring point has angle <=pi/6. Its squared distance is
            # convex in r, and both endpoint distances are <1000 (max 988.52).
            # Missed scans, another channel's silence, or a mere planned visit
            # cannot certify emptiness. Observed sources always retain retries.
            silent=set(state.silent_points)
            if (0.,0.) in silent and all(p in silent for p in self._ring_plan):
                return False
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
        result['search_layout']={
            'mode':self._hybrid_mode,
            'ring_radius_m':self.ring_radius,
            'ring_stations':6,
            'phase_candidates':12,
            'selected_ring':self._ring_plan,
        }
        result['q3_parameters']={
            'approach_fraction':self.config.approach_fraction,
            'lateral_fraction':self.config.lateral_fraction,
            'dogleg_angle_deg':self.config.dogleg_offset_deg,
            'dogleg_range_fraction':self.config.dogleg_range_fraction,
            'clearance_guarantee_radius_m':19.9,
            'public_source_count_max':SOURCE_COUNT_MAX,
            'all_historical_bearings':True,
            'per_channel_continuous_coverage_certificate':True,
            'clear_pose_considers_next_leg':True,
            'in_scan_source_count_guard':True,
        }
        return result
