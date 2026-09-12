"""Combinations of separately measured Q3 improvements; development only."""
import math
from src.b_q3_geometry_experiment import Q3GeometryStrategy
from src.b_q3_clearance_experiment import ClearanceStrategy,KnownCountStrategy
from src.b_q3_refine_experiment import RefineStrategy
from src.b_q3_multistart_experiment import multistart_route


class Fast(Q3GeometryStrategy):
    pass


class FastClear(Q3GeometryStrategy,ClearanceStrategy,KnownCountStrategy):
    pass


class FastPig(FastClear):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config.piggyback_coverage=True
        self.config.piggyback_min_gain_cells=150
        self.config.piggyback_max_per_stop=20


class FastRefine(FastClear,RefineStrategy):
    pass


class MultiMixin:
    source_dependencies=('src.b_q3_multistart_experiment',)
    def _next_waypoint(self):
        waypoints=[]
        kinds=[]
        for state in self.channels.values():
            if not state.active:continue
            if state.estimate is not None:
                waypoints.append(state.estimate);kinds.append(('target',state))
            else:
                point=self._seed_waypoint(state)
                if point is not None:
                    waypoints.append(point);kinds.append(('seed',state))
        if not (self.config.cover_defer and waypoints):
            for station in self._cover_plan():
                waypoints.append(station);kinds.append(('station',None))
        if not waypoints:return False
        order=multistart_route(self.pos,waypoints,improve=self.config.route_improve,starts=4,or_each=True)
        head=order[0] if order else waypoints[0]
        pick=min(range(len(waypoints)),key=lambda i:math.dist(waypoints[i],head))
        kind,state=kinds[pick]
        self._next_hint=order[1] if len(order)>1 else None
        if kind=='station':return self._visit_cover_station(waypoints[pick])
        if not self._home_channel(state):
            state.exhausted=True
            self._invalidate_cover_plan()
        elif kind=='target':
            self._piggyback(exclude=state.channel)
            self._invalidate_cover_plan()
        return True


class FastMulti(MultiMixin,Fast):
    pass


class FastClearMulti(MultiMixin,FastClear):
    pass


class FastPigMulti(MultiMixin,FastPig):
    pass


class FastRefineMulti(MultiMixin,FastRefine):
    pass
