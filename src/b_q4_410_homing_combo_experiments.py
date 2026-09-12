"""Frozen homing combination candidates, kept separate from pilot sources."""
from .b_q4_410_rescue_experiments import FullRescueStrategy, region_cover_points
from .b_q4_410_homing_experiments import DoglegLongStrategy,Dogleg6Strategy
from .b_sim.q4_fast import bounded_region
from .b_sim.route import order_route


class FullLongStrategy(FullRescueStrategy,DoglegLongStrategy):
    pass


class FullSixStrategy(FullRescueStrategy,Dogleg6Strategy):
    pass


class RegionRayFirstStrategy(FullRescueStrategy):
    def _ray_blind_clear(self,state):
        if len(state.bearings)>=2 and not state.ray_swept:
            polygon=bounded_region(state.bearings)
            if polygon:
                for point in order_route(self.pos,region_cover_points(polygon)):
                    if self._stop():
                        return state.cleared
                    if self._clear_at(point,state.channel):
                        state.ray_swept=True
                        return True
        return super()._ray_blind_clear(state)
