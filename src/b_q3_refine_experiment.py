"""Extra high-value bearings from stops already on the Q3 route."""
import math
from src.b_sim.strategy import DogStrategy


class RefineStrategy(DogStrategy):
    refine_diameter=150
    refine_limit=3

    def _piggyback(self,exclude=None):
        super()._piggyback(exclude)
        candidates=[]
        for st in self.channels.values():
            if st.channel==exclude or not st.active or st.estimate is None:
                continue
            if st.diameter<self.refine_diameter:
                continue
            distance=math.dist(self.pos,st.estimate)
            # Q3 is omnidirectional; this bound guarantees the next reply is
            # positive for every location in the current uncertainty polygon.
            if distance+st.cover_radius>990:
                continue
            if min(math.dist(self.pos,(b.x,b.y)) for b in st.bearings)<100:
                continue
            candidates.append(st)
        for st in sorted(candidates,key=lambda st:-st.diameter)[:self.refine_limit]:
            if self._stop():break
            self._probe_channel_at(self.pos,st)


class Refine300Strategy(RefineStrategy):
    refine_diameter=300


class Refine80Strategy(RefineStrategy):
    refine_diameter=80
