"""Development-only Q4 visibility gating; public geometry and observations only."""
import math
from src.b_sim.arena import RECEPTION_MAX_M
from src.b_sim.q4_fast import Q4FastStrategy


class DirectionalSkip(Q4FastStrategy):
    def _skip_probe(self, state, station):
        if super()._skip_probe(state, station):
            return True
        feasible = state.feasible
        if feasible is None or not feasible.headings:
            return False
        limit = RECEPTION_MAX_M + feasible._half
        return not any(
            math.dist(feasible.cells[i], station) <= limit
            and mask & feasible._visible_mask(feasible.cells[i], *station,
                                              conservative_drop=False)
            for i, mask in feasible.alive.items())
