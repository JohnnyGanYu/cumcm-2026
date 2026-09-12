"""Final combined development candidate, before independent validation."""
from src.b_q3_200_hybrid_search import Hybrid
from src.b_q3_200_clear_experiments import RouteClear
from src.b_sim.arena import SOURCE_COUNT_MAX


class Selected(Hybrid, RouteClear):
    def _probe_channel_at(self, point, state):
        # Recheck after each observation: a station's scan list can become stale.
        if (not state.bearings and not state.cleared
                and sum(bool(s.bearings) or s.cleared for s in self.channels.values())
                >= SOURCE_COUNT_MAX):
            return
        super()._probe_channel_at(point, state)
