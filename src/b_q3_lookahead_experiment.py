"""Development-only coverage credit for upcoming Q3 homing stops."""
from __future__ import annotations
from src.b_sim.strategy import DogStrategy
from src.b_sim.feasible import cover_route


class FutureCoverStrategy(DogStrategy):
    scan_gain = 1
    predict_single = False

    def _cover_plan(self):
        cfg=self.config
        pending=[st for st in self.channels.values() if self._search_target(st)]
        if not pending:
            return []
        anchors=[]
        for st in self.channels.values():
            if not st.active:
                continue
            point=st.estimate
            if point is None and self.predict_single:
                point=self._seed_waypoint(st)
            if point is not None:
                anchors.append(point)
        if not anchors:
            return super()._cover_plan()
        copies=[]
        for st in pending:
            fs=st.feasible.copy()
            for point in anchors:
                fs.apply_silence(*point)
            copies.append(fs)
        # Hypothetical constraints affect only planning, never channel belief.
        # When actual homing/observations differ, remaining coverage is replanned.
        return cover_route(copies,self.pos,cell_step=cfg.feasible_cell_step_m,
                           max_stations=cfg.cover_max_stations,min_gain=1,
                           travel_weight_m=cfg.cover_travel_weight_m,
                           gain_tolerance=cfg.cover_gain_tolerance,
                           use_ring_layouts=cfg.cover_ring_layouts,anchors=anchors)

    def _home_channel(self,state):
        result=super()._home_channel(state)
        if result:
            for target in self._scan_order(list(self.channels.values())):
                if self._stop():break
                if self._search_target(target) and target.feasible.coverage_gain(*self.pos)>=self.scan_gain:
                    self._probe_channel_at(self.pos,target)
            self._invalidate_cover_plan()
        return result


class Future40Strategy(FutureCoverStrategy):
    scan_gain=40


class Future80Strategy(FutureCoverStrategy):
    scan_gain=80


class FutureSingleStrategy(FutureCoverStrategy):
    predict_single=True


class ClearSweepStrategy(FutureCoverStrategy):
    def _cover_plan(self):
        return DogStrategy._cover_plan(self)
