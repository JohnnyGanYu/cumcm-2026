"""Opportunistic scans only when an existing full search station can be replaced."""
from __future__ import annotations

import math

from src.b_sim.q3_fast import Q3FastStrategy


class ReplacementScan(Q3FastStrategy):
    min_gain = 1

    def _piggyback(self, exclude=None):
        super()._piggyback(exclude)
        self._replace_station()

    def _replace_station(self):
        cache = self._cover_cache
        if not cache or not cache[1] or self._stop():
            return
        todo = [s for s in self.channels.values() if self._search_target(s)
                and s.feasible is not None]
        if not todo:
            return
        if max(s.feasible.coverage_gain(*self.pos) for s in todo) < self.min_gain:
            return
        stations = cache[1]
        baseline = [s.feasible.uncovered_by(stations) for s in todo]
        replaceable = []
        for idx, station in enumerate(stations):
            if math.dist(station, self.pos) < 1:
                continue
            trial = [self.pos] + [p for k,p in enumerate(stations) if k != idx]
            if all(not (s.feasible.uncovered_by(trial)-residue)
                   for s,residue in zip(todo,baseline)):
                replaceable.append(idx)
        if not replaceable:
            return
        idx = max(replaceable,key=lambda k:math.dist(self.pos,stations[k]))
        self.cover_visits += 1
        for state in self._scan_order(todo):
            if self._stop():
                break
            if self._search_target(state) and not self._skip_probe(state,self.pos):
                if state.exhausted:
                    state.probe_rounds += 1
                self._probe_channel_at(self.pos,state)
        self.events.append('search station replaced by actual homing stop')
        self._invalidate_cover_plan()


class Replacement80(ReplacementScan):
    min_gain = 80

