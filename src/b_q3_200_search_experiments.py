"""Offline-only Q3 search experiments with conservative continuous station moves."""
from __future__ import annotations

import math

from src.b_sim.q3_fast import Q3FastStrategy, nearest_cover_point
from src.b_sim.feasible import guaranteed_reach
from src.b_sim.route import order_route, path_length


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 2:
        return points
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower = []
    for p in points:
        while len(lower) > 1 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) > 1 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


class ContinuousStations(Q3FastStrategy):
    """Allocate every remaining cell, then shorten stations within its disk cover."""
    include_seeds = True
    iterations = 2

    def _cover_plan(self):
        cache = self._cover_cache
        original = super()._cover_plan()
        # Reuse production cache lifecycle; only optimize a newly built plan.
        if not original or self._cover_cache is cache:
            return original
        pending = [s for s in self.channels.values() if self._search_target(s)
                   and s.feasible is not None]
        if not pending:
            return original
        cells = sorted(set(p for s in pending for p in s.feasible.alive_points()))
        radius = guaranteed_reach(self.config.feasible_cell_step_m) - 0.01
        # Keep uncovered residue for the unchanged fallback; never make it larger.
        groups = [[] for _ in original]
        for cell in cells:
            idx = min(range(len(original)), key=lambda k:math.dist(original[k], cell))
            if math.dist(original[idx], cell) <= radius:
                groups[idx].append(cell)
        hulls = [convex_hull(g) for g in groups]
        anchors = [s.estimate or (self._seed_waypoint(s) if self.include_seeds else None)
                   for s in self.channels.values() if s.active]
        anchors = [p for p in anchors if p is not None]
        stations = list(original)
        for _ in range(self.iterations):
            tour = order_route(self.pos, stations + anchors)
            for k, hull in enumerate(hulls):
                if not hull:
                    continue
                pos = tour.index(stations[k]) if stations[k] in tour else -1
                if pos < 0:
                    continue
                prev = tour[pos-1] if pos else self.pos
                nxt = tour[pos+1] if pos+1 < len(tour) else prev
                targets = [prev, nxt, *((prev[0]+f*(nxt[0]-prev[0]),
                                        prev[1]+f*(nxt[1]-prev[1])) for f in (.25,.5,.75))]
                choices = [stations[k]]
                choices.extend(p for t in targets
                               if (p := nearest_cover_point(hull,t,radius)) is not None)
                chosen = min(choices,key=lambda p:math.dist(prev,p)+math.dist(p,nxt))
                stations[k] = chosen
                tour[pos] = chosen
        before = path_length(self.pos, order_route(self.pos, original+anchors))
        after = path_length(self.pos, order_route(self.pos, stations+anchors))
        if after > before + 1e-6:
            return original
        # Directly verify each live conservative feasible set preserves coverage.
        if any(set(s.feasible.uncovered_by(stations)) - set(s.feasible.uncovered_by(original))
               for s in pending):
            return original
        self._cover_cache = (self._cover_cache[0], stations)
        return list(stations)


class ContinuousLocalized(ContinuousStations):
    include_seeds = False
