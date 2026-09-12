"""Offline Q4 homing candidates. Only public observations inform actions."""
from __future__ import annotations

import math
from .b_sim.arena import ARENA_RADIUS, CLEAR_RADIUS_M
from .b_sim.geometry import centroid, wedge_half_planes
from .b_sim.q4_fast import Q4FastStrategy, bounded_region


class RayBoundStrategy(Q4FastStrategy):
    """Retain the old ray coverage, skip only points proven unable to clear."""

    nearest_ray_start = False

    def _ray_blind_clear(self, state):
        if not state.bearings or state.ray_swept:
            return False
        state.ray_swept = True
        original_attempts = state.clear_attempts
        last = state.bearings[-1]
        a = math.radians(last.svd_deg)
        ux, uy = math.cos(a), math.sin(a)
        planes = [p for b in state.bearings for p in wedge_half_planes(b, 1.005)]
        points = []
        r = self.config.ray_clear_step_m
        while r <= self.config.ray_clear_max_range_m:
            offsets = [0.0]
            if r * math.radians(1.0) > self.config.ray_clear_lateral_m:
                offsets = [0.0, self.config.ray_clear_lateral_m, -self.config.ray_clear_lateral_m]
            for off in offsets:
                point = (last.x + ux*r - uy*off, last.y + uy*r + ux*off)
                if math.hypot(*point) > ARENA_RADIUS + CLEAR_RADIUS_M:
                    continue
                # Wedge normals have unit length. A rejected centre is >20m
                # from a half-plane containing every legally possible source.
                if any(nx*point[0]+ny*point[1]-c > CLEAR_RADIUS_M + 1e-6
                       for nx, ny, c in planes):
                    continue
                points.append(point)
            r += self.config.ray_clear_step_m
        if self.nearest_ray_start and points:
            # Reverse the ordered sweep when its far end is closer. Each old
            # candidate remains; only impossible clear centres were removed.
            if math.dist(self.pos, points[-1]) < math.dist(self.pos, points[0]):
                points.reverse()
        for point in points:
            if self._stop() or state.cleared:
                return state.cleared
            if state.clear_attempts-original_attempts >= self.config.ray_clear_max_attempts:
                return state.cleared
            if self._clear_at(point, state.channel):
                return True
        return state.cleared


class RayNearestStrategy(RayBoundStrategy):
    nearest_ray_start = True


class DoglegStrategy(Q4FastStrategy):
    dogleg_angle = 24.0
    length_fraction = .65

    def _dogleg(self, state):
        last = state.bearings[-1]
        if math.dist(self.pos, (last.x,last.y)) > 30:
            before = len(state.bearings)
            self._probe_channel_at(self.pos,state)
            if state.cleared or len(state.bearings)>before:
                return True
        last = state.bearings[-1]
        centre = state.feasible.centroid() if state.feasible else None
        expected = math.dist((last.x,last.y),centre) if centre else 600
        length = max(100,min(700,expected*self.length_fraction))
        for offset in (self.dogleg_angle,-self.dogleg_angle,0.0):
            angle = math.radians(last.svd_deg+offset)
            candidate=(last.x+length*math.cos(angle),last.y+length*math.sin(angle))
            before=len(state.bearings)
            self._probe_channel_at(candidate,state)
            if state.cleared or len(state.bearings)>before:
                return True
            if self._stop():
                break
        return False


class Dogleg6Strategy(DoglegStrategy):
    dogleg_angle=6.0


class Dogleg18Strategy(DoglegStrategy):
    dogleg_angle=18.0


class Dogleg30Strategy(DoglegStrategy):
    dogleg_angle=30.0


class Dogleg45Strategy(DoglegStrategy):
    dogleg_angle=45.0


class DoglegShortStrategy(DoglegStrategy):
    dogleg_angle=12.0
    length_fraction=.35


class DoglegLongStrategy(DoglegStrategy):
    dogleg_angle=12.0
    length_fraction=.85


class StepStraightStrategy(Q4FastStrategy):
    def _step_closer(self,state,distance):
        ex,ey=state.estimate
        ux=(ex-self.pos[0])/max(distance,1e-9)
        uy=(ey-self.pos[1])/max(distance,1e-9)
        advance=min(distance*.8,max(0,distance-15))
        self._probe_channel_at((self.pos[0]+ux*advance,self.pos[1]+uy*advance),state)
        return True
