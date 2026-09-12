"""Q3 development: tighter, still conservative square-cell silence proofs."""
from __future__ import annotations

from functools import lru_cache
import math

from src.b_sim.q3_fast import Q3FastStrategy


@lru_cache(maxsize=65536)
def covered_square(cx, cy, half, observations, depth):
    # A disk is convex: containing all four corners certifies the whole square.
    if any((abs(cx-px)+half)**2 + (abs(cy-py)+half)**2 <= 999.999**2
           for px, py in observations):
        return True
    # An uncovered centre is a witness; never infer emptiness from sampling.
    if not depth or all(math.hypot(cx-px, cy-py) > 999.999 for px, py in observations):
        return False
    q = half/2
    return all(covered_square(cx+dx*q, cy+dy*q, q, observations, depth-1)
               for dx in (-1, 1) for dy in (-1, 1))


class ExactSquares(Q3FastStrategy):
    depth = 0

    def _probe_channel_at(self, point, state):
        before = len(state.silent_points)
        super()._probe_channel_at(point, state)
        fs = state.feasible
        if (len(state.silent_points) == before or fs is None or fs.headings
                or not self.config.use_silence_constraint):
            return
        observations = tuple(state.silent_points)
        fs.alive = {i: mask for i, mask in fs.alive.items()
                    if not covered_square(*fs.cells[i], fs.step/2, observations, self.depth)}


class UnionSquares(ExactSquares):
    depth = 3

