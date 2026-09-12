"""Q3 development-only piggyback baseline thresholds.

Only the existing minimum-baseline parameter changes. Every inherited guard,
measurement budget, and coverage rule stays in force.
"""
from __future__ import annotations

from dataclasses import replace

from src.b_sim.q3_fast import Q3FastStrategy


class Piggyback100(Q3FastStrategy):
    minimum_baseline_m = 100.0

    def __init__(self, client, config=None):
        super().__init__(client, config)
        self.config = replace(
            self.config, piggyback_min_baseline_m=self.minimum_baseline_m)


class Piggyback400(Piggyback100):
    minimum_baseline_m = 400.0


class Piggyback600(Piggyback100):
    minimum_baseline_m = 600.0


class PiggybackZero(Piggyback100):
    minimum_baseline_m = 0.0
