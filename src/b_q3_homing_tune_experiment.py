"""Frozen Q3 geometry derivatives for local homing parameter development.

Only public observations from Q3GeometryStrategy feed decisions. All budgets,
fallbacks, source-count uncertainty and clear success checks stay inherited.
"""
from src.b_q3_geometry_experiment import Q3GeometryStrategy


class RapidApproach(Q3GeometryStrategy):
    fraction = .9
    lateral = .08
    aggressive_clear = 250.


class GentleApproach(Q3GeometryStrategy):
    fraction = .85
    lateral = .04


class LongerBaseline(Q3GeometryStrategy):
    dogleg_fraction = .8
    dogleg_max = 900.
    aggressive_clear = 250.


class BalancedHoming(Q3GeometryStrategy):
    fraction = .85
    lateral = .04
    dogleg_fraction = .8
    dogleg_max = 900.
    aggressive_clear = 250.
