"""Frozen development combinations of Q3 homing candidates."""
from src.b_q3_200_clear_experiments import Early250, RouteClear


class RouteEarly250(Early250, RouteClear):
    source_dependencies=('src.b_q3_200_clear_experiments',)
