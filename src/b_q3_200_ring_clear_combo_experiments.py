"""Final Q3 development combinations: certified six-station ring + safe clear pose."""
from src.b_q3_200_analytic_experiments import Continuous1130, Continuous1150
from src.b_q3_200_clear_experiments import RouteClear


class Continuous1130RouteClear(Continuous1130, RouteClear):
    source_dependencies=('src.b_q3_200_analytic_experiments', 'src.b_q3_200_clear_experiments')


class Continuous1150RouteClear(Continuous1150, RouteClear):
    source_dependencies=('src.b_q3_200_analytic_experiments', 'src.b_q3_200_clear_experiments')
