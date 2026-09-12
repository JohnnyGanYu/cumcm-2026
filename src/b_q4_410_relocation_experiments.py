"""Last bounded Q4 development experiment; inherit the frozen v3 strategy."""
from .b_sim.q4_fast import Q4FastStrategy


class Relocate200Strategy(Q4FastStrategy):
    relocate_inner=True
    relocate_radius=200.0


class Relocate300Strategy(Q4FastStrategy):
    relocate_inner=True
    relocate_radius=300.0


class Relocate450Strategy(Q4FastStrategy):
    relocate_inner=True
    relocate_radius=450.0
