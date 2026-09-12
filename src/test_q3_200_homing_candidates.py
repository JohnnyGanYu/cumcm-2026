"""Safety checks for development-only Q3 route-aware clearance."""
import math

import pytest

from src.b_q3_200_clear_experiments import RouteClear
from src.b_q3_220_client import InProcessClient
from src.b_sim.arena import Arena, Case, Jammer
from src.b_sim.q3_fast import nearest_cover_point
from src.b_sim.q4_fast import bounded_region


@pytest.mark.parametrize('hint', [(1500, 500), (500, 1500), (1500, 1500), (600, 600), None])
def test_route_aware_clear_remains_physically_successful_and_bounds_two_legs(hint):
    source = (1000, 1000)
    arena = Arena(Case([Jammer(1, *source, 1500)], error_salt='q3-route-clear', seed=0))
    client = InProcessClient(arena)
    dog = RouteClear(client)
    client.enter()
    state = dog.channels[1]
    dog._probe_channel_at((700, 700), state)
    dog._probe_channel_at((700, 1300), state)
    polygon = bounded_region(state.bearings)
    start = dog.pos
    baseline = nearest_cover_point(polygon, start)
    assert baseline is not None
    dog._next_hint = hint
    assert dog._clear_at(state.estimate, 1)
    assert arena.case.cleared_count == 1
    assert all(math.dist(dog.pos, vertex) <= 19.9 + 1e-8 for vertex in polygon)
    if hint is None:
        assert dog.pos == baseline
    else:
        proposed = math.dist(start, dog.pos) + math.dist(dog.pos, hint)
        original = math.dist(start, baseline) + math.dist(baseline, hint)
        # Finite ternary search can be within a few micrometres of an endpoint.
        assert proposed <= original + 1e-4
