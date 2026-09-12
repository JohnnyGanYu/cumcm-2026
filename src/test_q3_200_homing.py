"""Production Q3 safety checks for route-aware clearing and the public count cap."""
import math

import pytest

from src.b_q3_220_client import InProcessClient
from src.b_sim.arena import Arena, Case, Jammer
from src.b_sim.feasible import FeasibleSet
from src.b_sim.q3_fast import Q3FastStrategy, nearest_cover_point
from src.b_sim.q4_fast import bounded_region


@pytest.mark.parametrize('hint', [(1500, 500), (500, 1500), (1500, 1500), (600, 600), None])
def test_production_route_clear_is_safe_and_bounds_both_travel_legs(hint):
    source = (1000, 1000)
    arena = Arena(Case([Jammer(1, *source, 1500)], error_salt='q3-route-clear', seed=0))
    client = InProcessClient(arena)
    dog = Q3FastStrategy(client)
    client.enter()
    state = dog.channels[1]
    dog._probe_channel_at((700, 700), state)
    dog._probe_channel_at((700, 1300), state)
    polygon = bounded_region(state.bearings)
    start = dog.pos
    previous_clear_point = nearest_cover_point(polygon, start)
    assert previous_clear_point is not None
    dog._next_hint = hint
    assert dog._clear_at(state.estimate, 1)
    assert arena.case.cleared_count == 1
    assert all(math.dist(dog.pos, vertex) <= 19.9 + 1e-8 for vertex in polygon)
    if hint is None:
        assert dog.pos == previous_clear_point
    else:
        proposed = math.dist(start, dog.pos) + math.dist(dog.pos, hint)
        previous = math.dist(start, previous_clear_point) + math.dist(previous_clear_point, hint)
        # Finite ternary search may finish a few micrometres from an endpoint.
        assert proposed <= previous + 1e-4
        if hint == (600, 600):
            # This geometry has a useful off-axis exit: reverting to the old
            # nearest-only clear point would lose this actual travel saving.
            assert proposed < previous - 0.01


def observed_source_strategy(count=16, last_source_near=False):
    sources = [Jammer(channel,
                      400 * math.cos(2 * math.pi * channel / count),
                      400 * math.sin(2 * math.pi * channel / count), 1500)
               for channel in range(1, count + 1)]
    if last_source_near:
        sources[-1].x = sources[-1].y = 0.0
    arena = Arena(Case(sources, error_salt='q3-observed-count-cap', seed=0))
    client = InProcessClient(arena)
    dog = Q3FastStrategy(client)
    for state in dog.channels.values():
        state.feasible = FeasibleSet(step=dog.config.feasible_cell_step_m, headings=0)
    client.enter()
    dog._sweep_at((0.0, 0.0), range(1, 16))
    assert sum(bool(state.bearings) or state.cleared for state in dog.channels.values()) == 15
    return dog, client


@pytest.mark.parametrize('entrypoint', ['sweep', 'cover_station'])
@pytest.mark.parametrize('last_source_near', [False, True])
def test_station_stops_unknown_probe_calls_immediately_after_sixteenth_observed_source(
        entrypoint, last_source_near):
    dog, client = observed_source_strategy(last_source_near=last_source_near)
    previous_log_size = len(client.log)
    previous_measure_calls = dog.measure_calls
    if entrypoint == 'sweep':
        dog._sweep_at((0.0, 0.0), range(16, 21))
    else:
        # The station computes its pending list before channel 16 is discovered.
        assert dog._visit_cover_station((0.0, 0.0))
    measured_channels = [row['request']['channel'] for row in client.log[previous_log_size:]
                         if row['path'] == '/measure']
    assert measured_channels == [16]
    assert dog.measure_calls == previous_measure_calls + 1
    assert bool(dog.channels[16].bearings) or dog.channels[16].cleared
    if last_source_near:
        assert dog.channels[16].cleared
        assert not dog.channels[16].bearings  # A successful near clear counts too.
    for channel in range(17, 21):
        assert dog.channels[channel].measure_count == 0
        assert dog.channels[channel].silent_points == []


def test_fifteen_distinct_observed_channels_still_probe_an_unknown_channel():
    dog, client = observed_source_strategy(count=15)
    # Repeated bearings on one source do not consume more of the public cap.
    dog.channels[1].bearings.extend([dog.channels[1].bearings[0]] * 20)
    previous_log_size = len(client.log)
    previous_measure_calls = dog.measure_calls
    point = (50.0, 0.0)
    dog._probe_channel_at(point, dog.channels[16])
    assert [row['request']['channel'] for row in client.log[previous_log_size:]
            if row['path'] == '/measure'] == [16]
    assert dog.measure_calls == previous_measure_calls + 1
    assert dog.channels[16].silent_points == [point]


@pytest.mark.parametrize('exhausted', [False, True])
def test_sixteenth_observed_source_does_not_block_remeasurement_of_an_unresolved_source(exhausted):
    dog, client = observed_source_strategy()
    dog._probe_channel_at((0.0, 0.0), dog.channels[16])
    state = dog.channels[1]
    assert not state.cleared
    state.exhausted = exhausted
    previous_bearings = len(state.bearings)
    previous_measure_calls = dog.measure_calls
    previous_log_size = len(client.log)
    dog._probe_channel_at((100.0, 0.0), state)
    assert [row['request']['channel'] for row in client.log[previous_log_size:]
            if row['path'] == '/measure'] == [1]
    assert dog.measure_calls == previous_measure_calls + 1
    assert len(state.bearings) == previous_bearings + 1
    assert not state.cleared
