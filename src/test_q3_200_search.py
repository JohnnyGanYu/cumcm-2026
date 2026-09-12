"""Q3 search safety: physical coverage, observed silence, and unresolved retries."""
import math

import pytest

from src.b_q3_220_client import InProcessClient
from src.b_sim.arena import Arena, Case, Jammer, RECEPTION_MIN_M
from src.b_sim.feasible import FeasibleSet
from src.b_sim.geometry import Bearing
from src.b_sim.q3_fast import Q3FastStrategy
from src.b_sim.strategy import DogStrategy, StrategyConfig


def ring_for(dog, phase=0.0):
    return [(dog.ring_radius*math.cos(phase+k*math.pi/3),
             dog.ring_radius*math.sin(phase+k*math.pi/3)) for k in range(6)]


def unresolved_dog():
    # Disable only the old cell elimination so these tests isolate the new
    # continuous certificate; all silence records still come from real actions.
    arena = Arena(Case([], error_salt='q3-search-certificate', seed=0))
    client = InProcessClient(arena)
    dog = Q3FastStrategy(client, StrategyConfig(use_silence_constraint=False))
    for state in dog.channels.values():
        state.feasible = FeasibleSet(step=75, headings=0)
    dog._hybrid_mode = 'ring'
    dog._ring_plan = ring_for(dog)
    client.enter()
    return dog


@pytest.mark.parametrize('phase', [0.0, math.pi/19, math.pi/6])
def test_minimum_reception_radius_covers_boundary_and_interior(phase):
    dog = unresolved_dog()
    ring = ring_for(dog, phase)
    stations = [(0.0,0.0),*ring]
    # Include outer-boundary bisectors, origin/ring Voronoi vertices, and
    # interior annulus samples. Detection is checked by the unchanged arena.
    critical_radii = [0.0, dog.ring_radius/math.sqrt(3), 999.999, 1000.001,
                      dog.ring_radius, 1500.0, 1800.0]
    for radius in critical_radii:
        for i in range(72):
            theta = phase+i*math.pi/36
            source = (radius*math.cos(theta), radius*math.sin(theta))
            nearest = min(stations,key=lambda p:math.dist(p,source))
            arena = Arena(Case([Jammer(1,*source,RECEPTION_MIN_M)],
                               error_salt='q3-search-boundary',seed=0))
            arena.enter()
            result = arena.measure(*nearest,1)
            assert result['measure_result'] in ('direction','near')
    worst_boundary = math.sqrt(1800**2+dog.ring_radius**2
                               -2*1800*dog.ring_radius*math.cos(math.pi/6))
    assert worst_boundary < RECEPTION_MIN_M-11.0


def test_certificate_requires_all_seven_real_observations_of_that_channel():
    dog = unresolved_dog()
    first, second = dog.channels[1], dog.channels[2]
    stations = [(0.0,0.0),*dog._ring_plan]
    for point in stations:
        dog._probe_channel_at(point,first)
    for point in stations[:-1]:
        dog._probe_channel_at(point,second)
    # A global station-visit record is not a per-channel observation record.
    dog._ring_visited.update(dog._ring_plan)
    assert first.feasible.possible and second.feasible.possible
    assert not dog._search_target(first)
    assert dog._search_target(second)
    dog._probe_channel_at(stations[-1],second)
    assert not dog._search_target(second)


def test_repeated_silence_at_one_station_cannot_replace_missing_stations():
    dog = unresolved_dog()
    state = dog.channels[1]
    for _ in range(7):
        dog._probe_channel_at((0.0,0.0),state)
    assert len(state.silent_points) == 7
    assert dog._search_target(state)


def test_ring_silences_without_actual_origin_observation_do_not_certify_empty():
    dog = unresolved_dog()
    state = dog.channels[1]
    for point in dog._ring_plan:
        dog._probe_channel_at(point,state)
    assert dog._search_target(state)


@pytest.mark.parametrize('cleared_count',[0,10,11,15])
def test_source_count_lower_bound_does_not_replace_observation_certificate(cleared_count):
    dog = unresolved_dog()
    for channel in range(1,cleared_count+1):
        dog.channels[channel].cleared = True
    state = dog.channels[20]
    for point in [(0.0,0.0),*dog._ring_plan[:-1]]:
        dog._probe_channel_at(point,state)
    assert dog._search_target(state)
    dog._probe_channel_at(dog._ring_plan[-1],state)
    assert not dog._search_target(state)


def test_observed_exhausted_source_retains_retry_after_ring_silences():
    dog = unresolved_dog()
    state = dog.channels[1]
    # This deliberately inconsistent history checks priority: evidence that a
    # source exists must never be discarded by a later empty-channel shortcut.
    state.bearings = [Bearing(0.0,0.0,45.0)]
    state.exhausted = True
    state.silent_points = [(0.0,0.0),*dog._ring_plan]
    assert dog._search_target(state)
    assert not state.abandoned
    assert state.bearings


def test_skipped_channel_observation_preserves_search_and_adaptive_fallback(monkeypatch):
    dog = unresolved_dog()
    state = dog.channels[1]
    for other in dog.channels.values():
        if other is not state:
            other.abandoned = True
    dog._probe_channel_at((0.0,0.0),state)
    skipped = dog._ring_plan[0]
    monkeypatch.setattr(dog,'_skip_probe',lambda st,p:p == skipped)
    for point in dog._ring_plan:
        dog._visit_cover_station(point)
    assert skipped in dog._ring_visited
    assert skipped not in state.silent_points
    assert dog._search_target(state)
    sentinel = [(1600.0,123.0)]
    monkeypatch.setattr(DogStrategy,'_cover_plan',lambda self:list(sentinel))
    assert dog._cover_plan() == sentinel
