"""Route completeness, open-tour cost and source-count stopping regressions."""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from src.b_sim.q4_fast import Q4FastStrategy as CountStrategy, Q4FastStrategy as RouteStrategy
from src.b_sim.arena import SOURCE_COUNT_MAX
from src.b_sim.geometry import Bearing
from src.b_sim.route import nearest_neighbour, order_route, path_length
from src.b_sim.strategy import StrategyConfig


def route_strategy(improve=True):
    dog = RouteStrategy(SimpleNamespace(log=[], virtual_time_s=0),
                        StrategyConfig(directional_probe=True, route_improve=improve))
    dog.pos = (-432.25, 187.5)
    return dog


def count_strategy():
    dog = CountStrategy(SimpleNamespace(log=[], virtual_time_s=0))
    for state in dog.channels.values():
        state.feasible = SimpleNamespace(possible=True)
    return dog


@pytest.mark.parametrize("count", [0, 1, 2, 16, 25, 40])
def test_multistart_preserves_all_stops_and_open_tour_cost(count):
    rng = random.Random(8410 + count)
    points = [(rng.uniform(-2200, 2200), rng.uniform(-2200, 2200))
              for _ in range(count)]
    original = list(points)
    dog = route_strategy()
    start = dog.pos
    old_order = order_route(start, points, improve=True)
    order = dog._route_order(points)
    assert len(order) == count
    assert set(order) == set(points)
    assert points == original
    assert dog.pos == start
    assert path_length(start, order) <= path_length(start, old_order) + 1e-6
    # An open tour must neither insert a return to start nor alter its anchor.
    if order:
        manual = math.dist(start, order[0]) + sum(
            math.dist(a, b) for a, b in zip(order, order[1:]))
        assert path_length(start, order) == pytest.approx(manual)


def test_multistart_deduplicates_same_position_without_losing_distinct_stops():
    dog = route_strategy()
    points = [(100.0, 0.0), (100.0, 0.0), (100.0000001, 0.0),
              (100.00001, 0.0), (0.0, 300.0), (0.0, 300.0)]
    order = dog._route_order(points)
    assert len(order) == 3
    assert set(order) == {(100.0, 0.0), (100.00001, 0.0), (0.0, 300.0)}


def test_disabled_improvement_retains_original_nearest_neighbour():
    rng = random.Random(8411)
    points = [(rng.uniform(-2000, 2000), rng.uniform(-2000, 2000))
              for _ in range(40)]
    dog = route_strategy(improve=False)
    assert dog._route_order(points) == nearest_neighbour(dog.pos, points)


def test_known_source_cap_closes_unknown_search_before_all_sources_clear():
    dog = count_strategy()
    for channel in range(1, SOURCE_COUNT_MAX + 1):
        dog.channels[channel].bearings = [Bearing(0.0, 0.0, channel * 10.0)]
    assert dog._cleared_count() == 0
    assert not dog._search_target(dog.channels[SOURCE_COUNT_MAX + 1])
    # Failure to home an observed source still permits changing observation site.
    observed = dog.channels[1]
    observed.exhausted = True
    assert dog._search_target(observed)
    observed.probe_rounds = dog.config.max_probe_rounds
    assert not dog._search_target(observed)


def test_near_clears_without_bearings_count_toward_distinct_source_cap():
    dog = count_strategy()
    for channel in range(1, 9):
        dog.channels[channel].cleared = True
    for channel in range(9, SOURCE_COUNT_MAX + 1):
        dog.channels[channel].bearings = [Bearing(0.0, 0.0, channel * 10.0)]
    assert dog._cleared_count() == 8
    assert not dog._search_target(dog.channels[SOURCE_COUNT_MAX + 1])


def test_fifteen_identified_sources_do_not_close_unknown_channels():
    dog = count_strategy()
    for channel in range(1, SOURCE_COUNT_MAX):
        dog.channels[channel].bearings = [Bearing(0.0, 0.0, channel * 10.0)]
    assert dog._search_target(dog.channels[SOURCE_COUNT_MAX])


def test_repeated_bearings_of_one_channel_do_not_exhaust_source_cap():
    dog = count_strategy()
    dog.channels[1].bearings = [Bearing(float(k), 0.0, 0.0)
                                for k in range(SOURCE_COUNT_MAX + 1)]
    assert dog._search_target(dog.channels[2])
