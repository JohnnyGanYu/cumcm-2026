"""Safety and degeneracy regressions for bounded bearing homing."""

import math
import random

import pytest

from .geometry import Bearing
from .homing import bounded_estimate, certified_clear_point, source_region


def _contains(polygon, point, tolerance=1e-6):
    return all((b[0] - a[0]) * (point[1] - a[1])
               - (b[1] - a[1]) * (point[0] - a[0]) >= -tolerance
               for a, b in zip(polygon, polygon[1:] + polygon[:1]))


def _bearing(observer, source, error=0):
    return Bearing(*observer, math.degrees(math.atan2(source[1] - observer[1],
                                                     source[0] - observer[0])) + error)


def test_nearly_parallel_bearings_remain_bounded_by_source_prior():
    observations = [Bearing(0, 0, 0), Bearing(100, 0, 0)]
    point, diameter = bounded_estimate(observations)
    assert math.hypot(*point) <= 1801
    assert diameter < 1801
    assert certified_clear_point(observations) is None


@pytest.mark.parametrize('source', [(1800, 0), (0, -1800),
                                   (1800 * math.cos(0.037), 1800 * math.sin(0.037))])
def test_outer_polygon_keeps_boundary_source_at_extreme_bearing_errors(source):
    observers = [(2200, -600), (-300, 1200), (0, 0)]
    bearings = [_bearing(p, source, e) for p, e in zip(observers, [-1, 1, -1])]
    polygon = source_region(bearings)
    assert polygon and _contains(polygon, source)


def test_random_valid_bounded_observations_never_exclude_true_source():
    rng = random.Random(71831)
    for _ in range(300):
        angle = rng.uniform(-math.pi, math.pi)
        radius = rng.choice([1800.0, 1800 * math.sqrt(rng.random())])
        source = (radius * math.cos(angle), radius * math.sin(angle))
        observations = []
        for _ in range(6):
            observer = (rng.uniform(-2400, 2400), rng.uniform(-2400, 2400))
            observations.append(_bearing(observer, source, rng.uniform(-1, 1)))
        polygon = source_region(observations)
        assert polygon and _contains(polygon, source, tolerance=1e-5)
        point = certified_clear_point(observations)
        if point is not None:
            assert math.dist(point, source) <= 18.0 + 1e-6


def test_certificate_covers_all_vertices_and_actual_source():
    source = (1000, 1000)
    bearings = [_bearing((500, 1000), source, -1),
                _bearing((1000, 600), source, 1),
                _bearing((850, 850), source, -0.5)]
    point = certified_clear_point(bearings)
    assert point is not None
    assert math.dist(point, source) <= 18
    assert all(math.dist(point, vertex) <= 18 for vertex in source_region(bearings))


def test_inconsistent_or_missing_observations_do_not_produce_clear_certificate():
    inconsistent = [Bearing(0, 0, 0), Bearing(100, 100, 90)]
    assert source_region(inconsistent) == []
    assert bounded_estimate(inconsistent) is None
    assert certified_clear_point(inconsistent) is None
    assert bounded_estimate([Bearing(0, 0, 0)]) is None
    assert certified_clear_point([]) is None


def test_repeated_same_location_is_not_a_new_geometric_baseline():
    bearings = [Bearing(0, 0, 21)] * 20
    assert certified_clear_point(bearings) is None
    assert bounded_estimate(bearings)[1] > 1700


def test_nonfinite_observation_cannot_certify_or_localize():
    observations = [Bearing(0, 0, 0), Bearing(1, 1, float('nan'))]
    assert bounded_estimate(observations) is None
    assert certified_clear_point(observations) is None
