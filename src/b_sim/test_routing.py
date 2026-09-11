"""Routing proxy contracts independent of simulator source truth."""

import math

import pytest

from src.b_sim.geometry import Bearing
from src.b_sim.routing import single_bearing_proxy


@pytest.mark.parametrize("bearing", [
    Bearing(0, 0, 0), Bearing(0, 0, 135), Bearing(1700, 0, 0),
    Bearing(1900, 0, 180), Bearing(-600, 800, 225),
])
def test_proxy_stays_inside_public_source_and_reception_bounds(bearing):
    point = single_bearing_proxy(bearing, [])
    assert point is not None
    assert math.hypot(*point) <= 1800.0 + 1e-9
    assert math.dist(point, (bearing.x, bearing.y)) <= 1500.0 + 1e-9


def test_origin_silence_changes_only_the_ordering_proxy():
    bearing = Bearing(0, 0, 0)
    silent_points = [(0.0, 0.0)]
    ordinary = single_bearing_proxy(bearing, [])
    conditioned = single_bearing_proxy(bearing, silent_points)
    assert ordinary is not None and conditioned is not None
    assert ordinary[0] < 1000.0 < conditioned[0] < 1500.0
    assert conditioned[1] == pytest.approx(0.0, abs=1e-9)
    assert silent_points == [(0.0, 0.0)]
    assert bearing == Bearing(0, 0, 0)


def test_outward_boundary_ray_no_longer_projects_900m_outside_source_domain():
    point = single_bearing_proxy(Bearing(1700, 0, 0), [])
    assert point is not None
    assert 1700.0 < point[0] < 1800.0


def test_sampling_no_candidate_returns_none_for_nominal_fallback():
    assert single_bearing_proxy(Bearing(0, 0, 0), [(750.0, 0.0)]) is None
    assert single_bearing_proxy(Bearing(4000, 0, 0), []) is None


def test_silence_order_and_duplicates_do_not_change_proxy():
    bearing = Bearing(1100, 0, 150)
    points = [(0.0, 0.0), (-1200.0, 300.0)]
    first = single_bearing_proxy(bearing, points)
    assert first == single_bearing_proxy(bearing, list(reversed(points)) + points)


@pytest.mark.parametrize("bearing, points", [
    (Bearing(float('nan'), 0, 0), []),
    (Bearing(0, 0, float('inf')), []),
    (Bearing(0, 0, 0), [(float('nan'), 0)]),
])
def test_nonfinite_observations_return_fallback(bearing, points):
    assert single_bearing_proxy(bearing, points) is None
