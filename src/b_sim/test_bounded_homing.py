"""Contracts for the optional bounded homing region, independent of routing."""

import math
from dataclasses import asdict

import pytest

from src.b_sim.geometry import Bearing, estimate_position, localization_polygon, polygon_diameter
from src.b_sim.homing import bounded_estimate, source_region
from src.b_sim.strategy import ChannelState, DogStrategy, StrategyConfig


def bearing_at(observer, source, error=0.0):
    return Bearing(*observer, math.degrees(math.atan2(source[1] - observer[1], source[0] - observer[0])) + error)


def assert_encloses(polygon, point):
    assert polygon
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        assert cross >= -1e-6


@pytest.mark.parametrize("angle", [0.0, 1.0, 22.5, 91.0, 179.0, 250.0, 359.0])
def test_outer_source_region_keeps_boundary_sources_at_extreme_errors(angle):
    source = (1800 * math.cos(math.radians(angle)), 1800 * math.sin(math.radians(angle)))
    observations = [bearing_at((0, 0), source, 1.0), bearing_at((2100, 1900), source, -1.0)]
    polygon = source_region(observations)
    assert_encloses(polygon, source)
    estimate, diameter, cover = bounded_estimate(observations)
    assert math.dist(estimate, source) <= cover + 1e-6
    assert cover == pytest.approx(max(math.dist(estimate, p) for p in polygon))
    assert diameter == pytest.approx(polygon_diameter(polygon)[0])


def test_all_bearings_are_used_even_when_first_is_older_than_recent_four():
    source = (500.0, 0.0)
    observations = [bearing_at((0, 100), source)] + [Bearing(0, 0, 0)] * 4
    estimate, diameter, _ = bounded_estimate(observations)
    assert math.dist(estimate, source) < 30.0
    recent_diameter = bounded_estimate(observations[-4:])[1]
    assert recent_diameter > 1500.0
    assert diameter < recent_diameter / 4


def test_inconsistent_region_resets_estimate_and_both_radii_without_ls_fallback():
    state = ChannelState(1, bounded_homing=True)
    state.add_bearing(0, 0, 0)
    state.add_bearing(0, 100, -math.degrees(math.atan2(100, 500)))
    assert state.estimate is not None
    state.add_bearing(-100, 0, 180)
    assert state.estimate is None
    assert math.isinf(state.diameter) and math.isinf(state.cover_radius)
    assert not state.cleared and not state.abandoned


def test_single_bearing_keeps_existing_second_baseline_acquisition():
    state = ChannelState(1, bounded_homing=True)
    state.add_bearing(0, 0, 0)
    assert state.estimate is None
    assert math.isinf(state.diameter) and math.isinf(state.cover_radius)
    assert bounded_estimate([]) is None


def test_parallel_bearings_are_bounded_by_source_prior():
    observations = [Bearing(0, 0, 0), Bearing(100, 0, 0)]
    estimate, diameter, cover = bounded_estimate(observations)
    assert 100 < estimate[0] < 1801
    assert 1500 < diameter < 1801
    assert 20 < cover < 1801


def test_legacy_channel_state_preserves_existing_estimate_and_cover_radius():
    observations = [bearing_at(p, (600, 400)) for p in [(0, 0), (250, -50), (-100, 600), (700, 100), (400, 800)]]
    state = ChannelState(1)
    for b in observations:
        state.add_bearing(b.x, b.y, b.svd_deg)
    polygon = localization_polygon(observations[-4:])
    estimate = estimate_position(observations[-4:]) or estimate_position(observations)
    assert not state.bounded_homing
    assert state.estimate == estimate
    assert state.diameter == polygon_diameter(polygon)[0]
    assert state.cover_radius == max(math.dist(estimate, p) for p in polygon)


@pytest.mark.parametrize("enabled", [False, True])
def test_strategy_passes_only_opt_in_to_all_channel_states(enabled):
    strategy = DogStrategy(object(), StrategyConfig(bounded_homing=enabled))
    assert len(strategy.channels) == 20
    assert all(state.bounded_homing is enabled for state in strategy.channels.values())
    state = strategy.channels[1]
    assert not state.verdict_empty
    state.abandoned = True
    assert state.verdict_empty


def test_nonfinite_observations_do_not_create_small_cover_radius():
    assert bounded_estimate([Bearing(0, 0, 0), Bearing(float('nan'), 100, 0)]) is None


def assert_effective_policy(config, bounded, certified, monkeypatch):
    """Verify resolved report values and that both production gates use them."""
    serialized = asdict(config)
    assert serialized['bounded_homing'] is bounded
    assert serialized['certified_silence'] is certified
    strategy = DogStrategy(object(), config)
    assert all(state.bounded_homing is bounded for state in strategy.channels.values())

    # A positive certificate stub isolates the policy gate from geometry.
    # Disabled policy must not invoke or accept it, even when it would pass.
    certificate_calls = []

    def positive_certificate(points, directional):
        certificate_calls.append((tuple(points), directional))
        return True

    monkeypatch.setattr('src.b_sim.silence_coverage.silence_certificate', positive_certificate)
    state = strategy.channels[1]
    state.silent_points = [(125.0, -300.0)]
    assert strategy._certify_silence(state) is certified
    assert state.certified_empty is certified
    expected_calls = [(((125.0, -300.0),), config.directional_probe)] if certified else []
    assert certificate_calls == expected_calls


@pytest.mark.parametrize('directional, expected', [(False, True), (True, False)])
def test_default_policy_enables_q3_and_preserves_q4_gates(directional, expected, monkeypatch):
    config = StrategyConfig(directional_probe=directional)
    assert_effective_policy(config, expected, expected, monkeypatch)


@pytest.mark.parametrize('directional', [False, True])
@pytest.mark.parametrize('field', ['bounded_homing', 'certified_silence'])
@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_field_override_remains_independent_of_other_auto_default(
    directional, field, enabled, monkeypatch,
):
    config = StrategyConfig(directional_probe=directional, **{field: enabled})
    expected = {'bounded_homing': not directional, 'certified_silence': not directional}
    expected[field] = enabled
    assert_effective_policy(config, expected['bounded_homing'], expected['certified_silence'], monkeypatch)
