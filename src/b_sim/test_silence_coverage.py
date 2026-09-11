"""Continuous absence certificates, including geometric failure injection."""

import math
import random

import pytest

from . import silence_coverage as coverage


@pytest.mark.parametrize("kind", ["rings25", "mesh25"])
def test_templates_cover_source_disk_and_every_half_plane(kind):
    stations = coverage.certified_stations(kind)
    assert len(stations) == 25
    assert coverage.silence_certificate(stations, directional=True)
    # Independent boundary stress: an outward beam can only be seen by a
    # receiver outside the source disk, which the former inner-ring layout lacks.
    for k in range(720):
        angle = math.tau * k / 720
        sx, sy = 1800 * math.cos(angle), 1800 * math.sin(angle)
        ux, uy = math.cos(angle), math.sin(angle)
        assert any(math.hypot(px - sx, py - sy) <= 1000
                   and (px - sx) * ux + (py - sy) * uy >= 0 for px, py in stations)


def test_missing_outer_station_keeps_an_outward_source_possible():
    stations = coverage.certified_stations("rings25")
    # The source at (1800, 0), heading east, is invisible after (1870, 0)
    # is omitted. A finite source grid must never certify it absent.
    missing = [p for p in stations if p != (1870.0, 0.0)]
    assert not any(math.dist((1800, 0), p) <= 1000 and p[0] >= 1800 for p in missing)
    assert not coverage.silence_certificate(missing, directional=True)


def test_inside_only_stations_cannot_exclude_outward_boundary_source():
    points = [(r * math.cos(k * math.tau / 72), r * math.sin(k * math.tau / 72))
              for r in (700, 1200, 1750) for k in range(72)]
    assert not coverage.silence_certificate(points, directional=True)


def test_omni_proof_has_a_different_visibility_model():
    stations = [(0.0, 0.0)] + [(1150 * math.cos(k * math.tau / 6), 1150 * math.sin(k * math.tau / 6))
                               for k in range(6)]
    assert coverage.silence_certificate(stations)
    assert not coverage.silence_certificate(stations, directional=True)
    assert not coverage.silence_certificate([(0, 0)])


def test_real_additional_observations_do_not_break_known_layout_example():
    rng = random.Random(31)
    points = coverage.certified_stations() + [(rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)) for _ in range(20)]
    # Extra points create long Delaunay edges; the local continuous certificate
    # must be able to use receivers outside an individual Delaunay triangle.
    assert coverage.silence_certificate(points, directional=True)


def test_flat_outer_triangle_keeps_complete_observed_template_as_witness():
    # Actual clearing stop for the radius-1800, heading-135-degree edge case.
    # Bowyer-Watson omits a tiny outer triangle and the mesh audit fails closed,
    # although the original 25 silent stations still prove absence.
    extra = (-1273.7590070260906, 1279.4052531895484)
    stations = coverage.certified_stations("rings25")
    assert coverage.silence_certificate(stations + [extra], directional=True)


def test_template_fallback_never_supplies_an_unobserved_station():
    extra = (-1273.7590070260906, 1279.4052531895484)
    stations = [point for point in coverage.certified_stations("rings25")
                if point != (1870.0, 0.0)]
    # The same extra stop does not see an eastward source at (1800, 0).
    # Knowing the template must not manufacture its missing east observation.
    assert not coverage.silence_certificate(stations + [extra], directional=True)


@pytest.mark.parametrize("damage", ["missing", "reversed", "duplicate"])
def test_damaged_triangulation_fails_closed(monkeypatch, damage):
    original = coverage._triangulate

    def broken(points):
        triangles = original(points)
        # Select an interior triangle, so removing it creates a real hole.
        i = next(i for i, tri in enumerate(triangles) if all(math.hypot(*p) < 1800 for p in tri))
        if damage == "missing":
            del triangles[i]
        elif damage == "reversed":
            triangles[i] = tuple(reversed(triangles[i]))
        else:
            triangles.append(triangles[i])
        return triangles

    coverage._certificate_cached.cache_clear()
    with monkeypatch.context() as context:
        context.setattr(coverage, "_triangulate", broken)
        assert not coverage.silence_certificate(coverage.certified_stations(), True)
    coverage._certificate_cached.cache_clear()


def test_coincident_collinear_and_nonfinite_inputs_fail_closed():
    assert not coverage.silence_certificate([(0, 0)] * 25, True)
    assert not coverage.silence_certificate([(x, 0) for x in range(-2000, 2001, 100)], True)
    assert not coverage.silence_certificate([(float("nan"), 0)], True)
    assert not coverage.silence_certificate([(float("inf"), 0)], False)


def test_near_coincident_observations_do_not_erase_a_real_gap():
    stations = [p for p in coverage.certified_stations() if p != (1870.0, 0.0)]
    points = stations + [(x + 1e-9, y - 1e-9) for x, y in stations]
    assert not coverage.silence_certificate(points, True)


def test_incremental_gain_is_not_used_as_a_proof():
    state = coverage.SilenceCoverage(directional=True)
    assert state.gain((0, 0)) > 0
    assert not state.complete
    state.add((0, 0))
    state.add((0, 0))
    assert len(state.points) == 1
    assert state.gain((0, 0)) == 0
    assert state.uncovered_cells
    for point in coverage.certified_stations():
        state.add(point)
    assert state.complete


def test_cache_reuses_identical_channel_histories_without_rounding():
    coverage._certificate_cached.cache_clear()
    stations = coverage.certified_stations()
    assert coverage.silence_certificate(stations, True)
    assert coverage.silence_certificate(reversed(stations), True)
    assert coverage._certificate_cached.cache_info().hits == 1
    changed = stations[:-1] + [(stations[-1][0] + 1e-8, stations[-1][1])]
    coverage.silence_certificate(changed, True)
    assert coverage._certificate_cached.cache_info().misses == 2
