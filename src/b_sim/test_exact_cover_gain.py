"""Exact gain and complete-plan equivalence against the 83a3064 baseline."""

import math
from pathlib import Path
import random
import subprocess
import types

import pytest

from src.b_sim import feasible


@pytest.fixture(scope='module')
def original():
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ['git', '-C', str(root), 'show', '83a3064:src/b_sim/feasible.py'], text=True, encoding='utf-8'
    )
    module = types.ModuleType('src.b_sim._exact_gain_baseline')
    module.__package__ = 'src.b_sim'
    exec(compile(source, '<83a3064:src/b_sim/feasible.py>', 'exec'), module.__dict__)
    return module


def old_gain(work, station, step, reach, headings):
    points, index = feasible._lattice(step)
    x, y = points[station]
    i, j = int(round(x / step)), int(round(y / step))
    offsets, masks = feasible._beam_stencil(step, reach, headings)
    total = 0
    for di, dj, bucket in offsets:
        mask = work.get(index.get((i - di, j - dj), -1), 0)
        if mask:
            total += (mask & masks[bucket]).bit_count()
    return total


@pytest.mark.parametrize('step, headings, reception', [
    (100.0, 12, 1000.0), (120.0, 12, 1000.0), (100.0, 6, 1000.0),
    (100.0, 0, 1000.0), (180.0, 12, 900.0), (100.0, 12, 600.0),
])
def test_random_and_boundary_gains_are_bit_for_bit_equal(step, headings, reception):
    rng = random.Random(20260912)
    points, index = feasible._lattice(step)
    reach = feasible.guaranteed_reach(step, reception)
    all_mask = (1 << (headings + 1)) - 1
    # Include outer lattice cells as well as source cells to exercise every slot.
    work = {cell: rng.randint(0, all_mask) for cell in rng.sample(range(len(points)), len(points) // 2)}
    stations = set(rng.sample(range(len(points)), min(80, len(points))))
    stations.update([0, len(points) - 1, index[(0, 0)]])
    stations.update(i for i, p in enumerate(points) if math.hypot(*p) >= feasible.STATION_RADIUS_M - step / 2)
    for iteration in range(3):
        bits = feasible._work_mode_bits(work, headings)
        for station in sorted(stations):
            actual = (bits & feasible._station_mode_bits(step, reach, headings, station)).bit_count()
            assert actual == old_gain(work, station, step, reach, headings)
        # Changed masks and deleted cells must be reflected in the next encoding.
        work = {cell: mask & rng.randint(0, all_mask) for cell, mask in work.items() if rng.random() > .3}


def test_empty_work_and_cache_are_bounded():
    assert feasible._work_mode_bits({}, 12) == 0
    assert feasible._station_mode_bits.cache_info().maxsize == 8192


@pytest.mark.parametrize('seed, headings, tolerance, weight, max_stations', [
    (7, 12, .9, 20., 8), (13, 12, 0., 20., 6), (23, 12, 1., 20., 8),
    (31, 6, .9, .01, 5), (43, 0, .9, 20., 6), (47, 12, .9, 20., 0),
])
def test_complete_cover_route_matches_original(original, seed, headings, tolerance, weight, max_stations):
    step = 120.0
    rng = random.Random(seed)
    old_sets, new_sets = [], []
    for channel in range(3):
        old = original.FeasibleSet(step, headings)
        new = feasible.FeasibleSet(step, headings)
        if channel:
            selected = rng.sample(list(old.alive), 45 + 15 * channel)
            old.alive = {cell: rng.randint(1, old._all_mask) for cell in selected}
        for _ in range(3):
            old.apply_silence(rng.uniform(-2000, 2000), rng.uniform(-2000, 2000))
        new.alive = dict(old.alive)
        old_sets.append(old)
        new_sets.append(new)
    current = (rng.uniform(-2000, 2000), rng.uniform(-2000, 2000))
    kwargs = dict(cell_step=step, max_stations=max_stations, min_gain=3,
                  gain_tolerance=tolerance, travel_weight_m=weight,
                  anchors=[(0., 0.), (1100., -700.)])
    expected = original.cover_route(old_sets, current, **kwargs)
    before = [dict(fs.alive) for fs in new_sets]
    assert feasible.cover_route(new_sets, current, **kwargs) == expected
    assert [fs.alive for fs in new_sets] == before
    # Repeated calls must be identical with a warm station-mask cache.
    assert feasible.cover_route(new_sets, current, **kwargs) == expected


def test_no_alive_modes_returns_same_empty_plan(original):
    old = original.FeasibleSet(headings=12)
    new = feasible.FeasibleSet(headings=12)
    old.alive.clear()
    new.alive.clear()
    assert feasible.cover_route([new], (0., 0.)) == original.cover_route([old], (0., 0.)) == []
