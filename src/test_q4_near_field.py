"""Protect directional truth when a silent probe is inside one source cell."""
from src.b_sim.feasible import FeasibleSet, _beam_stencil


def test_silent_probe_retains_near_field_directional_hypotheses():
    feasible = FeasibleSet(step=200, headings=12)
    index = feasible.cells.index((0.0, 0.0))
    all_modes = (1 << 13) - 1
    feasible.alive = {index: all_modes}
    assert feasible.coverage_gain(0, 0) == 1
    assert feasible.apply_silence(0, 0) == 1
    assert feasible.alive == {index: all_modes & ~1}
    assert feasible.detection_possible(0, 0)
    # Q3 still removes its sole omnidirectional hypothesis after silence.
    omni = FeasibleSet(step=200, headings=0)
    omni.alive = {index: 1}
    assert omni.apply_silence(0, 0) == 1
    assert not omni.possible


def test_near_field_stencil_does_not_claim_directional_coverage():
    offsets, masks = _beam_stencil(200, 800, 12)
    bucket = next(bucket for di, dj, bucket in offsets if di == dj == 0)
    assert masks[bucket] == 1
