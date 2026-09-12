"""Tests for immutable, fail-closed source/configuration freezing."""
from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from src.b_eval.provenance import (
    DEFAULT_EXECUTION_BOUNDS,
    EXECUTION_BOUND_FIELDS,
    GATE_VERSION,
    PHYSICS_RULES_VERSION,
    FrozenManifest,
    build_manifest,
    freeze_problem_configurations,
    manifest_digest,
    verify_manifest_digest,
    write_manifest,
)
from src.b_sim.strategy import StrategyConfig
from src.pack_dist_windows import RUNTIME_MODULES


def _sources(prefix: str) -> dict[str, bytes]:
    return {
        f"src/b_sim/{name}": f"{prefix}:{name}\n".encode()
        for name in RUNTIME_MODULES
    }


def _environment(worker_threads: int = 4) -> dict:
    return {
        "python": {"implementation": "CPython", "version": "3.11.15"},
        "os": {"system": "TestOS", "release": "1", "version": "1.0", "machine": "x86_64"},
        "cpu": {"model": "Test CPU", "logical_count": 8},
        "worker_threads": worker_threads,
    }


def _manifest(**kwargs) -> FrozenManifest:
    values = {
        "baseline_commit": "a" * 40,
        "candidate_commit": "b" * 40,
        "baseline_runtime_files": _sources("baseline"),
        "candidate_runtime_files": _sources("candidate"),
        "baseline_config_class": StrategyConfig,
        "candidate_config_class": StrategyConfig,
        "worker_threads": 4,
        "environment": _environment(),
    }
    values.update(kwargs)
    return build_manifest(**values)


def _bounds(**updates: object) -> dict[str, object]:
    return {**DEFAULT_EXECUTION_BOUNDS, **updates}


def _problem_bounds(**q3_updates: object) -> dict[str, dict[str, object]]:
    return {"q3": _bounds(**q3_updates), "q4": _bounds()}


def test_manifest_is_complete_immutable_and_source_sensitive():
    manifest = _manifest(candidate_overrides={"q3": {"approach_fraction": 0.7}})
    data = manifest.to_dict()

    assert verify_manifest_digest(data)
    assert data["provenance"]["physics_rules_version"] == PHYSICS_RULES_VERSION
    assert data["provenance"]["gate_version"] == GATE_VERSION
    assert data["provenance"]["environment"]["worker_threads"] == 4
    assert len(data["provenance"]["baseline"]["runtime_source_sha256"]) == 64
    assert len(data["provenance"]["candidate"]["runtime_source_sha256"]) == 64
    assert data["provenance"]["baseline"]["runtime_source_sha256"] != data["provenance"]["candidate"]["runtime_source_sha256"]

    expected_fields = set(asdict(StrategyConfig()))
    assert set(data["configurations"]) == {"q3", "q4"}
    for problem in ("q3", "q4"):
        section = data["configurations"][problem]
        for arm in ("baseline", "candidate"):
            assert set(section[arm]["strategy_config"]) == expected_fields
            assert set(section[arm]["execution_bounds"]) == set(EXECUTION_BOUND_FIELDS)
    assert data["configurations"]["q3"]["candidate"]["strategy_config"]["directional_probe"] is False
    assert data["configurations"]["q4"]["candidate"]["strategy_config"]["directional_probe"] is True
    assert data["configurations"]["q3"]["candidate"]["strategy_config"]["approach_fraction"] == 0.7

    with pytest.raises(TypeError):
        manifest.data["schema_version"] = 2
    with pytest.raises(TypeError):
        manifest.data["configurations"]["q3"]["candidate"]["strategy_config"]["sweep_origin"] = False

    changed = _manifest(candidate_runtime_files=_sources("changed"))
    assert changed.sha256 != manifest.sha256


def test_repeated_manifest_generation_has_identical_digest_and_bytes(tmp_path):
    first = _manifest()
    second = _manifest()
    assert first.sha256 == second.sha256
    assert first.to_dict() == second.to_dict()

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_manifest(first_path, first)
    write_manifest(second_path, second)
    assert first_path.read_bytes() == second_path.read_bytes()
    loaded = json.loads(first_path.read_text(encoding="utf-8"))
    assert loaded["manifest_sha256"] == manifest_digest(loaded)

    loaded["provenance"]["gate_version"] = "tampered"
    assert not verify_manifest_digest(loaded)
    with pytest.raises(ValueError, match="invalid digest"):
        write_manifest(tmp_path / "tampered.json", loaded)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "10"])
def test_execution_bounds_require_positive_integers(value):
    with pytest.raises(ValueError, match="positive integer"):
        freeze_problem_configurations(
            StrategyConfig,
            StrategyConfig,
            execution_bounds=_problem_bounds(loop_iterations_max=value),
        )


def test_execution_bound_schema_rejects_unknown_missing_and_conflicting_fields():
    unknown = _problem_bounds()
    unknown["q3"]["loop_limit_typo"] = 10
    with pytest.raises(ValueError, match="Unknown execution bound"):
        freeze_problem_configurations(StrategyConfig, StrategyConfig, execution_bounds=unknown)

    missing = _problem_bounds()
    del missing["q3"]["recursion_depth_max"]
    with pytest.raises(ValueError, match="Missing execution bound"):
        freeze_problem_configurations(StrategyConfig, StrategyConfig, execution_bounds=missing)

    conflict = _problem_bounds(candidate_expansions_total_max=10)
    with pytest.raises(ValueError, match="conflicts"):
        freeze_problem_configurations(StrategyConfig, StrategyConfig, execution_bounds=conflict)


def test_configuration_schema_rejects_unknown_modes_types_and_non_finite_values():
    with pytest.raises(ValueError, match="Unknown problem configuration"):
        freeze_problem_configurations(
            StrategyConfig,
            StrategyConfig,
            candidate_overrides={"q5": {}},
        )
    with pytest.raises(ValueError, match="Unknown StrategyConfig"):
        freeze_problem_configurations(
            StrategyConfig,
            StrategyConfig,
            candidate_overrides={"q3": {"approach_fracton": 0.7}},
        )
    with pytest.raises(ValueError, match="conflicts with q3"):
        freeze_problem_configurations(
            StrategyConfig,
            StrategyConfig,
            candidate_overrides={"q3": {"directional_probe": True}},
        )
    with pytest.raises(ValueError, match="conflicts with q4"):
        freeze_problem_configurations(
            StrategyConfig,
            StrategyConfig,
            candidate_overrides={"q4": {"directional_probe": False}},
        )
    with pytest.raises(ValueError, match="must be int"):
        freeze_problem_configurations(
            StrategyConfig,
            StrategyConfig,
            candidate_overrides={"q3": {"max_total_measures": 900.0}},
        )
    with pytest.raises(ValueError, match="Non-finite"):
        freeze_problem_configurations(
            StrategyConfig,
            StrategyConfig,
            candidate_overrides={"q4": {"approach_fraction": float("inf")}},
        )


def test_worker_count_and_runtime_source_closure_are_fail_closed():
    with pytest.raises(ValueError, match="worker_threads"):
        _manifest(worker_threads=0, environment=_environment(0))
    with pytest.raises(ValueError, match="conflicts"):
        _manifest(worker_threads=2, environment=_environment(4))

    incomplete = _sources("baseline")
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="delivery closure"):
        _manifest(baseline_runtime_files=incomplete)
