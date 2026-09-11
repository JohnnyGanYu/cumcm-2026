"""Audit-method tests: observation-only evidence and explicit denominators."""

import pytest

from .b_silence_audit import (
    audit, closed_boundary_checks, monte_carlo_case, observation_decision, summarize,
)


def test_decision_only_passes_actual_prefix_and_never_reads_future():
    seen = []

    def certificate(points, *, directional):
        assert directional
        seen.append(list(points))
        return len(points) == 2

    def observations():
        yield (0.0, 0.0), "no_signal"
        yield (100.0, 200.0), "no_signal"
        raise AssertionError("unmeasured future evidence must not be accessed")

    result = observation_decision(observations(), certificate)
    assert result["empty_verdict"]
    assert result["first_empty_verdict_prefix"] == 2
    assert seen == [[(0.0, 0.0)], [(0.0, 0.0), (100.0, 200.0)]]


@pytest.mark.parametrize("positive", ["direction", "near"])
def test_positive_observation_prevents_any_subsequent_empty_verdict(positive):
    result = observation_decision([((0, 0), positive), ((100, 0), "no_signal")],
                                  certificate=lambda *args, **kwargs: pytest.fail("positive source cannot be judged empty"))
    assert result["outcome"] == "observed_signal"
    assert not result["empty_verdict"]
    assert result["certificate_checks"] == 0


def test_unknown_response_is_not_treated_as_silence():
    with pytest.raises(ValueError, match="Unrecognized"):
        observation_decision([((0, 0), "invalid")])


def test_denominators_separate_empty_verdicts_from_present_sources():
    def row(present, outcome):
        return {"source_present": present, "outcome": outcome,
                "empty_verdict": outcome == "certified_empty",
                "false_empty": present and outcome == "certified_empty", "certificate_checks": 2,
                "first_empty_verdict_prefix": 2 if outcome == "certified_empty" else None}
    report = summarize([row(True, "certified_empty"), row(True, "observed_signal"),
                        row(True, "unresolved"), row(False, "certified_empty"), row(False, "unresolved")])
    assert report["false_empty_count"] == 1
    assert report["empty_verdict_count"] == 2
    assert report["false_empty_over_empty_verdicts"] == .5
    assert report["false_empty_over_source_present"] == pytest.approx(1 / 3)
    assert report["empty_channel_certificate_trigger_rate"] == .5
    assert summarize([row(True, "unresolved")])["false_empty_over_empty_verdicts"] is None


def test_case_generation_is_reproducible_with_real_world_source_counts():
    first, second = monte_carlo_case(0, 2026091200), monte_carlo_case(0, 2026091200)
    assert first["observation_sha256"] == second["observation_sha256"]
    assert first["route_sha256"] == second["route_sha256"]
    assert first["target_truth_for_scoring_only"] == second["target_truth_for_scoring_only"]
    assert first["source_present"]
    assert 10 <= first["world_source_count"] <= 16
    assert not monte_carlo_case(1, 2026091200)["source_present"]


def test_small_study_is_labeled_smoke_and_refuses_missing_station_proofs():
    report = audit(8, boundary_angles=4)
    assert report["scale"] == "smoke_only"
    assert report["monte_carlo"]["source_present_count"] == 4
    assert report["monte_carlo"]["source_absent_count"] == 4
    assert report["acceptance"]["passed"]
    assert report["boundary"]["case_count"] == 12
    assert report["groups"]["boundary/missing_outer_present"]["source_present_unresolved_count"] == 4
    assert report["groups"]["boundary/missing_outer_empty"]["empty_channel_unresolved_count"] == 4


def test_closed_reception_and_beam_boundaries_match_mock_physics():
    assert all(row["passed"] for row in closed_boundary_checks())


def test_zero_error_upper_uses_case_denominators_not_repeated_prefix_checks():
    rows = [{"study": "monte_carlo", "source_present": i < 10,
             "outcome": "unresolved" if i < 10 else "certified_empty",
             "empty_verdict": i >= 10, "false_empty": False,
             "certificate_checks": 99,
             "first_empty_verdict_prefix": 99 if i >= 10 else None} for i in range(15)]
    result = summarize(rows)
    assert result["zero_false_empty_95pct_upper_given_source_present"] == pytest.approx(1 - .05 ** (1 / 10))
    assert result["zero_false_empty_95pct_upper_among_empty_verdicts"] == pytest.approx(1 - .05 ** (1 / 5))
    for row in rows:
        row["study"] = "boundary"
    assert summarize(rows)["zero_false_empty_95pct_upper_given_source_present"] is None
