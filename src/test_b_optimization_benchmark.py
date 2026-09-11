"""Guard against invalid comparisons and fast-transport scoring drift."""
from copy import deepcopy

from .b_optimization_benchmark import (
    aggregate, frozen_versions, gate, local_client, overrides_for, paired_ci, regressions,
    run_episode, selected_configurations, resume_rows, write_report, failure_events,
)


def episode(cleared=(1, 2), missed=(), mean=100):
    return {"seed": 1, "directional": 0, "cleared_channels": list(cleared),
            "missed_channels": list(missed), "cleared": len(cleared),
            "total_jammers": len(cleared) + len(missed), "full_clear": not missed,
            "mean_clear_time_s": mean, "total_virtual_time_s": mean * len(cleared),
            "measure_calls": 10, "clear_calls": 3, "move_distance_m": 500,
            "exception": None, "aborted_events": [], "finished": True}


def test_gate_rejects_channel_regression_even_with_same_clear_count():
    pair = {"seed": 1, "directional": 0,
            "baseline": episode((1, 2), (3,)), "candidate": episode((1, 3), (2,), 80)}
    assert regressions(pair) == [2]
    assert not gate([pair], 1, [])["passed"]
    summary = aggregate([pair])["d0"]
    assert summary["candidate"]["full_clear_cases"] == 0
    assert summary["candidate"]["mean_clear_time_s"]["mean"] == 80
    assert summary["clear_regressions"] == [{"seed": 1, "new_missed_channels": [2]}]


def test_gate_rejects_incomplete_aborted_and_http_mismatch():
    pair = {"seed": 1, "directional": 0, "baseline": episode(), "candidate": episode()}
    assert gate([pair], 1, [])["passed"]
    assert not gate([pair], 2, [])["passed"]
    bad = deepcopy(pair)
    bad["candidate"]["aborted_events"] = ["aborted: connection failure"]
    assert not gate([bad], 1, [])["passed"]
    checks = [{"directional": 0, "seed": 1, "arm": "candidate", "equivalent": False}]
    assert not gate([pair], 1, checks)["passed"]


def test_config_overrides_and_bootstrap_are_deterministic():
    assert overrides_for({"approach_fraction": .8}, 0) == {"approach_fraction": .8}
    assert overrides_for({"common": {"approach_fraction": .8}, "d5": {"approach_fraction": .7}}, 5) == {"approach_fraction": .7}
    assert paired_ci([-20, -20, -20]) == [-20, -20]
    assert paired_ci([-1, -5, 2]) == paired_ci([-1, -5, 2])


def test_local_dispatch_matches_http_action_sequence():
    import threading
    with frozen_versions("d83c40d") as (bundles, metadata):
        code = bundles["baseline"]
        assert metadata["baseline_commit"].startswith("d83c40d")
        arenas = [code["arena"].Arena(code["arena"].generate_case(930000)) for _ in range(2)]
        sessions = [code["server"].Session("benchmark-test", arena) for arena in arenas]
        direct = local_client(code["client"], code["server"], sessions[0])
        server = code["server"].make_server(sessions[1], port=0)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        wire = code["client"].RobotClient("benchmark-test", base_url=f"http://127.0.0.1:{server.server_address[1]}")
        try:
            for client in (direct, wire):
                client.enter()
                client.measure(0, 0, 1)
                client.measure(100.5, 300, 3)
                client.clear(100.5, 300, 3)
                client.measure(0, 0, 1)
                client.exit()
            assert arenas[0].robot.trace == arenas[1].robot.trace
            assert arenas[0].stats() == arenas[1].stats()
            assert direct.virtual_time_s == wire.virtual_time_s
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_frozen_modules_configs_and_concurrent_episodes_are_isolated():
    from concurrent.futures import ThreadPoolExecutor
    with frozen_versions("d83c40d") as (bundles, metadata):
        baseline = bundles["baseline"]["strategy"]
        candidate = bundles["candidate"]["strategy"]
        assert baseline.DogStrategy is not candidate.DogStrategy
        assert baseline.StrategyConfig is not candidate.StrategyConfig
        overrides = {"approach_fraction": .71}
        selected = selected_configurations(bundles, [0], overrides)
        assert selected["d0"]["candidate"]["approach_fraction"] == .71
        assert selected["d0"]["baseline"]["approach_fraction"] == baseline.StrategyConfig().approach_fraction
        assert candidate.StrategyConfig().approach_fraction != .71
        # Repeated episodes share cached modules but no mutable arena/channel
        # state. Distinct client logs must produce identical physical traces.
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: run_episode(bundles, 930000, 0, "candidate", overrides), range(4)))
        assert len({r["trace_sha256"] for r in results}) == 1
        assert all(r["config"] == selected["d0"]["candidate"] for r in results)
        assert all(r["finished"] and not r["exception"] and not r["aborted_events"] for r in results)
        assert all("completion_basis" in r and "unresolved_channels" in r for r in results)


def test_full_configuration_rejects_unknown_fields_before_any_run():
    import pytest
    with frozen_versions("d83c40d") as (bundles, _):
        with pytest.raises(ValueError, match="Unknown StrategyConfig"):
            selected_configurations(bundles, [0], {"approach_fracton": .8})
        with pytest.raises(ValueError, match="directional_probe"):
            selected_configurations(bundles, [5], {"directional_probe": False})


def test_continuous_certificate_cache_is_parallel_and_coordinate_exact():
    from concurrent.futures import ThreadPoolExecutor
    from .b_sim.silence_coverage import certified_stations, silence_certificate
    points = certified_stations("rings25")
    missing_outer = points[:13]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: silence_certificate(points if i % 2 == 0 else missing_outer, True), range(16)))
    assert results == [i % 2 == 0 for i in range(16)]
    assert silence_certificate(points, True)
    # A caller's list mutation must not corrupt an immutable cached key/result.
    points.clear()
    assert not silence_certificate(points, True)
    assert silence_certificate(certified_stations("rings25"), True)


def test_report_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    original = Path.replace
    attempts = []
    def blocked_once(self, target):
        attempts.append(target)
        if len(attempts) == 1:
            raise PermissionError("simulated sharing violation")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", blocked_once)
    target = tmp_path / "report.json"
    target.write_text('{"old": true}', encoding="utf-8")
    write_report(target, {"complete": False, "rows": [1]})
    assert len(attempts) == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": False, "rows": [1]}
    assert not target.with_suffix(".json.tmp").exists()


def test_resume_rejects_changed_provenance_and_duplicate_or_unplanned_pairs():
    import pytest
    report = {key: "same" for key in (
        "schema_version", "baseline_commit", "candidate_source_sha256", "baseline_source_sha256",
        "transport", "variant_overrides", "evaluation_stage", "selected_configurations",
        "http_checks_per_stratum", "metric_definition", "physics", "bootstrap")}
    report.update(seed_plan={"seeds": [1, 2], "directionals": [0, 5]}, complete=False,
                  rows=[{"seed": 1, "directional": 0}])
    assert resume_rows(deepcopy(report), report) == report["rows"]
    for key in ("candidate_source_sha256", "evaluation_stage", "selected_configurations", "http_checks_per_stratum"):
        previous = deepcopy(report)
        previous[key] = "changed"
        with pytest.raises(ValueError, match=key):
            resume_rows(previous, report)
    for rows in ([{"seed": 1, "directional": 0}] * 2, [{"seed": 3, "directional": 0}]):
        previous = {**report, "rows": rows}
        with pytest.raises(ValueError, match="Duplicate or unplanned"):
            resume_rows(previous, report)
    with pytest.raises(ValueError, match="already complete"):
        resume_rows({**report, "complete": True}, report)


def test_swallowed_protocol_failures_and_global_budget_stops_fail_gate():
    for event in ("measure stopped: rejected", "clear stopped: rejected", "aborted: rejected",
                  "real time budget reached", "virtual time budget reached"):
        pair = {"seed": 1, "directional": 0, "baseline": episode(), "candidate": episode()}
        pair["candidate"]["aborted_events"] = failure_events([event])
        assert not gate([pair], 1, [])["passed"]
    assert failure_events(["ch1: measure budget exhausted", "cleared channel 1"]) == []


def test_final_local_evaluation_requires_http_check(monkeypatch):
    import pytest
    from .b_optimization_benchmark import main
    monkeypatch.setattr("sys.argv", ["benchmark", "--stage", "final", "--http-checks", "0", "--out", "unused.json"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
