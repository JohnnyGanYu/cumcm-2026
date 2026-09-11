"""Audit directional empty-channel certificates using actual silent prefixes.

Run ``python -m src.b_silence_audit --out results/b_silence_audit.json``.
The default study has 4000 independently generated, valid 10--16-source worlds,
plus deterministic boundary stress cases. One preselected channel is evaluated
per world. This is a certificate audit, not a whole-strategy clearing benchmark.
Truth is used to generate observations and score decisions, never to make them.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import Callable, Iterable

from .b_sim.arena import Arena, Case, Jammer, generate_case
from .b_sim.silence_coverage import certified_stations, silence_certificate

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("rings25_complete", "mesh25_complete", "rings25_missing_outer", "inside_random")
Point = tuple[float, float]
Observation = tuple[Point, str]


def observation_decision(
    observations: Iterable[Observation],
    certificate: Callable = silence_certificate,
) -> dict:
    """Consume only public observations; never inspect a source or the future.

    A positive observation permanently blocks empty classification. Calls stop
    at the first certificate or positive observation, so every checked list is
    an actually measured all-silent prefix on this one channel.
    """
    silent: list[Point] = []
    checked = 0
    trace: list[Observation] = []
    outcome = "unresolved"
    for point, result in observations:
        trace.append((point, result))
        if result in ("direction", "near"):
            outcome = "observed_signal"
            break
        if result != "no_signal":
            raise ValueError(f"Unrecognized observation result: {result}")
        silent.append(point)
        checked += 1
        if certificate(silent, directional=True):
            outcome = "certified_empty"
            break
    return {"outcome": outcome, "empty_verdict": outcome == "certified_empty",
            "observation_count": len(trace), "silent_prefix_count": len(silent),
            "certificate_checks": checked,
            "first_empty_verdict_prefix": len(silent) if outcome == "certified_empty" else None,
            "observation_sha256": hashlib.sha256(
                json.dumps(trace, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def _rotate(points: Iterable[Point], angle: float) -> list[Point]:
    c, s = math.cos(angle), math.sin(angle)
    return [(c * x - s * y, s * x + c * y) for x, y in points]


def planned_route(profile: str, seed: int) -> list[Point]:
    """Select a route without any source, channel result or truth input."""
    rng = random.Random(seed)
    if profile == "inside_random":
        points = []
        for _ in range(rng.randint(8, 24)):
            r, theta = 1750 * math.sqrt(rng.random()), rng.uniform(0, math.tau)
            points.append((r * math.cos(theta), r * math.sin(theta)))
        return points
    kind = "mesh25" if profile == "mesh25_complete" else "rings25"
    points = certified_stations(kind)
    if profile == "rings25_missing_outer":
        del points[13 + rng.randrange(12)]
    elif profile not in ("rings25_complete", "mesh25_complete"):
        raise ValueError(f"Unknown profile: {profile}")
    # Continuous rotations stress hull/triangulation arithmetic; shuffling makes
    # different actual prefixes available before the same complete route ends.
    points = _rotate(points, rng.uniform(0, math.tau))
    rng.shuffle(points)
    return points


def _evaluate_world(case: Case, channel: int, route: list[Point], **labels) -> dict:
    arena = Arena(case)
    arena.enter()

    def observations():
        for point in route:
            response = arena.measure(*point, channel)
            yield point, response["measure_result"]

    started = time.perf_counter()
    try:
        decision = observation_decision(observations())
    finally:
        arena.exit()
    # Truth is inspected only after the observation-only decision returns.
    source = next((jammer for jammer in case.jammers if jammer.channel == channel), None)
    truth = None if source is None else {
        "x": source.x, "y": source.y, "reception_radius_m": source.reception_radius_m,
        "directional_deg": source.directional_deg}
    return {**labels, **decision, "channel": channel, "world_source_count": case.total,
            "source_present": source is not None, "target_truth_for_scoring_only": truth,
            "false_empty": source is not None and decision["empty_verdict"],
            "planned_station_count": len(route),
            "route_sha256": hashlib.sha256(json.dumps(route, separators=(",", ":")).encode()).hexdigest(),
            "wall_time_s": time.perf_counter() - started}


def monte_carlo_case(index: int, seed_start: int) -> dict:
    seed = seed_start + index
    selection_rng = random.Random(seed ^ 0x71C39)
    count = selection_rng.randint(10, 16)
    case = generate_case(seed, count=count, directional_count=count)
    occupied = {j.channel for j in case.jammers}
    source_present = index % 2 == 0
    candidates = sorted(occupied if source_present else set(range(1, 21)) - occupied)
    channel = selection_rng.choice(candidates)
    profile = PROFILES[(index // 2) % len(PROFILES)]
    route_seed = seed ^ 0x5D239817
    route = planned_route(profile, route_seed)
    return _evaluate_world(case, channel, route, study="monte_carlo", profile=profile,
                           case_index=index, world_seed=seed, route_seed=route_seed)


def boundary_cases(angle_count: int, seed_start: int) -> list[dict]:
    rows = []
    for k in range(angle_count):
        theta = math.tau * k / angle_count
        direction = math.degrees(theta)
        for variant in ("complete_present", "missing_outer_present", "missing_outer_empty"):
            seed = seed_start + 3 * k + (0 if variant == "complete_present" else
                                        1 if variant == "missing_outer_present" else 2)
            case = generate_case(seed, count=10, directional_count=10)
            channel = case.jammers[0].channel
            if variant.endswith("_empty"):
                # Choose a truly absent channel while retaining all 10 sources.
                channel = min(set(range(1, 21)) - {j.channel for j in case.jammers})
            else:
                case.jammers[0] = Jammer(channel, 1800 * math.cos(theta), 1800 * math.sin(theta),
                                         1000, direction)
            if variant == "complete_present":
                route = certified_stations("rings25")
            else:
                # Deleting the outer receiver directly ahead leaves an actual
                # outward source possible, for both occupied and empty channels.
                route = _rotate([p for p in certified_stations("rings25") if p != (1870.0, 0.0)], theta)
            rows.append(_evaluate_world(case, channel, route, study="boundary", profile=variant,
                                        angle_deg=direction, world_seed=seed,
                                        expected_outcome=("observed_signal" if variant == "complete_present"
                                                          else "unresolved")))
    return rows


def closed_boundary_checks() -> list[dict]:
    """Independent protocol checks at exactly 1000 m and +/-90-degree beams."""
    source = Jammer(1, 1800, 0, 1000, 0)
    examples = [((2800.0, 0.0), "direction"), ((1800.0, 1000.0), "direction"),
                ((1800.0, -1000.0), "direction"), ((2800.000001, 0.0), "no_signal"),
                ((1799.999999, 999.0), "no_signal"), ((1800.0, 0.0), "near")]
    rows = []
    for point, expected in examples:
        case = generate_case(200, count=10, directional_count=10)
        # Keep distinct channels and the official total count in every fixture.
        other = [j for j in case.jammers if j.channel != 1][:9]
        if len(other) < 9:
            raise AssertionError("fixture requires nine unrelated channels")
        arena = Arena(Case([source] + other, "closed-boundary", 200))
        arena.enter()
        actual = arena.measure(*point, 1)["measure_result"]
        arena.exit()
        rows.append({"position": point, "expected": expected, "actual": actual, "passed": actual == expected})
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    present = sum(row["source_present"] for row in rows)
    absent = total - present
    verdicts = sum(row["empty_verdict"] for row in rows)
    false_empty = sum(row["false_empty"] for row in rows)
    true_empty = verdicts - false_empty
    triggered_at = [row["first_empty_verdict_prefix"] for row in rows if row["empty_verdict"]]
    stochastic = bool(rows) and all(row.get("study") == "monte_carlo" for row in rows)

    def zero_error_upper(denominator):
        # Exact one-sided zero-success binomial bound, when the independent
        # per-case Bernoulli sampling assumptions stated in the report apply.
        return -math.expm1(math.log(.05) / denominator) if stochastic and not false_empty and denominator else None

    return {
        "case_count": total, "source_present_count": present, "source_absent_count": absent,
        "empty_verdict_count": verdicts, "false_empty_count": false_empty,
        "false_empty_over_empty_verdicts": false_empty / verdicts if verdicts else None,
        "false_empty_over_source_present": false_empty / present if present else None,
        "zero_false_empty_95pct_upper_given_source_present": zero_error_upper(present),
        "zero_false_empty_95pct_upper_among_empty_verdicts": zero_error_upper(verdicts),
        "zero_error_bound_scope": ("one-sided 1 - 0.05**(1/n); model-conditional independent-case reference; see sampling limits"
                                   if stochastic else "not applied to deterministic boundary fixtures or unspecified sampling"),
        "true_empty_verdict_count": true_empty,
        "empty_channel_certificate_trigger_rate": true_empty / absent if absent else None,
        "source_present_observed_count": sum(row["source_present"] and row["outcome"] == "observed_signal" for row in rows),
        "empty_channel_unresolved_count": sum(not row["source_present"] and row["outcome"] == "unresolved" for row in rows),
        "source_present_unresolved_count": sum(row["source_present"] and row["outcome"] == "unresolved" for row in rows),
        "observed_signal_on_empty_count": sum(not row["source_present"] and row["outcome"] == "observed_signal" for row in rows),
        "certificate_prefix_check_count": sum(row["certificate_checks"] for row in rows),
        "mean_observations_at_empty_verdict": statistics.fmean(triggered_at) if triggered_at else None,
    }


def audit(cases: int = 4000, seed_start: int = 2026091200, boundary_angles: int = 360,
          progress: Callable | None = None) -> dict:
    if cases <= 0 or cases % 8 or boundary_angles < 4:
        raise ValueError("cases must be a positive multiple of 8; boundary_angles must be >=4")
    rows = []
    started = time.perf_counter()
    for index in range(cases):
        rows.append(monte_carlo_case(index, seed_start))
        if progress and (index + 1) % 200 == 0:
            progress(f"Monte Carlo: {index + 1}/{cases}, elapsed {time.perf_counter() - started:.1f}s")
    boundary = boundary_cases(boundary_angles, seed_start + cases + 100000)
    closed = closed_boundary_checks()
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows + boundary:
        groups[row["study"] + "/" + row["profile"]].append(row)
    boundary_failures = [row for row in boundary if row["outcome"] != row["expected_outcome"]]
    false_empty = [row for row in rows + boundary if row["false_empty"]]
    false_signal = [row for row in rows + boundary if not row["source_present"] and row["outcome"] == "observed_signal"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                     for name in ("src/b_sim/arena.py", "src/b_sim/silence_coverage.py", "src/b_silence_audit.py")}
    return {
        "schema_version": 1, "complete": True,
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_hashes,
        "scope": "directional continuous-certificate audit; one channel per scene, not full-strategy clearance performance",
        "scale": "4000_or_more_monte_carlo_cases" if cases >= 4000 else "smoke_only",
        "seed_start": seed_start,
        "sampling": {
            "monte_carlo_case_count": cases,
            "world": "independent local Arena worlds; total sources uniform integer 10..16; all sources directional; distinct random channels",
            "source_position": "uniform area in closed radius-1800 disk via r=1800*sqrt(U), theta=2*pi*U",
            "source_reception_m": "uniform [1000,1500]", "source_heading_deg": "uniform [0,360)",
            "target_channel": "exactly 50% occupied, 50% absent; uniformly selected within each condition; evaluation balance is not an operational prior",
            "routes": "four profiles equally frequent; template routes independently rotated and shuffled; missing-outer deletes one of 12 outer receivers; inside_random has 8..24 uniform-area points within radius1750",
            "boundary": f"{boundary_angles} equally spaced angles, source radius1800, reception1000, outward heading; complete-present/missing-present/missing-empty each {boundary_angles} cases",
            "decision_input": "only one channel's actual no_signal prefix; direction/near ends the decision before any further certificate call",
        },
        "denominators": {
            "false_empty_over_empty_verdicts": "false_empty_count / empty_verdict_count; null if no empty verdict",
            "false_empty_over_source_present": "false_empty_count / source_present_count; false-positive rate conditional on an occupied target channel",
            "empty_channel_certificate_trigger_rate": "true_empty_verdict_count / source_absent_count; includes deliberately incomplete routes",
            "prefix_checks": "diagnostic repeated checks within cases, not independent samples; no confidence interval treats prefixes as independent",
            "zero_error_bound": "when zero errors are observed, 1-0.05**(1/n), with n separately equal to occupied cases or empty verdicts; null for zero denominator or deterministic boundary fixtures",
        },
        "monte_carlo": summarize(rows), "boundary": summarize(boundary),
        "groups": {name: summarize(group) for name, group in sorted(groups.items())},
        "closed_boundary_checks": closed,
        "acceptance": {"passed": not false_empty and not false_signal and not boundary_failures and all(c["passed"] for c in closed),
                       "false_empty_count": len(false_empty), "observed_signal_on_empty_count": len(false_signal),
                       "boundary_outcome_failure_count": len(boundary_failures)},
        "limitations": ["Zero sampled false-empty decisions is empirical evidence, not a proof of zero risk or a formal-server result.",
                        "One independently generated scene contributes one selected channel; repeated silent prefixes from the same scene are dependent and never increase the confidence-bound denominator.",
                        "The one-sided zero-error bound assumes independent Bernoulli cases with a common conditional error probability within each random-route stratum. Pooled bounds are model-conditional references for this balanced design, not a distribution-free guarantee for deployment or deterministic boundary fixtures.",
                        "Safety depends on the published minimum reception radius, source disk and closed 180-degree beam assumptions.",
                        "The mathematical certificate proves absence; inability to construct a certificate means unresolved, including numerical fail-closed cases.",
                        "Route generation and channel balance are audit distributions; certificate trigger rates are not production-strategy trigger rates."],
        "counterexamples": false_empty + false_signal + boundary_failures,
        "wall_time_s": time.perf_counter() - started,
        "rows": rows + boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=4000)
    parser.add_argument("--seed-start", type=int, default=2026091200)
    parser.add_argument("--boundary-angles", type=int, default=360)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.cases, args.seed_start, args.boundary_angles,
                   progress=lambda message: print(message, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({name: report[name] for name in ("scale", "monte_carlo", "boundary", "acceptance", "wall_time_s")}, indent=2))
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
