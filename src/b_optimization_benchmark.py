"""Paired, frozen-source offline evaluation; never contacts the official service.

python -m src.b_optimization_benchmark --episodes 30 --seed-start 930000 \
    --out results/b_optimization_development.json

Both arms use the baseline simulator. ``local`` calls the same HTTP handler
dispatcher in process; it is checked against real loopback HTTP by default.
Failed cases remain in the report. Exit code 2 means the acceptance gate failed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mean_clear_time_s", "total_virtual_time_s", "measure_calls", "clear_calls", "move_distance_m")


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def source_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        digest.update(name.encode("utf-8") + b"\0" + content.replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


@contextmanager
def frozen_versions(baseline_ref: str):
    """Do not let baseline imports, or concurrent working-tree edits, drift."""
    base_hash = git("rev-parse", "--verify", baseline_ref + "^{commit}").decode().strip()
    paths = git("ls-tree", "-r", "--name-only", base_hash, "src/b_sim").decode().splitlines()
    base_files = {p.removeprefix("src/b_sim/"): git("show", f"{base_hash}:{p}")
                  for p in paths if p.endswith(".py") and not Path(p).name.startswith("test_")}
    candidate_files = {p.relative_to(ROOT / "src/b_sim").as_posix(): p.read_bytes()
                       for p in (ROOT / "src/b_sim").rglob("*.py")
                       if not p.name.startswith("test_")}
    with tempfile.TemporaryDirectory(prefix="cumcm-paired-") as directory:
        bundles = {}
        for arm, files in (("baseline", base_files), ("candidate", candidate_files)):
            package_name = "_cumcm_" + arm + "_" + Path(directory).name.replace("-", "_")
            folder = Path(directory) / package_name
            folder.mkdir()
            for name, data in files.items():
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            spec = importlib.util.spec_from_file_location(package_name, folder / "__init__.py",
                                                          submodule_search_locations=[str(folder)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            spec.loader.exec_module(package)
            bundles[arm] = {name: importlib.import_module(package_name + "." + name)
                            for name in ("arena", "client", "server", "strategy")}
            bundles[arm]["package_name"] = package_name
        try:
            yield bundles, {"baseline_commit": base_hash,
                            "working_tree_head": git("rev-parse", "HEAD").decode().strip(),
                            "baseline_source_sha256": source_digest(base_files),
                            "candidate_source_sha256": source_digest(candidate_files)}
        finally:
            for bundle in bundles.values():
                prefix = bundle["package_name"]
                for key in list(sys.modules):
                    if key == prefix or key.startswith(prefix + "."):
                        sys.modules.pop(key)


def local_client(client_module, server_module, session):
    """Same validation/dispatch as HTTP; strategy receives only protocol replies."""
    handler = object.__new__(server_module.Handler)
    handler.session = session

    class LocalClient(client_module.RobotClient):
        def _post(self, path, payload):
            try:
                status, body = handler._dispatch(path, payload)
            except server_module.BadRequest:
                status, body = 400, handler._envelope(False)
            except server_module.Conflict:
                status, body = 409, handler._envelope(False)
            except server_module.Rejected:
                status, body = 200, handler._envelope(False)
            # JSON roundtrip preserves the wire representation and independent logs.
            body = json.loads(json.dumps(body, allow_nan=False))
            self.log.append({"ts": time.time(), "path": path, "request_id": payload["request_id"],
                             "request": dict(payload), "http_status": status, "response": body})
            if status != 200 or body.get("accepted") is not True:
                raise client_module.ProtocolError(f"{path} rejected http={status} body={body}")
            self.virtual_time_s = float(body["virtual_time_s"])
            return body

    return LocalClient(session.robot_id)


def failure_events(events):
    return [event for event in events if event.startswith((
        "aborted:", "measure stopped:", "clear stopped:",
        "real time budget reached", "virtual time budget reached"))]


def run_episode(bundles, seed: int, directional: int, arm: str, overrides=None, transport="local"):
    """The strategy cannot access seed, source count or arena statistics."""
    engine = bundles["baseline"]
    code = bundles[arm]
    count = 16 if directional == 16 else None
    arena = engine["arena"].Arena(engine["arena"].generate_case(seed, count=count, directional_count=directional))
    session = engine["server"].Session("paired-offline", arena, real_limit_s=1200)
    server = thread = None
    config = code["strategy"].StrategyConfig(directional_probe=directional > 0)
    for key, value in (overrides or {}).items():
        if key not in asdict(config):
            raise ValueError(f"Unknown StrategyConfig field: {key}")
        if key == "directional_probe" and value != (directional > 0):
            raise ValueError("directional_probe must match the evaluation stratum")
        setattr(config, key, value)
    started = time.monotonic()
    caught = None
    strategy = None
    summary = {}
    try:
        if transport == "http":
            server = engine["server"].make_server(session, port=0)
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
            thread.start()
            client = code["client"].RobotClient(session.robot_id,
                                                base_url=f"http://127.0.0.1:{server.server_address[1]}")
        elif transport == "local":
            client = local_client(code["client"], engine["server"], session)
        else:
            raise ValueError(f"Unknown transport: {transport}")
        strategy = code["strategy"].DogStrategy(client, config)
        summary = strategy.run()
    except Exception:
        caught = traceback.format_exc()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    stats = arena.stats()
    successes = [e for e in arena.robot.trace if e.get("clear_result") == "success"]
    events = strategy.events if strategy is not None else []
    trace_json = json.dumps(arena.robot.trace, sort_keys=True, separators=(",", ":"), allow_nan=False)
    stats.update(seed=seed, directional=directional, explicit_count=count, arm=arm, transport=transport,
                 wall_time_s=time.monotonic() - started, config=asdict(config),
                 missed_channels=sorted(j.channel for j in arena.case.jammers if not j.cleared),
                 cleared_channels=sorted(j.channel for j in arena.case.jammers if j.cleared),
                 full_clear=arena.case.cleared_count == arena.case.total,
                 after_last_clear_s=(arena.robot.virtual_time_s - successes[-1]["virtual_time_s"]
                                     if successes else None),
                 trace_sha256=hashlib.sha256(trace_json.encode()).hexdigest(),
                 completion_basis=summary.get("completion_basis"),
                 certified_empty_channels=summary.get("certified_empty_channels"),
                 unresolved_channels=summary.get("unresolved_channels"),
                 coverage_skipped_stations=summary.get("coverage_skipped_stations"),
                 exception=caught, aborted_events=failure_events(events),
                 finished=arena.robot.finished)
    return stats


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def paired_ci(differences, repeats=2000):
    """Percentile bootstrap over paired episodes; negative means faster."""
    if not differences:
        return None
    rng = random.Random(20260911)
    sampled = [statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(repeats)]
    return [percentile(sampled, 0.025), percentile(sampled, 0.975)]


def regressions(pair):
    return sorted(set(pair["baseline"]["cleared_channels"]) - set(pair["candidate"]["cleared_channels"]))


def aggregate(rows):
    output = {}
    for directional in sorted({r["directional"] for r in rows}):
        group = [r for r in rows if r["directional"] == directional]
        section = {"episodes": len(group), "seed_start": min(r["seed"] for r in group),
                   "clear_regressions": [{"seed": p["seed"], "new_missed_channels": regressions(p)}
                                         for p in group if regressions(p)]}
        for arm in ("baseline", "candidate"):
            episodes = [p[arm] for p in group]
            summary = {"full_clear_cases": sum(e["full_clear"] for e in episodes),
                       "cleared": sum(e["cleared"] for e in episodes),
                       "total_jammers": sum(e["total_jammers"] for e in episodes),
                       "missed_cases": [{"seed": e["seed"], "channels": e["missed_channels"]}
                                        for e in episodes if e["missed_channels"]]}
            for metric in METRICS:
                values = [e[metric] for e in episodes if e[metric] is not None]
                # Missing scores invalidate the report, not silently improve its mean.
                summary[metric] = {"mean": statistics.fmean(values) if len(values) == len(episodes) else None,
                                   "p90": percentile(values, .9) if len(values) == len(episodes) else None}
            section[arm] = summary
        differences = [p["candidate"]["mean_clear_time_s"] - p["baseline"]["mean_clear_time_s"]
                       for p in group if p["candidate"]["mean_clear_time_s"] is not None
                       and p["baseline"]["mean_clear_time_s"] is not None]
        section["paired_mean_delta_s"] = statistics.fmean(differences) if len(differences) == len(group) else None
        section["paired_delta_95pct_ci_s"] = paired_ci(differences) if len(differences) == len(group) else None
        section["paired_better_tie_worse"] = {"better": sum(d < -1e-6 for d in differences),
                                               "tie": sum(abs(d) <= 1e-6 for d in differences),
                                               "worse": sum(d > 1e-6 for d in differences)}
        output[f"d{directional}"] = section
    return output


def gate(rows, expected, http_checks):
    reasons = []
    if len(rows) != expected:
        reasons.append(f"incomplete: expected {expected} pairs, got {len(rows)}")
    for pair in rows:
        key = f"d{pair['directional']}/seed={pair['seed']}"
        for arm in ("baseline", "candidate"):
            e = pair[arm]
            if e["exception"] or e["aborted_events"] or not e["finished"] or e["mean_clear_time_s"] is None:
                reasons.append(f"{key}/{arm}: exception, aborted, unfinished or undefined score")
        if regressions(pair):
            reasons.append(f"{key}: candidate newly missed channels {regressions(pair)}")
    for check in http_checks:
        if not check["equivalent"]:
            reasons.append(f"HTTP mismatch: d{check['directional']}/seed={check['seed']}/{check['arm']}")
    return {"passed": not reasons, "reasons": reasons,
            "scope": "completion, errors, per-case cleared-channel retention and HTTP transport equivalence; speed targets reported separately"}


def overrides_for(variant, directional):
    if not any(key in variant for key in ("common", "d0", "d5", "d10", "d16")):
        return dict(variant)
    return {**variant.get("common", {}), **variant.get(f"d{directional}", {})}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Keep each raw pair on one line so a complete reproducibility artifact
    # does not bury a code review under hundreds of thousands of JSON lines.
    sections = []
    for key, value in report.items():
        if key == "rows":
            encoded = "[\n" + ",\n".join("    " + json.dumps(row, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False) for row in value) + "\n  ]"
        else:
            encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).replace("\n", "\n  ")
        sections.append("  " + json.dumps(key) + ": " + encoded)
    temporary.write_text("{\n" + ",\n".join(sections) + "\n}\n", encoding="utf-8")
    # Windows readers/virus scanners can temporarily hold the destination open.
    # A transient sharing violation must not strand a whole batch in the pool.
    for attempt in range(50):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.05)


def resume_rows(previous, report):
    """Reuse observations only when the entire frozen evaluation still matches."""
    for key in ("schema_version", "baseline_commit", "candidate_source_sha256", "baseline_source_sha256",
                "transport", "seed_plan", "variant_overrides", "evaluation_stage", "selected_configurations",
                "http_checks_per_stratum", "metric_definition", "physics", "bootstrap"):
        if key not in previous or previous[key] != report[key]:
            raise ValueError(f"Cannot resume: {key} changed or missing")
    if previous.get("complete"):
        raise ValueError("Cannot resume an already complete report")
    allowed = {(seed, d) for seed in report["seed_plan"]["seeds"]
               for d in report["seed_plan"]["directionals"]}
    keys = [(r["seed"], r["directional"]) for r in previous["rows"]]
    if len(set(keys)) != len(keys) or not set(keys) <= allowed:
        raise ValueError("Duplicate or unplanned pairs in resumed report")
    return previous["rows"]


def selected_configurations(bundles, directionals, variant):
    """Record the complete configuration before observing any episode result."""
    configs = {}
    for directional in directionals:
        configs[f"d{directional}"] = {}
        for arm in ("baseline", "candidate"):
            config = asdict(bundles[arm]["strategy"].StrategyConfig(directional_probe=directional > 0))
            overrides = overrides_for(variant, directional) if arm == "candidate" else {}
            if set(overrides) - set(config):
                raise ValueError(f"Unknown StrategyConfig fields: {sorted(set(overrides) - set(config))}")
            if "directional_probe" in overrides and overrides["directional_probe"] != (directional > 0):
                raise ValueError("directional_probe must match the evaluation stratum")
            config.update(overrides)
            configs[f"d{directional}"][arm] = config
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30, help="paired episodes per directional stratum")
    parser.add_argument("--seed-start", type=int, default=930000)
    parser.add_argument("--seeds", help="explicit comma-separated seeds, replacing episodes/seed-start")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="d83c40d")
    parser.add_argument("--resume", action="store_true", help="resume an incomplete report only with identical sources/configuration/seeds")
    parser.add_argument("--directionals", default="0,5,10,16")
    parser.add_argument("--variant", default="default", help="default, JSON object, or path to config override JSON")
    parser.add_argument("--transport", choices=("local", "http"), default="local")
    parser.add_argument("--http-checks", type=int, default=1, help="HTTP equivalence cases per stratum; 0 is development-only")
    parser.add_argument("--stage", choices=("development", "validation", "final"), default="development",
                        help="audit label; changing code/config after viewing results requires a fresh held-out seed set")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(args.seed_start, args.seed_start + args.episodes))
    directionals = [int(d) for d in args.directionals.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or args.workers < 1 or args.http_checks < 0:
        parser.error("require unique nonempty seeds, workers >= 1 and http-checks >= 0")
    if not directionals or len(set(directionals)) != len(directionals) or any(d not in (0, 5, 10, 16) for d in directionals):
        parser.error("directionals must be a unique nonempty subset of 0,5,10,16")
    if args.stage != "development" and args.transport == "local" and args.http_checks == 0:
        parser.error("validation/final local evaluations require at least one HTTP check per stratum")
    variant = {} if args.variant == "default" else json.loads(args.variant if args.variant.lstrip().startswith("{")
                                                            else Path(args.variant).read_text(encoding="utf-8-sig"))
    expected = len(seeds) * len(directionals)
    rows, checks = [], []
    with frozen_versions(args.baseline_ref) as (bundles, metadata):
        report = {"schema_version": 2, "complete": False, **metadata, "transport": args.transport,
                  "evaluation_stage": args.stage,
                  "http_checks_per_stratum": args.http_checks, "variant_overrides": variant,
                  "selected_configurations": selected_configurations(bundles, directionals, variant),
                  "seed_plan": {"seeds": seeds, "directionals": directionals,
                                "d16_count": 16, "other_counts": "generated 10..16, not passed to strategy"},
                  "metric_definition": "per-episode total virtual time / successful clear count; arithmetic mean across all episodes, including misses",
                  "physics": "baseline Arena and protocol dispatcher for both arms; local mock only",
                  "bootstrap": {"paired": True, "repeats": 2000, "seed": 20260911, "confidence": .95},
                  "rows": rows, "http_checks": checks}
        if args.resume:
            previous = json.loads(args.out.read_text(encoding="utf-8"))
            rows.extend(resume_rows(previous, report))
            report["resumed_pairs"] = len(rows)
        write_report(args.out, report)

        def pair_run(seed, directional):
            return {"seed": seed, "directional": directional,
                    "baseline": run_episode(bundles, seed, directional, "baseline", transport=args.transport),
                    "candidate": run_episode(bundles, seed, directional, "candidate",
                                               overrides_for(variant, directional), args.transport)}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            finished_pairs = {(r["seed"], r["directional"]) for r in rows}
            if len(finished_pairs) != len(rows):
                raise ValueError("Duplicate pairs in resumed report")
            futures = {pool.submit(pair_run, seed, d): (seed, d) for d in directionals for seed in seeds
                       if (seed, d) not in finished_pairs}
            for future in as_completed(futures):
                pair = future.result()
                pair["new_missed_channels"] = regressions(pair)
                rows.append(pair)
                rows.sort(key=lambda r: (r["directional"], r["seed"]))
                # Checkpoint in batches: rewriting a multi-MB JSON for every
                # episode dominates Windows execution and invites sharing locks.
                if len(rows) % 25 == 0 or len(rows) == expected:
                    write_report(args.out, report)
                print(f"{len(rows)}/{expected} d={pair['directional']} seed={pair['seed']} "
                      f"M={pair['baseline']['mean_clear_time_s']} -> {pair['candidate']['mean_clear_time_s']} "
                      f"clear={pair['baseline']['cleared']} -> {pair['candidate']['cleared']}", flush=True)
        if args.transport == "local":
            for d in directionals:
                for seed in seeds[:args.http_checks]:
                    local = next(r for r in rows if r["seed"] == seed and r["directional"] == d)
                    for arm in ("baseline", "candidate"):
                        http = run_episode(bundles, seed, d, arm,
                                           overrides_for(variant, d) if arm == "candidate" else {}, "http")
                        equivalent = (http["trace_sha256"] == local[arm]["trace_sha256"]
                                      and http["cleared_channels"] == local[arm]["cleared_channels"]
                                      and not http["exception"] and not http["aborted_events"] and http["finished"])
                        checks.append({"seed": seed, "directional": d, "arm": arm,
                                       "equivalent": equivalent, "http_result": http})
                        write_report(args.out, report)
        report["complete"] = len(rows) == expected
        report["aggregate"] = aggregate(rows)
        report["acceptance"] = gate(rows, expected, checks)
        report["release_validation"] = args.transport == "http" or (args.http_checks > 0 and bool(checks))
        report["targets"] = {key: {"target_s": 300 if key == "d0" else 500,
                                    "mean_s": value["candidate"]["mean_clear_time_s"]["mean"],
                                    "met": value["candidate"]["mean_clear_time_s"]["mean"] is not None
                                    and value["candidate"]["mean_clear_time_s"]["mean"] <= (300 if key == "d0" else 500),
                                    "full_clear_cases": value["candidate"]["full_clear_cases"],
                                    "episodes": value["episodes"]}
                             for key, value in report["aggregate"].items()}
        write_report(args.out, report)
        print(json.dumps({"acceptance": report["acceptance"], "targets": report["targets"]}, ensure_ascii=False, indent=2))
        return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
