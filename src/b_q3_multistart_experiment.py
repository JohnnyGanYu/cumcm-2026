"""Offline Q3 development experiment: alternate NN starts for open tours.

Only the experiment worker's strategy-module route binding is replaced; the
original binding is restored in a finally block. Physics and budgets are intact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def multistart_route(start, points, *, improve=True, max_points=24,
                     starts=4, or_each=True):
    from src.b_sim.route import (distance, nearest_neighbour, order_route,
                                or_opt, path_length, two_opt)
    # Retain the actual legacy result as an incumbent: alternate starts can
    # never make the planned open tour longer for the same observed waypoints.
    incumbent = order_route(start, points, improve=improve, max_points=max_points)
    if not improve or not 2 < len(incumbent) <= max_points:
        return incumbent
    best_cost = path_length(start, incumbent)
    unique = sorted(incumbent)
    firsts = sorted(unique, key=lambda p: (distance(start, p), p))
    if starts > 0:
        firsts = firsts[:starts]
    for first in firsts:
        route = [first, *nearest_neighbour(first, [p for p in unique if p != first])]
        route = two_opt(start, route)
        if or_each:
            route = or_opt(start, route)
        cost = path_length(start, route)
        if cost < best_cost - 1e-9:
            incumbent, best_cost = route, cost
    if not or_each:
        incumbent = or_opt(start, incumbent)
    return incumbent


def episode(seed, variant):
    from src.b_eval.dev_measure import InProcessClient
    from src.b_sim.arena import Arena, generate_case
    import src.b_sim.strategy as strategy_module
    original = strategy_module.order_route
    calls = 0
    route_wall = 0.0
    planned_saved = 0.0
    options = {
        "baseline": None,
        "ms4": (4, True),
        "ms8": (8, True),
        "msall2opt": (0, False),
    }
    mode = options[variant]

    def altered(start, points, **kwargs):
        nonlocal calls, route_wall, planned_saved
        from src.b_sim.route import path_length
        begin = time.monotonic()
        base = original(start, points, **kwargs)
        result = multistart_route(start, points, starts=mode[0], or_each=mode[1], **kwargs)
        calls += 1
        route_wall += time.monotonic() - begin
        planned_saved += path_length(start, base) - path_length(start, result)
        return result

    arena = Arena(generate_case(seed, directional_count=0))
    strategy = strategy_module.DogStrategy(InProcessClient(arena))
    started = time.monotonic()
    try:
        if mode is not None:
            strategy_module.order_route = altered
        strategy.run()
    finally:
        strategy_module.order_route = original
    return dict(seed=seed, variant=variant, **arena.stats(),
                wall_s=time.monotonic()-started, route_calls=calls,
                route_wall_s=route_wall, planned_saved_m=planned_saved,
                cover_visits=strategy.cover_visits,
                missed_channels=[j.channel for j in arena.case.jammers if not j.cleared])


def aggregate(rows):
    result = {}
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        good = [row for row in group if "error" not in row]
        result[variant] = {
            "episodes": len(group), "errors": len(group)-len(good),
            "total_jammers": sum(row["total_jammers"] for row in good),
            "cleared": sum(row["cleared"] for row in good),
            "full_clear_episodes": sum(not row["missed_channels"] for row in good),
            **{key: statistics.mean(row[key] for row in good) if good else None
               for key in ("mean_clear_time_s", "move_distance_m", "measure_calls",
                           "wall_s", "route_wall_s", "cover_visits")},
        }
    return result


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(7001,7009)))
    ap.add_argument("--variants", nargs="+", default=["baseline","ms4","ms8","msall2opt"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", default="results/q3_220_multistart_development.json")
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(episode(args.seeds[0], args.variants[0]), ensure_ascii=True))
        return
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    sources = [Path(p) for p in ("src/b_sim/strategy.py", "src/b_sim/arena.py",
                                "src/b_sim/route.py", __file__)]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    frozen = hashes()
    tasks = [(seed, variant) for seed in args.seeds for variant in args.variants]
    rows = []
    output = Path(args.out)
    def work(task):
        seed, variant = task
        try:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "src.b_q3_multistart_experiment",
                 "--worker", "--seeds", str(seed), "--variants", variant],
                capture_output=True, text=True, encoding="utf-8", check=True,
                timeout=args.timeout, env=env)
            return json.loads(proc.stdout)
        except Exception as exc:
            return {"seed": seed, "variant": variant, "error": str(exc),
                    "stderr": getattr(exc, "stderr", None)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work, task) for task in tasks]):
            row = future.result()
            rows.append(row)
            print(json.dumps({k:row.get(k) for k in (
                "seed","variant","mean_clear_time_s","cleared","total_jammers","wall_s","error")}), flush=True)
            payload = {"seeds":args.seeds, "variants":args.variants,
                       "expected_records":len(tasks), "completed_records":len(rows),
                       "source_sha256":frozen, "source_unchanged":hashes()==frozen,
                       "records":sorted(rows,key=lambda r:(r["seed"],r["variant"])),
                       "summary":aggregate(rows)}
            output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
            if not payload["source_unchanged"]:
                raise RuntimeError("Runtime source changed during evaluation")


if __name__ == "__main__":
    main()
