"""批量离线演练：在本地 mock 模拟器上跑策略，统计清除率与平均定位清除时间。

用法：
  python -m src.b_sim.rehearse --episodes 30                # 问题 3（全向）
  python -m src.b_sim.rehearse --episodes 30 --directional 5 # 问题 4（含定向）

输出 results/b_rehearsal_*.json，含每局种子、真值、统计与失败频道，便于定位策略缺陷。
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

from .arena import Arena, generate_case
from .client import RobotClient
from .server import Session, make_server
from .strategy import DogStrategy, StrategyConfig

ROBOT_ID = "rehearsal"


def run_episode(seed: int, *, directional: int = 0, count: int | None = None,
                config: StrategyConfig | None = None, real_limit_s: float = 1200.0) -> dict:
    """跑一局：起 mock 服务 -> 执行策略 -> 收集真值对照。"""
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=real_limit_s)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strategy = DogStrategy(client, config)
        summary = strategy.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    wall_s = time.monotonic() - started

    stats = arena.stats()
    truth = case.truth_table()
    missed = [row["channel"] for row in truth if not row["cleared"]]
    return {
        "seed": seed,
        "total_jammers": stats["total_jammers"],
        "directional_jammers": stats["directional_jammers"],
        "cleared": stats["cleared"],
        "cleared_ratio": stats["cleared_ratio"],
        "total_virtual_time_s": round(stats["total_virtual_time_s"], 3),
        "mean_clear_time_s": None if stats["mean_clear_time_s"] is None else round(stats["mean_clear_time_s"], 3),
        "measure_calls": stats["measure_calls"],
        "clear_calls": stats["clear_calls"],
        "move_distance_m": round(stats["move_distance_m"], 1),
        "program_wall_s": round(wall_s, 2),
        "missed_channels": missed,
        "events": summary["events"],
        "truth": truth,
    }


def aggregate(episodes: list[dict]) -> dict:
    ratios = [e["cleared_ratio"] for e in episodes]
    means = [e["mean_clear_time_s"] for e in episodes if e["mean_clear_time_s"] is not None]
    virtual = [e["total_virtual_time_s"] for e in episodes]
    wall = [e["program_wall_s"] for e in episodes]
    full = sum(1 for e in episodes if e["cleared_ratio"] >= 1.0 - 1e-9)
    return {
        "episodes": len(episodes),
        "full_clear_episodes": full,
        "full_clear_rate": full / len(episodes) if episodes else 0.0,
        "cleared_ratio_mean": round(statistics.fmean(ratios), 4) if ratios else 0.0,
        "cleared_ratio_min": round(min(ratios), 4) if ratios else 0.0,
        "mean_clear_time_s_mean": round(statistics.fmean(means), 2) if means else None,
        "mean_clear_time_s_max": round(max(means), 2) if means else None,
        "total_virtual_time_s_mean": round(statistics.fmean(virtual), 1) if virtual else None,
        "total_virtual_time_s_max": round(max(virtual), 1) if virtual else None,
        "program_wall_s_mean": round(statistics.fmean(wall), 2) if wall else None,
        "program_wall_s_max": round(max(wall), 2) if wall else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B 题离线批量演练")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--directional", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--real-limit", type=float, default=1200.0)
    parser.add_argument("--target-diameter", type=float, default=None)
    parser.add_argument("--scan-stations", type=int, default=None)
    parser.add_argument("--scan-radius", type=float, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    config = StrategyConfig(directional_probe=args.directional > 0)
    if args.target_diameter is not None:
        config.target_diameter_m = args.target_diameter
    if args.scan_stations is not None:
        config.scan_stations = args.scan_stations
    if args.scan_radius is not None:
        config.scan_ring_radius = args.scan_radius

    episodes = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        episode = run_episode(seed, directional=args.directional, count=args.count,
                              config=config, real_limit_s=args.real_limit)
        episodes.append(episode)
        print(f"seed={seed} total={episode['total_jammers']} "
              f"cleared={episode['cleared']} ratio={episode['cleared_ratio']:.3f} "
              f"mean_t={episode['mean_clear_time_s']} virt={episode['total_virtual_time_s']:.0f}s "
              f"wall={episode['program_wall_s']}s missed={episode['missed_channels']}", flush=True)

    summary = aggregate(episodes)
    print("\n== aggregate ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    label = args.label or (f"q4_dir{args.directional}" if args.directional else "q3")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"b_rehearsal_{label}.json"
    payload = {
        "config": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "directional": args.directional,
            "count": args.count,
            "real_limit_s": args.real_limit,
            "strategy": vars(config),
        },
        "summary": summary,
        "episodes": episodes,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
