"""批量离线演练：在本地 mock 模拟器上跑策略，统计清除率与平均定位清除时间。

用法：
  python -m src.b_sim.rehearse --episodes 30                # 问题 3（全向）
  python -m src.b_sim.rehearse --episodes 30 --directional 5 # 问题 4（含定向）

输出 results/b_rehearsal_*.json，含每局种子、真值、统计与失败频道，便于定位策略缺陷。

批量规模（R6.3）：Q3（全向）至少跑 60 局（`--episodes 60`），Q4（含定向源）至少
跑 40 局（`--episodes 40 --directional N`）。`aggregate()` 汇总平均定位清除时间 `T̄`
（`mean_clear_time_s_mean`）、清除率（`clear_rate` = 累计已清除 / 真实源总数，另有
逐局全清率 `full_clear_rate`）与程序运行时长（`program_wall_s_mean/_max`）。

官方校准流程（R6.4 / R6.5）：本地 mock 仿真偏乐观（时间更短、检测更少）。已测得
离线↔官方在评分项 `T̄` 上的相对偏差 Q3 ≈ 5.4%、Q4 ≈ 9.6%（详见
`official_runs/README.md`）。据此在正式测试的时间预期上按官方偏大值留保守余量。
任何离线与官方实测不一致处，一律**以官方模拟器实测为最终判据**，并据此重新校准
离线仿真参数——本地仿真只用于无限次调参，不作最终判定。
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
    """汇总一批彩排局的评分项（R6.3）。

    产出字段（供 official_runs 校准与论文 Table 1 对齐）：
      - episodes / full_clear_episodes / full_clear_rate：局数与全清率（清除率）
      - cleared_total / total_jammers_total：累计已清除个数与真实源总数
      - cleared_ratio_mean / cleared_ratio_min：逐局清除率的均值/下界
      - mean_clear_time_s_mean / _max：平均定位清除时间 T̄（评分项）的均值/最坏
      - total_virtual_time_s_mean / _max：虚拟时间
      - program_wall_s_mean / _max：程序运行时长（现实秒，上限 1200 s）
    """
    ratios = [e["cleared_ratio"] for e in episodes]
    means = [e["mean_clear_time_s"] for e in episodes if e["mean_clear_time_s"] is not None]
    virtual = [e["total_virtual_time_s"] for e in episodes]
    wall = [e["program_wall_s"] for e in episodes]
    cleared_total = sum(e["cleared"] for e in episodes)
    total_jammers = sum(e["total_jammers"] for e in episodes)
    full = sum(1 for e in episodes if e["cleared_ratio"] >= 1.0 - 1e-9)
    return {
        "episodes": len(episodes),
        "full_clear_episodes": full,
        "full_clear_rate": full / len(episodes) if episodes else 0.0,
        "cleared_total": cleared_total,
        "total_jammers_total": total_jammers,
        "clear_rate": round(cleared_total / total_jammers, 4) if total_jammers else 0.0,
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
    parser.add_argument("--out", default=None, help="直接指定输出文件路径")
    parser.add_argument("--outer-ring", dest="outer_ring", action="store_true",
                        help="启用外环补扫（救近边界定向源）")
    parser.add_argument("--no-outer-ring", dest="outer_ring", action="store_false")
    parser.set_defaults(outer_ring=None)
    parser.add_argument("--outer-radius", type=float, default=None)
    parser.add_argument("--outer-stations", type=int, default=None)
    parser.add_argument("--home-batch", type=int, default=None,
                        help="扫过几个站位后统一归航（1=逐站归航）")
    parser.add_argument("--approach", type=float, default=None,
                        help="每次前进占剩余距离的比例（默认0.62）")
    parser.add_argument("--lateral", type=float, default=None,
                        help="前进时横向偏移比例（默认0.16）")
    args = parser.parse_args()

    config = StrategyConfig(directional_probe=args.directional > 0)
    if args.target_diameter is not None:
        config.target_diameter_m = args.target_diameter
    if args.scan_stations is not None:
        config.scan_stations = args.scan_stations
    if args.scan_radius is not None:
        config.scan_ring_radius = args.scan_radius
    if args.outer_ring is not None:
        config.use_outer_ring = args.outer_ring
    if args.outer_radius is not None:
        config.outer_ring_radius = args.outer_radius
    if args.outer_stations is not None:
        config.outer_ring_stations = args.outer_stations
    if args.home_batch is not None:
        config.home_batch_stations = args.home_batch
    if args.approach is not None:
        config.approach_fraction = args.approach
    if args.lateral is not None:
        config.lateral_fraction = args.lateral

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
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
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
