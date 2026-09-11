"""按策略阶段拆分移动距离：定位"绕路"到底发生在哪个阶段。

`timeline.py` 回答"时间花在哪一类动作上"（答案：移动约 80%）。本模块回答下一个
问题——"这些移动发生在哪个阶段"。做法是在 `DogStrategy` 的阶段入口打标记，记录
`arena.robot.trace` 的下标区间，再按区间累加距离。

用法：python -m src.b_sim.phase_probe --episodes 12
"""

from __future__ import annotations

import argparse
import math
import statistics
import threading
from collections import defaultdict

from .arena import SPEED_MPS, Arena, generate_case
from .client import RobotClient
from .server import Session, make_server
from .strategy import DogStrategy, StrategyConfig

PHASES = ["origin_sweep", "homing", "cover_station", "piggyback",
          "probe_ring", "inner_retry", "belief_search", "blind_clear"]


def profile(seed: int, *, directional: int = 0,
            config: StrategyConfig | None = None) -> dict:
    case = generate_case(seed, directional_count=directional)
    arena = Arena(case)
    session = Session("phase", arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    marks: list[tuple[int, str]] = []

    def wrap(name: str, func):
        def inner(*a, **kw):
            marks.append((len(arena.robot.trace), name))
            try:
                return func(*a, **kw)
            finally:
                marks.append((len(arena.robot.trace), "__end__"))
        return inner

    try:
        client = RobotClient("phase", base_url=f"http://127.0.0.1:{port}")
        strategy = DogStrategy(client, config)
        strategy._sweep_at = wrap("origin_sweep", strategy._sweep_at)
        strategy._home_channel = wrap("homing", strategy._home_channel)
        strategy._visit_cover_station = wrap("cover_station", strategy._visit_cover_station)
        strategy._piggyback = wrap("piggyback", strategy._piggyback)
        strategy._probe_ring = wrap("probe_ring", strategy._probe_ring)
        strategy._inner_retry = wrap("inner_retry", strategy._inner_retry)
        strategy._belief_search = wrap("belief_search", strategy._belief_search)
        strategy._blind_clear_sweep = wrap("blind_clear", strategy._blind_clear_sweep)
        strategy.run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    # 把 trace 下标映射到"最内层未结束的阶段"
    owner: list[str] = ["<none>"] * len(arena.robot.trace)
    stack: list[str] = []
    cursor = 0
    for index, name in marks:
        for k in range(cursor, min(index, len(owner))):
            owner[k] = stack[-1] if stack else "<none>"
        cursor = max(cursor, min(index, len(owner)))
        if name == "__end__":
            if stack:
                stack.pop()
        else:
            stack.append(name)
    for k in range(cursor, len(owner)):
        owner[k] = stack[-1] if stack else "<none>"

    dist: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    x, y = 0.0, 0.0
    for k, ev in enumerate(arena.robot.trace):
        dist[owner[k]] += math.hypot(ev["x"] - x, ev["y"] - y)
        calls[owner[k]] += 1
        x, y = ev["x"], ev["y"]
    stats = arena.stats()
    return {"seed": seed, "dist": dict(dist), "calls": dict(calls),
            "total_dist": stats["move_distance_m"],
            "mean_t": stats["mean_clear_time_s"],
            "cleared": stats["cleared"], "jammers": stats["total_jammers"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="按阶段拆分移动距离")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--directional", type=int, default=0)
    args = parser.parse_args()
    seed_start = args.seed_start if args.seed_start is not None else (
        7001 if args.directional else 5000)
    config = StrategyConfig(directional_probe=args.directional > 0)

    rows = [profile(seed_start + i, directional=args.directional, config=config)
            for i in range(args.episodes)]
    keys = PHASES + ["<none>"]
    print(f"{'seed':>6} {'T̄':>6} {'总距离':>8} " + " ".join(f"{k:>13}" for k in keys))
    for r in rows:
        print(f"{r['seed']:>6} {(r['mean_t'] or 0):>6.0f} {r['total_dist']:>8.0f} "
              + " ".join(f"{r['dist'].get(k, 0.0):>13.0f}" for k in keys))
    print("\n== 均值（米 / 占比 / 折算秒）==")
    total = statistics.fmean(r["total_dist"] for r in rows)
    for k in keys:
        m = statistics.fmean(r["dist"].get(k, 0.0) for r in rows)
        c = statistics.fmean(r["calls"].get(k, 0) for r in rows)
        if m or c:
            print(f"{k:>14}: {m:9.0f} m  {m / total * 100:5.1f}%  "
                  f"{m / SPEED_MPS:7.0f} s  动作 {c:5.1f} 次")
    print(f"{'合计':>14}: {total:9.0f} m         {total / SPEED_MPS:7.0f} s")


if __name__ == "__main__":
    main()
