"""顺路覆盖审计：现有策略的检测停车点，已经"顺手"覆盖了目标区域的多少？

动机（对齐 `feasible.py`）：每次 no_signal 检测都从目标区域挖掉一个半径 ≥1000 m 的
圆盘。策略在归航清除途中本来就要停很多次车，如果每次停车都顺手把还没定位的频道扫一
遍（"机会式搭载检测"，边际成本仅 5-6 s ≡ 25-30 m 位移），这些圆盘就是**免费**的覆盖。

本模块回放一局的检测停车点，统计：

* 只用原点 → 残余可行域比例（理论 ≈ 1 − 1000²/1800² ≈ 69%）；
* 用上全部停车点 → 残余可行域比例（衡量"搭载检测能免掉多少专门环扫"）；
* 补齐残余可行域所需的定向站位数与巡回长度（`feasible.cover_route`）。

用法：python -m src.b_sim.coverage_audit --episodes 12
"""

from __future__ import annotations

import argparse
import statistics
import threading

from .arena import Arena, generate_case
from .client import RobotClient
from .feasible import FeasibleSet, cover_route
from .route import path_length
from .server import Session, make_server
from .strategy import DogStrategy, StrategyConfig


def audit(seed: int, *, directional: int = 0,
          config: StrategyConfig | None = None) -> dict:
    case = generate_case(seed, directional_count=directional)
    arena = Arena(case)
    session = Session("audit", arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient("audit", base_url=f"http://127.0.0.1:{port}")
        DogStrategy(client, config).run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    stops = [(ev["x"], ev["y"]) for ev in arena.robot.trace if ev["action"] == "measure"]

    full = FeasibleSet()
    total_cells = len(full)

    origin_only = FeasibleSet()
    origin_only.apply_silence(0.0, 0.0)

    all_stops = FeasibleSet()
    for x, y in stops:
        all_stops.apply_silence(x, y)

    route = cover_route([all_stops], (0.0, 0.0), max_stations=12)
    return {
        "seed": seed,
        "stops": len(stops),
        "total_cells": total_cells,
        "alive_origin_only": len(origin_only),
        "alive_all_stops": len(all_stops),
        "residual_origin": len(origin_only) / total_cells,
        "residual_all_stops": len(all_stops) / total_cells,
        "cover_stations": len(route),
        "cover_route_m": path_length((0.0, 0.0), route),
        "move_distance_m": arena.stats()["move_distance_m"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="顺路覆盖审计")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--directional", type=int, default=0)
    args = parser.parse_args()
    seed_start = args.seed_start if args.seed_start is not None else (
        7001 if args.directional else 5000)
    config = StrategyConfig(directional_probe=args.directional > 0)

    rows = [audit(seed_start + i, directional=args.directional, config=config)
            for i in range(args.episodes)]
    print(f"{'seed':>6} {'停车':>5} {'仅原点残余':>11} {'全停车点残余':>13} "
          f"{'补齐站位':>9} {'补齐巡回m':>10}")
    for r in rows:
        print(f"{r['seed']:>6} {r['stops']:>5} {r['residual_origin']:>10.1%} "
              f"{r['residual_all_stops']:>12.1%} {r['cover_stations']:>9} "
              f"{r['cover_route_m']:>10.0f}")
    print("\n== 均值 ==")
    print(f"停车点数                 {statistics.fmean(r['stops'] for r in rows):8.1f}")
    print(f"仅原点残余可行域         {statistics.fmean(r['residual_origin'] for r in rows):8.1%}")
    print(f"全停车点残余可行域       {statistics.fmean(r['residual_all_stops'] for r in rows):8.1%}")
    print(f"补齐所需站位             {statistics.fmean(r['cover_stations'] for r in rows):8.1f}")
    print(f"补齐巡回长度             {statistics.fmean(r['cover_route_m'] for r in rows):8.0f} m")
    print(f"（对照）实际总移动距离   {statistics.fmean(r['move_distance_m'] for r in rows):8.0f} m")


if __name__ == "__main__":
    main()
