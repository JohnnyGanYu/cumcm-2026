"""边界场景专项：用当前交付策略重测六档（源数 x 定向占比）。

RACEDAY 第 3 节那张表是赛日判断「T̄ 是否正常、要不要重打」的唯一依据，
但原表数字无对应结果文件、也未标注策略版本，量级明显属旧策略时期
（Q4 全定向 1168.4 s，而 v1.0.5 洁净环境实测约 410 s）。
照旧表判断会把真实退化当成正常。这里用交付策略重测并落盘 JSON。
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.q3_fast import Q3FastStrategy
from .b_sim.q4_fast import Q4FastStrategy
from .b_sim.strategy import StrategyConfig

ROBOT_ID = "202600000000"
OUT = Path("results/b_boundary_matrix.json")

# (总源数, 定向源数) —— 取源数与定向占比的两端，对应 RACEDAY 第 3 节六档
CELLS = ((16, 0), (10, 0), (16, 8), (16, 16), (10, 5), (10, 10))


def _serve(arena, port: int):
    """起一个真实 HTTP mock server。

    不要用「就地覆写 client._post」的捷径：client 的重试、计数与协议校验都在
    HTTP 层之上，覆写会让 _seq 不递增、策略以为已扫完，整局零检测且不报错
    （实测 run() 在 0.006 s 内静默返回）。走真实服务才是官方链路。
    """
    from .b_sim.server import Session, make_server

    session = Session(robot_id=ROBOT_ID, arena=arena)
    httpd = make_server(session, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd

def one_run(seed: int, total: int, directional: int, port: int) -> dict:
    case = generate_case(seed=seed, count=total, directional_count=directional)
    arena = Arena(case)
    httpd = _serve(arena, port)
    try:
        cfg = StrategyConfig(directional_probe=directional > 0)
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        cls = Q4FastStrategy if directional > 0 else Q3FastStrategy
        strategy = cls(client, cfg)
        t0 = time.time()
        summary = strategy.run()
        wall = round(time.time() - t0, 3)
    finally:
        httpd.shutdown()
        httpd.server_close()
    return {
        "seed": seed,
        "total_jammers": total,
        "directional": directional,
        "cleared": summary["cleared_count"],
        "virtual_time_s": summary["virtual_time_s"],
        "mean_clear_time_s": summary["mean_clear_time_s"],
        "measure_calls": summary["measure_calls"],
        "clear_calls": summary["clear_calls"],
        "strategy_profile": summary.get("strategy_profile"),
        "wall_s": wall,
    }

def aggregate(rows: list[dict]) -> dict:
    """池化 T̄ = 总虚拟时间 / 总清除数，与官方评分口径一致。"""
    cleared = sum(r["cleared"] for r in rows)
    virtual = sum(r["virtual_time_s"] for r in rows)
    expected = sum(r["total_jammers"] for r in rows)
    return {
        "runs": len(rows),
        "cleared": cleared,
        "expected": expected,
        "clear_ratio": round(cleared / expected, 6) if expected else None,
        "pooled_mean_clear_time_s": round(virtual / cleared, 2) if cleared else None,
        "mean_measure_calls": round(sum(r["measure_calls"] for r in rows) / len(rows), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="边界场景专项：六档批量评测")
    ap.add_argument("--episodes", type=int, default=40, help="每档局数")
    ap.add_argument("--seed-base", type=int, default=21000)
    ap.add_argument("--port-base", type=int, default=2400)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    payload = {
        "episodes_per_cell": args.episodes,
        "seed_base": args.seed_base,
        "cells": [],
    }
    port = args.port_base
    for total, directional in CELLS:
        rows = []
        for k in range(args.episodes):
            seed = args.seed_base + total * 1000 + directional * 100 + k
            rows.append(one_run(seed, total, directional, port))
            port += 1
        agg = aggregate(rows)
        agg["total_jammers"] = total
        agg["directional"] = directional
        agg["strategy_profile"] = rows[0]["strategy_profile"]
        payload["cells"].append({"summary": agg, "runs": rows})
        label = f"{total} 源 / {directional} 定向"
        print(f'{label:18s} 清除 {agg["cleared"]}/{agg["expected"]}'
              f'  池化 T̄ {agg["pooled_mean_clear_time_s"]} s'
              f'  检测 {agg["mean_measure_calls"]}'
              f'  profile {agg["strategy_profile"]}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    pooled = [c["summary"]["pooled_mean_clear_time_s"] for c in payload["cells"]]
    ratios = [c["summary"]["clear_ratio"] for c in payload["cells"]]
    print()
    print(f"指标区间 {min(pooled)} – {max(pooled)} s，跨度 {max(pooled)/min(pooled):.1f} 倍")
    print(f"最低清除率 {min(ratios):.4f}")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
