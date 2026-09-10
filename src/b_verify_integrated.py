"""验证集成版：确认 strategy.py 里的 `_revive_nearby` 复现了实验脚本测到的收益。

对照方式：同一份 `DogStrategy`，只用配置开关区分
  off : nearby_revive_radius_m = 0.0   （等价于集成前的行为）
  on  : nearby_revive_radius_m = 900.0 （默认值）

这比"实验子类 vs 原类"更严格：两臂走的是完全同一份代码路径，
排除了子类改写引入差异的可能。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-verify"


def run_one(seed: int, *, directional: int, radius: float,
            count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional > 0,
                         nearby_revive_radius_m=radius)
    try:
        client = RobotClient(ROBOT_ID, base_url="http://127.0.0.1:" + str(port))
        strat = DogStrategy(client, cfg)
        strat.run()
        revives = sum(strat._revive_counts.values())
    finally:
        httpd.shutdown()
        httpd.server_close()
    s = arena.stats()
    return {"seed": seed, "arm": "on" if radius > 0 else "off",
            "total": s["total_jammers"], "cleared": s["cleared"],
            "ratio": s["cleared_ratio"], "virtual_s": s["total_virtual_time_s"],
            "mean_clear_time_s": s["mean_clear_time_s"],
            "move_m": s["move_distance_m"], "revives": revives,
            "missed": [r["channel"] for r in case.truth_table() if not r["cleared"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=91000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--label", default="int")
    args = ap.parse_args()

    rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        for radius in (0.0, 900.0):
            r = run_one(seed, directional=args.directional, radius=radius,
                        count=args.count)
            rows.append(r)
            print(f"seed={seed} {r['arm']:3s} cleared={r['cleared']}/{r['total']} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"revives={r['revives']} missed={r['missed']}", flush=True)

    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["arm"]] = r
    d = [p["on"]["mean_clear_time_s"] - p["off"]["mean_clear_time_s"]
         for p in pairs.values()
         if p.get("on", {}).get("mean_clear_time_s") and p.get("off", {}).get("mean_clear_time_s")]
    base = st.fmean(p["off"]["mean_clear_time_s"] for p in pairs.values()
                    if p.get("off", {}).get("mean_clear_time_s"))
    n = len(d)
    m = st.fmean(d)
    se = st.stdev(d) / math.sqrt(n) if n > 1 else 0.0
    report = {
        "episodes": n,
        "off_mean_clear_time_s": round(base, 1),
        "on_mean_clear_time_s": round(base + m, 1),
        "d_mean_clear_time_s": round(m, 1),
        "d_pct": round(m / base * 100, 2),
        "median_s": round(st.median(d), 1),
        "ci95": [round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)] if se else None,
        "t": round(m / se, 2) if se else None,
        "better": sum(1 for x in d if x < -1),
        "worse": sum(1 for x in d if x > 1),
        "tie": sum(1 for x in d if abs(x) <= 1),
        "missed_off": sum(len(p["off"]["missed"]) for p in pairs.values() if "off" in p),
        "missed_on": sum(len(p["on"]["missed"]) for p in pairs.values() if "on" in p),
        "full_clear_off": round(sum(1 for p in pairs.values() if p["off"]["ratio"] >= 1 - 1e-9) / len(pairs), 4),
        "full_clear_on": round(sum(1 for p in pairs.values() if p["on"]["ratio"] >= 1 - 1e-9) / len(pairs), 4),
        "mean_revives_on": round(st.fmean(p["on"]["revives"] for p in pairs.values()), 2),
    }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_verify_integrated_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
