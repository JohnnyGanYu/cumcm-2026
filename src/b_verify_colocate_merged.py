"""独立验证：生产路径里已合入的 colocate 是否真的生效。

合入生产的是 strategy.py 内联实现 + colocate_probe 开关，而我此前验证的是
src/b_colocate.py 的子类覆盖实现。两者代码不同，必须直接对生产路径做开关 A/B，
不能拿子类结论替代。

用法:
  python -m src.b_verify_colocate_merged --episodes 40 --directional 0 --label q3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from scipy import stats as sps

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-verify"


def run_one(seed: int, *, directional: int, colo: bool,
            count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        cfg = StrategyConfig(directional_probe=directional > 0,
                             colocate_probe=colo)
        strat = DogStrategy(client, cfg)
        strat.run()
        s = arena.stats()
        return {
            "seed": seed,
            "cleared": s["cleared"],
            "total": s["total_jammers"],
            "clear_ratio": s["cleared_ratio"],
            "mean_clear_time_s": s["mean_clear_time_s"],
            "move_m": s["move_distance_m"],
            "measures": s["measure_calls"],
            "colo_stops": getattr(strat, "colo_stops", 0),
            "colo_measures": getattr(strat, "colo_measures", 0),
            "colo_new_estimates": getattr(strat, "colo_new_estimates", 0),
        }
    finally:
        httpd.shutdown()
        httpd.server_close()


def paired(base: list[dict], var: list[dict], key: str) -> dict:
    pairs = [(b[key], v[key]) for b, v in zip(base, var)
             if b[key] is not None and v[key] is not None]
    d = [v - b for b, v in pairs]
    n = len(d)
    if n < 2:
        return {"n_pairs": n}
    bm = st.fmean(b for b, _ in pairs)
    dm = st.fmean(d)
    sd = st.stdev(d)
    t_res = sps.ttest_rel([v for _, v in pairs], [b for b, _ in pairs])
    try:
        w_p = float(sps.wilcoxon([v for _, v in pairs], [b for b, _ in pairs]).pvalue)
    except ValueError:
        w_p = 1.0
    half = 1.959963985 * sd / math.sqrt(n) if n else 0.0
    return {
        "n_pairs": n,
        "baseline_mean": round(bm, 1),
        "d_mean": round(dm, 1),
        "d_pct": round(dm / bm * 100, 2) if bm else None,
        "ci95": [round(dm - half, 1), round(dm + half, 1)],
        "t": round(float(t_res.statistic), 3),
        "p_ttest": float(f"{float(t_res.pvalue):.3g}"),
        "p_wilcoxon": float(f"{w_p:.3g}"),
        "better": sum(1 for x in d if x < 0),
        "worse": sum(1 for x in d if x > 0),
        "tie": sum(1 for x in d if x == 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=350100)
    ap.add_argument("--label", default="verify")
    a = ap.parse_args()

    seeds = [a.seed_start + i for i in range(a.episodes)]
    off, on = [], []
    for sd in seeds:
        r_off = run_one(sd, directional=a.directional, colo=False, count=a.count)
        r_on = run_one(sd, directional=a.directional, colo=True, count=a.count)
        off.append(r_off)
        on.append(r_on)
        print(f"seed={sd} OFF meanT={r_off['mean_clear_time_s']:.0f} move={r_off['move_m']:.0f} "
              f"cl={r_off['cleared']}/{r_off['total']} | "
              f"ON meanT={r_on['mean_clear_time_s']:.0f} move={r_on['move_m']:.0f} "
              f"cl={r_on['cleared']}/{r_on['total']} est+={r_on['colo_new_estimates']}", flush=True)

    def agg(rows):
        return {
            "episodes": len(rows),
            "clear_rate": round(st.fmean(r["clear_ratio"] for r in rows), 4),
            "full_clear": sum(1 for r in rows if r["cleared"] == r["total"]),
            "meanT": round(st.fmean(r["mean_clear_time_s"] for r in rows
                                    if r["mean_clear_time_s"] is not None), 1),
            "move_m": round(st.fmean(r["move_m"] for r in rows), 1),
            "measures": round(st.fmean(r["measures"] for r in rows), 1),
            "colo_new_estimates_total": sum(r["colo_new_estimates"] for r in rows),
        }

    report = {
        "config": {"directional": a.directional, "count": a.count,
                   "seed_start": a.seed_start, "episodes": a.episodes},
        "colocate_off": agg(off),
        "colocate_on": agg(on),
        "paired_on_vs_off": {k: paired(off, on, k) for k in
                             ("mean_clear_time_s", "move_m", "cleared", "measures")},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_verify_colocate_merged_{a.label}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"report": report, "off": off, "on": on}, fh,
                  ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
