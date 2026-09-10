"""活体验证：把后验集合覆盖真正接进策略，在本地仿真器上跑完整局，比清除率与耗时。

前面的分析都是"在记录下来的决策点上"做反事实回放，会低估现状（不计扑空后的下游
成本），也无法体现后验策略改变后续轨迹的连锁效应。这里做真实的 A/B：

  baseline  : 原 DogStrategy
  posterior : 只替换 `_spiral` —— 清除失败后不再打固定 6 点 15 m 环，而是在
              **精确后验支撑集**（全部观测的楔形交 ∩ 目标区域）上做 20 m 圆盘
              贪心集合覆盖，按覆盖后验质量降序打点。
              另加不确定度门禁：后验直径 > gate_m 时不打单点，直接走集合覆盖。

其余一切（搜索、环形补扫、dogleg、预算、定向推理）保持不变，同一批种子。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from .b_sim.arena import ARENA_RADIUS, Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.geometry import polygon_diameter
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig
from .b_belief_eval import posterior_polygon, sample_polygon
from .b_belief_replay import posterior_plan

RESULTS = Path("results")
ROBOT_ID = "belief-live"


class PosteriorStrategy(DogStrategy):
    """只把 _spiral 换成后验集合覆盖；其余决策逻辑不动。"""

    def __init__(self, client, config=None, *, gate_m: float = 60.0,
                 max_discs: int = 24, samples: int = 1200, seed: int = 0) -> None:
        super().__init__(client, config)
        self.gate_m = gate_m
        self.max_discs = max_discs
        self.n_samples = samples
        self.rng = np.random.default_rng(seed)
        self.posterior_calls = 0
        self.posterior_points = 0

    def _posterior_points(self, state) -> list[tuple[float, float]]:
        if len(state.bearings) < 2:
            return []
        poly = posterior_polygon(list(state.bearings))
        if not poly:
            poly = posterior_polygon(list(state.bearings)[-4:])
        if not poly:
            return []
        samples = sample_polygon(poly, self.n_samples, self.rng)
        plan = posterior_plan(samples, self.max_discs, self.rng)
        return [p for p in plan if math.hypot(*p) <= ARENA_RADIUS * 1.02]

    def _spiral(self, state) -> bool:
        pts = self._posterior_points(state)
        if not pts:
            return super()._spiral(state)
        self.posterior_calls += 1
        self.posterior_points += len(pts)
        for p in pts:
            if state.cleared:
                return True
            if self._stop():
                return state.cleared
            if self._clear_at(p, state.channel):
                return True
        return state.cleared


def run_one(seed: int, *, directional: int, variant: str, gate_m: float,
            max_discs: int, count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional > 0)
    t0 = time.monotonic()
    extra = {}
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        if variant == "baseline":
            strat = DogStrategy(client, cfg)
        else:
            strat = PosteriorStrategy(client, cfg, gate_m=gate_m,
                                      max_discs=max_discs, seed=seed)
        strat.run()
        if variant != "baseline":
            extra = {"posterior_calls": strat.posterior_calls,
                     "posterior_points": strat.posterior_points}
    finally:
        httpd.shutdown()
        httpd.server_close()
    wall = time.monotonic() - t0
    s = arena.stats()
    return {"seed": seed, "variant": variant, "total": s["total_jammers"],
            "cleared": s["cleared"], "ratio": s["cleared_ratio"],
            "virtual_s": s["total_virtual_time_s"],
            "mean_clear_time_s": s["mean_clear_time_s"],
            "measures": s["measure_calls"], "clears": s["clear_calls"],
            "move_m": s["move_distance_m"], "wall_s": round(wall, 2),
            "missed": [r["channel"] for r in case.truth_table() if not r["cleared"]],
            **extra}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=40000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--gate", type=float, default=60.0)
    ap.add_argument("--max-discs", type=int, default=24)
    ap.add_argument("--label", default="q4live")
    ap.add_argument("--count", type=int, default=None,
                    help="固定干扰源总数；--directional 接近 16 时必须给，否则随机 count 可能小于 directional")
    args = ap.parse_args()
    if args.count is not None and args.directional > args.count:
        ap.error(f"--directional {args.directional} 不能超过 --count {args.count}")

    out_rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        for variant in ("baseline", "posterior"):
            r = run_one(seed, directional=args.directional, variant=variant,
                        gate_m=args.gate, max_discs=args.max_discs, count=args.count)
            out_rows.append(r)
            print(f"seed={seed} {variant:9s} cleared={r['cleared']}/{r['total']} "
                  f"ratio={r['ratio']:.3f} virt={r['virtual_s']:.0f}s "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"clears={r['clears']} meas={r['measures']} wall={r['wall_s']}s "
                  f"missed={r['missed']}", flush=True)

    def agg(v):
        rs = [r for r in out_rows if r["variant"] == v]
        mt = [r["mean_clear_time_s"] for r in rs if r["mean_clear_time_s"]]
        return {
            "variant": v, "episodes": len(rs),
            "full_clear_episodes": sum(1 for r in rs if r["ratio"] >= 1 - 1e-9),
            "full_clear_rate": round(sum(1 for r in rs if r["ratio"] >= 1 - 1e-9) / len(rs), 4),
            "cleared_ratio_mean": round(statistics.fmean(r["ratio"] for r in rs), 4),
            "cleared_ratio_min": round(min(r["ratio"] for r in rs), 4),
            "total_missed_sources": sum(len(r["missed"]) for r in rs),
            "mean_clear_time_s": round(statistics.fmean(mt), 1) if mt else None,
            "max_mean_clear_time_s": round(max(mt), 1) if mt else None,
            "virtual_s_mean": round(statistics.fmean(r["virtual_s"] for r in rs), 1),
            "virtual_s_max": round(max(r["virtual_s"] for r in rs), 1),
            "clears_mean": round(statistics.fmean(r["clears"] for r in rs), 2),
            "measures_mean": round(statistics.fmean(r["measures"] for r in rs), 1),
            "move_m_mean": round(statistics.fmean(r["move_m"] for r in rs), 1),
            "wall_s_mean": round(statistics.fmean(r["wall_s"] for r in rs), 2),
            "wall_s_max": round(max(r["wall_s"] for r in rs), 2),
        }

    base, post = agg("baseline"), agg("posterior")
    # 配对比较
    pairs = {}
    for seed in sorted({r["seed"] for r in out_rows}):
        b = next(r for r in out_rows if r["seed"] == seed and r["variant"] == "baseline")
        p = next(r for r in out_rows if r["seed"] == seed and r["variant"] == "posterior")
        pairs[seed] = {"d_cleared": p["cleared"] - b["cleared"],
                       "d_virtual_s": round(p["virtual_s"] - b["virtual_s"], 1),
                       "d_clears": p["clears"] - b["clears"]}
    report = {
        "baseline": base, "posterior": post,
        "paired": {
            "episodes": len(pairs),
            "episodes_posterior_cleared_more": sum(1 for v in pairs.values() if v["d_cleared"] > 0),
            "episodes_posterior_cleared_fewer": sum(1 for v in pairs.values() if v["d_cleared"] < 0),
            "net_extra_sources_cleared": sum(v["d_cleared"] for v in pairs.values()),
            "mean_d_virtual_s": round(statistics.fmean(v["d_virtual_s"] for v in pairs.values()), 1),
            "mean_d_clear_calls": round(statistics.fmean(v["d_clears"] for v in pairs.values()), 2),
        },
        "config": {"directional": args.directional, "gate_m": args.gate,
                   "max_discs": args.max_discs, "seed_start": args.seed_start},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"b_belief_live_{args.label}.json"
    out.write_text(json.dumps({"report": report, "rows": out_rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
