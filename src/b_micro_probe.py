"""判据可行性诊断：在单示向度决策点上，把"可证明上界 r_max"与"真实距离 r"对照。

真值只用于**事后诊断**，策略本身从不读它（见 b_micro_opt.py）。回答三个问题：
  Q1 优化 (a) 在离线仿真里为什么零效果？-> 统计 _spiral 进入次数与成功后调用数。
  Q2 单示向度决策点上真实距离分布如何？多少比例其实已在 573 m / 1146 m 内？
  Q3 可证明上界 r_max 有多松？松到永远碰不到阈值吗？
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading
from concurrent.futures import ProcessPoolExecutor

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import ChannelState, DogStrategy, StrategyConfig
from .b_micro_opt import THRESHOLD_573, THRESHOLD_1146, MicroStrategy

ROBOT_ID = "dog-probe"


class ProbeStrategy(MicroStrategy):
    """在每个单示向度决策点记录 (r_max_provable, r_true, lateral_error)。"""

    def __init__(self, *a, truth=None, **kw):
        super().__init__(*a, **kw)
        self.truth = truth or {}
        self.decision_points: list[dict] = []
        self.spiral_entries = 0
        self.spiral_clear_calls = 0
        self.post_success_calls = 0

    def _spiral(self, state):
        self.spiral_entries += 1
        before = state.clear_attempts
        r = super()._spiral(state)
        self.spiral_clear_calls += state.clear_attempts - before
        return r

    def _clear_at(self, point, channel):
        if self.channels[channel].cleared:
            self.post_success_calls += 1
        return super()._clear_at(point, channel)

    def _home_channel(self, state: ChannelState) -> bool:
        # 在进入原逻辑前，若处于"单示向度、尚无交会估计"状态，记录一个决策点
        if state.bearings and state.estimate is None and not state.cleared:
            last = state.bearings[-1]
            tr = self.truth.get(state.channel)
            if tr is not None:
                r_true = math.hypot(tr["x"] - last.x, tr["y"] - last.y)
                true_brg = math.degrees(math.atan2(tr["y"] - last.y, tr["x"] - last.x))
                err = (last.svd_deg - true_brg + 180.0) % 360.0 - 180.0
                lateral = r_true * math.tan(math.radians(abs(err)))
                self.decision_points.append({
                    "channel": state.channel,
                    "r_max_provable": round(self._distance_upper_bound(state), 1),
                    "r_max_c0_only": 1500.0,
                    "r_true": round(r_true, 1),
                    "bearing_err_deg": round(err, 3),
                    "lateral_offset_m": round(lateral, 2),
                    "n_silent": len(state.silent_points),
                    "n_bearings": len(state.bearings),
                    "directional": bool(tr.get("directional_deg") is not None),
                })
        return super()._home_channel(state)


def run_one(args) -> dict:
    seed, directional, count = args
    case = generate_case(seed, count=count, directional_count=directional)
    truth = {j.channel: {"x": j.x, "y": j.y, "directional_deg": j.directional_deg,
                         "reception_radius_m": j.reception_radius_m}
             for j in case.jammers}
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strat = ProbeStrategy(client, StrategyConfig(directional_probe=directional > 0),
                             truth=truth, opt_a=True, opt_b=False)
        strat.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    s = arena.stats()
    return {"seed": seed, "cleared": s["cleared"], "total": s["total_jammers"],
            "clear_calls": s["clear_calls"],
            "spiral_entries": strat.spiral_entries,
            "spiral_clear_calls": strat.spiral_clear_calls,
            "post_success_calls": strat.post_success_calls,
            "decision_points": strat.decision_points}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jobs = [(args.seed_start + i, args.directional, args.count) for i in range(args.episodes)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run_one, jobs, chunksize=2):
            rows.append(r)
            print(f"seed={r['seed']} cleared={r['cleared']}/{r['total']} "
                  f"clears={r['clear_calls']} spiral_entries={r['spiral_entries']} "
                  f"spiral_calls={r['spiral_clear_calls']} post_succ={r['post_success_calls']} "
                  f"dp={len(r['decision_points'])}", flush=True)

    dp = [d for r in rows for d in r["decision_points"]]
    n = len(dp)
    def frac(pred):
        return round(sum(1 for d in dp if pred(d)) / n, 4) if n else None
    summary = {
        "episodes": len(rows),
        "cleared_total": sum(r["cleared"] for r in rows),
        "jammers_total": sum(r["total"] for r in rows),
        "clear_calls_mean": round(st.fmean([r["clear_calls"] for r in rows]), 2),
        "spiral_entries_total": sum(r["spiral_entries"] for r in rows),
        "spiral_clear_calls_total": sum(r["spiral_clear_calls"] for r in rows),
        "post_success_clear_calls_total": sum(r["post_success_calls"] for r in rows),
        # --- Q2/Q3 ---
        "single_bearing_decision_points": n,
        "r_true_min": round(min(d["r_true"] for d in dp), 1) if n else None,
        "r_true_median": round(st.median([d["r_true"] for d in dp]), 1) if n else None,
        "r_true_mean": round(st.fmean([d["r_true"] for d in dp]), 1) if n else None,
        "frac_r_true_le_573": frac(lambda d: d["r_true"] <= THRESHOLD_573),
        "frac_r_true_le_1146": frac(lambda d: d["r_true"] <= THRESHOLD_1146),
        "rmax_provable_min": round(min(d["r_max_provable"] for d in dp), 1) if n else None,
        "rmax_provable_median": round(st.median([d["r_max_provable"] for d in dp]), 1) if n else None,
        "frac_rmax_le_573": frac(lambda d: d["r_max_provable"] <= THRESHOLD_573),
        "frac_rmax_le_1146": frac(lambda d: d["r_max_provable"] <= THRESHOLD_1146),
        # 上界松紧比
        "rmax_over_rtrue_median": round(st.median([d["r_max_provable"] / d["r_true"]
                                                  for d in dp if d["r_true"] > 1]), 3) if n else None,
        # 单示向度直清的可行性：横向偏差 vs 清除半径 20 m
        "lateral_offset_median_m": round(st.median([d["lateral_offset_m"] for d in dp]), 2) if n else None,
        "lateral_offset_max_m": round(max(d["lateral_offset_m"] for d in dp), 2) if n else None,
        "frac_lateral_le_20": frac(lambda d: d["lateral_offset_m"] <= 20.0),
        # 若真在 573 内，横向是否一定 <=20？（理论应为真）
        "frac_lateral_le_20_given_rtrue_le_573": (
            round(sum(1 for d in dp if d["r_true"] <= THRESHOLD_573 and d["lateral_offset_m"] <= 20.0)
                  / max(1, sum(1 for d in dp if d["r_true"] <= THRESHOLD_573)), 4) if n else None),
        "frac_lateral_le_20_given_rtrue_le_1146": (
            round(sum(1 for d in dp if d["r_true"] <= THRESHOLD_1146 and d["lateral_offset_m"] <= 20.0)
                  / max(1, sum(1 for d in dp if d["r_true"] <= THRESHOLD_1146)), 4) if n else None),
    }
    label = f"q{4 if args.directional else 3}"
    out = args.out or f"results/b_micro_probe_{label}.json"
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"summary": summary, "episodes": rows},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== summary ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
