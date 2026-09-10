"""因子分解：把"改用完整后验"的收益拆成两个独立因子，各自单独计价。

反事实回放暴露了一件事：收益的绝对多数来自"用全部观测"而不是"维护完整后验"。
必须把两者分开，否则会把一个免费的一行修改（去掉 `bearings[-4:]`）的功劳
记到一个需要重写状态表示的大改造头上。

2x2 设计（观测集合 x 清除点选取），在同一批真实决策点上评估：

              点估计(质心)          后验集合覆盖
  最近4条      A = 现状              B = 只换清除规划
  全部观测     C = 只换观测窗口       D = 两者都换

每格评估：单发命中率、打满计划的命中率、期望清除调用数、后验直径。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .b_sim.arena import CLEAR_RADIUS_M
from .b_sim.geometry import Bearing, centroid, polygon_diameter
from .b_belief_eval import disc_mass, posterior_polygon, sample_polygon
from .b_belief_replay import cost_of_plan, posterior_plan, spiral_points

RESULTS = Path("results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="q4")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--max-records", type=int, default=500)
    ap.add_argument("--max-discs", type=int, default=40)
    args = ap.parse_args()

    payload = json.loads(
        (RESULTS / f"b_belief_records_{args.label}.json").read_text(encoding="utf-8"))
    recs = [r for r in payload["records"] if r["kind"] == "clear" and not r["pre_cleared"]]
    if args.max_records and len(recs) > args.max_records:
        idx = np.linspace(0, len(recs) - 1, args.max_records).astype(int)
        recs = [recs[i] for i in idx]
    rng = np.random.default_rng(23)

    cells = {k: [] for k in ("A_cur", "B_post_last4", "C_centroid_all", "D_post_all")}
    diag = []

    for i, rec in enumerate(recs):
        b_all = [Bearing(*b) for b in rec["bearings"]]
        b4 = b_all[-4:]
        poly4 = posterior_polygon(b4)
        polyall = posterior_polygon(b_all)
        if not poly4 or not polyall:
            continue
        truth = (rec["truth"][0], rec["truth"][1])
        start = (rec["dog_pos"][0], rec["dog_pos"][1])
        s4 = sample_polygon(poly4, args.samples, rng)
        sall = sample_polygon(polyall, args.samples, rng)

        est4 = centroid(poly4)
        estall = centroid(polyall)

        planA = [est4] + spiral_points(*est4)
        planB = posterior_plan(s4, args.max_discs, rng)
        planC = [estall] + spiral_points(*estall)
        planD = posterior_plan(sall, args.max_discs, rng)

        for key, plan, est, samples in (
            ("A_cur", planA, est4, s4),
            ("B_post_last4", planB, planB[0] if planB else est4, s4),
            ("C_centroid_all", planC, estall, sall),
            ("D_post_all", planD, planD[0] if planD else estall, sall),
        ):
            res = cost_of_plan(plan, start, truth)
            cells[key].append({
                "first_shot_hit": math.hypot(est[0] - truth[0], est[1] - truth[1]) <= CLEAR_RADIUS_M,
                "first_point_err_m": math.hypot(est[0] - truth[0], est[1] - truth[1]),
                "plan_hit": res["hit"], "calls": res["calls"], "seconds": res["seconds"],
                "plan_len": len(plan),
                "p_hit_first": disc_mass(samples, est),
                "directional": rec["directional"],
            })
        diag.append({
            "seed": rec["seed"], "channel": rec["channel"], "directional": rec["directional"],
            "n_bearings": rec["n_bearings"],
            "diam_last4_m": polygon_diameter(poly4)[0],
            "diam_all_m": polygon_diameter(polyall)[0],
            "est_err_last4_m": math.hypot(est4[0] - truth[0], est4[1] - truth[1]),
            "est_err_all_m": math.hypot(estall[0] - truth[0], estall[1] - truth[1]),
            "observed_clear_ok": rec["clear_ok"],
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    def summarise(rs):
        return {
            "n": len(rs),
            "first_shot_hit_rate": round(statistics.fmean(1.0 if r["first_shot_hit"] else 0.0 for r in rs), 4),
            "plan_hit_rate": round(statistics.fmean(1.0 if r["plan_hit"] else 0.0 for r in rs), 4),
            "mean_calls": round(statistics.fmean(r["calls"] for r in rs), 2),
            "mean_seconds": round(statistics.fmean(r["seconds"] for r in rs), 1),
            "mean_plan_len": round(statistics.fmean(r["plan_len"] for r in rs), 2),
            "mean_first_point_err_m": round(statistics.fmean(r["first_point_err_m"] for r in rs), 2),
            "mean_p_hit_first_point": round(statistics.fmean(r["p_hit_first"] for r in rs), 4),
        }

    report = {"label": args.label, "grid": {}, "grid_directional_only": {}}
    for k, rs in cells.items():
        report["grid"][k] = summarise(rs)
        d = [r for r in rs if r["directional"]]
        if d:
            report["grid_directional_only"][k] = summarise(d)

    a, b, c, d = (report["grid"][k]["plan_hit_rate"] for k in
                  ("A_cur", "B_post_last4", "C_centroid_all", "D_post_all"))
    report["factor_decomposition_plan_hit_rate"] = {
        "baseline_A_current": a,
        "effect_of_posterior_planning_only_B_minus_A": round(b - a, 4),
        "effect_of_all_observations_only_C_minus_A": round(c - a, 4),
        "effect_of_both_D_minus_A": round(d - a, 4),
        "interaction": round(d - b - c + a, 4),
    }
    # 观测窗口本身对后验尺度/点估计误差的影响
    report["window_effect"] = {
        "mean_diam_last4_m": round(statistics.fmean(r["diam_last4_m"] for r in diag), 1),
        "mean_diam_all_m": round(statistics.fmean(r["diam_all_m"] for r in diag), 1),
        "mean_est_err_last4_m": round(statistics.fmean(r["est_err_last4_m"] for r in diag), 2),
        "mean_est_err_all_m": round(statistics.fmean(r["est_err_all_m"] for r in diag), 2),
        "median_diam_last4_m": round(statistics.median(r["diam_last4_m"] for r in diag), 1),
        "median_diam_all_m": round(statistics.median(r["diam_all_m"] for r in diag), 1),
        "frac_all_strictly_tighter": round(statistics.fmean(
            1.0 if r["diam_all_m"] < r["diam_last4_m"] - 1e-6 else 0.0 for r in diag), 4),
        "frac_est_err_improved_by_all": round(statistics.fmean(
            1.0 if r["est_err_all_m"] < r["est_err_last4_m"] - 1e-6 else 0.0 for r in diag), 4),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"b_belief_ablation_{args.label}.json"
    out.write_text(json.dumps({"report": report, "diag": diag}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
