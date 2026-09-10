"""决策级反事实回放：在真实记录的信念状态上，把"现状点估计计划"与"后验计划"对打。

比误差分布更有说服力的是**决策代价**。对每一个真实发生过的清除决策点，两种计划：

现状计划（读 strategy.py 得到的确切行为）
  candidates = [estimate] + spiral(6 点, 半径 15 m, 绕 estimate)
  按顺序调用 /clear，命中即停。

后验计划
  在精确后验支撑集（楔形交 ∩ 目标区域，后验在其上均匀）上做 20 m 圆盘贪心集合覆盖，
  按覆盖后验质量降序调用 /clear，命中即停。

两者都用**真值**判定命中，都计入移动时间（5 m/s）与 /clear 动作耗时
（成功 5 s / 失败 3 s），从记录下来的机器狗真实位置出发，按最近邻顺序走点。

同时评估"不确定度门禁"：现状 `_home_channel` 只看"到 estimate 的距离 <= 55 m"就清除，
完全不看后验尺度。这里统计后验直径能否预测清除失败（AUC + 最优阈值）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .b_sim.arena import CLEAR_FAIL_SECONDS, CLEAR_RADIUS_M, CLEAR_SUCCESS_SECONDS, SPEED_MPS
from .b_sim.geometry import Bearing, polygon_diameter
from .b_belief_eval import (
    best_disc,
    posterior_polygon,
    sample_polygon,
)

RESULTS = Path("results")
SPIRAL_RADIUS_M = 15.0
SPIRAL_POINTS = 6


def spiral_points(cx: float, cy: float, radius: float = SPIRAL_RADIUS_M,
                  n: int = SPIRAL_POINTS) -> list[tuple[float, float]]:
    """复刻 strategy._spiral 的补点几何。"""
    out = []
    for k in range(n):
        theta = 2.0 * math.pi * k / n
        out.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return out


def posterior_plan(samples: np.ndarray, max_discs: int, rng: np.random.Generator,
                   n_cand: int = 350) -> list[tuple[float, float]]:
    """贪心集合覆盖，返回按增益降序的清除点序列。"""
    m = len(samples)
    cand = samples if m <= n_cand else samples[rng.choice(m, n_cand, replace=False)]
    d = np.hypot(samples[:, 0][None, :] - cand[:, 0][:, None],
                 samples[:, 1][None, :] - cand[:, 1][:, None])
    hit = d <= CLEAR_RADIUS_M
    uncovered = np.ones(m, dtype=bool)
    plan: list[tuple[float, float]] = []
    for _ in range(max_discs):
        gains = hit[:, uncovered].sum(axis=1)
        k = int(np.argmax(gains))
        if gains[k] == 0:
            break
        plan.append((float(cand[k, 0]), float(cand[k, 1])))
        uncovered[np.where(uncovered)[0][hit[k, uncovered]]] = False
        if not uncovered.any():
            break
    return plan


def cost_of_plan(plan: list[tuple[float, float]], start: tuple[float, float],
                 truth: tuple[float, float]) -> dict:
    """按给定顺序执行 /clear，命中即停。返回调用数、耗时、是否命中。"""
    pos = start
    seconds = 0.0
    for i, p in enumerate(plan, 1):
        seconds += math.hypot(p[0] - pos[0], p[1] - pos[1]) / SPEED_MPS
        pos = p
        if math.hypot(p[0] - truth[0], p[1] - truth[1]) <= CLEAR_RADIUS_M:
            return {"hit": True, "calls": i, "seconds": seconds + CLEAR_SUCCESS_SECONDS}
        seconds += CLEAR_FAIL_SECONDS
    return {"hit": False, "calls": len(plan), "seconds": seconds}


def auc(scores: list[float], labels: list[int]) -> float | None:
    """ROC AUC（Mann-Whitney U），labels 1 = 正类（失败）。"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="q4")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--max-discs", type=int, default=40)
    ap.add_argument("--max-records", type=int, default=600)
    args = ap.parse_args()

    payload = json.loads((RESULTS / f"b_belief_records_{args.label}.json").read_text(encoding="utf-8"))
    recs = [r for r in payload["records"] if r["kind"] == "clear" and not r["pre_cleared"]]
    if args.max_records and len(recs) > args.max_records:
        idx = np.linspace(0, len(recs) - 1, args.max_records).astype(int)
        recs = [recs[i] for i in idx]
    rng = np.random.default_rng(7)

    rows = []
    for i, rec in enumerate(recs):
        bearings4 = [Bearing(*b) for b in rec["bearings"]][-4:]
        bearings_all = [Bearing(*b) for b in rec["bearings"]]
        poly = posterior_polygon(bearings4)
        poly_all = posterior_polygon(bearings_all)
        if not poly or not poly_all:
            continue
        truth = (rec["truth"][0], rec["truth"][1])
        start = (rec["dog_pos"][0], rec["dog_pos"][1])
        est = rec["estimate"]
        if est is None:
            continue

        samples = sample_polygon(poly, args.samples, rng)
        samples_all = sample_polygon(poly_all, args.samples, rng)

        cur_plan = [tuple(est)] + spiral_points(est[0], est[1])
        post_plan = posterior_plan(samples, args.max_discs, rng)
        post_plan_all = posterior_plan(samples_all, args.max_discs, rng)

        cur = cost_of_plan(cur_plan, start, truth)
        post = cost_of_plan(post_plan, start, truth)
        post_all = cost_of_plan(post_plan_all, start, truth)

        diam = polygon_diameter(poly)[0]
        diam_all = polygon_diameter(poly_all)[0]
        best_p, _ = best_disc(samples, rng=rng)

        rows.append({
            "seed": rec["seed"], "channel": rec["channel"],
            "directional": rec["directional"], "caller": rec["caller"],
            "n_bearings": rec["n_bearings"],
            "observed_clear_ok": rec["clear_ok"],
            "est_err_m": math.hypot(est[0] - truth[0], est[1] - truth[1]),
            "post_diam_m": diam, "post_diam_all_m": diam_all,
            "p_hit_best": best_p,
            "cur_hit": cur["hit"], "cur_calls": cur["calls"], "cur_seconds": cur["seconds"],
            "post_hit": post["hit"], "post_calls": post["calls"], "post_seconds": post["seconds"],
            "post_plan_len": len(post_plan),
            "postall_hit": post_all["hit"], "postall_calls": post_all["calls"],
            "postall_seconds": post_all["seconds"], "postall_plan_len": len(post_plan_all),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    def agg(rs, name):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "hit_rate_current": round(statistics.fmean(1.0 if r["cur_hit"] else 0.0 for r in rs), 4),
            "hit_rate_posterior": round(statistics.fmean(1.0 if r["post_hit"] else 0.0 for r in rs), 4),
            "hit_rate_posterior_allobs": round(
                statistics.fmean(1.0 if r["postall_hit"] else 0.0 for r in rs), 4),
            "mean_calls_current": round(statistics.fmean(r["cur_calls"] for r in rs), 2),
            "mean_calls_posterior": round(statistics.fmean(r["post_calls"] for r in rs), 2),
            "mean_calls_posterior_allobs": round(statistics.fmean(r["postall_calls"] for r in rs), 2),
            "mean_seconds_current": round(statistics.fmean(r["cur_seconds"] for r in rs), 1),
            "mean_seconds_posterior": round(statistics.fmean(r["post_seconds"] for r in rs), 1),
            "mean_seconds_posterior_allobs": round(
                statistics.fmean(r["postall_seconds"] for r in rs), 1),
            # 只在"现状计划整体扑空"的子集上看后验能救回多少
            "current_miss_count": sum(1 for r in rs if not r["cur_hit"]),
            "posterior_rescues_of_current_miss": sum(
                1 for r in rs if not r["cur_hit"] and r["post_hit"]),
            "posterior_loses_where_current_hit": sum(
                1 for r in rs if r["cur_hit"] and not r["post_hit"]),
        }

    all_rows = rows
    dir_rows = [r for r in rows if r["directional"]]
    omni_rows = [r for r in rows if not r["directional"]]

    # 不确定度门禁的判别力：后验直径 vs 实测清除失败
    labels = [0 if r["observed_clear_ok"] else 1 for r in rows]
    gate = {
        "auc_post_diameter_predicts_observed_fail": None if len(set(labels)) < 2 else round(
            auc([r["post_diam_m"] for r in rows], labels), 4),
        "auc_est_err_predicts_fail_oracle": None if len(set(labels)) < 2 else round(
            auc([r["est_err_m"] for r in rows], labels), 4),
        "auc_neg_phit_predicts_fail": None if len(set(labels)) < 2 else round(
            auc([-r["p_hit_best"] for r in rows], labels), 4),
    }
    # 门禁阈值扫描：若"后验直径 > T 就先再测一次而不是清除"
    scan = []
    for T in (40, 60, 80, 120, 200, 400):
        gated = [r for r in rows if r["post_diam_m"] > T]
        if not gated:
            continue
        fails_avoided = sum(1 for r in gated if not r["observed_clear_ok"])
        good_blocked = sum(1 for r in gated if r["observed_clear_ok"])
        scan.append({
            "diameter_threshold_m": T,
            "decisions_deferred": len(gated),
            "failed_clears_avoided": fails_avoided,
            "successful_clears_wrongly_deferred": good_blocked,
            "precision": round(fails_avoided / len(gated), 4),
            "recall_of_all_fails": round(
                fails_avoided / max(1, sum(labels)), 4),
            "net_seconds_saved_est": round(
                fails_avoided * CLEAR_FAIL_SECONDS - good_blocked * 0.0, 1),
        })

    report = {
        "label": args.label,
        "overall": agg(all_rows, "all"),
        "directional": agg(dir_rows, "dir"),
        "omnidirectional": agg(omni_rows, "omni"),
        "by_caller": {c: agg([r for r in rows if r["caller"] == c], c)
                      for c in sorted({r["caller"] for r in rows})},
        "uncertainty_gate": gate,
        "gate_threshold_scan": scan,
        "observed_fail_rate": round(statistics.fmean(labels), 4),
        "note_current_plan": f"[estimate] + spiral({SPIRAL_POINTS} 点 @ {SPIRAL_RADIUS_M} m)",
        "note_posterior_plan": f"20 m 圆盘贪心集合覆盖，上限 {args.max_discs} 个点",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"b_belief_replay_{args.label}.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
