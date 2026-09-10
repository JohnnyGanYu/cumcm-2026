"""端到端账：现状"归航-清除"环路的真实代价 vs 后验计划的真实代价。

`b_belief_replay` 只给现状计划记了它自己那 7 次调用的钱，没记"扑空之后回到
_home_channel 循环再测、再走、再打"的下游代价，因此**低估**了现状成本。这里改用
真实记录做端到端账：

对每个 (seed, channel)，取该频道所有新鲜 clear 记录（按发生顺序）：
  现状实际代价 = 该频道全部 clear 调用的动作耗时 + 相邻清除点之间的移动耗时
                 + 首次清除尝试之后发生的所有 measure 的耗时（5 s/次 + 移动）
后验计划代价 = 在**该频道首次达到清除条件时**的信念上做集合覆盖，一次性打完。

后者的关键是：后验计划在第一次决策点就知道"要打 k 个点"，不需要打一发、扑空、
再测、再打。所以省下的正是中间那些 measure 和往返。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from .b_sim.arena import (
    CLEAR_FAIL_SECONDS,
    CLEAR_RADIUS_M,
    CLEAR_SUCCESS_SECONDS,
    MEASURE_SECONDS,
    SPEED_MPS,
)
from .b_sim.geometry import Bearing, polygon_diameter
from .b_belief_eval import posterior_polygon, sample_polygon
from .b_belief_replay import cost_of_plan, posterior_plan

RESULTS = Path("results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="q4")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--max-discs", type=int, default=60)
    args = ap.parse_args()

    payload = json.loads(
        (RESULTS / f"b_belief_records_{args.label}.json").read_text(encoding="utf-8"))
    records = payload["records"]
    episodes = payload["episodes"]
    rng = np.random.default_rng(11)

    # 按 (seed, channel) 分组，保持记录顺序
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for idx, rec in enumerate(records):
        rec["_idx"] = idx
        groups[(rec["seed"], rec["channel"])].append(rec)

    rows = []
    for (seed, ch), recs in sorted(groups.items()):
        clears = [r for r in recs if r["kind"] == "clear" and not r["pre_cleared"]]
        if not clears:
            continue
        first = clears[0]
        first_idx = first["_idx"]
        truth = (first["truth"][0], first["truth"][1])
        start = (first["dog_pos"][0], first["dog_pos"][1])

        # ---- 现状真实代价 ----
        pos = start
        cur_seconds = 0.0
        cur_clear_calls = 0
        cur_measures_after = 0
        # 该频道在首次清除尝试之后发生的所有动作（clear + measure），按记录顺序
        for r in recs:
            if r["_idx"] < first_idx:
                continue
            p = (r["clear_point"][0], r["clear_point"][1]) if r["kind"] == "clear" \
                else (r["bearings"][-1][0], r["bearings"][-1][1])
            cur_seconds += math.hypot(p[0] - pos[0], p[1] - pos[1]) / SPEED_MPS
            pos = p
            if r["kind"] == "clear":
                cur_clear_calls += 1
                cur_seconds += CLEAR_SUCCESS_SECONDS if r["clear_ok"] else CLEAR_FAIL_SECONDS
            else:
                cur_measures_after += 1
                cur_seconds += MEASURE_SECONDS
        cur_hit = any(r["clear_ok"] for r in clears)

        # ---- 后验计划代价（在首次决策点的信念上一次性规划）----
        bearings4 = [Bearing(*b) for b in first["bearings"]][-4:]
        bearings_all = [Bearing(*b) for b in first["bearings"]]
        poly = posterior_polygon(bearings4)
        poly_all = posterior_polygon(bearings_all)
        if not poly or not poly_all:
            continue
        s4 = sample_polygon(poly, args.samples, rng)
        sall = sample_polygon(poly_all, args.samples, rng)
        plan4 = posterior_plan(s4, args.max_discs, rng)
        planall = posterior_plan(sall, args.max_discs, rng)
        post = cost_of_plan(plan4, start, truth)
        postall = cost_of_plan(planall, start, truth)

        rows.append({
            "seed": seed, "channel": ch, "directional": first["directional"],
            "n_bearings_at_first_clear": first["n_bearings"],
            "post_diam_m": polygon_diameter(poly)[0],
            "cur_hit": cur_hit, "cur_clear_calls": cur_clear_calls,
            "cur_measures_after_first_clear": cur_measures_after,
            "cur_seconds": cur_seconds,
            "post_hit": post["hit"], "post_calls": post["calls"], "post_seconds": post["seconds"],
            "postall_hit": postall["hit"], "postall_calls": postall["calls"],
            "postall_seconds": postall["seconds"],
        })

    def agg(rs):
        if not rs:
            return {"n": 0}
        cs = statistics.fmean(r["cur_seconds"] for r in rs)
        ps = statistics.fmean(r["post_seconds"] for r in rs)
        pas = statistics.fmean(r["postall_seconds"] for r in rs)
        return {
            "n_channel_episodes": len(rs),
            "hit_rate_current": round(statistics.fmean(1.0 if r["cur_hit"] else 0.0 for r in rs), 4),
            "hit_rate_posterior": round(statistics.fmean(1.0 if r["post_hit"] else 0.0 for r in rs), 4),
            "hit_rate_posterior_allobs": round(
                statistics.fmean(1.0 if r["postall_hit"] else 0.0 for r in rs), 4),
            "mean_seconds_current": round(cs, 1),
            "mean_seconds_posterior": round(ps, 1),
            "mean_seconds_posterior_allobs": round(pas, 1),
            "seconds_saved_per_channel": round(cs - ps, 1),
            "seconds_saved_per_channel_allobs": round(cs - pas, 1),
            "relative_time_saving": round(1.0 - ps / cs, 4) if cs > 0 else None,
            "relative_time_saving_allobs": round(1.0 - pas / cs, 4) if cs > 0 else None,
            "mean_clear_calls_current": round(
                statistics.fmean(r["cur_clear_calls"] for r in rs), 2),
            "mean_clear_calls_posterior": round(statistics.fmean(r["post_calls"] for r in rs), 2),
            "mean_measures_after_first_clear_current": round(
                statistics.fmean(r["cur_measures_after_first_clear"] for r in rs), 2),
            "p90_seconds_current": round(sorted(r["cur_seconds"] for r in rs)[int(0.9 * (len(rs) - 1))], 1),
            "p90_seconds_posterior": round(sorted(r["post_seconds"] for r in rs)[int(0.9 * (len(rs) - 1))], 1),
        }

    dir_rows = [r for r in rows if r["directional"]]
    omni_rows = [r for r in rows if not r["directional"]]
    # 只看现状"打了不止一发"的频道（即真的绕过路的）
    hard = [r for r in rows if r["cur_clear_calls"] > 1]

    n_ep = len(episodes)
    total_cur = sum(r["cur_seconds"] for r in rows)
    total_post = sum(r["post_seconds"] for r in rows)
    total_postall = sum(r["postall_seconds"] for r in rows)
    obs_virtual = statistics.fmean(e["total_virtual_time_s"] for e in episodes)

    report = {
        "label": args.label,
        "overall": agg(rows),
        "directional": agg(dir_rows),
        "omnidirectional": agg(omni_rows),
        "hard_cases_current_needed_multiple_clears": agg(hard),
        "episode_level": {
            "episodes": n_ep,
            "observed_mean_virtual_time_s": round(obs_virtual, 1),
            "observed_mean_clear_calls": round(
                statistics.fmean(e["clear_calls"] for e in episodes), 2),
            "observed_full_clear_rate": round(sum(
                1 for e in episodes if e["cleared_ratio"] >= 1 - 1e-9) / n_ep, 4),
            "sum_current_homing_seconds": round(total_cur, 1),
            "sum_posterior_seconds": round(total_post, 1),
            "sum_posterior_allobs_seconds": round(total_postall, 1),
            "mean_saving_per_episode_s": round((total_cur - total_post) / n_ep, 1),
            "mean_saving_per_episode_allobs_s": round((total_cur - total_postall) / n_ep, 1),
            "saving_as_pct_of_observed_episode_time": round(
                (total_cur - total_post) / n_ep / obs_virtual, 4),
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"b_belief_endtoend_{args.label}.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
