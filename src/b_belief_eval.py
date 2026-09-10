"""量化 B 题当前"单点估计"的误差分布，并评估改用完整后验（粒子/网格信念）的收益。

背景
----
现状：`ChannelState` 对每个频道只维护
    estimate = centroid(localization_polygon(bearings[-4:]))       # 退化时用最小二乘
    diameter = polygon_diameter(...)                                # 计算了但从不参与决策
`_home_channel` 的清除决策只看"机器狗到 estimate 的距离 <= clear_distance_m(55)"，
不看不确定度；清除失败后用固定的 6 点 / 15 m 螺旋补点。

关键观察（本脚本的理论基础）
--------------------------
仿真器的示向度误差严格落在 [-1, 1] 度内（`arena.bearing_error_deg`），
因此对**均匀先验 + 有界均匀误差**，真值的精确后验支撑集就是各观测楔形的交集
（`localization_polygon`）再与目标区域求交，且后验在该凸集上均匀。
=> 现有点估计（面积质心）恰好就是后验均值；
=> "完整后验"的增益因此不在点位本身，而在四处：
   (A) 知道后验的尺度，能判断"该清除还是该再测"；
   (B) 清除点应最大化"20 m 圆盘覆盖的后验质量"，而不是取后验均值；
   (C) 补点应按后验做集合覆盖，而不是固定半径螺旋；
   (D) 应使用**全部**观测而不是最近 4 条。

本脚本两步走
-----------
mode=record   跑真实演练（本地 mock 服务 + 真实策略），在每次估计刷新、每次 /clear
              调用处记录 (观测集合, 点估计, 真值, 结果)。
mode=analyze  离线对每条记录重建精确后验，算出上面 A-D 四项的真实数字。

运行：
  python -m src.b_belief_eval --mode record  --episodes 40 --directional 0 --label q3
  python -m src.b_belief_eval --mode record  --episodes 30 --directional 5 --label q4
  python -m src.b_belief_eval --mode analyze --label q3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .b_sim.arena import (
    ARENA_RADIUS,
    CLEAR_FAIL_SECONDS,
    CLEAR_RADIUS_M,
    CLEAR_SUCCESS_SECONDS,
    MEASURE_SECONDS,
    SPEED_MPS,
    Arena,
    generate_case,
)
from .b_sim.client import RobotClient
from .b_sim.geometry import (
    Bearing,
    clip_polygon,
    localization_polygon,
    polygon_diameter,
    wedge_half_planes,
)
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

RESULTS = Path("results")
ROBOT_ID = "belief-eval"


# ======================================================================
# 第一步：带真值记录的演练
# ======================================================================


class RecordingStrategy(DogStrategy):
    """在原策略上挂真值探针，不改变任何决策逻辑。"""

    def __init__(self, client, config, case, records: list[dict], seed: int) -> None:
        super().__init__(client, config)
        self.case = case
        self.records = records
        self.seed = seed

    def _snapshot(self, state, kind: str, extra: dict) -> None:
        jammer = self.case.by_channel(state.channel)
        if jammer is None or len(state.bearings) < 2:
            return
        rec = {
            "seed": self.seed,
            "kind": kind,
            "channel": state.channel,
            "directional": jammer.is_directional,
            "truth": [jammer.x, jammer.y],
            "reception_radius_m": jammer.reception_radius_m,
            "bearings": [[b.x, b.y, b.svd_deg] for b in state.bearings],
            "estimate": None if state.estimate is None else list(state.estimate),
            "reported_diameter": state.diameter if math.isfinite(state.diameter) else None,
            "dog_pos": list(self.pos),
            "n_bearings": len(state.bearings),
            "n_silent": len(state.silent_points),
        }
        rec.update(extra)
        self.records.append(rec)

    def _probe_channel_at(self, point, state) -> None:
        n_before = len(state.bearings)
        super()._probe_channel_at(point, state)
        if len(state.bearings) > n_before and state.estimate is not None:
            self._snapshot(state, "estimate", {})

    def _clear_at(self, point, channel):
        state = self.channels[channel]
        jammer = self.case.by_channel(channel)
        pre_cleared = state.cleared or (jammer is not None and jammer.cleared)
        caller = sys._getframe(1).f_code.co_name
        move_m = math.hypot(point[0] - self.pos[0], point[1] - self.pos[1])
        ok = super()._clear_at(point, channel)
        if jammer is not None and len(state.bearings) >= 2:
            self._snapshot(
                state,
                "clear",
                {
                    "clear_point": list(point),
                    "clear_ok": bool(ok),
                    "pre_cleared": bool(pre_cleared),
                    "caller": caller,
                    "clear_move_m": move_m,
                    "clear_err_m": math.hypot(point[0] - jammer.x, point[1] - jammer.y),
                },
            )
        return ok


def record_episode(seed: int, *, directional: int, config: StrategyConfig,
                   records: list[dict]) -> dict:
    case = generate_case(seed, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    n0 = len(records)
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strategy = RecordingStrategy(client, config, case, records, seed)
        strategy.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    stats = arena.stats()
    return {
        "seed": seed,
        "total_jammers": stats["total_jammers"],
        "cleared": stats["cleared"],
        "cleared_ratio": stats["cleared_ratio"],
        "total_virtual_time_s": stats["total_virtual_time_s"],
        "mean_clear_time_s": stats["mean_clear_time_s"],
        "measure_calls": stats["measure_calls"],
        "clear_calls": stats["clear_calls"],
        "move_distance_m": stats["move_distance_m"],
        "records": len(records) - n0,
        "missed_channels": [r["channel"] for r in case.truth_table() if not r["cleared"]],
    }


def do_record(args) -> None:
    config = StrategyConfig(directional_probe=args.directional > 0)
    records: list[dict] = []
    episodes = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        t0 = time.monotonic()
        ep = record_episode(seed, directional=args.directional, config=config, records=records)
        ep["wall_s"] = round(time.monotonic() - t0, 2)
        episodes.append(ep)
        print(f"seed={seed} cleared={ep['cleared']}/{ep['total_jammers']} "
              f"ratio={ep['cleared_ratio']:.3f} clears={ep['clear_calls']} "
              f"measures={ep['measure_calls']} recs={ep['records']} "
              f"wall={ep['wall_s']}s missed={ep['missed_channels']}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"b_belief_records_{args.label}.json"
    out.write_text(json.dumps({
        "meta": {"episodes": args.episodes, "seed_start": args.seed_start,
                 "directional": args.directional},
        "episodes": episodes,
        "records": records,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}  ({len(records)} records)")


# ======================================================================
# 第二步：精确后验重建与收益分析
# ======================================================================


def _arena_polygon(n: int = 180, radius: float = ARENA_RADIUS) -> list[tuple[float, float]]:
    """目标区域圆盘的内接 n 边形（n=180 时半径误差 0.27 m，可忽略）。"""
    return [(radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n))
            for k in range(n)]


_ARENA_POLY = _arena_polygon()


def posterior_polygon(bearings: list[Bearing]) -> list[tuple[float, float]]:
    """真值的精确后验支撑集：各楔形之交 ∩ 目标区域。后验在其上均匀。"""
    poly = _ARENA_POLY
    for b in bearings:
        for plane in wedge_half_planes(b):
            poly = clip_polygon(poly, plane)
            if not poly:
                return []
    return poly


def polygon_area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def sample_polygon(poly, n: int, rng: np.random.Generator) -> np.ndarray:
    """凸多边形内均匀采样（扇形三角化 + 按面积加权 + 重心坐标）。"""
    p0 = np.asarray(poly[0], dtype=float)
    pts = np.asarray(poly, dtype=float)
    a = pts[1:-1] - p0
    b = pts[2:] - p0
    areas = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2.0
    total = areas.sum()
    if total <= 0:
        return np.repeat(p0[None, :], n, axis=0)
    idx = rng.choice(len(areas), size=n, p=areas / total)
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return p0 + a[idx] * u[:, None] + b[idx] * v[:, None]


def disc_mass(samples: np.ndarray, center, radius: float = CLEAR_RADIUS_M) -> float:
    d = np.hypot(samples[:, 0] - center[0], samples[:, 1] - center[1])
    return float((d <= radius).mean())


def best_disc(samples: np.ndarray, radius: float = CLEAR_RADIUS_M,
              n_cand: int = 400, rng: np.random.Generator | None = None):
    """单个 20 m 圆盘能覆盖的最大后验质量，及其圆心。候选圆心取后验样本子集。"""
    rng = rng or np.random.default_rng(0)
    m = len(samples)
    cand = samples if m <= n_cand else samples[rng.choice(m, n_cand, replace=False)]
    d = np.hypot(samples[:, 0][None, :] - cand[:, 0][:, None],
                 samples[:, 1][None, :] - cand[:, 1][:, None])
    cover = (d <= radius).mean(axis=1)
    k = int(np.argmax(cover))
    return float(cover[k]), (float(cand[k, 0]), float(cand[k, 1]))


def greedy_cover(samples: np.ndarray, targets=(0.95, 1.0), radius: float = CLEAR_RADIUS_M,
                 n_cand: int = 400, max_discs: int = 60,
                 rng: np.random.Generator | None = None) -> dict:
    """贪心集合覆盖：需要多少个 20 m 清除点才能覆盖给定比例的后验质量。"""
    rng = rng or np.random.default_rng(0)
    m = len(samples)
    cand = samples if m <= n_cand else samples[rng.choice(m, n_cand, replace=False)]
    d = np.hypot(samples[:, 0][None, :] - cand[:, 0][:, None],
                 samples[:, 1][None, :] - cand[:, 1][:, None])
    hit = d <= radius
    uncovered = np.ones(m, dtype=bool)
    out: dict[str, int | None] = {f"discs_{t}": None for t in targets}
    curve = []
    for step in range(1, max_discs + 1):
        gains = hit[:, uncovered].sum(axis=1)
        k = int(np.argmax(gains))
        if gains[k] == 0:
            break
        uncovered[np.where(uncovered)[0][hit[k, uncovered]]] = False
        covered = 1.0 - uncovered.mean()
        curve.append(round(covered, 4))
        for t in targets:
            if out[f"discs_{t}"] is None and covered >= t - 1e-9:
                out[f"discs_{t}"] = step
        if all(out[f"discs_{t}"] is not None for t in targets):
            break
    out["cover_curve"] = curve
    return out


def analyse_record(rec: dict, rng: np.random.Generator, n_samples: int = 2500) -> dict | None:
    bearings_all = [Bearing(*b) for b in rec["bearings"]]
    bearings_used = bearings_all[-4:]           # 策略实际用于估计的子集
    truth = np.array(rec["truth"], dtype=float)

    poly_used = posterior_polygon(bearings_used)
    poly_all = posterior_polygon(bearings_all)
    if not poly_used or not poly_all:
        return None

    samples_used = sample_polygon(poly_used, n_samples, rng)
    samples_all = sample_polygon(poly_all, n_samples, rng)

    d_used = polygon_diameter(poly_used)[0]
    d_all = polygon_diameter(poly_all)[0]

    est = rec["estimate"]
    out: dict = {
        "seed": rec["seed"], "channel": rec["channel"], "kind": rec["kind"],
        "directional": rec["directional"], "n_bearings": rec["n_bearings"],
        "truth_in_support": bool(
            np.hypot(samples_used[:, 0] - truth[0], samples_used[:, 1] - truth[1]).min() < 1e9
            and _inside(poly_used, truth)),
        "truth_in_support_all": bool(_inside(poly_all, truth)),
        # (D) 用全部观测 vs 只用最近 4 条
        "post_diam_used_m": d_used,
        "post_diam_all_m": d_all,
        "post_area_used_m2": polygon_area(poly_used),
        "post_area_all_m2": polygon_area(poly_all),
        # 现状点估计
        "est_err_m": None if est is None else float(np.hypot(est[0] - truth[0], est[1] - truth[1])),
        "reported_diameter_m": rec.get("reported_diameter"),
    }
    # (A) 点估计处的命中概率 = 后验质量
    if est is not None:
        out["p_hit_point_est"] = disc_mass(samples_used, est)
    # (B) 后验最优单点
    best_p, best_c = best_disc(samples_used, rng=rng)
    out["p_hit_best_single"] = best_p
    out["best_single_err_m"] = float(np.hypot(best_c[0] - truth[0], best_c[1] - truth[1]))
    best_p_all, best_c_all = best_disc(samples_all, rng=rng)
    out["p_hit_best_single_allobs"] = best_p_all
    out["best_single_allobs_err_m"] = float(np.hypot(best_c_all[0] - truth[0],
                                                    best_c_all[1] - truth[1]))
    # (C) 集合覆盖代价
    cov = greedy_cover(samples_used, rng=rng)
    out["discs_95"] = cov["discs_0.95"]
    out["discs_100"] = cov["discs_1.0"]
    cov_all = greedy_cover(samples_all, rng=rng)
    out["discs_95_allobs"] = cov_all["discs_0.95"]
    out["discs_100_allobs"] = cov_all["discs_1.0"]

    if rec["kind"] == "clear":
        out["clear_ok"] = rec["clear_ok"]
        out["pre_cleared"] = rec["pre_cleared"]
        out["caller"] = rec["caller"]
        out["clear_err_m"] = rec["clear_err_m"]
        out["clear_move_m"] = rec["clear_move_m"]
        out["p_hit_actual_clear_point"] = disc_mass(samples_used, rec["clear_point"])
    return out


def _inside(poly, p, eps: float = 1e-6) -> bool:
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
        if abs(cross) < eps:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def q(values, p):
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return round(s[k], 2)


def describe(name, values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 3),
        "p50": q(vals, 0.50), "p75": q(vals, 0.75), "p90": q(vals, 0.90),
        "p95": q(vals, 0.95), "p99": q(vals, 0.99), "max": round(max(vals), 2),
    }


def do_analyze(args) -> None:
    src = RESULTS / f"b_belief_records_{args.label}.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    records = payload["records"]
    episodes = payload["episodes"]
    rng = np.random.default_rng(12345)

    # 只分析"新鲜"的记录：pre_cleared 的 clear 调用是纯浪费，另行统计
    rows: list[dict] = []
    skipped_empty = 0
    subset = records
    if args.max_records and len(records) > args.max_records:
        idx = np.linspace(0, len(records) - 1, args.max_records).astype(int)
        subset = [records[i] for i in idx]
    for i, rec in enumerate(subset):
        r = analyse_record(rec, rng, n_samples=args.samples)
        if r is None:
            skipped_empty += 1
            continue
        rows.append(r)
        if (i + 1) % 200 == 0:
            print(f"  analysed {i+1}/{len(subset)}", flush=True)

    est_rows = [r for r in rows if r["kind"] == "estimate"]
    clr_rows = [r for r in rows if r["kind"] == "clear" and not r.get("pre_cleared")]
    fresh_clr = clr_rows

    report: dict = {
        "label": args.label,
        "n_records_total": len(records),
        "n_analysed": len(rows),
        "n_support_empty": skipped_empty,
        "episode_baseline": {
            "episodes": len(episodes),
            "full_clear_rate": round(
                sum(1 for e in episodes if e["cleared_ratio"] >= 1 - 1e-9) / len(episodes), 4),
            "cleared_ratio_mean": round(statistics.fmean(e["cleared_ratio"] for e in episodes), 4),
            "mean_clear_time_s": round(statistics.fmean(
                e["mean_clear_time_s"] for e in episodes if e["mean_clear_time_s"]), 1),
            "clear_calls_mean": round(statistics.fmean(e["clear_calls"] for e in episodes), 1),
            "measure_calls_mean": round(statistics.fmean(e["measure_calls"] for e in episodes), 1),
            "cleared_mean": round(statistics.fmean(e["cleared"] for e in episodes), 2),
        },
    }

    # ---- 校验：后验支撑集是否真的包含真值（模型正确性自检）----
    report["sanity"] = {
        "truth_in_support_used_rate": round(
            statistics.fmean(1.0 if r["truth_in_support"] else 0.0 for r in rows), 4),
        "truth_in_support_all_rate": round(
            statistics.fmean(1.0 if r["truth_in_support_all"] else 0.0 for r in rows), 4),
    }

    # ---- (A) 现状单点估计的误差分布 ----
    report["A_point_estimate_error"] = {
        "all_snapshots": describe("est_err", [r["est_err_m"] for r in rows]),
        "at_clear_decisions": describe("est_err", [r["est_err_m"] for r in fresh_clr]),
        "frac_err_gt_clear_radius": round(statistics.fmean(
            1.0 if (r["est_err_m"] or 0) > CLEAR_RADIUS_M else 0.0 for r in rows), 4),
        "frac_err_gt_clear_radius_at_clear": round(statistics.fmean(
            1.0 if (r["est_err_m"] or 0) > CLEAR_RADIUS_M else 0.0 for r in fresh_clr), 4)
            if fresh_clr else None,
        "by_n_bearings": {
            str(n): describe("e", [r["est_err_m"] for r in rows if r["n_bearings"] == n])
            for n in sorted({r["n_bearings"] for r in rows})[:8]
        },
    }

    # ---- 后验尺度分布 ----
    report["posterior_scale"] = {
        "diameter_last4_m": describe("d", [r["post_diam_used_m"] for r in rows]),
        "diameter_allobs_m": describe("d", [r["post_diam_all_m"] for r in rows]),
        "frac_diameter_le_40m_last4": round(statistics.fmean(
            1.0 if r["post_diam_used_m"] <= 2 * CLEAR_RADIUS_M else 0.0 for r in rows), 4),
        "frac_diameter_le_40m_allobs": round(statistics.fmean(
            1.0 if r["post_diam_all_m"] <= 2 * CLEAR_RADIUS_M else 0.0 for r in rows), 4),
    }

    # ---- (B) 命中概率：点估计 vs 后验最优点 ----
    p_pt = [r.get("p_hit_point_est") for r in rows if r.get("p_hit_point_est") is not None]
    p_best = [r["p_hit_best_single"] for r in rows]
    p_best_all = [r["p_hit_best_single_allobs"] for r in rows]
    report["B_hit_probability"] = {
        "point_estimate": describe("p", p_pt),
        "best_single_disc_last4": describe("p", p_best),
        "best_single_disc_allobs": describe("p", p_best_all),
        "mean_gain_best_vs_point": round(statistics.fmean(p_best) - statistics.fmean(p_pt), 4),
        "mean_gain_allobs_vs_point": round(statistics.fmean(p_best_all) - statistics.fmean(p_pt), 4),
        "at_clear_decisions": {
            "actual_clear_point": describe(
                "p", [r["p_hit_actual_clear_point"] for r in fresh_clr]),
            "best_single_disc": describe("p", [r["p_hit_best_single"] for r in fresh_clr]),
        },
    }

    # ---- 实测清除调用的分解与"可预测的浪费" ----
    all_clear = [r for r in rows if r["kind"] == "clear"]
    pre = [r for r in all_clear if r.get("pre_cleared")]
    fail = [r for r in fresh_clr if not r["clear_ok"]]
    succ = [r for r in fresh_clr if r["clear_ok"]]
    lowp_fail = [r for r in fail if r["p_hit_actual_clear_point"] < 0.05]
    report["C_clear_call_breakdown"] = {
        "clear_records_analysed": len(all_clear),
        "post_clear_redundant": len(pre),
        "fresh": len(fresh_clr),
        "fresh_success": len(succ),
        "fresh_fail": len(fail),
        "fresh_fail_rate": round(len(fail) / len(fresh_clr), 4) if fresh_clr else None,
        "fail_with_posterior_phit_lt_5pct": len(lowp_fail),
        "frac_of_fails_predictable": round(len(lowp_fail) / len(fail), 4) if fail else None,
        "mean_phit_of_failed_calls": round(
            statistics.fmean(r["p_hit_actual_clear_point"] for r in fail), 4) if fail else None,
        "mean_phit_of_successful_calls": round(
            statistics.fmean(r["p_hit_actual_clear_point"] for r in succ), 4) if succ else None,
        "wasted_seconds_post_clear": round(
            sum(CLEAR_FAIL_SECONDS + r["clear_move_m"] / SPEED_MPS for r in pre), 1),
        "wasted_seconds_failed_fresh": round(
            sum(CLEAR_FAIL_SECONDS + r["clear_move_m"] / SPEED_MPS for r in fail), 1),
    }

    # ---- (D) 覆盖计划代价：后验驱动的补点数 vs 现状固定 6 点螺旋 ----
    report["D_coverage_plan"] = {
        "discs_to_cover_95pct_last4": describe("k", [r["discs_95"] for r in rows]),
        "discs_to_cover_100pct_last4": describe("k", [r["discs_100"] for r in rows]),
        "discs_to_cover_95pct_allobs": describe("k", [r["discs_95_allobs"] for r in rows]),
        "discs_to_cover_100pct_allobs": describe("k", [r["discs_100_allobs"] for r in rows]),
        "frac_needing_le_6_discs_last4": round(statistics.fmean(
            1.0 if (r["discs_95"] or 99) <= 6 else 0.0 for r in rows), 4),
        "frac_needing_le_6_discs_allobs": round(statistics.fmean(
            1.0 if (r["discs_95_allobs"] or 99) <= 6 else 0.0 for r in rows), 4),
        "note": "现状 _spiral 固定 6 点 / 半径 15 m，只能覆盖直径 <= ~70 m 的后验",
    }

    # ---- 决策论收益估算 ----
    # 现状：到 estimate 距离 <= 55 m 即清除，成功概率 = p_hit_point_est
    # 后验策略：若 best_single 的 p_hit 足够高则清除，否则先再测一次
    if fresh_clr:
        cur_success = statistics.fmean(1.0 if r["clear_ok"] else 0.0 for r in fresh_clr)
        cur_phit = statistics.fmean(r["p_hit_actual_clear_point"] for r in fresh_clr)
        best_phit = statistics.fmean(r["p_hit_best_single"] for r in fresh_clr)
        best_phit_all = statistics.fmean(r["p_hit_best_single_allobs"] for r in fresh_clr)
        # 期望清除调用次数 = 1 / p_hit（几何分布，粗略上界；用覆盖计划更准）
        report["E_decision_value"] = {
            "observed_fresh_clear_success_rate": round(cur_success, 4),
            "mean_phit_current_clear_point": round(cur_phit, 4),
            "mean_phit_posterior_best_point": round(best_phit, 4),
            "mean_phit_posterior_best_point_allobs": round(best_phit_all, 4),
            "relative_phit_gain_best_vs_current": round(
                best_phit / cur_phit - 1.0, 4) if cur_phit > 0 else None,
            "relative_phit_gain_allobs_vs_current": round(
                best_phit_all / cur_phit - 1.0, 4) if cur_phit > 0 else None,
            "expected_clear_calls_current_1_over_p": round(1.0 / cur_phit, 2) if cur_phit > 0 else None,
            "expected_clear_calls_posterior_plan_median": q(
                [r["discs_100"] for r in fresh_clr if r["discs_100"]], 0.5),
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"b_belief_analysis_{args.label}.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["record", "analyze"], required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed-start", type=int, default=20000)
    p.add_argument("--directional", type=int, default=0)
    p.add_argument("--label", default="q3")
    p.add_argument("--samples", type=int, default=2500)
    p.add_argument("--max-records", type=int, default=1200)
    args = p.parse_args()
    if args.mode == "record":
        do_record(args)
    else:
        do_analyze(args)


if __name__ == "__main__":
    main()
