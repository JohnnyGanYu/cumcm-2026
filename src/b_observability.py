"""B 题可观测性天花板量化。

回答四个问题：
1. 漏失源里 "从未拿到任何示向度"（可观测性失败）vs "拿到过示向度但没清除"（策略失败）的比例。
2. 每个从未被观测到的源，其"可见区"（区域内能收到它信号的点集）占目标区域面积的比例分布。
3. 现有站位集合的可见率，以及把站位推到 60/100 个、或按覆盖增益贪心主动选点后的可见率上限。
4. 完全不可观测（可见区 ∩ 目标区域为空）的源占比 = 清除率硬上限的补。

方法学要点：
- "可见区"用 arena.Jammer.covers() 的真值逻辑，在目标区域内做 20 m 网格采样统计。
  为速度做了 numpy 向量化，但 `--self-check` 会逐点比对向量化结果与 covers() 是否完全一致。
- 纯几何分析不跑仿真（更快更干净）；只有 Q1 需要跑完整对局，用 arena.robot.trace
  打点判断"是否拿到过示向度"，不改 strategy.py。
- 区分 "可观测"（存在区域内的点能收到）与 "被覆盖"（现有站位里有点能收到）。

子命令：
  geom      纯几何：可见区分布 + 各布局可见率 + 贪心主动选点上限
  episodes  跑完整对局：漏失归因（可观测性失败 vs 策略失败）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .b_sim.arena import (
    ARENA_RADIUS,
    DIRECTIONAL_HALF_ANGLE_DEG,
    Arena,
    Jammer,
    generate_case,
)

RESULTS = "results"

# ---------------------------------------------------------------- 网格


def arena_grid(step: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """目标区域（半径 1800 m 圆盘）内的规则网格点。"""
    axis = np.arange(-ARENA_RADIUS, ARENA_RADIUS + 1e-9, step)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    inside = gx * gx + gy * gy <= ARENA_RADIUS * ARENA_RADIUS
    return gx[inside], gy[inside]


def visible_mask(j: Jammer, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """向量化的 covers() + 距离判定。返回 bool 掩码。"""
    dx, dy = gx - j.x, gy - j.y
    in_range = dx * dx + dy * dy <= j.reception_radius_m ** 2
    if j.directional_deg is None:
        return in_range
    bearing = np.degrees(np.arctan2(dy, dx)) % 360.0
    diff = np.abs(bearing - (j.directional_deg % 360.0))
    gap = np.minimum(diff, 360.0 - diff)
    return in_range & (gap <= DIRECTIONAL_HALF_ANGLE_DEG)


def exact_visible_area_m2(j: Jammer, n_theta: int = 4096) -> float:
    """解析计算可见区面积（不受网格分辨率限制）。

    可见区 = 源接收圆（定向源再交 180 度扇区）∩ 目标区域圆盘。
    用以源为极点的极坐标积分：对每个角度 θ，可见半径 = min(接收半径, 到区域边界的距离)，
    只在扇区内的 θ 上积分。这解决了 20 m 网格分辨率下限（0.0039% 面积）
    无法区分"真的为空"与"小于一个格子"的问题。
    """
    rho = math.hypot(j.x, j.y)          # 源到区域中心距离
    thetas = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    if j.directional_deg is not None:
        h = math.radians(DIRECTIONAL_HALF_ANGLE_DEG)
        c = math.radians(j.directional_deg)
        d = np.abs((thetas - c + math.pi) % (2 * math.pi) - math.pi)
        thetas = thetas[d <= h]
        if thetas.size == 0:
            return 0.0
    # 沿方向 θ，射线与区域边界圆(半径 R, 圆心距源 rho)的交点距离
    # 源指向圆心的方向角
    if rho > 0:
        phi = math.atan2(-j.y, -j.x)
    else:
        phi = 0.0
    cosa = np.cos(thetas - phi)
    disc = ARENA_RADIUS ** 2 - rho ** 2 * (1.0 - cosa ** 2)
    t = np.where(disc >= 0.0, rho * cosa + np.sqrt(np.maximum(disc, 0.0)), 0.0)
    t = np.maximum(t, 0.0)              # 源在区域内时前向交点即出界距离
    rad = np.minimum(t, j.reception_radius_m)
    dtheta = 2 * math.pi / n_theta
    return float(0.5 * np.sum(rad ** 2) * dtheta)


ARENA_AREA_M2 = math.pi * ARENA_RADIUS ** 2


def self_check(n: int = 40000, seed: int = 1) -> dict:
    """逐点比对向量化判定与 arena 原生 covers()/distance_to()。"""
    rng = random.Random(seed)
    gx, gy = arena_grid(60.0)
    bad = 0
    checked = 0
    for _ in range(n // 200):
        case = generate_case(rng.randrange(10 ** 6), count=16, directional_count=8)
        j = rng.choice(case.jammers)
        idx = rng.sample(range(len(gx)), 200)
        vm = visible_mask(j, gx[idx], gy[idx])
        for k, i in enumerate(idx):
            truth = j.covers(float(gx[i]), float(gy[i])) and \
                j.distance_to(float(gx[i]), float(gy[i])) <= j.reception_radius_m
            checked += 1
            if bool(vm[k]) != truth:
                bad += 1
    return {"pairs_checked": checked, "mismatches": bad}


# ---------------------------------------------------------------- 站位布局


def ring(radius: float, n: int, phase: float = 0.0) -> list[tuple[float, float]]:
    return [(radius * math.cos(2 * math.pi * k / n + phase),
             radius * math.sin(2 * math.pi * k / n + phase)) for k in range(n)]


def layouts() -> dict[str, list[tuple[float, float]]]:
    """现有站位集合 + 加密到 60/100 站位的规则布局。"""
    origin = [(0.0, 0.0)]
    inner = ring(1250.0, 12)
    out_q3 = ring(1750.0, 14, math.pi / 14)
    out_q4 = ring(1750.0, 18, math.pi / 18)
    return {
        "L01_origin": origin,
        "L13_origin+1250x12": origin + inner,
        "L27_Q3_+1750x14": origin + inner + out_q3,
        "L31_Q4_+1750x18": origin + inner + out_q4,
        "L60_4ring": origin + ring(600.0, 8) + ring(1100.0, 14)
        + ring(1500.0, 18, math.pi / 18) + ring(1780.0, 19),
        "L100_5ring": origin + ring(450.0, 8) + ring(850.0, 14)
        + ring(1230.0, 21, math.pi / 21) + ring(1560.0, 28)
        + ring(1790.0, 28, math.pi / 28),
    }


def covered_by(j: Jammer, stations: list[tuple[float, float]]) -> bool:
    """现有站位里是否存在能收到该源信号的点（真值 covers()，不采样）。"""
    r2 = j.reception_radius_m ** 2
    for sx, sy in stations:
        if (sx - j.x) ** 2 + (sy - j.y) ** 2 <= r2 and j.covers(sx, sy):
            return True
    return False


# ---------------------------------------------------------------- 几何分析


def source_pool(n_cases: int, directional_count: int, count, seed0: int) -> list[tuple[int, Jammer]]:
    pool = []
    for i in range(n_cases):
        seed = seed0 + i
        case = generate_case(seed, count=count, directional_count=directional_count)
        for j in case.jammers:
            pool.append((seed, j))
    return pool


def pctl(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_geom(args) -> None:
    gx, gy = arena_grid(args.step)
    n_grid = len(gx)
    lays = layouts()
    scen = [("Q3_omni", 0, None), ("Q4_d5", 5, None), ("Q4_d10", 10, None),
            ("Q4_d16", 16, 16)]
    out = {"grid": {"step_m": args.step, "points": n_grid,
                    "arena_radius_m": ARENA_RADIUS},
           "self_check": self_check() if args.self_check else None,
           "layout_sizes": {k: len(v) for k, v in lays.items()},
           "scenarios": {}}

    for name, dcount, count in scen:
        pool = source_pool(args.cases, dcount, count, args.seed_start)
        rows = []
        for seed, j in pool:
            vm = visible_mask(j, gx, gy)
            vis_frac = float(vm.sum()) / n_grid
            rec = {"seed": seed, "channel": j.channel,
                   "dir": None if j.directional_deg is None else round(j.directional_deg, 2),
                   "r": round(j.reception_radius_m, 1),
                   "rho": round(math.hypot(j.x, j.y), 1),
                   "vis_frac": vis_frac,
                   "observable": vis_frac > 0.0}
            for lname, st_ in lays.items():
                rec[lname] = covered_by(j, st_)
            rows.append(rec)

        n = len(rows)
        dirs = [r for r in rows if r["dir"] is not None]
        omni = [r for r in rows if r["dir"] is None]
        summary = {
            "sources": n, "directional": len(dirs), "omni": len(omni),
            "observable_pct": round(100.0 * sum(r["observable"] for r in rows) / n, 3),
            "unobservable_pct": round(100.0 * sum(not r["observable"] for r in rows) / n, 3),
            "coverage_pct": {ln: round(100.0 * sum(r[ln] for r in rows) / n, 3)
                             for ln in lays},
        }
        if dirs:
            summary["directional_coverage_pct"] = {
                ln: round(100.0 * sum(r[ln] for r in dirs) / len(dirs), 3) for ln in lays}
            summary["directional_observable_pct"] = round(
                100.0 * sum(r["observable"] for r in dirs) / len(dirs), 3)
        # 现有 Q4 布局漏掉的源的可见区分布
        key = "L31_Q4_+1750x18" if dcount else "L13_origin+1250x12"
        missed = [r for r in rows if not r[key]]
        vf = [r["vis_frac"] * 100 for r in missed]
        summary["uncovered_by_" + key] = {
            "count": len(missed), "pct_of_sources": round(100.0 * len(missed) / n, 3),
            "vis_frac_pct": {"p10": round(pctl(vf, .10), 4), "p50": round(pctl(vf, .50), 4),
                             "p90": round(pctl(vf, .90), 4),
                             "max": round(max(vf), 4) if vf else None,
                             "mean": round(st.fmean(vf), 4) if vf else None,
                             "zero_count": sum(1 for x in vf if x == 0.0)},
        }
        allvf = [r["vis_frac"] * 100 for r in rows]
        summary["all_sources_vis_frac_pct"] = {
            "p10": round(pctl(allvf, .10), 3), "p50": round(pctl(allvf, .50), 3),
            "p90": round(pctl(allvf, .90), 3), "min": round(min(allvf), 4)}
        out["scenarios"][name] = summary
        print(f"[geom] {name}: n={n} observable={summary['observable_pct']}% "
              f"L13={summary['coverage_pct']['L13_origin+1250x12']}% "
              f"L31={summary['coverage_pct']['L31_Q4_+1750x18']}% "
              f"L60={summary['coverage_pct']['L60_4ring']}% "
              f"L100={summary['coverage_pct']['L100_5ring']}%", flush=True)
        # 明细只存少量（漏失样本）以控制文件体积
        out["scenarios"][name]["uncovered_examples"] = sorted(
            [{k: r[k] for k in ("seed", "channel", "dir", "r", "rho", "vis_frac", "observable")}
             for r in missed], key=lambda r: r["vis_frac"])[:40]

    path = os.path.join(RESULTS, f"b_observability_geom_{args.label}.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)
    print("wrote " + path)


# ---------------------------------------------------------------- 贪心主动选点


def run_greedy(args) -> None:
    """按覆盖增益贪心选站位：给出"任意布局 + K 个站位"能达到的可见率上限（近似）。

    候选站位 = 目标区域内 step_cand 网格；目标 = 训练源池。
    贪心是最大覆盖问题的 (1-1/e) 近似，且这里同时报 holdout 可见率，避免过拟合。
    """
    cx, cy = arena_grid(args.step_cand)
    print(f"[greedy] candidates={len(cx)}", flush=True)
    scen = [("Q3_omni", 0, None), ("Q4_d5", 5, None), ("Q4_d10", 10, None),
            ("Q4_d16", 16, 16)]
    out = {"candidate_stations": int(len(cx)), "step_cand_m": args.step_cand,
           "train_cases": args.cases, "holdout_cases": args.cases, "scenarios": {}}
    for name, dcount, count in scen:
        train = source_pool(args.cases, dcount, count, args.seed_start)
        hold = source_pool(args.cases, dcount, count, args.seed_start + 500000)

        def matrix(pool):
            M = np.zeros((len(cx), len(pool)), dtype=bool)
            for k, (_, j) in enumerate(pool):
                M[:, k] = visible_mask(j, cx, cy)
            return M

        Mt, Mh = matrix(train), matrix(hold)
        obs_t = Mt.any(axis=0)
        chosen: list[int] = []
        curve = []
        covered = np.zeros(Mt.shape[1], dtype=bool)
        for k in range(args.max_stations):
            gain = (Mt & ~covered).sum(axis=1)
            best = int(np.argmax(gain))
            if gain[best] == 0:
                break
            chosen.append(best)
            covered |= Mt[best]
            if len(chosen) in (1, 5, 9, 13, 20, 31, 40, 60, 80, 100, 150, 200):
                hc = Mh[chosen].any(axis=0)
                curve.append({"stations": len(chosen),
                              "train_pct": round(100.0 * covered.mean(), 3),
                              "holdout_pct": round(100.0 * hc.mean(), 3)})
        hc = Mh[chosen].any(axis=0)
        out["scenarios"][name] = {
            "sources_train": int(Mt.shape[1]), "sources_holdout": int(Mh.shape[1]),
            "observable_train_pct": round(100.0 * obs_t.mean(), 3),
            "observable_holdout_pct": round(100.0 * Mh.any(axis=0).mean(), 3),
            "greedy_curve": curve,
            "greedy_final_stations": len(chosen),
            "greedy_final_train_pct": round(100.0 * covered.mean(), 3),
            "greedy_final_holdout_pct": round(100.0 * hc.mean(), 3),
            "chosen_first20": [[round(float(cx[i]), 1), round(float(cy[i]), 1)]
                               for i in chosen[:20]],
        }
        print(f"[greedy] {name}: obs={out['scenarios'][name]['observable_train_pct']}% "
              f"curve={[(c['stations'], c['holdout_pct']) for c in curve]}", flush=True)
    path = os.path.join(RESULTS, f"b_observability_greedy_{args.label}.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)
    print("wrote " + path)


# ---------------------------------------------------------------- 完整对局


def _episode(task) -> dict:
    seed, dcount, count, step = task
    # 延迟导入，避免子进程重复开销进入主模块
    from .b_sim.client import RobotClient
    from .b_sim.server import Session, make_server
    from .b_sim.strategy import DogStrategy, StrategyConfig

    case = generate_case(seed, count=count, directional_count=dcount)
    arena = Arena(case)
    session = Session("dog-obs", arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient("dog-obs", base_url="http://127.0.0.1:" + str(port))
        DogStrategy(client, StrategyConfig(directional_probe=dcount > 0)).run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    # 每个频道是否拿到过示向度/near（= 是否被实际观测到）
    seen: dict[int, int] = {}
    for e in arena.robot.trace:
        if e["action"] == "measure" and e.get("measure_result") in ("direction", "near"):
            seen[e["channel"]] = seen.get(e["channel"], 0) + 1
    s = arena.stats()
    gx, gy = arena_grid(step)
    n_grid = len(gx)
    missed = []
    for j in case.jammers:
        if j.cleared:
            continue
        vf = float(visible_mask(j, gx, gy).sum()) / n_grid
        missed.append({
            "channel": j.channel, "directional": j.directional_deg is not None,
            "bearings_obtained": seen.get(j.channel, 0),
            "never_observed": seen.get(j.channel, 0) == 0,
            "vis_frac_pct": round(100.0 * vf, 5),
            "observable": vf > 0.0,
            "rho": round(math.hypot(j.x, j.y), 1),
            "r": round(j.reception_radius_m, 1),
        })
    return {"seed": seed, "dcount": dcount, "total": s["total_jammers"],
            "cleared": s["cleared"], "ratio": s["cleared_ratio"],
            "mean_clear_time_s": s["mean_clear_time_s"],
            "measure_calls": s["measure_calls"], "missed": missed}


def run_episodes(args) -> None:
    scen = [("Q3_omni", 0, None), ("Q4_d5", 5, None), ("Q4_d10", 10, None),
            ("Q4_d16", 16, 16)]
    if args.only:
        scen = [s for s in scen if s[0] in args.only.split(",")]
    tasks, meta = [], []
    for name, dcount, count in scen:
        for i in range(args.episodes):
            tasks.append((args.seed_start + i, dcount, count, args.step))
            meta.append(name)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_episode, tasks, chunksize=4)):
            r["scenario"] = meta[k]
            rows.append(r)
            if (k + 1) % 50 == 0:
                print(f"[ep] {k+1}/{len(tasks)}", flush=True)

    report = {"episodes_total": len(rows), "step_m": args.step, "scenarios": {}}
    for name, _, _ in scen:
        rs = [r for r in rows if r["scenario"] == name]
        if not rs:
            continue
        allm = [m for r in rs for m in r["missed"]]
        never = [m for m in allm if m["never_observed"]]
        strat = [m for m in allm if not m["never_observed"]]
        vf = [m["vis_frac_pct"] for m in never]
        tot_src = sum(r["total"] for r in rs)
        mct = [r["mean_clear_time_s"] for r in rs if r["mean_clear_time_s"]]
        report["scenarios"][name] = {
            "episodes": len(rs), "sources": tot_src,
            "cleared": sum(r["cleared"] for r in rs),
            "clear_rate_pct": round(100.0 * sum(r["cleared"] for r in rs) / tot_src, 3),
            "full_clear_episodes_pct": round(
                100.0 * sum(1 for r in rs if r["ratio"] >= 1 - 1e-9) / len(rs), 2),
            "mean_clear_time_s": round(st.fmean(mct), 1) if mct else None,
            "missed_total": len(allm),
            "missed_per_episode": round(len(allm) / len(rs), 4),
            "miss_attribution": {
                "observability_failure_never_observed": len(never),
                "strategy_failure_observed_not_cleared": len(strat),
                "never_observed_pct_of_misses": round(100.0 * len(never) / len(allm), 2) if allm else None,
                "strategy_pct_of_misses": round(100.0 * len(strat) / len(allm), 2) if allm else None,
            },
            "never_observed_directional_share_pct": round(
                100.0 * sum(1 for m in never if m["directional"]) / len(never), 2) if never else None,
            "never_observed_vis_frac_pct": {
                "p50": round(pctl(vf, .5), 4), "p90": round(pctl(vf, .9), 4),
                "max": round(max(vf), 4) if vf else None,
                "mean": round(st.fmean(vf), 4) if vf else None,
                "n_zero_unobservable": sum(1 for x in vf if x == 0.0),
                "n_below_0p5pct": sum(1 for x in vf if x < 0.5),
            },
            "strategy_failure_vis_frac_pct": {
                "p50": round(pctl([m["vis_frac_pct"] for m in strat], .5), 4) if strat else None,
                "mean": round(st.fmean([m["vis_frac_pct"] for m in strat]), 4) if strat else None,
            },
            "worst_misses": sorted(never, key=lambda m: m["vis_frac_pct"])[:10],
        }
        a = report["scenarios"][name]
        print(f"[ep] {name}: eps={a['episodes']} clear={a['clear_rate_pct']}% "
              f"miss={a['missed_total']} never_obs={len(never)} "
              f"({a['miss_attribution']['never_observed_pct_of_misses']}%) "
              f"visfrac p50={a['never_observed_vis_frac_pct']['p50']}%", flush=True)

    path = os.path.join(RESULTS, f"b_observability_episodes_{args.label}.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print("wrote " + path)


def run_reachable(args) -> None:
    """实用可观测性：站位半径受限时还剩多少源可观测。

    动机：所有"极难观测"的源可见区都只剩区域最外缘的一小片（witness 点 ρ≈1798 m）。
    机器狗贴着 1800 m 边界巡逻既不现实也不安全，所以真正的天花板要看
    "站位限制在 ρ ≤ R_cap 时可观测比例"。R_cap 是工程可选的巡逻外沿。
    """
    caps = [float(c) for c in args.caps.split(",")]
    scen = [("Q3_omni", 0, None), ("Q4_d5", 5, None), ("Q4_d10", 10, None),
            ("Q4_d16", 16, 16)]
    out = {"cases": args.cases, "caps_m": caps, "step_m": args.step, "scenarios": {}}
    for name, dcount, count in scen:
        pool = source_pool(args.cases, dcount, count, args.seed_start)
        res = {}
        for cap in caps:
            gx, gy = arena_grid(args.step)
            keep = gx * gx + gy * gy <= cap * cap
            cgx, cgy = gx[keep], gy[keep]
            ok = 0
            for _, j in pool:
                # 该源在 ρ<=cap 的子区域内是否存在可接收点（网格判据 + 边缘细化）
                if visible_mask(j, cgx, cgy).any():
                    ok += 1
                    continue
                # 网格漏判兜底：沿扇区角度密扫 cap 圆周与内部若干半径
                found = False
                nth = 2000
                ths = np.linspace(0, 2 * math.pi, nth, endpoint=False)
                if j.directional_deg is not None:
                    h = math.radians(DIRECTIONAL_HALF_ANGLE_DEG)
                    c0 = math.radians(j.directional_deg)
                    d0 = np.abs((ths - c0 + math.pi) % (2 * math.pi) - math.pi)
                    ths = ths[d0 <= h]
                for frac in (1.0, 0.98, 0.9, 0.75, 0.5, 0.25, 0.1):
                    r = j.reception_radius_m * frac
                    px, py = j.x + r * np.cos(ths), j.y + r * np.sin(ths)
                    m = (px * px + py * py <= cap * cap)
                    if m.any():
                        found = True
                        break
                ok += int(found)
            res[str(int(cap))] = round(100.0 * ok / len(pool), 4)
        out["scenarios"][name] = {"sources": len(pool), "observable_pct_by_cap": res}
        print(f"[reach] {name}: {res}", flush=True)
    path = os.path.join(RESULTS, f"b_observability_reachable_{args.label}.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)
    print("wrote " + path)


def run_exact(args) -> None:
    """解析面积版可见区分析。

    目的：20 m 网格的分辨率下限是区域面积的 0.0039%，无法区分"可见区真的为空"
    与"可见区小于一个格子"。这里用 exact_visible_area_m2 精确算，并与网格互校。
    """
    gx, gy = arena_grid(args.step)
    n_grid = len(gx)
    lays = layouts()
    scen = [("Q3_omni", 0, None), ("Q4_d5", 5, None), ("Q4_d10", 10, None),
            ("Q4_d16", 16, 16)]
    out = {"arena_area_m2": round(ARENA_AREA_M2, 1),
           "grid_cell_share_pct": round(100.0 * args.step ** 2 / ARENA_AREA_M2, 5),
           "cases": args.cases, "scenarios": {}}
    calib = []
    for name, dcount, count in scen:
        pool = source_pool(args.cases, dcount, count, args.seed_start)
        exact_pct, grid_pct, cov = [], [], {ln: [] for ln in lays}
        for _, j in pool:
            a = exact_visible_area_m2(j) / ARENA_AREA_M2 * 100.0
            exact_pct.append(a)
            if len(grid_pct) < args.calib:
                g = 100.0 * float(visible_mask(j, gx, gy).sum()) / n_grid
                grid_pct.append(g)
                calib.append((a, g))
            for ln, sts in lays.items():
                cov[ln].append(covered_by(j, sts))
        n = len(exact_pct)
        zero = sum(1 for a in exact_pct if a <= 0.0)
        tiny = sum(1 for a in exact_pct if 0.0 < a < 0.01)
        key = "L31_Q4_+1750x18"
        unc = [exact_pct[i] for i in range(n) if not cov[key][i]]
        out["scenarios"][name] = {
            "sources": n,
            "exact_unobservable_count": zero,
            "exact_unobservable_pct": round(100.0 * zero / n, 5),
            "exact_area_below_0p01pct_count": tiny,
            "exact_vis_area_pct": {
                "min": round(min(exact_pct), 6), "p1": round(pctl(exact_pct, .01), 4),
                "p10": round(pctl(exact_pct, .10), 3), "p50": round(pctl(exact_pct, .50), 3),
                "p90": round(pctl(exact_pct, .90), 3), "max": round(max(exact_pct), 3)},
            "uncovered_by_L31_exact_vis_area_pct": {
                "count": len(unc), "pct_of_sources": round(100.0 * len(unc) / n, 4),
                "p50": round(pctl(unc, .5), 4) if unc else None,
                "p90": round(pctl(unc, .9), 4) if unc else None,
                "max": round(max(unc), 4) if unc else None,
                "mean": round(st.fmean(unc), 4) if unc else None,
                "n_zero": sum(1 for a in unc if a <= 0.0),
                "n_below_0p05pct": sum(1 for a in unc if a < 0.05)},
            "coverage_pct": {ln: round(100.0 * sum(cov[ln]) / n, 4) for ln in lays},
        }
        s = out["scenarios"][name]
        print(f"[exact] {name}: n={n} unobs={s['exact_unobservable_pct']}% "
              f"(<0.01%: {tiny}) L31={s['coverage_pct'][key]}% "
              f"unc_p50={s['uncovered_by_L31_exact_vis_area_pct']['p50']}%", flush=True)
    if calib:
        err = [abs(a - g) for a, g in calib]
        rel = [abs(a - g) / a for a, g in calib if a > 0.5]
        out["grid_vs_exact_calibration"] = {
            "pairs": len(calib), "mean_abs_diff_pct_pts": round(st.fmean(err), 5),
            "max_abs_diff_pct_pts": round(max(err), 5),
            "mean_rel_err_where_area_gt_0p5pct": round(st.fmean(rel), 5) if rel else None}
        print("[exact] calib " + json.dumps(out["grid_vs_exact_calibration"]), flush=True)
    path = os.path.join(RESULTS, f"b_observability_exact_{args.label}.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)
    print("wrote " + path)


def run_rim(args) -> None:
    """边缘专项：漏失源几乎全在 rho≈1760-1800 m 的区域边缘，测边缘加密的边际收益。

    也按源到中心距离 rho 分箱给出条件可见率，定位失效到底集中在哪。
    """
    origin = [(0.0, 0.0)]
    inner = ring(1250.0, 12)
    base31 = origin + inner + ring(1750.0, 18, math.pi / 18)
    cands = {
        "L31_base": base31,
        "L31+1795x18": base31 + ring(1795.0, 18),
        "L31+1795x36": base31 + ring(1795.0, 36),
        "L31+1799x72": base31 + ring(1799.0, 72),
        "L31+1799x144": base31 + ring(1799.0, 144),
        "rim_only_1799x360": origin + ring(1799.0, 360),
    }
    scen = [("Q4_d5", 5, None), ("Q4_d10", 10, None), ("Q4_d16", 16, 16)]
    bins = [(0, 800), (800, 1400), (1400, 1700), (1700, 1750),
            (1750, 1780), (1780, 1800)]
    out = {"cases": args.cases, "layout_sizes": {k: len(v) for k, v in cands.items()},
           "scenarios": {}}
    for name, dcount, count in scen:
        pool = source_pool(args.cases, dcount, count, args.seed_start)
        res = {ln: 0 for ln in cands}
        binstat = {f"{a}-{b}": {"n": 0, "L31": 0, "best": 0} for a, b in bins}
        for _, j in pool:
            rho = math.hypot(j.x, j.y)
            hits = {ln: covered_by(j, sts) for ln, sts in cands.items()}
            for ln, v in hits.items():
                res[ln] += v
            for a, b in bins:
                if a <= rho < b or (b == 1800 and rho >= a):
                    k = f"{a}-{b}"
                    binstat[k]["n"] += 1
                    binstat[k]["L31"] += hits["L31_base"]
                    binstat[k]["best"] += hits["L31+1799x144"]
                    break
        n = len(pool)
        out["scenarios"][name] = {
            "sources": n,
            "coverage_pct": {ln: round(100.0 * res[ln] / n, 4) for ln in cands},
            "by_rho_bin": {k: {"n": v["n"],
                               "share_of_sources_pct": round(100.0 * v["n"] / n, 3),
                               "L31_cov_pct": round(100.0 * v["L31"] / v["n"], 3) if v["n"] else None,
                               "L31+1799x144_cov_pct": round(100.0 * v["best"] / v["n"], 3) if v["n"] else None}
                          for k, v in binstat.items()},
        }
        print(f"[rim] {name}: " + json.dumps(out["scenarios"][name]["coverage_pct"]), flush=True)
    path = os.path.join(RESULTS, f"b_observability_rim_{args.label}.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)
    print("wrote " + path)


def run_twomode(args) -> None:
    """把"未被覆盖"拆成两种失效模式，并各自给对症布局。

    rim 分析暴露出第二种模式：rho<800 m 的近心源在 L31 下也有 ~0.8-0.9% 未被覆盖，
    且边缘加密对它完全无效。原因是接收半径下限 1000 m 加 180 度扇区后，
    近心定向源的可见区可能整片落在"内环 1250 m 与原点之间"的环带里，
    而 L31 在该环带没有站位（只有原点与 1250/1750 两环）。

    对症布局：加一圈中半径环（如 700 m x 8）。这里直接量化其收益。
    """
    origin = [(0.0, 0.0)]
    inner = ring(1250.0, 12)
    base31 = origin + inner + ring(1750.0, 18, math.pi / 18)
    cands = {
        "L31_base": base31,
        "L31+700x8": base31 + ring(700.0, 8),
        "L31+700x8+1795x18": base31 + ring(700.0, 8) + ring(1795.0, 18),
        "L31+600x6+1000x10+1795x24": base31 + ring(600.0, 6) + ring(1000.0, 10)
        + ring(1795.0, 24),
    }
    scen = [("Q3_omni", 0, None), ("Q4_d5", 5, None), ("Q4_d10", 10, None),
            ("Q4_d16", 16, 16)]
    out = {"cases": args.cases,
           "layout_sizes": {k: len(v) for k, v in cands.items()}, "scenarios": {}}
    for name, dcount, count in scen:
        pool = source_pool(args.cases, dcount, count, args.seed_start)
        res = {ln: 0 for ln in cands}
        # 未被 L31 覆盖的源按 rho 归入 near(<1000) / mid / rim(>1700)
        modes = {"near_lt1000": 0, "mid_1000_1700": 0, "rim_gt1700": 0}
        recovered = {ln: dict(near_lt1000=0, mid_1000_1700=0, rim_gt1700=0)
                     for ln in cands}
        for _, j in pool:
            rho = math.hypot(j.x, j.y)
            hits = {ln: covered_by(j, sts) for ln, sts in cands.items()}
            for ln, v in hits.items():
                res[ln] += v
            if not hits["L31_base"]:
                m = ("near_lt1000" if rho < 1000 else
                     "rim_gt1700" if rho > 1700 else "mid_1000_1700")
                modes[m] += 1
                for ln in cands:
                    if hits[ln]:
                        recovered[ln][m] += 1
        n = len(pool)
        out["scenarios"][name] = {
            "sources": n,
            "coverage_pct": {ln: round(100.0 * res[ln] / n, 4) for ln in cands},
            "L31_uncovered_by_mode": modes,
            "L31_uncovered_total": sum(modes.values()),
            "recovered_from_L31_misses": recovered,
        }
        print(f"[2mode] {name}: cov=" + json.dumps(out["scenarios"][name]["coverage_pct"])
              + " modes=" + json.dumps(modes), flush=True)
    path = os.path.join(RESULTS, f"b_observability_twomode_{args.label}.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)
    print("wrote " + path)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("geom")
    g.add_argument("--cases", type=int, default=300)
    g.add_argument("--seed-start", type=int, default=700000)
    g.add_argument("--step", type=float, default=20.0)
    g.add_argument("--label", default="main")
    g.add_argument("--self-check", action="store_true")
    g.set_defaults(func=run_geom)

    gr = sub.add_parser("greedy")
    gr.add_argument("--cases", type=int, default=200)
    gr.add_argument("--seed-start", type=int, default=800000)
    gr.add_argument("--step-cand", type=float, default=100.0)
    gr.add_argument("--max-stations", type=int, default=200)
    gr.add_argument("--label", default="main")
    gr.set_defaults(func=run_greedy)

    x = sub.add_parser("exact")
    x.add_argument("--cases", type=int, default=2000)
    x.add_argument("--seed-start", type=int, default=700000)
    x.add_argument("--step", type=float, default=20.0)
    x.add_argument("--calib", type=int, default=2000)
    x.add_argument("--label", default="main")
    x.set_defaults(func=run_exact)

    e = sub.add_parser("episodes")
    e.add_argument("--episodes", type=int, default=200)
    e.add_argument("--seed-start", type=int, default=600000)
    e.add_argument("--step", type=float, default=25.0)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--only", default=None)
    e.add_argument("--label", default="main")
    e.set_defaults(func=run_episodes)

    tm = sub.add_parser("twomode")
    tm.add_argument("--cases", type=int, default=2000)
    tm.add_argument("--seed-start", type=int, default=700000)
    tm.add_argument("--label", default="main")
    tm.set_defaults(func=run_twomode)

    rm = sub.add_parser("rim")
    rm.add_argument("--cases", type=int, default=2000)
    rm.add_argument("--seed-start", type=int, default=700000)
    rm.add_argument("--label", default="main")
    rm.set_defaults(func=run_rim)

    rc = sub.add_parser("reachable")
    rc.add_argument("--cases", type=int, default=1000)
    rc.add_argument("--seed-start", type=int, default=700000)
    rc.add_argument("--step", type=float, default=20.0)
    rc.add_argument("--caps", default="1800,1780,1750,1700,1650,1600,1500")
    rc.add_argument("--label", default="main")
    rc.set_defaults(func=run_reachable)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
