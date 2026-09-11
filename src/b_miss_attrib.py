"""漏失源的真值归因 + 信息可达上限（离线诊断，不修改 src/b_sim/*）。

目的
----
`results/b_budget_verdict.md` 已确认：Q3 全向零漏失，Q4 定向残余漏失 0.13%-0.4%。
本模块只回答「这 0.4% 到底为什么漏、理论上能不能救」。

三层上限口径（互相嵌套，越往下越强）
------------------------------------
U_hard   : 该源的可见区（距离 <= r_recv ∩ 180° 辐射扇区 ∩ 目标区域内）是否非空。
           几何上几乎恒为真 —— 半平面与圆盘的交在源附近必非空，除非源恰在边界
           且辐射轴恰为外法线。用「源附近多半径多角度采样」判定非空，而不是用
           粗网格（粗网格会把 0.01% 的窄可见区误判成空）。
           **/clear 只看距离、与辐射方向无关**，所以只要 U_hard 成立，
           该源原则上可清除 => U_hard 就是信息可达上限。
U_layout : 规划站位集合（原点 + 内环 1250×12 + 定向外环 1750×18 + 内圈 4 点）
           中是否至少一个站位可收到信号。这是「靠固定布局发现」的上限。
U_traj   : 机器狗**实际站过的任意一点**是否可收到信号（含归航点、belief 补测点）。

漏失原因分类（互斥，按优先级判定）
----------------------------------
A_unobservable   U_hard 为假 —— 真正的信息缺陷（可见区为空）。
B_never_heard    U_hard 真但该频道所有检测点都收不到信号（可观测性存在却没被采样）。
                 细分 B1 = U_layout 也为假（布局缺口）、B2 = U_layout 真但没在那站测
                 （策略提前放弃 / 判空 / 预算门禁）。
C_homing_failed  收到过信号（near 或示向度）却没清掉：交会失败 / 丢信号 / 预算耗尽。
D_blind_miss     全程沉默，但 belief 候选非空且发过 /clear，只是没覆盖到真位置。

用法
----
  python -m src.b_miss_attrib deep   --plan known          # 复现已知漏失种子并归因
  python -m src.b_miss_attrib bound  --episodes 2000       # 纯几何上限（不跑仿真）
  python -m src.b_miss_attrib scan   --episodes 200 --directional 10
  python -m src.b_miss_attrib selftest
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .b_sim.arena import (
    ARENA_RADIUS,
    CLEAR_RADIUS_M,
    DIRECTIONAL_HALF_ANGLE_DEG,
    Arena,
    angular_gap_deg,
    generate_case,
    norm_deg,
)
from .b_sim.client import RobotClient
from .b_sim.directional import GRID_STEP_M, HEADING_STEP_DEG
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-miss"
RESULTS = Path(__file__).resolve().parent.parent / "results"

# 已知漏失样本（来自 results/b_budget_wide_d{5,10,16}.json 的 missed_episodes）
KNOWN_MISSES: list[tuple[int, int, int | None]] = (
    [(s, 5, None) for s in (7047, 7053, 7126, 7171)]
    + [(s, 10, None) for s in (7013, 7016, 7040, 7053, 7074, 7122, 7158, 7160, 7175, 7186)]
    + [(s, 16, 16) for s in (7028, 7124, 7175)]
)


# ------------------------------------------------------------------ 观察者


class MissProbe(DogStrategy):
    """只读观察者：所有钩子先调 super()，再记录，返回值原样转发。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.measure_log: dict[int, list[tuple]] = {}
        self.clear_log: dict[int, list[tuple]] = {}
        self.visited: list[tuple[float, float]] = [(0.0, 0.0)]
        self.first_signal_vt: dict[int, float] = {}
        self.first_bearing_vt: dict[int, float] = {}
        self.clear_vt: dict[int, float] = {}
        self.path: dict[int, list[str]] = {}
        self.phase = "sweep"

    # -- 钩子（只观察） --
    def _measure(self, point, channel):
        r = super()._measure(point, channel)
        vt = self.client.virtual_time_s
        self.visited.append((point[0], point[1]))
        if r is not None:
            self.measure_log.setdefault(channel, []).append(
                (point[0], point[1], r.result, r.svd_deg, vt, self.phase))
            if r.result != "no_signal":
                self.first_signal_vt.setdefault(channel, vt)
            if r.result == "direction":
                self.first_bearing_vt.setdefault(channel, vt)
        return r

    def _clear_at(self, point, channel):
        ok = super()._clear_at(point, channel)
        self.visited.append((point[0], point[1]))
        self.clear_log.setdefault(channel, []).append(
            (point[0], point[1], ok, self.client.virtual_time_s, self.phase))
        if ok:
            self.clear_vt.setdefault(channel, self.client.virtual_time_s)
        return ok

    def _note(self, ch, tag):
        self.path.setdefault(ch, []).append(tag)

    def _dogleg(self, state):
        r = super()._dogleg(state)
        self._note(state.channel, f"dogleg={'ok' if r else 'FAIL'}")
        return r

    def _ray_blind_clear(self, state):
        r = super()._ray_blind_clear(state)
        self._note(state.channel, f"ray={'ok' if r else 'fail'}")
        return r

    def _home_channel(self, state):
        r = super()._home_channel(state)
        self._note(state.channel, f"home={'ok' if r else 'GIVEUP'}")
        return r

    # -- 阶段标注（同样只调 super()，不改行为） --
    def _probe_ring(self):
        self.phase = "ring"
        try:
            return super()._probe_ring()
        finally:
            self.phase = "post_ring"

    def _belief_search(self, *a, **kw):
        self.phase = "belief"
        try:
            return super()._belief_search(*a, **kw)
        finally:
            self.phase = "post_belief"

    def _blind_clear_sweep(self, *a, **kw):
        self.phase = "blind"
        try:
            return super()._blind_clear_sweep(*a, **kw)
        finally:
            self.phase = "post_blind"


# ------------------------------------------------------------------ 几何（只在观察侧）


def receivable(sx, sy, r_recv, ddeg, px, py) -> bool:
    if math.hypot(px - sx, py - sy) > r_recv:
        return False
    if ddeg is None:
        return True
    if px == sx and py == sy:
        return True
    b = norm_deg(math.degrees(math.atan2(py - sy, px - sx)))
    return angular_gap_deg(b, ddeg) <= DIRECTIONAL_HALF_ANGLE_DEG


def planned_stations(directional: bool, cfg: StrategyConfig) -> list[tuple[float, float]]:
    pts = [(0.0, 0.0)] if cfg.sweep_origin else []
    layers = [(cfg.probe_ring_radius, cfg.probe_ring_stations, 0.0)]
    if directional or cfg.use_outer_ring:
        n = cfg.directional_ring_stations if directional else cfg.outer_ring_stations
        layers.append((cfg.outer_ring_radius, n, math.pi / max(1, n)))
    for radius, count, phase in layers:
        pts += [(radius * math.cos(2 * math.pi * k / count + phase),
                 radius * math.sin(2 * math.pi * k / count + phase))
                for k in range(count)]
    pts += [(0.0, 0.0), (300.0, 0.0), (-150.0, 260.0), (-150.0, -260.0)]
    return pts


_FINE_RADII = (0.5, 2.0, 5.0, 20.0, 60.0, 150.0, 400.0, 900.0)


def visible_nonempty(sx, sy, r_recv, ddeg) -> tuple[bool, float]:
    """可见区（∩目标区域）是否非空；返回 (非空?, 最近可见点到源的距离 m)。

    用「源附近多半径 × 扇区内多角度」采样，能分辨 0.01% 量级的窄可见区，
    这是粗网格做不到的。全向源只要在区域内就恒非空（源自身位置可见）。
    """
    if ddeg is None:
        return True, 0.0
    for d in _FINE_RADII:
        if d > r_recv:
            break
        # 扇区内角度采样（含轴线与两侧接近 90° 的边缘）
        for off in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 89.0, 89.9,
                    -15.0, -30.0, -45.0, -60.0, -75.0, -85.0, -89.0, -89.9):
            a = math.radians(ddeg + off)
            px, py = sx + d * math.cos(a), sy + d * math.sin(a)
            if math.hypot(px, py) <= ARENA_RADIUS:
                return True, d
    return False, float("inf")


def visible_area_frac(sx, sy, r_recv, ddeg, step: float = 20.0) -> float:
    """可见区占目标区域面积的比例（20 m 网格蒙特卡洛式估计）。"""
    n = int(ARENA_RADIUS / step)
    inside = 0
    ok = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            px, py = i * step, j * step
            if px * px + py * py > ARENA_RADIUS * ARENA_RADIUS:
                continue
            inside += 1
            if receivable(sx, sy, r_recv, ddeg, px, py):
                ok += 1
    return ok / inside if inside else 0.0


def edge_outward(sx, sy, ddeg) -> tuple[float, float]:
    """(到区域边界的距离, 辐射轴与「朝区域外」方向的夹角)。夹角 0 = 正朝外。"""
    r = math.hypot(sx, sy)
    edge = ARENA_RADIUS - r
    if ddeg is None or r < 1e-9:
        return edge, float("nan")
    outward = norm_deg(math.degrees(math.atan2(sy, sx)))
    return edge, angular_gap_deg(ddeg, outward)


# ------------------------------------------------------------------ 单局


def run_one(seed: int, directional: int, count: int | None = None,
            cls=MissProbe, overrides: dict | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional > 0)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    try:
        strat = cls(RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}"), cfg)
        strat.run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    stats = arena.stats()
    out = {
        "seed": seed, "directional": directional,
        "total": stats["total_jammers"], "cleared": stats["cleared"],
        "mean_clear_time_s": (None if stats["mean_clear_time_s"] is None
                              else round(stats["mean_clear_time_s"], 1)),
        "virtual_s": round(stats["total_virtual_time_s"], 1),
        "measure_calls": stats["measure_calls"], "clear_calls": stats["clear_calls"],
        "missed_channels": [r["channel"] for r in case.truth_table() if not r["cleared"]],
    }
    if not isinstance(strat, MissProbe):
        return out

    # ==== run 之后的离线归因：读真值不影响策略 ====
    stations = planned_stations(directional > 0, cfg)
    ev = " | ".join(strat.events)
    budget_hit = ("virtual time" in ev) or ("real time" in ev)
    out["budget_hit"] = budget_hit
    details = []
    for row in case.truth_table():
        if row["cleared"]:
            continue
        ch, sx, sy = row["channel"], row["x"], row["y"]
        r_recv, ddeg = row["reception_radius_m"], row["directional_deg"]
        mlog = strat.measure_log.get(ch, [])
        clog = strat.clear_log.get(ch, [])
        state = strat.channels[ch]

        got_signal = any(m[2] != "no_signal" for m in mlog)
        got_bearing = any(m[2] == "direction" for m in mlog)
        recv_pts = [m for m in mlog if receivable(sx, sy, r_recv, ddeg, m[0], m[1])]
        in_range = [m for m in mlog if math.hypot(m[0] - sx, m[1] - sy) <= r_recv]
        u_layout = any(receivable(sx, sy, r_recv, ddeg, px, py) for px, py in stations)
        u_traj = any(receivable(sx, sy, r_recv, ddeg, px, py) for px, py in strat.visited)
        u_hard, nearest_vis = visible_nonempty(sx, sy, r_recv, ddeg)
        edge_d, aim_out = edge_outward(sx, sy, ddeg)

        # belief 是否还保留了真状态
        belief_alive = state.belief is not None and state.belief.alive
        truth_in_belief = None
        belief_positions = None
        if state.belief is not None:
            pos = state.belief.positions
            pi_true = min(range(len(pos)),
                          key=lambda i: math.hypot(pos[i][0] - sx, pos[i][1] - sy))
            hi_true = int(round(norm_deg(ddeg if ddeg is not None else 0.0)
                                / HEADING_STEP_DEG)) % len(state.belief.headings)
            pis = {p for p, _, _ in state.belief.candidates}
            truth_in_belief = {
                "pos_cell_alive": pi_true in pis,
                "pos_heading_alive": (pi_true, hi_true, 0) in state.belief.candidates,
                "truth_cell_dist_m": round(math.hypot(pos[pi_true][0] - sx,
                                                      pos[pi_true][1] - sy), 1),
            }
            belief_positions = len(state.belief.candidate_positions())

        # 分类
        if got_signal or got_bearing:
            cause = "C_homing_failed"
        elif not u_hard:
            cause = "A_unobservable"
        elif clog:
            cause = "D_blind_miss"
        elif not u_layout:
            cause = "B1_layout_gap"
        else:
            cause = "B2_not_sampled"
        if budget_hit and cause != "C_homing_failed":
            cause += "+budget"

        details.append({
            "seed": seed, "channel": ch, "cause": cause,
            "x": round(sx, 1), "y": round(sy, 1),
            "r_from_origin": round(math.hypot(sx, sy), 1),
            "r_recv": round(r_recv, 1),
            "directional": ddeg is not None,
            "directional_deg": None if ddeg is None else round(ddeg, 1),
            "dist_to_edge_m": round(edge_d, 1),
            "aim_outward_gap_deg": None if ddeg is None else round(aim_out, 1),
            "u_hard": u_hard, "u_layout": u_layout, "u_traj": u_traj,
            "nearest_visible_m": None if nearest_vis == float("inf") else nearest_vis,
            "visible_area_frac": round(visible_area_frac(sx, sy, r_recv, ddeg), 5),
            "measures_on_ch": len(mlog), "measures_in_range": len(in_range),
            "measures_receivable": len(recv_pts),
            "got_signal": got_signal, "got_bearing": got_bearing,
            "bearings": len(state.bearings), "silent_points": len(state.silent_points),
            "clear_calls": len(clog),
            "min_clear_dist_m": (None if not clog else
                                 round(min(math.hypot(c[0] - sx, c[1] - sy) for c in clog), 1)),
            "min_measure_dist_m": (None if not mlog else
                                   round(min(math.hypot(m[0] - sx, m[1] - sy) for m in mlog), 1)),
            "min_visited_dist_m": round(min(math.hypot(p[0] - sx, p[1] - sy)
                                            for p in strat.visited), 1),
            "belief_alive": belief_alive, "belief_positions": belief_positions,
            "truth_in_belief": truth_in_belief,
            "state_exhausted": state.exhausted, "verdict_empty": state.verdict_empty,
            "path_tail": strat.path.get(ch, [])[-8:],
            "phases_measured": sorted({m[5] for m in mlog}),
            "phases_cleared": sorted({c[4] for c in clog}),
            "t_first_signal": None if ch not in strat.first_signal_vt else round(strat.first_signal_vt[ch]),
            "events": [e for e in strat.events if e.startswith(f"ch{ch}:")],
        })
    out["missed_detail"] = details
    return out


def resolve_strategy(name: str):
    if name == "MissProbe":
        return MissProbe
    if name == "DogStrategy":
        return DogStrategy
    from . import b_miss_rescue
    cls = getattr(b_miss_rescue, name, None)
    if cls is None:
        raise KeyError(f"unknown strategy: {name}")
    return cls


def _worker(args):
    seed, directional, count, clsname, overrides = args
    cls = resolve_strategy(clsname)
    try:
        return run_one(seed, directional, count, cls, overrides)
    except Exception as exc:  # noqa: BLE001
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------ 纯几何上限（不跑仿真）


def bound_only(episodes: int, seed_start: int, directional: int,
               count: int | None) -> dict:
    """只算几何量：可见区非空比例、U_layout 比例、可见面积占比分布。"""
    cfg = StrategyConfig(directional_probe=directional > 0)
    stations = planned_stations(directional > 0, cfg)
    n_src = n_dir = 0
    hard_empty = 0
    layout_ok = 0
    fracs: list[float] = []
    tiny = Counter()
    for i in range(episodes):
        case = generate_case(seed_start + i, count=count, directional_count=directional)
        for row in case.truth_table():
            sx, sy = row["x"], row["y"]
            r_recv, ddeg = row["reception_radius_m"], row["directional_deg"]
            n_src += 1
            u_hard, _ = visible_nonempty(sx, sy, r_recv, ddeg)
            if not u_hard:
                hard_empty += 1
            if any(receivable(sx, sy, r_recv, ddeg, px, py) for px, py in stations):
                layout_ok += 1
            if ddeg is not None:
                n_dir += 1
                f = visible_area_frac(sx, sy, r_recv, ddeg, step=40.0)
                fracs.append(f)
                for thr in (0.001, 0.005, 0.01, 0.05):
                    if f < thr:
                        tiny[f"frac<{thr}"] += 1
    return {
        "episodes": episodes, "seed_start": seed_start, "directional": directional,
        "sources": n_src, "directional_sources": n_dir,
        "u_hard_nonempty": n_src - hard_empty,
        "u_hard_rate": round((n_src - hard_empty) / n_src, 6) if n_src else None,
        "u_layout_ok": layout_ok,
        "u_layout_rate": round(layout_ok / n_src, 5) if n_src else None,
        "dir_visible_frac_mean": round(st.fmean(fracs), 5) if fracs else None,
        "dir_visible_frac_median": round(st.median(fracs), 5) if fracs else None,
        "dir_visible_frac_p05": (round(sorted(fracs)[max(0, int(0.05 * len(fracs)) - 1)], 5)
                                 if fracs else None),
        "dir_visible_frac_min": round(min(fracs), 6) if fracs else None,
        "narrow_counts": dict(tiny),
        "narrow_rates": {k: round(v / n_dir, 5) for k, v in tiny.items()} if n_dir else {},
    }


# ------------------------------------------------------------------ 聚合


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    missed = [m for r in ok for m in r.get("missed_detail", [])]
    cause = Counter(m["cause"] for m in missed)
    src_total = sum(r["total"] for r in ok)
    cleared = sum(r["cleared"] for r in ok)
    hard_bound = src_total - sum(1 for m in missed if not m["u_hard"])
    layout_bound = sum(1 for m in missed if m["u_layout"])
    return {
        "episodes": len(ok), "errors": errs[:5], "error_count": len(errs),
        "sources": src_total, "cleared": cleared,
        "clear_rate_raw": f"{cleared}/{src_total}",
        "clear_rate": round(cleared / src_total, 5) if src_total else None,
        "missed": len(missed),
        "cause_distribution": dict(cause),
        "u_hard_bound_sources": hard_bound,
        "u_hard_bound_raw": f"{hard_bound}/{src_total}",
        "missed_with_u_hard": sum(1 for m in missed if m["u_hard"]),
        "missed_with_u_layout": layout_bound,
        "missed_with_u_traj": sum(1 for m in missed if m["u_traj"]),
        "gap_to_hard_bound_sources": hard_bound - cleared,
        "gap_to_hard_bound_pct": (round((hard_bound - cleared) / src_total * 100, 4)
                                  if src_total else None),
        "visible_frac_of_missed": sorted(m["visible_area_frac"] for m in missed),
        "mean_clear_time_s": (round(st.fmean(r["mean_clear_time_s"] for r in ok
                                             if r["mean_clear_time_s"] is not None), 1)
                              if ok else None),
    }


# ------------------------------------------------------------------ 自检


def selftest(seeds: list[int], directional: int) -> dict:
    checks = []
    for seed in seeds:
        ref = {}
        for cls in (DogStrategy, MissProbe):
            r = run_one(seed, directional, None, cls)
            ref[cls.__name__] = {k: r[k] for k in
                                 ("cleared", "total", "virtual_s", "measure_calls",
                                  "clear_calls", "missed_channels")}
        checks.append({"seed": seed, "identical": ref["DogStrategy"] == ref["MissProbe"],
                       **ref})
    return {"all_identical": all(c["identical"] for c in checks), "checks": checks}


# ------------------------------------------------------------------ CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["deep", "bound", "scan", "selftest"])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-start", type=int, default=7000)
    ap.add_argument("--directional", type=int, default=10)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--strategy", default="MissProbe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.monotonic()

    if args.mode == "selftest":
        res = selftest([7047, 7053, 7126], args.directional)
        print(json.dumps({"all_identical": res["all_identical"]}, ensure_ascii=False))
        for c in res["checks"]:
            print(f"  seed {c['seed']}: identical={c['identical']} "
                  f"{c['DogStrategy']} vs {c['MissProbe']}")
        out = res
    elif args.mode == "bound":
        out = bound_only(args.episodes, args.seed_start, args.directional, args.count)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if args.mode == "deep":
            tasks = [(s, d, c, args.strategy, None) for s, d, c in KNOWN_MISSES]
        else:
            tasks = [(args.seed_start + i, args.directional, args.count,
                      args.strategy, None) for i in range(args.episodes)]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_worker, t) for t in tasks]
            for f in as_completed(futs):
                rows.append(f.result())
        rows.sort(key=lambda r: (r.get("directional", 0), r["seed"]))
        agg = aggregate(rows)
        out = {"config": vars(args), "aggregate": agg, "rows": rows}
        print(json.dumps({k: v for k, v in agg.items()
                          if k not in ("visible_frac_of_missed", "errors")},
                         ensure_ascii=False, indent=2))
        for r in rows:
            for m in r.get("missed_detail", []):
                print(f"  d{r['directional']} seed {m['seed']} ch{m['channel']:>2} "
                      f"{m['cause']:<16} u_hard={m['u_hard']} u_layout={m['u_layout']} "
                      f"u_traj={m['u_traj']} visfrac={m['visible_area_frac']:.5f} "
                      f"aim_out={m['aim_outward_gap_deg']} edge={m['dist_to_edge_m']} "
                      f"bear={m['bearings']} clears={m['clear_calls']} "
                      f"minclr={m['min_clear_dist_m']} minvis={m['min_visited_dist_m']} "
                      f"belief={m['belief_positions']} truth_in={m['truth_in_belief']}")

    out["elapsed_s"] = round(time.monotonic() - t0, 1)
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = RESULTS.parent / p
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
