"""带真值的漏失归因诊断：为什么某个干扰源没被清掉，以及布局决定的上限在哪。

设计原则（严格分离「诊断观察者」与「被测策略」）
------------------------------------------------
1. `AttribStrategy` 只 override 三个底层动作钩子（`_measure` / `_clear_at` /
   `_home_channel`），每个钩子的第一件事就是调用 `super()`，随后把返回值原样
   转发；钩子内部只往观察者自己的 dict 里 append，从不修改 `ChannelState`、
   `config`、`self.pos`、返回值或任何策略可见的量。
2. 观察者手上没有真值：`AttribStrategy` 的构造函数不接收 case/arena/jammer；
   真值只在 `strat.run()` 返回之后，在 `run_one()` 里从 `case.truth_table()`
   读出来做离线 join。策略进程内没有任何路径能读到真值。
3. 自检：`--selftest` 用同一 seed 分别跑 `DogStrategy` 与 `AttribStrategy`，
   断言清除频道集合、虚拟时间、measure/clear 次数完全一致 —— 挂钩子没有
   改变行为，因此归因结论对原策略成立。

归因类别
--------
A cover_gap        从未有任何检测点落入其接收半径（覆盖缺口，纯几何）
B directional_mask 有检测点落入接收半径，但落在辐射扇区之外（定向遮蔽）
C no_localization  收到过信号但示向度不足以定位（交会失败/丢信号）
D clear_failed     定位出了估计位置，清除尝试全部落空
E abandoned        被策略主动放弃（exhausted / verdict_empty / 预算门禁）
F budget           虚拟时间或现实时间预算耗尽导致整局提前结束

上限口径
--------
U_layout : 该源能否从「规划站位集合」（Q3 = 原点+1250×12；Q4 = 再加 1750×18）
           中至少一个站位收到信号。这是**布局决定的上限**。
U_traj   : 该源能否从机器狗**实际站过的任意一点**收到信号（含归航/信念搜索点）。
missed = layout_loss(不在 U_layout 内) + strategy_loss(在 U_layout 内却漏掉)

注意：/clear 只看距离、不看辐射方向，所以盲清除可以清掉 U_layout 之外的源，
U_layout 不是清除数的硬上界，而是「靠信号发现」的上界。两者都报。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from .b_sim.arena import (
    ARENA_RADIUS,
    DIRECTIONAL_HALF_ANGLE_DEG,
    NEAR_THRESHOLD_M,
    Arena,
    angular_gap_deg,
    generate_case,
    norm_deg,
)
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-attrib"

# ---------------------------------------------------------------- 观察者


class AttribStrategy(DogStrategy):
    """只读观察者。所有钩子先调 super()，再记录，返回值原样转发。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # 每个频道的检测点：(x, y, result, svd_deg, virtual_time_s)
        self.measure_log: dict[int, list[tuple]] = {}
        # 机器狗站过的所有位置（measure + clear），用于 U_traj
        self.visited: list[tuple[float, float]] = [(0.0, 0.0)]
        self.clear_log: dict[int, int] = {}
        self.clear_vt: dict[int, float] = {}
        self.first_signal_vt: dict[int, float] = {}
        self.first_bearing_vt: dict[int, float] = {}
        self.home_calls: dict[int, int] = {}
        self.home_giveup: dict[int, int] = {}

    def _measure(self, point, channel):
        r = super()._measure(point, channel)          # 先执行真实动作
        vt = self.client.virtual_time_s
        self.visited.append((point[0], point[1]))
        if r is not None:
            self.measure_log.setdefault(channel, []).append(
                (point[0], point[1], r.result, r.svd_deg, vt))
            if r.result != "no_signal":
                self.first_signal_vt.setdefault(channel, vt)
            if r.result == "direction":
                self.first_bearing_vt.setdefault(channel, vt)
        return r                                       # 原样转发

    def _clear_at(self, point, channel):
        ok = super()._clear_at(point, channel)
        self.visited.append((point[0], point[1]))
        self.clear_log[channel] = self.clear_log.get(channel, 0) + 1
        if ok:
            self.clear_vt.setdefault(channel, self.client.virtual_time_s)
        return ok

    def _home_channel(self, state):
        ch = state.channel
        self.home_calls[ch] = self.home_calls.get(ch, 0) + 1
        r = super()._home_channel(state)
        if not r:
            self.home_giveup[ch] = self.home_giveup.get(ch, 0) + 1
        return r


# ---------------------------------------------------------------- 几何工具（只在观察侧用）


def planned_stations(directional: bool, cfg: StrategyConfig) -> list[tuple[float, float]]:
    """复刻 DogStrategy 的规划站位集合（原点 + 内环 [+ 外环]）。"""
    pts = [(0.0, 0.0)] if cfg.sweep_origin else []
    layers = [(cfg.probe_ring_radius, cfg.probe_ring_stations, 0.0)]
    if directional or cfg.use_outer_ring:
        n = cfg.directional_ring_stations if directional else cfg.outer_ring_stations
        layers.append((cfg.outer_ring_radius, n, math.pi / max(1, n)))
    for radius, count, phase in layers:
        pts += [(radius * math.cos(2 * math.pi * k / count + phase),
                 radius * math.sin(2 * math.pi * k / count + phase))
                for k in range(count)]
    # _inner_retry 的内圈四点也是无条件规划好的（不依赖真值），一并计入布局
    pts += [(0.0, 0.0), (300.0, 0.0), (-150.0, 260.0), (-150.0, -260.0)]
    return pts


def receivable(sx, sy, r_recv, ddeg, px, py) -> bool:
    """在 (px,py) 对该源做检测能否收到信号（距离 + 辐射扇区）。"""
    if math.hypot(px - sx, py - sy) > r_recv:
        return False
    if ddeg is None:
        return True
    if px == sx and py == sy:
        return True
    b = norm_deg(math.degrees(math.atan2(py - sy, px - sx)))
    return angular_gap_deg(b, ddeg) <= DIRECTIONAL_HALF_ANGLE_DEG


_GRID: np.ndarray | None = None


def arena_grid(step: float = 25.0) -> np.ndarray:
    """区域内均匀网格点，用于估计「可见区域面积占比」。"""
    global _GRID
    if _GRID is None:
        c = np.arange(-ARENA_RADIUS, ARENA_RADIUS + step, step)
        gx, gy = np.meshgrid(c, c)
        m = gx * gx + gy * gy <= ARENA_RADIUS ** 2
        _GRID = np.stack([gx[m], gy[m]], axis=1)
    return _GRID


def visible_area_fraction(sx, sy, r_recv, ddeg) -> float:
    """可见区域（区域内 ∩ 接收半径内 ∩ 辐射扇区内）占目标区域面积的比例。"""
    g = arena_grid()
    dx, dy = g[:, 0] - sx, g[:, 1] - sy
    ok = dx * dx + dy * dy <= r_recv * r_recv
    if ddeg is not None:
        ang = np.degrees(np.arctan2(dy, dx))
        d = np.abs(((ang - ddeg) + 180.0) % 360.0 - 180.0)
        ok &= d <= DIRECTIONAL_HALF_ANGLE_DEG
    return float(ok.sum()) / len(g)


# ---------------------------------------------------------------- 单局


def run_one(seed: int, *, directional: int, count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional > 0)
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strat = AttribStrategy(client, cfg)
        strat.run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    # ==== 以下全部是「run 之后」的离线归因，策略已结束，读真值不影响其行为 ====
    stations = planned_stations(directional > 0, cfg)
    visited = strat.visited
    ev = " | ".join(strat.events)
    global_budget = ("virtual time budget reached" in ev
                     or "real time budget reached" in ev)

    sources = []
    for row in case.truth_table():
        ch = row["channel"]
        sx, sy = row["x"], row["y"]
        r_recv = row["reception_radius_m"]
        ddeg = row["directional_deg"]
        mlog = strat.measure_log.get(ch, [])
        state = strat.channels[ch]

        got_signal = any(m[2] != "no_signal" for m in mlog)
        got_bearing = any(m[2] == "direction" for m in mlog)
        # 该频道实际检测过的点里，有没有落入接收半径的（不管扇区）
        in_range_pts = [m for m in mlog if math.hypot(m[0] - sx, m[1] - sy) <= r_recv]
        # 该频道实际检测过的点里，有没有本应收到信号的（距离+扇区都满足）
        recv_pts = [m for m in mlog if receivable(sx, sy, r_recv, ddeg, m[0], m[1])]

        u_layout = any(receivable(sx, sy, r_recv, ddeg, px, py) for px, py in stations)
        u_traj = any(receivable(sx, sy, r_recv, ddeg, px, py) for px, py in visited)

        min_d_measured = min((math.hypot(m[0] - sx, m[1] - sy) for m in mlog), default=None)
        min_d_station = min(math.hypot(px - sx, py - sy) for px, py in stations)

        cause = None
        if row["cleared"]:
            cause = "cleared"
        else:
            reason_ev = [e for e in strat.events if e.startswith(f"ch{ch}:")]
            if got_bearing:
                if state.estimate is not None and strat.clear_log.get(ch, 0) > 0:
                    cause = "D_clear_failed"
                else:
                    cause = "C_no_localization"
            elif got_signal:            # 只拿到 near 却没清掉，极罕见
                cause = "D_clear_failed"
            elif global_budget and not u_traj:
                cause = "F_budget"
            elif len(in_range_pts) > 0:
                cause = "B_directional_mask"
            elif u_layout:
                # 布局本可发现，但机器狗压根没在能发现它的站位上测这个频道
                cause = "E_abandoned"
            else:
                cause = "A_cover_gap"
            if cause in ("A_cover_gap", "B_directional_mask") and global_budget:
                cause = "F_budget"

        sources.append({
            "channel": ch,
            "cleared": row["cleared"],
            "cause": cause,
            "x": round(sx, 1), "y": round(sy, 1),
            "r_from_origin": round(math.hypot(sx, sy), 1),
            "r_recv": round(r_recv, 1),
            "directional": ddeg is not None,
            "directional_deg": None if ddeg is None else round(ddeg, 1),
            # 辐射轴与「源->原点」方向的夹角：0 = 朝区域中心辐射，180 = 朝外辐射
            "aim_inward_gap_deg": None if ddeg is None else round(
                angular_gap_deg(ddeg, norm_deg(math.degrees(math.atan2(-sy, -sx)))), 1),
            "visible_area_frac": round(visible_area_fraction(sx, sy, r_recv, ddeg), 4),
            "u_layout": u_layout,
            "u_traj": u_traj,
            "measures_on_ch": len(mlog),
            "measures_in_range": len(in_range_pts),
            "measures_receivable": len(recv_pts),
            "min_dist_measured_m": None if min_d_measured is None else round(min_d_measured),
            "min_dist_station_m": round(min_d_station),
            "clear_calls": strat.clear_log.get(ch, 0),
            "t_first_signal": _r(strat.first_signal_vt.get(ch)),
            "t_first_bearing": _r(strat.first_bearing_vt.get(ch)),
            "t_clear": _r(strat.clear_vt.get(ch)),
            "state_exhausted": state.exhausted,
            "state_verdict_empty": state.verdict_empty,
            "state_bearings": len(state.bearings),
            "state_silent": len(state.silent_points),
            "events": [e for e in strat.events if e.startswith(f"ch{ch}:")],
        })

    s = arena.stats()
    return {
        "seed": seed,
        "directional": directional,
        "total": s["total_jammers"],
        "cleared": s["cleared"],
        "mean_clear_time_s": None if s["mean_clear_time_s"] is None else round(s["mean_clear_time_s"], 1),
        "virtual_s": round(s["total_virtual_time_s"], 1),
        "measure_calls": s["measure_calls"],
        "clear_calls": s["clear_calls"],
        "u_layout_count": sum(1 for x in sources if x["u_layout"]),
        "u_traj_count": sum(1 for x in sources if x["u_traj"]),
        "global_budget_hit": global_budget,
        "planned_stations": len(stations),
        "sources": sources,
    }


def _r(v):
    return None if v is None else round(v)


# ---------------------------------------------------------------- 自检：钩子不改变行为


def selftest(seeds: list[int], directional: int) -> dict:
    out = []
    for seed in seeds:
        ref = {}
        for cls in (DogStrategy, AttribStrategy):
            case = generate_case(seed, directional_count=directional)
            arena = Arena(case)
            session = Session("selftest", arena, real_limit_s=1200.0)
            httpd = make_server(session, port=0)
            port = httpd.server_address[1]
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            try:
                client = RobotClient("selftest", base_url=f"http://127.0.0.1:{port}")
                cls(client, StrategyConfig(directional_probe=directional > 0)).run()
            finally:
                httpd.shutdown()
                httpd.server_close()
            s = arena.stats()
            ref[cls.__name__] = {
                "cleared": s["cleared"],
                "virtual_s": round(s["total_virtual_time_s"], 6),
                "measure_calls": s["measure_calls"],
                "clear_calls": s["clear_calls"],
                "cleared_channels": sorted(j.channel for j in case.jammers if j.cleared),
            }
        same = ref["DogStrategy"] == ref["AttribStrategy"]
        out.append({"seed": seed, "identical": same, **ref})
    return {"all_identical": all(o["identical"] for o in out), "checks": out}


# ---------------------------------------------------------------- 聚合


CAUSES = ["A_cover_gap", "B_directional_mask", "C_no_localization",
          "D_clear_failed", "E_abandoned", "F_budget"]


def aggregate(rows: list[dict]) -> dict:
    allsrc = [s for r in rows for s in r["sources"]]
    missed = [s for s in allsrc if not s["cleared"]]
    n_src = len(allsrc)
    cnt = Counter(s["cause"] for s in missed)
    cleared = sum(1 for s in allsrc if s["cleared"])
    u_layout = sum(1 for s in allsrc if s["u_layout"])
    u_traj = sum(1 for s in allsrc if s["u_traj"])
    layout_loss = [s for s in missed if not s["u_layout"]]
    strategy_loss = [s for s in missed if s["u_layout"]]

    def prof(g, name):
        if not g:
            return {"group": name, "n": 0}
        dirs = [s for s in g if s["directional"]]
        return {
            "group": name, "n": len(g),
            "directional_share": round(len(dirs) / len(g), 3),
            "r_from_origin_mean": round(st.fmean(s["r_from_origin"] for s in g)),
            "r_from_origin_median": round(st.median([s["r_from_origin"] for s in g])),
            "r_from_origin_min": round(min(s["r_from_origin"] for s in g)),
            "r_recv_mean": round(st.fmean(s["r_recv"] for s in g)),
            "r_recv_median": round(st.median(s["r_recv"] for s in g)),
            "visible_area_frac_mean": round(st.fmean(s["visible_area_frac"] for s in g), 4),
            "visible_area_frac_median": round(st.median(s["visible_area_frac"] for s in g), 4),
            "aim_inward_gap_deg_mean": (round(st.fmean(s["aim_inward_gap_deg"] for s in dirs))
                                        if dirs else None),
            "aim_outward_share": (round(st.fmean(1.0 if d["aim_inward_gap_deg"] > 90 else 0.0
                                                 for d in dirs), 3) if dirs else None),
            "min_dist_station_m_mean": round(st.fmean(s["min_dist_station_m"] for s in g)),
        }

    return {
        "episodes": len(rows),
        "sources_total": n_src,
        "cleared_total": cleared,
        "clear_rate": round(cleared / n_src, 4) if n_src else None,
        "full_clear_episodes": sum(1 for r in rows if r["cleared"] == r["total"]),
        "full_clear_rate": round(sum(1 for r in rows if r["cleared"] == r["total"]) / len(rows), 4),
        "missed_total": len(missed),
        "missed_per_episode": round(len(missed) / len(rows), 3),
        "layout_upper_bound_sources": u_layout,
        "layout_upper_bound_rate": round(u_layout / n_src, 4) if n_src else None,
        "traj_upper_bound_sources": u_traj,
        "traj_upper_bound_rate": round(u_traj / n_src, 4) if n_src else None,
        "cleared_beyond_layout_bound": sum(
            1 for s in allsrc if s["cleared"] and not s["u_layout"]),
        "layout_loss_sources": len(layout_loss),
        "strategy_loss_sources": len(strategy_loss),
        "attribution": {c: {"n": cnt.get(c, 0),
                            "share_of_missed": (round(cnt.get(c, 0) / len(missed), 4)
                                                if missed else 0.0),
                            "per_1000_sources": round(cnt.get(c, 0) / n_src * 1000, 2)}
                        for c in CAUSES},
        "mean_clear_time_s": round(st.fmean(r["mean_clear_time_s"] for r in rows
                                            if r["mean_clear_time_s"] is not None), 1),
        "measures_mean": round(st.fmean(r["measure_calls"] for r in rows), 1),
        "profiles": {
            "undiscoverable_by_layout(all)": prof([s for s in allsrc if not s["u_layout"]],
                                                  "not_u_layout"),
            "layout_loss(missed & not U_layout)": prof(layout_loss, "layout_loss"),
            "strategy_loss(missed & U_layout)": prof(strategy_loss, "strategy_loss"),
            "discoverable_baseline": prof([s for s in allsrc if s["u_layout"]], "u_layout"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="漏失归因（带真值的离线诊断）")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-start", type=int, default=310000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    label = args.label or (f"q4_d{args.directional}" if args.directional else "q3")
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)

    if args.selftest:
        rep = selftest([args.seed_start + i for i in range(3)], args.directional)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        (outdir / f"b_attrib_selftest_{label}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    rows = []
    for i in range(args.episodes):
        r = run_one(args.seed_start + i, directional=args.directional, count=args.count)
        rows.append(r)
        miss = [s["channel"] for s in r["sources"] if not s["cleared"]]
        print(f"seed={r['seed']} cleared={r['cleared']}/{r['total']} "
              f"U_layout={r['u_layout_count']} U_traj={r['u_traj_count']} "
              f"meanT={r['mean_clear_time_s']} missed={miss}", flush=True)

    # 先落盘原始 rows：跑一批要几分钟，聚合环节报错不该把它们丢掉
    out = outdir / f"b_attrib_{label}.json"
    out.write_text(json.dumps({"config": vars(args), "aggregate": None, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    agg = aggregate(rows)
    print("\n== aggregate ==")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    out.write_text(json.dumps({"config": vars(args), "aggregate": agg, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
