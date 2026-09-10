"""尾部诊断：mean_clear_time_s 的 55% 来自最慢的 3 个源，查清它们慢在哪。

评分指标是各源清除时刻的**平均**，所以尾部主导。实测（6 局，5 定向）：
  最慢 3 个源占清除时刻总和的 55.1%
  把它们压到中位数 => mean_clear_time_s 降 36.2%

候选原因，逐一用数据区分：
  R1 发现晚：源在环上很晚才被听到（远、或定向朝外）。=> 站位顺序/覆盖问题
  R2 发现早但清除晚：拿到示向度后归航拖沓（反复丢信号、预算耗尽走兜底）
  R3 全程沉默到最后：只能靠 blind/ray 兜底，本质是可观测性

对每个源记录：首次被检测到（拿到示向度）的虚拟时刻 t_first_bearing、
被清除的时刻 t_clear、以及两者之差。据此把尾部源归类。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-tail"


class TailStrategy(DogStrategy):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.first_bearing_vt: dict[int, float] = {}
        self.first_signal_vt: dict[int, float] = {}
        self.clear_vt: dict[int, float] = {}
        self.silent_counts: dict[int, int] = {}
        self.bearing_counts: dict[int, int] = {}
        self.home_attempts: dict[int, list[int]] = {}
        self.home_failures: dict[int, int] = {}

    def _measure(self, point, channel):
        r = super()._measure(point, channel)
        if r is not None:
            vt = self.client.virtual_time_s
            if r.result == "no_signal":
                self.silent_counts[channel] = self.silent_counts.get(channel, 0) + 1
            else:
                self.first_signal_vt.setdefault(channel, vt)
            if getattr(r, "has_direction", False):
                self.first_bearing_vt.setdefault(channel, vt)
                self.bearing_counts[channel] = self.bearing_counts.get(channel, 0) + 1
        return r

    def _clear_at(self, point, channel):
        ok = super()._clear_at(point, channel)
        if ok:
            self.clear_vt.setdefault(channel, self.client.virtual_time_s)
        return ok

    # 记录每个频道"何时被首次尝试归航"，以及它在等待期间机器狗在干什么
    def _home_channel(self, state):
        self.home_attempts.setdefault(state.channel, []).append(
            round(self.client.virtual_time_s))
        r = super()._home_channel(state)
        if not r:
            self.home_failures[state.channel] = self.home_failures.get(state.channel, 0) + 1
        return r


def run_one(seed: int, *, directional: int, count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strat = TailStrategy(client, StrategyConfig(directional_probe=directional > 0))
        strat.run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    truth = {r["channel"]: r for r in case.truth_table()}
    sources = []
    for ch, tr in truth.items():
        tc = strat.clear_vt.get(ch)
        tb = strat.first_bearing_vt.get(ch)
        sx, sy = tr["x"], tr["y"]
        sources.append({
            "channel": ch,
            "dist_from_origin": round(math.hypot(sx, sy)),
            "directional": tr.get("directional", False),
            "r_recv": round(tr.get("reception_radius_m", 0)),
            "t_first_bearing": None if tb is None else round(tb),
            "t_clear": None if tc is None else round(tc),
            "homing_lag_s": None if (tc is None or tb is None) else round(tc - tb),
            "silent_measures": strat.silent_counts.get(ch, 0),
            "bearings": strat.bearing_counts.get(ch, 0),
            "cleared": tr["cleared"],
            "home_attempts": strat.home_attempts.get(ch, []),
            "home_failures": strat.home_failures.get(ch, 0),
            "t_first_home_attempt": (strat.home_attempts.get(ch) or [None])[0],
            "n_home_attempts": len(strat.home_attempts.get(ch, [])),
        })
    s = arena.stats()
    return {"seed": seed, "mean_clear_time_s": s["mean_clear_time_s"],
            "cleared": s["cleared"], "total": s["total_jammers"],
            "sources": sources}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=81000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--label", default="tail")
    args = ap.parse_args()

    rows = []
    for i in range(args.episodes):
        r = run_one(args.seed_start + i, directional=args.directional, count=args.count)
        rows.append(r)
        print(f"seed={r['seed']} cleared={r['cleared']}/{r['total']} "
              f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])}",
              flush=True)

    # 把所有源按 t_clear 排名，取每局最慢 3 个作为"尾部"
    tail, head = [], []
    for r in rows:
        got = [s for s in r["sources"] if s["t_clear"] is not None]
        got.sort(key=lambda s: s["t_clear"])
        tail += got[-3:]
        head += got[:-3]

    def prof(g, name):
        if not g:
            return {}
        lag = [s["homing_lag_s"] for s in g if s["homing_lag_s"] is not None]
        tb = [s["t_first_bearing"] for s in g if s["t_first_bearing"] is not None]
        return {
            "group": name, "n": len(g),
            "mean_t_clear": round(st.fmean(s["t_clear"] for s in g)),
            "mean_t_first_bearing": round(st.fmean(tb)) if tb else None,
            "mean_homing_lag_s": round(st.fmean(lag)) if lag else None,
            "pct_of_t_clear_spent_before_first_bearing": (
                round(st.fmean(tb) / st.fmean(s["t_clear"] for s in g) * 100, 1)
                if tb else None),
            "mean_dist_from_origin": round(st.fmean(s["dist_from_origin"] for s in g)),
            "directional_share": round(st.fmean(1.0 if s["directional"] else 0.0 for s in g), 3),
            "mean_silent_measures": round(st.fmean(s["silent_measures"] for s in g), 1),
            "mean_bearings": round(st.fmean(s["bearings"] for s in g), 1),
            "never_got_bearing_share": round(
                st.fmean(1.0 if s["t_first_bearing"] is None else 0.0 for s in g), 3),
        }

    report = {
        "episodes": len(rows),
        "mean_clear_time_s": round(st.fmean(r["mean_clear_time_s"] for r in rows
                                            if r["mean_clear_time_s"]), 1),
        "tail_slowest3": prof(tail, "tail_slowest3"),
        "head_rest": prof(head, "head_rest"),
        "interpretation": (
            "若尾部的 pct_before_first_bearing 高 => 瓶颈是**发现**（站位/覆盖），"
            "而不是归航；若 mean_homing_lag_s 高 => 瓶颈是归航"),
    }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_tail_diag_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
