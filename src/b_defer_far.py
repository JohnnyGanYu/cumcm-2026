"""调度改进候选：远目标顺路化（defer_far），真实闭环配对 A/B。

机制（由 `src/b_deadhead_audit.py` 的归因得到）：
基线 `_walk_ring` 每扫完一个站位就 `_home_all()`，而 `_home_all` 会把**当时所有**
已定位的频道一次清完，按距离排序但不判断"值不值得现在去"。于是一个估计位置在
环对面的源会被立刻专程往返；而顺扫本来会在若干站之后走到它旁边。

改法：`_home_all` 里加一道准入判断——若存在**尚未访问的环站位**比当前位置更靠近
该源的估计位置（且优势超过 margin），就把它推迟，等顺扫走到那一段再顺路清除。
这不改变检测/清除/预算/复活逻辑，只改"何时去"。

安全网：
  1. 本环走完前把 `_remaining_stations` 清空，`_walk_ring` 尾部的 `_home_all` 因此
     不会再推迟任何目标；`run()` 里后续的 `_home_all` / `_inner_retry` 同理。
  2. 推迟不改 `state.exhausted`，目标始终留在 active 集合里。
  3. 每个频道推迟次数上限 `max_defers_per_channel`，防止在环上被反复踢皮球。

命中计数（区分"无效"与"从未触发"）：defers / deferred_channels / late_clears。

用法：
  python -m src.b_defer_far --episodes 40 --directional 0 --variants baseline,defer300 --label q3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from scipy import stats as sps

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import ChannelState, DogStrategy, StrategyConfig

ROBOT_ID = "dog-defer"


class DeferFarStrategy(DogStrategy):
    """候选：把"专程往返远目标"换成"顺扫路过时顺路清除"。"""

    def __init__(self, *a, defer_margin_m: float = 300.0,
                 defer_near_m: float = 250.0,
                 nominal_range_m: float = 900.0,
                 max_defers_per_channel: int = 3, **kw):
        super().__init__(*a, **kw)
        self.defer_margin_m = defer_margin_m
        self.defer_near_m = defer_near_m
        self.nominal_range_m = nominal_range_m
        self.max_defers_per_channel = max_defers_per_channel
        self._remaining_stations: list[tuple[float, float]] = []
        self._defer_counts: dict[int, int] = {}
        self.defers = 0
        self.deferred_channels: set[int] = set()
        self.late_clears = 0
        self.gate_calls = 0

    # ---- 目标位置代理 ----

    def _proxy_target(self, state: ChannelState) -> tuple[float, float] | None:
        """估计"这个频道的活儿大概在哪"，用于比较现在去还是以后顺路去。

        关键：`_home_all` 被调用时，刚发现的频道**只有一条示向度**，而
        `ChannelState._refresh` 要求 >= 2 条才给 `estimate`，所以此刻 estimate 恒为
        None（初版门禁因此一次都没触发：defers_total=0）。单条示向度已给出方向，
        只是距离未知；源必然落在该射线上、且因为能被收到而距观测点 <= 接收半径上限
        1500 m。故取射线上 `nominal_range_m` 处作为代理点——它只用于**排序/比较**
        （现在去 vs 以后去哪个近），不参与任何定位或清除计算，误差不影响正确性。
        """
        if state.estimate is not None:
            return state.estimate
        if not state.bearings:
            return None
        last = state.bearings[-1]
        rad = math.radians(last.svd_deg)
        return (last.x + self.nominal_range_m * math.cos(rad),
                last.y + self.nominal_range_m * math.sin(rad))

    # ---- 准入判断 ----

    def _should_home_now(self, state: ChannelState) -> bool:
        self.gate_calls += 1
        if not self._remaining_stations:
            return True
        if self._defer_counts.get(state.channel, 0) >= self.max_defers_per_channel:
            return True
        proxy = self._proxy_target(state)
        if proxy is None:
            return True
        ex, ey = proxy
        d_now = math.hypot(ex - self.pos[0], ey - self.pos[1])
        if d_now <= self.defer_near_m:
            return True          # 已经在旁边，直接清
        d_future = min(math.hypot(ex - sx, ey - sy)
                       for sx, sy in self._remaining_stations)
        if d_future + self.defer_margin_m < d_now:
            self._defer_counts[state.channel] = self._defer_counts.get(state.channel, 0) + 1
            self.defers += 1
            self.deferred_channels.add(state.channel)
            return False
        return True

    def _home_all(self) -> None:
        while not self._stop():
            targets = [st for st in self.channels.values() if st.active]
            if not targets:
                return
            targets.sort(key=lambda st: self._range_to(st))
            pick = next((st for st in targets if self._should_home_now(st)), None)
            if pick is None:
                return           # 全部推迟，交给后续站位顺路处理
            was_deferred = pick.channel in self.deferred_channels
            if self._home_channel(pick):
                if was_deferred:
                    self.late_clears += 1
            else:
                pick.exhausted = True

    # ---- 原点普扫后的首轮归航 ----

    def run(self) -> dict:
        """在进入环之前，把即将访问的环站位预告给门禁。

        必要性（实测归因 results/b_deadhead_audit_q3.json）：Q3 movement 里
        homing_origin 占 38.6%，是最大单一桶。它发生在 `run()` 的第一个
        `_home_all()`，此时 `_walk_ring` 还没执行过，`_remaining_stations` 为空，
        门禁一律放行——等于最大的那块开销完全不受调度改进影响。
        原点普扫结束后环站位集合已经完全确定（只由 config 决定），
        所以可以合法地提前预告。
        """
        if self.config.sweep_origin:
            rings = self._probe_stations()
            if rings:
                self._remaining_stations = list(rings[0])
        return super().run()

    # ---- 逐行复制基线 _walk_ring，仅维护 _remaining_stations ----

    def _walk_ring(self, ring: list[tuple[float, float]]) -> None:
        remaining = self._order_ring(ring)
        since_home = 0
        while remaining and not self._stop():
            todo = [st for st in self.channels.values() if self._probe_target(st)]
            if not todo:
                break
            station = remaining.pop(0)
            self._remaining_stations = remaining
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop():
                    return
                self._probe_channel_at(station, state)
            since_home += 1
            ready = any(st.active for st in self.channels.values())
            if ready and (since_home >= self.config.home_batch_stations or not remaining):
                self._home_all()
                since_home = 0
            if self._stop():
                return
            self._revive_nearby()
        # 本环结束：不再有"以后会更近"的站位，安全网收口
        self._remaining_stations = []
        if not self._stop():
            self._home_all()


VARIANTS: dict[str, dict] = {
    "defer150": {"defer_margin_m": 150.0},
    "defer300": {"defer_margin_m": 300.0},
    "defer600": {"defer_margin_m": 600.0},
    "defer300_near500": {"defer_margin_m": 300.0, "defer_near_m": 500.0},
}


def run_one(seed: int, *, directional: int, variant: str,
            count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional > 0)
    extra: dict = {}
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        if variant == "baseline":
            strat: DogStrategy = DogStrategy(client, cfg)
        else:
            strat = DeferFarStrategy(client, cfg, **VARIANTS[variant])
        strat.run()
        if isinstance(strat, DeferFarStrategy):
            extra = {"defers": strat.defers,
                     "deferred_channels": len(strat.deferred_channels),
                     "late_clears": strat.late_clears,
                     "gate_calls": strat.gate_calls}
    finally:
        httpd.shutdown()
        httpd.server_close()
    s = arena.stats()
    return {"seed": seed, "variant": variant, "total": s["total_jammers"],
            "cleared": s["cleared"], "ratio": s["cleared_ratio"],
            "virtual_s": s["total_virtual_time_s"],
            "mean_clear_time_s": s["mean_clear_time_s"],
            "measures": s["measure_calls"], "clears": s["clear_calls"],
            "move_m": s["move_distance_m"],
            "missed": [r["channel"] for r in case.truth_table() if not r["cleared"]],
            **extra}


def paired_block(pairs: list[dict], variant: str, key: str) -> dict | None:
    """配对检验：paired t + Wilcoxon signed-rank，报告均值/中位数差与 95% CI。"""
    d = [p[variant][key] - p["baseline"][key] for p in pairs
         if p[variant].get(key) is not None and p["baseline"].get(key) is not None]
    if len(d) < 2:
        return None
    n = len(d)
    m = st.fmean(d)
    sd = st.stdev(d)
    se = sd / math.sqrt(n)
    base = st.fmean(p["baseline"][key] for p in pairs if p["baseline"].get(key) is not None)
    t_res = sps.ttest_rel([p[variant][key] for p in pairs],
                          [p["baseline"][key] for p in pairs])
    try:
        w_res = sps.wilcoxon(d)
        w_p = float(w_res.pvalue)
    except ValueError:              # 全为 0 差值
        w_p = 1.0
    tcrit = float(sps.t.ppf(0.975, n - 1))
    return {
        "metric": key, "n_pairs": n,
        "baseline_mean": round(base, 1),
        "d_mean": round(m, 1),
        "d_pct_of_baseline": round(m / base * 100, 2) if base else None,
        "d_median": round(st.median(d), 1),
        "d_sd": round(sd, 1),
        "ci95": [round(m - tcrit * se, 1), round(m + tcrit * se, 1)],
        "paired_t": round(float(t_res.statistic), 3),
        "p_ttest": float(f"{float(t_res.pvalue):.3g}"),
        "p_wilcoxon": float(f"{w_p:.3g}"),
        "better": sum(1 for x in d if x < 0),
        "worse": sum(1 for x in d if x > 0),
        "tie": sum(1 for x in d if x == 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=320000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,defer300")
    ap.add_argument("--label", default="q3")
    args = ap.parse_args()
    if args.count is not None and args.directional > args.count:
        ap.error("--directional 不能超过 --count")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if "baseline" not in variants:
        ap.error("variants 必须含 baseline")
    for v in variants:
        if v != "baseline" and v not in VARIANTS:
            ap.error(f"未知变体 {v}")

    rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        for v in variants:
            r = run_one(seed, directional=args.directional, variant=v, count=args.count)
            rows.append(r)
            print(f"seed={seed} {v:16s} cleared={r['cleared']}/{r['total']} "
                  f"move={r['move_m']:.0f} virt={r['virtual_s']:.0f} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"defers={r.get('defers','-')}/{r.get('late_clears','-')} "
                  f"missed={r['missed']}", flush=True)

    def agg(v):
        rs = [r for r in rows if r["variant"] == v]
        mt = [r["mean_clear_time_s"] for r in rs if r["mean_clear_time_s"] is not None]
        return {"variant": v, "episodes": len(rs),
                "clear_rate": round(sum(r["cleared"] for r in rs) / sum(r["total"] for r in rs), 4),
                "full_clear_rate": round(sum(1 for r in rs if r["ratio"] >= 1 - 1e-9) / len(rs), 4),
                "total_missed": sum(len(r["missed"]) for r in rs),
                "move_m_mean": round(st.fmean(r["move_m"] for r in rs), 1),
                "virtual_s_mean": round(st.fmean(r["virtual_s"] for r in rs), 1),
                "mean_clear_time_s_mean": round(st.fmean(mt), 1) if mt else None,
                "measures_mean": round(st.fmean(r["measures"] for r in rs), 1),
                "defers_total": sum(r.get("defers", 0) for r in rs),
                "late_clears_total": sum(r.get("late_clears", 0) for r in rs)}

    by_seed: dict[int, dict] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["variant"]] = r

    report = {"config": {"directional": args.directional, "count": args.count,
                         "seed_start": args.seed_start, "episodes": args.episodes},
              "per_variant": {v: agg(v) for v in variants},
              "paired_vs_baseline": {}}
    for v in variants:
        if v == "baseline":
            continue
        pairs = [p for p in by_seed.values() if "baseline" in p and v in p]
        block = {}
        for key in ("move_m", "virtual_s", "mean_clear_time_s", "cleared", "measures"):
            b = paired_block(pairs, v, key)
            if b:
                block[key] = b
        block["n_pairs"] = len(pairs)
        block["net_extra_cleared"] = sum(p[v]["cleared"] - p["baseline"]["cleared"]
                                         for p in pairs)
        report["paired_vs_baseline"][v] = block

    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_defer_far_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
