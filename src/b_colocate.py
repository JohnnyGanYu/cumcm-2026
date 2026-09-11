"""调度改进：顺路共址补测（colocate），真实闭环配对 A/B。

## 归因链（全部实测，见 results/b_deadhead_audit_q3.json 与本文件 §打桩）

1. Q3 移动构成：homing_origin 38.6% / ring 32.0% / homing_ring 28.2% / blind 6.6%。
2. `_home_all` 的"就近优先"排序在关键时刻**完全退化**：打桩显示每次调用时所有目标的
   排序键都是 0.0（distinct=1）。原因是 `_range_to` 在 `estimate is None` 时回退到
   "到最后一条示向度观测点的距离"，而那条示向度刚好是在**当前位置**测的。
   => 实际执行顺序退化为频道号顺序。
3. 但单纯修排序（`src/b_raysort.py`，用射线代理点）实测**中性**：
   move +0.21%、meanT -0.27%，p>0.9。因为代理点半径未知，贪心最近邻≈随机。
4. 真正的结构性原因：原点普扫只在**一个点**取到示向度，所有频道 `estimate` 皆为 None
   （`_refresh` 要求 >=2 条）。于是每个频道都必须先走一次 `_dogleg`
   （`dogleg_length_m=380 m`）去挣第二条基线，才谈得上定位。
   十余个频道各付一次 380 m 的纯几何绕路，且此时排序还是退化的。

## 改法：共址补测（zero-movement information gain）

机器狗每次**停在某处**（刚清除完一个源、或刚走到环站位），对其余仍缺定位的频道
在**原地**各补一次检测。代价是 5 s 检测（+1 s 切频），**移动为 0**；收益是这些频道
立刻获得与原点相距很远的第二条基线 => 直接得到 `estimate`，从而
  (a) 不必再走 380 m 的 dogleg；
  (b) `_home_all` 的就近排序拿到真实距离，不再退化。

这是"调度"改进而非"策略"改进：不改检测/清除/预算/复活逻辑，不推迟任何清除，
只改**在已经付过移动费的位置上，顺手多做哪些观测**。

与已证伪候选的区别（三者都动调度，只有本候选不推迟清除且不增加移动）：
  - `survey_first`：先勘察后清除，meanT +34.6%（19/20 更差）——推迟了全部清除。
  - `b_defer_far` ：远目标顺路化，move +19.0%（8/8 更差）——推迟了部分清除。
  - `b_raysort`   ：修退化排序，move +0.21%（中性）——顺序变了但信息没变。
  - 本候选        ：信息提前到手，顺序自然正确，且不推迟任何清除。

变体：
  colo        : 仅在清除成功后的位置共址补测
  colo_ring   : additionally 在环站位也补测（环站位本来就要扫全部频道，故等价于基线）
  colo_cap4   : 每个位置最多补测 4 个频道，限制虚拟时间开销

用法：
  python -m src.b_colocate --episodes 40 --directional 0 --variants baseline,colo --label q3
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

ROBOT_ID = "dog-colo"


class ColocateStrategy(DogStrategy):
    """在已付过移动费的位置上顺手补测，把 estimate=None 变成可定位。"""

    def __init__(self, *a, max_per_stop: int = 99,
                 min_baseline_m: float = 250.0,
                 nominal_range_m: float = 900.0,
                 fix_sort: bool = True, **kw):
        super().__init__(*a, **kw)
        self.max_per_stop = max_per_stop
        self.min_baseline_m = min_baseline_m
        self.nominal_range_m = nominal_range_m
        self.fix_sort = fix_sort
        # 命中计数
        self.colo_stops = 0          # 触发共址补测的停靠点数
        self.colo_measures = 0       # 共址补测的检测次数
        self.colo_new_bearings = 0   # 其中拿到新示向度的次数
        self.colo_new_estimates = 0  # 因此首次获得 estimate 的频道数
        self.colo_near_clears = 0    # 补测直接返回 near 而顺手清掉的次数
        self._in_colocate = False

    # ---- 共址补测 ----

    def _colocate_probe(self) -> None:
        """对仍缺 estimate 的活跃频道，在当前位置原地补测。移动成本为 0。"""
        if self._in_colocate or self._stop():
            return
        here = self.pos
        cands: list[ChannelState] = []
        for state in self.channels.values():
            if state.cleared or state.verdict_empty or state.estimate is not None:
                continue
            if not state.bearings:
                continue          # 完全没信号的频道属于覆盖问题，交给环扫/信念搜索
            # 基线太短则三角化没意义（且浪费 5 s）
            last = state.bearings[-1]
            if math.hypot(last.x - here[0], last.y - here[1]) < self.min_baseline_m:
                continue
            cands.append(state)
        if not cands:
            return
        # 频道号顺序，切频代价与频道距离无关
        cands.sort(key=lambda s: s.channel)
        self._in_colocate = True
        self.colo_stops += 1
        try:
            for state in cands[: self.max_per_stop]:
                if self._stop() or state.cleared:
                    break
                had_est = state.estimate is not None
                before = len(state.bearings)
                was_cleared = state.cleared
                self.colo_measures += 1
                self._probe_channel_at(here, state)
                if len(state.bearings) > before:
                    self.colo_new_bearings += 1
                if not had_est and state.estimate is not None:
                    self.colo_new_estimates += 1
                if state.cleared and not was_cleared:
                    self.colo_near_clears += 1
        finally:
            self._in_colocate = False

    # ---- 挂到"刚清除完"这个时刻 ----

    def _clear_at(self, point, channel):
        ok = super()._clear_at(point, channel)
        if ok and not self._in_colocate:
            # 刚清掉一个源，机器狗已经在这里了：顺手给其他频道挣第二条基线
            self._colocate_probe()
        return ok

    # ---- 排序：拿到 estimate 后就不再退化；仍为无 estimate 者提供射线代理 ----

    def _proxy_range(self, state: ChannelState) -> float:
        if state.estimate is not None:
            return math.hypot(state.estimate[0] - self.pos[0],
                              state.estimate[1] - self.pos[1])
        if not state.bearings:
            return 1e9
        last = state.bearings[-1]
        rad = math.radians(last.svd_deg)
        px = last.x + self.nominal_range_m * math.cos(rad)
        py = last.y + self.nominal_range_m * math.sin(rad)
        return math.hypot(px - self.pos[0], py - self.pos[1])

    def _home_all(self) -> None:
        if not self.fix_sort:
            return super()._home_all()
        while not self._stop():
            targets = [st for st in self.channels.values() if st.active]
            if not targets:
                return
            pick = min(targets, key=self._proxy_range)
            if not self._home_channel(pick):
                pick.exhausted = True


VARIANTS: dict[str, dict] = {
    "colo": {},
    "colo_cap4": {"max_per_stop": 4},
    "colo_nosort": {"fix_sort": False},
    "colo_base600": {"min_baseline_m": 600.0},
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
            strat = ColocateStrategy(client, cfg, **VARIANTS[variant])
        strat.run()
        if isinstance(strat, ColocateStrategy):
            extra = {"colo_stops": strat.colo_stops,
                     "colo_measures": strat.colo_measures,
                     "colo_new_bearings": strat.colo_new_bearings,
                     "colo_new_estimates": strat.colo_new_estimates,
                     "colo_near_clears": strat.colo_near_clears}
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
    """配对检验：paired t + Wilcoxon signed-rank，含均值/中位数差与 95% CI。"""
    both = [p for p in pairs
            if p[variant].get(key) is not None and p["baseline"].get(key) is not None]
    if len(both) < 2:
        return None
    a = [p[variant][key] for p in both]
    b = [p["baseline"][key] for p in both]
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = st.fmean(d)
    sd = st.stdev(d)
    base = st.fmean(b)
    if sd == 0.0:
        t_stat, p_t, w_p, ci = 0.0, 1.0, 1.0, [0.0, 0.0]
    else:
        se = sd / math.sqrt(n)
        res = sps.ttest_rel(a, b)
        t_stat, p_t = float(res.statistic), float(res.pvalue)
        try:
            w_p = float(sps.wilcoxon(d).pvalue)
        except ValueError:
            w_p = 1.0
        tcrit = float(sps.t.ppf(0.975, n - 1))
        ci = [round(m - tcrit * se, 1), round(m + tcrit * se, 1)]
    return {
        "metric": key, "n_pairs": n,
        "baseline_mean": round(base, 1),
        "variant_mean": round(st.fmean(a), 1),
        "d_mean": round(m, 1),
        "d_pct_of_baseline": round(m / base * 100, 2) if base else None,
        "d_median": round(st.median(d), 1),
        "d_sd": round(sd, 1),
        "ci95": ci,
        "paired_t": round(t_stat, 3),
        "p_ttest": float(f"{p_t:.3g}"),
        "p_wilcoxon": float(f"{w_p:.3g}"),
        "better": sum(1 for x in d if x < 0),
        "worse": sum(1 for x in d if x > 0),
        "tie": sum(1 for x in d if x == 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=350000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,colo")
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
            print(f"seed={seed} {v:12s} cleared={r['cleared']}/{r['total']} "
                  f"move={r['move_m']:.0f} virt={r['virtual_s']:.0f} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"colo={r.get('colo_measures','-')}m/{r.get('colo_new_estimates','-')}est "
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
                "colo_stops_total": sum(r.get("colo_stops", 0) for r in rs),
                "colo_measures_total": sum(r.get("colo_measures", 0) for r in rs),
                "colo_new_bearings_total": sum(r.get("colo_new_bearings", 0) for r in rs),
                "colo_new_estimates_total": sum(r.get("colo_new_estimates", 0) for r in rs),
                "colo_near_clears_total": sum(r.get("colo_near_clears", 0) for r in rs)}

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
        block: dict = {"n_pairs": len(pairs)}
        for key in ("move_m", "virtual_s", "mean_clear_time_s", "cleared", "measures"):
            b = paired_block(pairs, v, key)
            if b:
                block[key] = b
        block["net_extra_cleared"] = sum(p[v]["cleared"] - p["baseline"]["cleared"]
                                         for p in pairs)
        report["paired_vs_baseline"][v] = block

    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_colocate_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
