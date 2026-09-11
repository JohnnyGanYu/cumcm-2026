"""调度改进：修复 `_home_all` 的退化排序（raysort），真实闭环配对 A/B。

## 根因（打桩实测，非推断）

`strategy.py::_home_all` 写的是"就近优先"：

    targets.sort(key=lambda st: self._range_to(st))

但 `_range_to` 在 `estimate is None` 时回退到"到最后一条示向度**观测点**的距离"：

    last = state.bearings[-1]
    return math.hypot(last.x - self.pos[0], last.y - self.pos[1])

而 `_home_all` 恰好在一批 `_probe_channel_at(station, ...)` 之后被调用 —— 那些示向度
就是**在当前位置**测到的，于是 `last.x,last.y == self.pos`，距离恒为 **0.0**。
打桩输出（seed 310000，`_home_all` 前 4 次调用）：每次所有目标的排序键都是 0.0，
distinct=1，排序完全退化，实际执行顺序退化成 `self.channels` 的插入顺序 = **频道号顺序**。

于是"就近优先"在最关键的时刻（原点普扫后 7 个目标待清、占 Q3 移动 38.6% 的
homing_origin 段）根本没生效：机器狗按 ch2, ch3, ch8, ch10... 的频道号乱序满场跑。

## 改法

单条示向度虽不能定位，但**给出了方向**。源必在该射线上，且距观测点 <= 接收半径上限。
故取射线上 `nominal_range_m` 处作为**排序代理点**，让 `_home_all` 真正按几何就近选取；
并在每清除一个后重新选（贪心最近邻），因为清除会改变当前位置。

关键区别于已被证伪的两个候选：
  - `survey_first`（+34.6%，19/20 更差）和 `defer_far`（本仓库 b_defer_far，+16% 更差）
    都**推迟**了清除时机，因此推高 mean_clear_time_s。
  - 本改法**不推迟任何事**：同一批目标、同一时刻、同样全部清除，只改**先后顺序**。
    工作量守恒，只有路径长度变化 => 不存在"越攒越慢"的失败模式。

代理点半径不敏感（它只参与比较，不参与定位/清除）；仍按 200/900/1500 m 三档实测。

用法：
  python -m src.b_raysort --episodes 40 --directional 0 --variants baseline,ray900 --label q3
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

ROBOT_ID = "dog-raysort"


class RaySortStrategy(DogStrategy):
    """只改 `_home_all` 的目标选取顺序，不改任何检测/清除/预算/复活逻辑。"""

    def __init__(self, *a, nominal_range_m: float = 900.0, **kw):
        super().__init__(*a, **kw)
        self.nominal_range_m = nominal_range_m
        # 命中计数：区分"无效"与"从未触发"
        self.sort_calls = 0            # _home_all 里做过排序决策的次数
        self.degenerate_calls = 0      # 基线排序键全相同（本改法真正改变顺序的场合）
        self.order_changed = 0         # 实际选了与基线不同的目标
        self.proxy_used = 0            # 用射线代理点的次数

    def _proxy_point(self, state: ChannelState) -> tuple[float, float] | None:
        """排序用的目标位置代理。有 estimate 用 estimate，否则用示向度射线上一点。"""
        if state.estimate is not None:
            return state.estimate
        if not state.bearings:
            return None
        self.proxy_used += 1
        last = state.bearings[-1]
        rad = math.radians(last.svd_deg)
        return (last.x + self.nominal_range_m * math.cos(rad),
                last.y + self.nominal_range_m * math.sin(rad))

    def _proxy_range(self, state: ChannelState) -> float:
        p = self._proxy_point(state)
        if p is None:
            return 1e9
        return math.hypot(p[0] - self.pos[0], p[1] - self.pos[1])

    def _home_all(self) -> None:
        while not self._stop():
            targets = [st for st in self.channels.values() if st.active]
            if not targets:
                return
            # 基线会选谁（用于统计顺序是否真的变了）
            base_pick = min(targets, key=lambda st: self._range_to(st))
            base_keys = {round(self._range_to(st), 6) for st in targets}
            if len(targets) > 1:
                self.sort_calls += 1
                if len(base_keys) == 1:
                    self.degenerate_calls += 1
            pick = min(targets, key=self._proxy_range)
            if len(targets) > 1 and pick.channel != base_pick.channel:
                self.order_changed += 1
            if not self._home_channel(pick):
                pick.exhausted = True


VARIANTS: dict[str, dict] = {
    "ray200": {"nominal_range_m": 200.0},
    "ray900": {"nominal_range_m": 900.0},
    "ray1500": {"nominal_range_m": 1500.0},
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
            strat = RaySortStrategy(client, cfg, **VARIANTS[variant])
        strat.run()
        if isinstance(strat, RaySortStrategy):
            extra = {"sort_calls": strat.sort_calls,
                     "degenerate_calls": strat.degenerate_calls,
                     "order_changed": strat.order_changed,
                     "proxy_used": strat.proxy_used}
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
    se = sd / math.sqrt(n) if sd else 0.0
    base = st.fmean(b)
    if sd == 0.0:
        t_stat, p_t, w_p, ci = 0.0, 1.0, 1.0, [0.0, 0.0]
    else:
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
    ap.add_argument("--seed-start", type=int, default=340000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,ray900")
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
            print(f"seed={seed} {v:9s} cleared={r['cleared']}/{r['total']} "
                  f"move={r['move_m']:.0f} virt={r['virtual_s']:.0f} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"deg={r.get('degenerate_calls','-')}/{r.get('sort_calls','-')} "
                  f"chg={r.get('order_changed','-')} missed={r['missed']}", flush=True)

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
                "sort_calls_total": sum(r.get("sort_calls", 0) for r in rs),
                "degenerate_calls_total": sum(r.get("degenerate_calls", 0) for r in rs),
                "order_changed_total": sum(r.get("order_changed", 0) for r in rs)}

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
    out = f"results/b_raysort_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
