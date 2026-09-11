"""组合验证：共址补测(colocate) x 顺扇排序(angsort) 是否可叠加。

## 为什么必须测这个

两个改进都已单独通过配对检验，但它们**攻击同一个症状的不同环节**：

  angsort  (`src/b_home_order.py`)：`_home_all` 排序键在 estimate 全为 None 时并列，
           稳定排序退化成频道号序。修法 = 并列时改用示向度方位角单调序。
           Q3 60 局 meanT -9.28%(p=1.2e-09)。
  colocate (`src/b_colocate.py`)：estimate 之所以为 None，是因为原点只在一个点采样。
           修法 = 在已付过移动费的停靠点原地给其余频道补测，0 移动成本拿到第二条基线。
           Q3 40 局 meanT -16.17%(p=4.5e-13)。

关键：**colocate 会消除 angsort 所修的那个并列**。拿到 estimate 后排序键不再退化，
angsort 的触发条件（`n_distinct_keys == 1`）本身就消失了。所以两者很可能是
**冗余而非互补**，朴素相加（-9.3% + -16.2% = -25.5%）几乎肯定是错的。

三种可能，必须实测区分：
  (a) 冗余：combo ≈ colocate，angsort 增益被吞掉 -> 只留 colocate，实现更简单
  (b) 互补：combo 显著优于 colocate -> 两个都要，因为 colocate 覆盖不到的频道
      （没信号、基线太短、被 cap 限制）仍走退化排序
  (c) 干扰：combo 差于 colocate -> 顺扇序与"就近取 estimate"冲突

判据是 combo vs colocate 的**配对检验**（不是各自 vs baseline 的百分比相减——
那是不同随机实现上的均值差，不构成配对证据）。

## 变体

  colo        : 只共址补测（排序用射线代理，等价于已验证的 colo）
  ang         : 只顺扇排序（复刻 b_home_order 的 angsort）
  combo       : 共址补测 + 残余并列时用顺扇序
  combo_angall: 共址补测 + 全程强制顺扇序（连有 estimate 的也按方位角），
                用于检验"顺扇"本身是否比"就近"更好，而非只作并列兜底

用法：
  python -m src.b_combo --episodes 40 --directional 0 --variants baseline,colo,ang,combo --label q3
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

ROBOT_ID = "dog-combo"


class ComboStrategy(DogStrategy):
    """共址补测与顺扇排序可独立开关，用于测叠加性。"""

    def __init__(self, *a,
                 colocate: bool = True,
                 sort_mode: str = "ray",   # ray | angle | angle_always | baseline
                 max_per_stop: int = 99,
                 min_baseline_m: float = 250.0,
                 nominal_range_m: float = 900.0,
                 **kw):
        super().__init__(*a, **kw)
        self.colocate = colocate
        self.sort_mode = sort_mode
        self.max_per_stop = max_per_stop
        self.min_baseline_m = min_baseline_m
        self.nominal_range_m = nominal_range_m
        # 命中计数
        self.colo_stops = 0
        self.colo_measures = 0
        self.colo_new_estimates = 0
        self.tie_batches = 0        # 排序键仍并列的批次（colocate 后应显著减少）
        self.reordered_batches = 0  # 并列且首选被改变的批次
        self.batches_multi = 0      # 多目标批次总数
        self._in_colocate = False

    # ---- 共址补测（与 b_colocate 同实现）----

    def _colocate_probe(self) -> None:
        if self._in_colocate or self._stop():
            return
        here = self.pos
        cands: list[ChannelState] = []
        for state in self.channels.values():
            if state.cleared or state.verdict_empty or state.estimate is not None:
                continue
            if not state.bearings:
                continue
            last = state.bearings[-1]
            if math.hypot(last.x - here[0], last.y - here[1]) < self.min_baseline_m:
                continue
            cands.append(state)
        if not cands:
            return
        cands.sort(key=lambda s: s.channel)
        self._in_colocate = True
        self.colo_stops += 1
        try:
            for state in cands[: self.max_per_stop]:
                if self._stop() or state.cleared:
                    break
                had = state.estimate is not None
                self.colo_measures += 1
                self._probe_channel_at(here, state)
                if not had and state.estimate is not None:
                    self.colo_new_estimates += 1
        finally:
            self._in_colocate = False

    def _clear_at(self, point, channel):
        ok = super()._clear_at(point, channel)
        if ok and self.colocate and not self._in_colocate:
            self._colocate_probe()
        return ok

    # ---- 排序 ----

    def _ray_proxy(self, state: ChannelState) -> float:
        last = state.bearings[-1]
        rad = math.radians(last.svd_deg)
        px = last.x + self.nominal_range_m * math.cos(rad)
        py = last.y + self.nominal_range_m * math.sin(rad)
        return math.hypot(px - self.pos[0], py - self.pos[1])

    def _order_key(self, state: ChannelState) -> float:
        if self.sort_mode == "angle_always":
            # 全程纯方位角序：连已定位的也按角度走，检验"顺扇"本身的价值
            if state.bearings:
                return state.bearings[-1].svd_deg
            return 1e9
        if state.estimate is not None:
            return math.hypot(state.estimate[0] - self.pos[0],
                              state.estimate[1] - self.pos[1])
        if not state.bearings:
            return 1e9
        if self.sort_mode == "angle":
            return state.bearings[-1].svd_deg
        return self._ray_proxy(state)

    def _home_all(self) -> None:
        if self.sort_mode == "baseline":
            return super()._home_all()
        while not self._stop():
            targets = [st for st in self.channels.values() if st.active]
            if not targets:
                return
            if len(targets) > 1:
                self.batches_multi += 1
                # 用**基线的**键判断是否仍并列：这是 angsort 的触发条件
                base_keys = {round(self._range_to(st), 6) for st in targets}
                if len(base_keys) == 1:
                    self.tie_batches += 1
                    old = min(targets, key=lambda s: s.channel).channel
                    new = min(targets, key=self._order_key).channel
                    if old != new:
                        self.reordered_batches += 1
            pick = min(targets, key=self._order_key)
            if not self._home_channel(pick):
                pick.exhausted = True


VARIANTS: dict[str, dict] = {
    "colo": {"colocate": True, "sort_mode": "ray"},
    "ang": {"colocate": False, "sort_mode": "angle"},
    "combo": {"colocate": True, "sort_mode": "angle"},
    "combo_angall": {"colocate": True, "sort_mode": "angle_always"},
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
            strat = ComboStrategy(client, cfg, **VARIANTS[variant])
        strat.run()
        if isinstance(strat, ComboStrategy):
            extra = {"colo_stops": strat.colo_stops,
                     "colo_measures": strat.colo_measures,
                     "colo_new_estimates": strat.colo_new_estimates,
                     "batches_multi": strat.batches_multi,
                     "tie_batches": strat.tie_batches,
                     "reordered_batches": strat.reordered_batches}
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


def paired_block(pairs: list[dict], variant: str, ref: str, key: str) -> dict | None:
    """variant 相对 ref 的配对检验（ref 不限于 baseline）。"""
    a = [p[variant][key] for p in pairs]
    b = [p[ref][key] for p in pairs]
    if len(a) < 2 or any(x is None for x in a + b):
        return None
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = st.fmean(d)
    sd = st.stdev(d)
    se = sd / math.sqrt(n) if n > 1 else 0.0
    base = st.fmean(b)
    out = {
        "metric": key, "ref": ref, "n_pairs": n,
        "ref_mean": round(base, 1),
        "variant_mean": round(st.fmean(a), 1),
        "d_mean": round(m, 1),
        "d_pct_of_ref": round(m / base * 100, 2) if base else None,
        "d_median": round(st.median(d), 1),
        "better": sum(1 for x in d if x < 0),
        "worse": sum(1 for x in d if x > 0),
        "tie": sum(1 for x in d if x == 0),
    }
    if sd == 0.0:
        out.update({"ci95": [0.0, 0.0], "paired_t": None,
                    "p_ttest": 1.0, "p_wilcoxon": 1.0,
                    "note": "所有配对差值恒为 0"})
        return out
    tcrit = float(sps.t.ppf(0.975, n - 1))
    tr = sps.ttest_rel(a, b)
    try:
        wp = float(sps.wilcoxon(d).pvalue)
    except ValueError:
        wp = 1.0
    out.update({
        "ci95": [round(m - tcrit * se, 1), round(m + tcrit * se, 1)],
        "paired_t": round(float(tr.statistic), 3),
        "p_ttest": float(f"{float(tr.pvalue):.3g}"),
        "p_wilcoxon": float(f"{wp:.3g}"),
        "cohen_dz": round(m / sd, 3),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=360000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,colo,ang,combo")
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
            print(f"seed={seed} {v:13s} cleared={r['cleared']}/{r['total']} "
                  f"move={r['move_m']:.0f} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"est+={r.get('colo_new_estimates','-')} "
                  f"tie/multi={r.get('tie_batches','-')}/{r.get('batches_multi','-')} "
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
                "colo_new_estimates_total": sum(r.get("colo_new_estimates", 0) for r in rs),
                "tie_batches_total": sum(r.get("tie_batches", 0) for r in rs),
                "batches_multi_total": sum(r.get("batches_multi", 0) for r in rs),
                "reordered_batches_total": sum(r.get("reordered_batches", 0) for r in rs)}

    by_seed: dict[int, dict] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["variant"]] = r

    METRICS = ("move_m", "virtual_s", "mean_clear_time_s", "cleared", "measures")
    report = {"config": {"directional": args.directional, "count": args.count,
                         "seed_start": args.seed_start, "episodes": args.episodes},
              "per_variant": {v: agg(v) for v in variants},
              "paired_vs_baseline": {},
              "paired_combo_vs_colo": {}}
    for v in variants:
        if v == "baseline":
            continue
        pairs = [p for p in by_seed.values() if "baseline" in p and v in p]
        block: dict = {"n_pairs": len(pairs)}
        for key in METRICS:
            bl = paired_block(pairs, v, "baseline", key)
            if bl:
                block[key] = bl
        block["net_extra_cleared"] = sum(p[v]["cleared"] - p["baseline"]["cleared"]
                                         for p in pairs)
        report["paired_vs_baseline"][v] = block

    # 叠加性判据：combo 相对 colo 的配对检验
    if "combo" in variants and "colo" in variants:
        pairs = [p for p in by_seed.values() if "colo" in p and "combo" in p]
        block = {"n_pairs": len(pairs)}
        for key in METRICS:
            bl = paired_block(pairs, "combo", "colo", key)
            if bl:
                block[key] = bl
        report["paired_combo_vs_colo"]["combo"] = block
    if "combo_angall" in variants and "colo" in variants:
        pairs = [p for p in by_seed.values() if "colo" in p and "combo_angall" in p]
        block = {"n_pairs": len(pairs)}
        for key in METRICS:
            bl = paired_block(pairs, "combo_angall", "colo", key)
            if bl:
                block[key] = bl
        report["paired_combo_vs_colo"]["combo_angall"] = block

    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_combo_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
