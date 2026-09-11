"""调度改进：修掉 `_home_all` 的排序键退化，真实闭环配对 A/B。

## 根因（打桩实测，`--diagnose` 可复现）

`DogStrategy._home_all` 每轮按 `self._range_to(st)` 排序取最近目标。但：

    def _range_to(self, state):
        if state.estimate is None:
            if state.bearings:
                last = state.bearings[-1]
                return math.hypot(last.x - self.pos[0], last.y - self.pos[1])   # <-- 观测点
            return 1e9
        return math.hypot(state.estimate[0] - self.pos[0], ...)

`estimate` 需要 >= 2 条示向度（`_refresh` 要求）才非 None。而环形补扫是"在同一个
站位上把所有待测频道扫一遍"，所以一批目标的**首条**示向度全部记在**同一个观测点**
上，且那个观测点就是机器狗当前位置 => 每个 key 都等于 0.0。

实测（30 局 Q3，74 个多目标批次）：**27/27** 抽样批次的活跃目标 `estimate` 全为
None，排序键互异值数恒为 1。`sorted` 是稳定排序，于是"按距离取最近"实际退化成
**按频道号升序**——频道号是题目随机分配的编号，与空间位置无关。

## 头room（真值巡回长度，74 批次均值）

    频道号序（现状）    3796 m
    示向度方位角序      2852 m   -24.9%
    真值最优（下界）    2179 m   -42.6%

## 修法

只改**并列时的次序**，不改任何检测/清除/预算/复活逻辑：`estimate is None` 时，
把目标投影到示向度射线上的一个标称点（观测点沿示向度方向前进 `ray_probe_m`），
用该点到当前位置的距离作排序键。`_home_all` 每轮重排 + 取首个，等价于在投影点集
上做贪心最近邻。

变体：
  raysort      : 标称距离 = 环半径量级（默认 700 m），仅修并列
  raysort350   : 标称距离更近（近源假设）
  raysort1200  : 标称距离更远（远源假设）
  angsort      : 直接按示向度方位角单调排序（不投影，纯扫角序）

`ray_probe_m` 的选择不影响"方向"信息，只影响不同频道投影点之间的相对远近；
标称距离越大越接近"纯方位角序"，越小越接近"观测点聚类序"。

用法：
  python -m src.b_home_order --diagnose --episodes 12
  python -m src.b_home_order --episodes 60 --directional 0 --variants baseline,raysort --label q3
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
import threading

from scipy import stats as sps

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import ChannelState, DogStrategy, StrategyConfig

ROBOT_ID = "dog-order"


class RaySortStrategy(DogStrategy):
    """并列时用示向度射线投影点排序，替代退化的频道号序。"""

    def __init__(self, *a, ray_probe_m: float = 700.0,
                 mode: str = "ray", **kw):
        super().__init__(*a, **kw)
        self.ray_probe_m = ray_probe_m
        self.mode = mode
        self.tie_batches = 0          # 命中计数：出现并列且被重排的批次数
        self.reordered_batches = 0    # 重排后首选目标确实变了的批次数

    def _order_key(self, state: ChannelState) -> float:
        if state.estimate is not None:
            return math.hypot(state.estimate[0] - self.pos[0],
                              state.estimate[1] - self.pos[1])
        if not state.bearings:
            return 1e9
        last = state.bearings[-1]
        if self.mode == "angle":
            # 纯扫角序：把方位角本身当排序键（0..360 -> 0..1e6 量级，与距离键分离）
            return last.svd_deg
        rad = math.radians(last.svd_deg)
        px = last.x + self.ray_probe_m * math.cos(rad)
        py = last.y + self.ray_probe_m * math.sin(rad)
        return math.hypot(px - self.pos[0], py - self.pos[1])

    def _home_all(self) -> None:
        while not self._stop():
            targets = [st for st in self.channels.values() if st.active]
            if not targets:
                return
            if len(targets) > 1:
                base_keys = {round(self._range_to(st), 6) for st in targets}
                if len(base_keys) == 1:
                    self.tie_batches += 1
                    old_first = min(targets, key=lambda s: s.channel).channel
                    new_first = min(targets, key=self._order_key).channel
                    if old_first != new_first:
                        self.reordered_batches += 1
            targets.sort(key=self._order_key)
            if not self._home_channel(targets[0]):
                targets[0].exhausted = True


VARIANTS: dict[str, dict] = {
    "raysort": {"ray_probe_m": 700.0, "mode": "ray"},
    "raysort350": {"ray_probe_m": 350.0, "mode": "ray"},
    "raysort1200": {"ray_probe_m": 1200.0, "mode": "ray"},
    "angsort": {"mode": "angle"},
}


# ---------------- 诊断：证明排序键退化 + 量化头room ----------------

class DiagProbe(DogStrategy):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.batches: list[dict] = []

    def _home_all(self):
        targets = [st for st in self.channels.values() if st.active]
        if len(targets) > 1:
            keys = [self._range_to(st) for st in targets]
            self.batches.append({
                "pos": self.pos,
                "n": len(targets),
                "n_with_estimate": sum(1 for s in targets if s.estimate is not None),
                "n_distinct_keys": len({round(k, 6) for k in keys}),
                "t": [{"ch": s.channel, "svd": s.bearings[-1].svd_deg} for s in targets],
            })
        return super()._home_all()


def _tour(order, pos, truth) -> float:
    cur, tot = pos, 0.0
    for ch in order:
        j = truth[ch]
        tot += math.dist(cur, (j.x, j.y))
        cur = (j.x, j.y)
    return tot


def diagnose(episodes: int, seed_start: int, directional: int) -> dict:
    batches = []
    for i in range(episodes):
        seed = seed_start + i
        case = generate_case(seed, directional_count=directional)
        truth = {j.channel: j for j in case.jammers}
        arena = Arena(case)
        sess = Session(ROBOT_ID, arena, real_limit_s=1200.0)
        h = make_server(sess, port=0)
        port = h.server_address[1]
        threading.Thread(target=h.serve_forever, daemon=True).start()
        try:
            s = DiagProbe(RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}"),
                          StrategyConfig(directional_probe=directional > 0))
            s.run()
        finally:
            h.shutdown()
            h.server_close()
        for b in s.batches:
            chs = [t["ch"] for t in b["t"]]
            ang = {t["ch"]: t["svd"] for t in b["t"]}
            L_ch = _tour(sorted(chs), b["pos"], truth)
            L_ang = _tour(sorted(chs, key=lambda c: ang[c]), b["pos"], truth)
            L_opt = (_tour(min(itertools.permutations(chs),
                              key=lambda o: _tour(o, b["pos"], truth)), b["pos"], truth)
                     if len(chs) <= 7 else None)
            batches.append({**{k: b[k] for k in ("n", "n_with_estimate", "n_distinct_keys")},
                            "L_channel_order_m": round(L_ch, 1),
                            "L_bearing_order_m": round(L_ang, 1),
                            "L_optimal_m": None if L_opt is None else round(L_opt, 1)})
    tied = [b for b in batches if b["n_distinct_keys"] == 1]
    cmp = [b for b in batches if b["L_optimal_m"] is not None]
    Lch = st.fmean(b["L_channel_order_m"] for b in batches)
    Lang = st.fmean(b["L_bearing_order_m"] for b in batches)
    return {
        "episodes": episodes, "directional": directional,
        "multi_target_batches": len(batches),
        "batches_all_keys_tied": len(tied),
        "tied_share": round(len(tied) / len(batches), 4) if batches else None,
        "batches_with_any_estimate": sum(1 for b in batches if b["n_with_estimate"] > 0),
        "batch_size_range": [min(b["n"] for b in batches), max(b["n"] for b in batches)]
        if batches else None,
        "truth_tour_channel_order_m_mean": round(Lch, 1),
        "truth_tour_bearing_order_m_mean": round(Lang, 1),
        "bearing_vs_channel_pct": round((Lang - Lch) / Lch * 100, 1),
        "truth_tour_optimal_m_mean": round(st.fmean(b["L_optimal_m"] for b in cmp), 1)
        if cmp else None,
        "optimal_vs_channel_pct": round(
            (st.fmean(b["L_optimal_m"] for b in cmp)
             - st.fmean(b["L_channel_order_m"] for b in cmp))
            / st.fmean(b["L_channel_order_m"] for b in cmp) * 100, 1) if cmp else None,
        "note": "L_* 为真值巡回长度，衡量批内顺序质量；不含检测/逼近开销，是顺序优化的乐观上限",
    }


# ---------------- 闭环 A/B ----------------

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
            extra = {"tie_batches": strat.tie_batches,
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


def paired_block(pairs: list[dict], variant: str, key: str) -> dict | None:
    a = [p[variant][key] for p in pairs]
    b = [p["baseline"][key] for p in pairs]
    if len(a) < 2 or any(x is None for x in a + b):
        return None
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = st.fmean(d)
    sd = st.stdev(d)
    se = sd / math.sqrt(n) if n > 1 else 0.0
    base = st.fmean(b)
    out = {
        "metric": key, "n_pairs": n,
        "baseline_mean": round(base, 1),
        "variant_mean": round(st.fmean(a), 1),
        "d_mean": round(m, 1),
        "d_pct_of_baseline": round(m / base * 100, 2) if base else None,
        "d_median": round(st.median(d), 1),
        "d_sd": round(sd, 1),
        "better": sum(1 for x in d if x < 0),
        "worse": sum(1 for x in d if x > 0),
        "tie": sum(1 for x in d if x == 0),
    }
    if sd == 0.0:
        out.update({"ci95": [0.0, 0.0], "paired_t": None,
                    "p_ttest": 1.0, "p_wilcoxon": 1.0,
                    "note": "所有配对差值恒为 0（变体与基线逐局完全一致）"})
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
    ap.add_argument("--diagnose", action="store_true", help="只跑根因诊断与头room量化")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-start", type=int, default=330000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,raysort")
    ap.add_argument("--label", default="q3")
    args = ap.parse_args()
    if args.count is not None and args.directional > args.count:
        ap.error("--directional 不能超过 --count")

    if args.diagnose:
        rep = diagnose(args.episodes, args.seed_start, args.directional)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        out = f"results/b_home_order_diag_{args.label}.json"
        with open(out, "w", encoding="utf-8") as h:
            json.dump(rep, h, ensure_ascii=False, indent=2)
        print(f"\nwrote {out}")
        return

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
                  f"tie/reord={r.get('tie_batches','-')}/{r.get('reordered_batches','-')} "
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
                "tie_batches_total": sum(r.get("tie_batches", 0) for r in rs),
                "reordered_batches_total": sum(r.get("reordered_batches", 0) for r in rs)}

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
            bl = paired_block(pairs, v, key)
            if bl:
                block[key] = bl
        block["net_extra_cleared"] = sum(p[v]["cleared"] - p["baseline"]["cleared"]
                                         for p in pairs)
        report["paired_vs_baseline"][v] = block

    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_home_order_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
