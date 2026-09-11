"""检验假设："对边界/窄扇区源的处理已经成熟，所以新思路没有空间了"。

## 为什么要测这个

连续四条新思路里三条被证伪（defer_far / colocate护栏 / GDOP门禁），
一种解释是"该拿的收益已经被现有兜底机制拿走了"。这个说法听起来合理，
但它是**可检验的**，不能停在感觉上：如果那些机制真的在承担关键工作，
**关掉它们应当显著变差**；如果关掉几乎无差别，说明它们其实是空转的
（那么"成熟"是假象，真正的原因在别处）。

这叫消融（ablation），方向与常规 A/B 相反：常规是"加东西看变好没有"，
这里是"减东西看变差多少"，用来量化既有机制的真实贡献。

## 被消融的机制（全部与边界/窄可见区源相关）

ray_off        : 关掉 `_ray_blind_clear`（方向已知、距离未知时沿射线铺点清除）。
                 这是 estimate 恒为 None 与丢信号两条路径的最终兜底。
narrow_off     : 关掉窄扇区加码（narrow_sector_extra_measures 22 -> 0，
                 narrow_sector_silent_tolerance 10 -> silent_tolerance）。
revive_off     : 关掉 `_revive_nearby`（就近复活已放弃的频道）。
all_off        : 同时关掉上述三项。

## 判据

对每个消融，看 **清除率** 与 **指标 T̄** 的配对变化：
  - 若关掉后清除率显著下降 => 该机制确实在承担关键工作，"成熟"说法成立。
  - 若关掉后几乎无差别 => 该机制在当前配置下是空转的，"成熟"不成立，
    需要另找三条思路失败的原因。

用法:
  python -m src.b_boundary_maturity --episodes 60 --directional 10
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

ROBOT_ID = "dog-ablate"


class NoRayStrategy(DogStrategy):
    """关掉射线盲扫兜底：估计不可用时直接放弃该频道。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ray_suppressed = 0

    def _ray_blind_clear(self, state: ChannelState) -> bool:
        self.ray_suppressed += 1
        return False


class NoReviveStrategy(DogStrategy):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.revive_suppressed = 0

    def _revive_nearby(self) -> None:
        self.revive_suppressed += 1
        return None


class NoRayNoReviveStrategy(NoRayStrategy):
    def _revive_nearby(self) -> None:
        return None


def make_config(directional: int, variant: str) -> StrategyConfig:
    cfg = StrategyConfig(directional_probe=directional > 0)
    if variant in ("narrow_off", "all_off"):
        cfg.narrow_sector_extra_measures = 0
        cfg.narrow_sector_silent_tolerance = cfg.silent_tolerance
    return cfg


def make_strategy(client, cfg, variant: str) -> DogStrategy:
    if variant == "ray_off":
        return NoRayStrategy(client, cfg)
    if variant == "revive_off":
        return NoReviveStrategy(client, cfg)
    if variant == "all_off":
        return NoRayNoReviveStrategy(client, cfg)
    return DogStrategy(client, cfg)          # prod / narrow_off


def run_one(seed: int, *, directional: int, variant: str,
            count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        cfg = make_config(directional, variant)
        strat = make_strategy(client, cfg, variant)
        strat.run()
        s = arena.stats()
        return {"seed": seed, "variant": variant,
                "cleared": s["cleared"], "total": s["total_jammers"],
                "clear_ratio": s["cleared_ratio"],
                "mean_clear_time_s": s["mean_clear_time_s"],
                "move_m": s["move_distance_m"],
                "measures": s["measure_calls"],
                "ray_suppressed": getattr(strat, "ray_suppressed", 0),
                "revive_suppressed": getattr(strat, "revive_suppressed", 0),
                "missed": [j.channel for j in case.jammers if not j.cleared]}
    finally:
        httpd.shutdown()
        httpd.server_close()


def paired(base: list[dict], var: list[dict], key: str) -> dict:
    pairs = [(b[key], v[key]) for b, v in zip(base, var)
             if b[key] is not None and v[key] is not None]
    d = [v - b for b, v in pairs]
    n = len(d)
    if n < 2:
        return {"n_pairs": n}
    bm = st.fmean(b for b, _ in pairs)
    dm = st.fmean(d)
    sd = st.stdev(d)
    if sd == 0.0:
        t_stat, p_t, p_w = 0.0, 1.0, 1.0
    else:
        r = sps.ttest_rel([v for _, v in pairs], [b for b, _ in pairs])
        t_stat, p_t = float(r.statistic), float(r.pvalue)
        try:
            p_w = float(sps.wilcoxon([v for _, v in pairs],
                                     [b for b, _ in pairs]).pvalue)
        except ValueError:
            p_w = 1.0
    half = 1.959963985 * sd / math.sqrt(n)
    return {"n_pairs": n, "baseline_mean": round(bm, 2),
            "d_mean": round(dm, 2),
            "d_pct": round(dm / bm * 100, 2) if bm else None,
            "ci95": [round(dm - half, 2), round(dm + half, 2)],
            "t": round(t_stat, 3),
            "p_ttest": float(f"{p_t:.3g}"), "p_wilcoxon": float(f"{p_w:.3g}"),
            "better": sum(1 for x in d if x < 0),
            "worse": sum(1 for x in d if x > 0),
            "tie": sum(1 for x in d if x == 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--directional", type=int, default=10)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=9000)
    ap.add_argument("--variants",
                    default="prod,ray_off,narrow_off,revive_off,all_off")
    ap.add_argument("--label", default="d10")
    a = ap.parse_args()

    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    runs: dict[str, list[dict]] = {v: [] for v in names}
    for i in range(a.episodes):
        sd = a.seed_start + i
        parts = [f"seed={sd}"]
        for v in names:
            r = run_one(sd, directional=a.directional, variant=v, count=a.count)
            runs[v].append(r)
            parts.append(f"{v}={r['cleared']}/{r['total']}")
        print("  ".join(parts), flush=True)

    def agg(rows):
        mts = [r["mean_clear_time_s"] for r in rows if r["mean_clear_time_s"] is not None]
        return {"episodes": len(rows),
                "clear_rate": round(st.fmean(r["clear_ratio"] for r in rows), 5),
                "full_clear": sum(1 for r in rows if r["cleared"] == r["total"]),
                "total_missed": sum(r["total"] - r["cleared"] for r in rows),
                "meanT": round(st.fmean(mts), 1) if mts else None,
                "move_m": round(st.fmean(r["move_m"] for r in rows), 1),
                "ray_suppressed": sum(r["ray_suppressed"] for r in rows),
                "revive_suppressed": sum(r["revive_suppressed"] for r in rows)}

    base = runs[names[0]]
    report = {"config": {"directional": a.directional, "count": a.count,
                         "seed_start": a.seed_start, "episodes": a.episodes},
              "per_variant": {v: agg(runs[v]) for v in names},
              "paired_vs_prod": {
                  v: {k: paired(base, runs[v], k) for k in
                      ("cleared", "mean_clear_time_s", "move_m")}
                  for v in names[1:]}}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_boundary_maturity_{a.label}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"report": report, "runs": runs}, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
