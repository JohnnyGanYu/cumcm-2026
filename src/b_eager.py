"""结构性修法：把"发现即归航"做彻底 —— 拿到示向度就归航到底，不让它退回环循环。

已确立的账（实测）：
  指标 mean_clear_time_s 当前 3807 s（按源平均口径）
  理论下界（每源在首次拿到示向度后 248 s 内清除）= 2414 s
  => 可省 1393 s = 37%

已证伪的修法及其原因：
  retry_soon  +6.8 s   原地重试是同样的失败几何
  clip        +60.5 s  裁剪后的 dog-leg 基线太短，交会更差
  near900     -13.7 s  有效但只吃到 1.2%：复活仍受环循环节奏支配

为什么现状慢：`_walk_ring` 的循环是"站位 -> 检测所有 probe_target 频道 -> _home_all"。
`_home_all` 只处理 `active`（有示向度且未 exhausted）的频道，且**一次只归航一个**
（`targets[0]`），归航失败即 exhausted，退回环循环等下一轮。于是"第 1 分钟听到、
第 17 次检测仍在听到"的源，可能要等整环走完才被再次认真处理。

本修法 V_eager：在 `_probe_channel_at` 拿到**首个**示向度的那一刻就地归航到底
（含射线兜底），成功即收工；失败才退回环循环。这把"发现"与"清除"在时间上绑定，
直接压缩 t_clear - t_first_bearing 这个滞后量。

代价：牺牲"就近批处理"（可能为一个远源专程往返）。这正是 `home_batch_stations`
实验当年测出"越攒越慢"的反面——那个实验说明"发现即归航"优于攒批，
本修法是把它推到极致。用实测判定。

变体：
  eager        : 首个示向度即就地归航到底
  eager_ray    : 在 eager 之上，归航失败立刻射线盲清除（方向已知时 /clear 只看距离）
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

ROBOT_ID = "dog-eager"


class EagerStrategy(DogStrategy):
    def __init__(self, *a, eager_ray: bool = False, **kw):
        super().__init__(*a, **kw)
        self.eager_ray = eager_ray
        self._in_eager = False
        self.eager_hits = 0
        self.eager_tries = 0

    def _probe_channel_at(self, point, state) -> None:
        had = bool(state.bearings)
        super()._probe_channel_at(point, state)
        if self._in_eager or state.cleared or self._stop():
            return
        # 刚拿到该频道的首个示向度 -> 立刻归航到底
        if not had and state.bearings:
            self._in_eager = True
            self.eager_tries += 1
            try:
                ok = self._home_channel(state)
                if not ok and self.eager_ray and state.bearings and not state.cleared:
                    if not state.ray_swept:
                        ok = self._ray_blind_clear(state)
                if ok:
                    self.eager_hits += 1
                elif not state.cleared:
                    state.exhausted = True
            finally:
                self._in_eager = False


def run_one(seed: int, *, directional: int, variant: str,
            count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional > 0)
    extra = {}
    try:
        client = RobotClient(ROBOT_ID, base_url="http://127.0.0.1:" + str(port))
        if variant == "baseline":
            strat = DogStrategy(client, cfg)
        else:
            strat = EagerStrategy(client, cfg, eager_ray=variant == "eager_ray")
        strat.run()
        if variant != "baseline":
            extra = {"eager_hits": strat.eager_hits, "eager_tries": strat.eager_tries}
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=81000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,eager,eager_ray")
    ap.add_argument("--label", default="eager")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        for v in variants:
            r = run_one(seed, directional=args.directional, variant=v, count=args.count)
            rows.append(r)
            print(f"seed={seed} {v:10s} cleared={r['cleared']}/{r['total']} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"virt={r['virtual_s']:.0f} move={r['move_m']:.0f} "
                  f"eager={r.get('eager_hits','-')}/{r.get('eager_tries','-')} "
                  f"missed={r['missed']}", flush=True)

    def agg(v):
        rs = [r for r in rows if r["variant"] == v]
        mt = [r["mean_clear_time_s"] for r in rs if r["mean_clear_time_s"]]
        return {"variant": v, "episodes": len(rs),
                "full_clear_rate": round(sum(1 for r in rs if r["ratio"] >= 1 - 1e-9) / len(rs), 4),
                "cleared_ratio_mean": round(st.fmean(r["ratio"] for r in rs), 4),
                "total_missed": sum(len(r["missed"]) for r in rs),
                "mean_clear_time_s": round(st.fmean(mt), 1) if mt else None,
                "virtual_s_mean": round(st.fmean(r["virtual_s"] for r in rs)),
                "move_m_mean": round(st.fmean(r["move_m"] for r in rs)),
                "measures_mean": round(st.fmean(r["measures"] for r in rs)),
                "clears_mean": round(st.fmean(r["clears"] for r in rs), 1)}

    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["variant"]] = r
    base_mean = st.fmean(p["baseline"]["mean_clear_time_s"] for p in pairs.values()
                         if "baseline" in p and p["baseline"]["mean_clear_time_s"])
    report = {"per_variant": {v: agg(v) for v in variants},
              "baseline_mean_clear_time_s": round(base_mean, 1),
              "paired_vs_baseline": {}}
    for v in variants:
        if v == "baseline":
            continue
        d = [p[v]["mean_clear_time_s"] - p["baseline"]["mean_clear_time_s"]
             for p in pairs.values() if "baseline" in p and v in p
             and p[v]["mean_clear_time_s"] and p["baseline"]["mean_clear_time_s"]]
        dc = [p[v]["cleared"] - p["baseline"]["cleared"] for p in pairs.values()
              if "baseline" in p and v in p]
        if not d:
            continue
        n = len(d)
        m = st.fmean(d)
        se = st.stdev(d) / math.sqrt(n) if n > 1 else 0.0
        report["paired_vs_baseline"][v] = {
            "episodes": n, "d_mean_clear_time_s": round(m, 1),
            "d_pct_of_baseline": round(m / base_mean * 100, 2),
            "median_s": round(st.median(d), 1),
            "ci95": [round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)] if se else None,
            "t": round(m / se, 2) if se else None,
            "better": sum(1 for x in d if x < -1),
            "worse": sum(1 for x in d if x > 1),
            "tie": sum(1 for x in d if abs(x) <= 1),
            "net_extra_cleared": sum(dc),
        }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_eager_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
