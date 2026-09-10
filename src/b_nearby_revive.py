"""按几何而非按时间调度重试：走到目标附近时再复活，而不是等整环走完。

诊断链（全部实测）：
  1. 指标 mean_clear_time_s 的 55% 来自最慢 3 个源
  2. 尾部源 = 第一次归航失败 (fails=1) 的源；头部源 fails=0
  3. 失败后要等 p50=6809 s 才有第二次归航，而第二次只需 248 s 就成功
  4. 等待原因：`_revive_exhausted` 只在 `_probe_ring` 走完**整个环**后才调用
  5. 已证伪的两个修法：
     - 立即重试（`b_tail_fix` retry_soon）：+6.8 s，无效。因为在原地重试
       是同样的失败几何。
     - 越界裁剪（`b_edge_fix` clip）：+60.5 s，更差。

本修法：复活的触发条件从"环走完"改为"机器狗当前位置离该源的估计位置足够近"。
理由：第二次归航之所以成功，是因为那时机器狗已经被环带到了目标附近（248 s 即完成）。
那就不必等环走完——每到一个环站位，检查是否有 exhausted 源的估计位置在
`revive_radius_m` 以内，若有就地复活并归航。这是"顺路处理"，不是"专程往返"。

变体：
  near600 / near900 / near1200 : 不同的就近半径
  ring_half                     : 环走到一半时也调用一次 _revive_exhausted（时间触发的折中）
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

ROBOT_ID = "dog-nr"


class NearbyReviveStrategy(DogStrategy):
    def __init__(self, *a, revive_radius_m: float = 900.0,
                 ring_half: bool = False, max_revives: int = 2, **kw):
        super().__init__(*a, **kw)
        self.revive_radius_m = revive_radius_m
        self.ring_half = ring_half
        self.max_revives = max_revives
        self._revive_counts: dict[int, int] = {}
        self.nearby_revive_hits = 0
        self.nearby_revive_tries = 0

    def _walk_ring(self, ring):
        remaining = self._order_ring(ring)
        since_home = 0
        n_total = len(remaining)
        walked = 0
        while remaining and not self._stop():
            todo = [s for s in self.channels.values() if self._probe_target(s)]
            if not todo:
                break
            station = remaining.pop(0)
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop():
                    return
                self._probe_channel_at(station, state)
            walked += 1
            since_home += 1
            ready = any(s.active for s in self.channels.values())
            if ready and (since_home >= self.config.home_batch_stations or not remaining):
                self._home_all()
                since_home = 0

            if self._stop():
                return
            # 就近复活：只处理"估计位置在当前位置附近"的 exhausted 源
            self._revive_nearby()
            if self.ring_half and walked == n_total // 2:
                self._revive_exhausted()
                self._home_all()

        if not self._stop():
            self._home_all()

    def _revive_nearby(self) -> None:
        cands = []
        for s in self.channels.values():
            if s.cleared or s.verdict_empty or not s.exhausted or not s.bearings:
                continue
            if self._revive_counts.get(s.channel, 0) >= self.max_revives:
                continue
            if s.estimate is None:
                continue
            d = math.hypot(s.estimate[0] - self.pos[0], s.estimate[1] - self.pos[1])
            if d <= self.revive_radius_m:
                cands.append((d, s))
        if not cands:
            return
        cands.sort(key=lambda kv: kv[0])
        _, s = cands[0]
        self._revive_counts[s.channel] = self._revive_counts.get(s.channel, 0) + 1
        self.nearby_revive_tries += 1
        s.exhausted = False
        s.consecutive_silent = 0
        s.homing_measures = 0
        s.clear_attempts = 0
        s.ray_swept = False
        if self._home_channel(s):
            self.nearby_revive_hits += 1
        else:
            s.exhausted = True


VARIANT_CFG = {
    "near600": {"revive_radius_m": 600.0},
    "near900": {"revive_radius_m": 900.0},
    "near1400": {"revive_radius_m": 1400.0},
    "ring_half": {"revive_radius_m": 0.0, "ring_half": True},
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
    extra = {}
    try:
        client = RobotClient(ROBOT_ID, base_url="http://127.0.0.1:" + str(port))
        if variant == "baseline":
            strat = DogStrategy(client, cfg)
        else:
            strat = NearbyReviveStrategy(client, cfg, **VARIANT_CFG[variant])
        strat.run()
        if variant != "baseline":
            extra = {"revive_hits": strat.nearby_revive_hits,
                     "revive_tries": strat.nearby_revive_tries}
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
    ap.add_argument("--seed-start", type=int, default=84000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,near900")
    ap.add_argument("--label", default="nr")
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
                  f"virt={r['virtual_s']:.0f} revive={r.get('revive_hits','-')}/{r.get('revive_tries','-')} "
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
                "measures_mean": round(st.fmean(r["measures"] for r in rs))}

    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["variant"]] = r
    base_mean = st.fmean(p["baseline"]["mean_clear_time_s"] for p in pairs.values()
                         if "baseline" in p and p["baseline"]["mean_clear_time_s"])
    report = {"per_variant": {v: agg(v) for v in variants}, "paired_vs_baseline": {},
              "baseline_mean_clear_time_s": round(base_mean, 1)}
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
    out = f"results/b_nearby_revive_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
