"""尾部修复实验：把"已发现但归航失败"的源尽早处理，压低 mean_clear_time_s。

诊断结论（`b_tail_diag`，10 局）：
  最慢 3 个源占清除时刻总和的 55%；把它们压到中位数 => 指标降 36%
  尾部 vs 头部：归航滞后 4761 s vs 1086 s（4.4x），沉默检测 21.3 vs 3.6
  典型样本：t=53..119 s 就拿到示向度，t≈9700 s 才被清除

机制（strategy.py）：
  `_home_all` 归航失败 -> `state.exhausted = True`
  `active` 属性排除 exhausted -> 该源不再被归航
  `_probe_target` = `not bearings or exhausted` -> 它变回补扫对象，
  但补扫要等**整个环走完**，之后才 `_revive_exhausted` 给第二次机会。
  于是"第 1 分钟就发现的源"要等到第 2 小时才处理。这就是尾部。

候选修复（都不改检测/清除物理，只改调度顺序）：
  V1 retry_soon   : 归航失败后不立即判死，先记 pending；在**下一个环站位**检测完
                    就重试一次归航（而不是等整环走完）。给每个源至多 R 次早期重试。
  V2 ray_earlier  : 归航失败且已有示向度时，立刻走射线盲清除（现状要等
                    clear/measure 预算耗尽或连续沉默触发）。方向已知时 /clear 只看
                    距离，沿射线铺点即可命中，代价是若干次 clear 调用（3 s each）。
  V3 both

指标：mean_clear_time_s（越低越好），同时必须不降清除率。
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

ROBOT_ID = "dog-tailfix"


class TailFixStrategy(DogStrategy):
    def __init__(self, *a, retry_soon: bool = False, ray_earlier: bool = False,
                 max_early_retries: int = 2, **kw):
        super().__init__(*a, **kw)
        self.retry_soon = retry_soon
        self.ray_earlier = ray_earlier
        self.max_early_retries = max_early_retries
        self._early_retries: dict[int, int] = {}
        self.early_retry_hits = 0
        self.early_ray_hits = 0

    # V2: 归航失败且有示向度 -> 立刻射线盲清除
    def _home_all(self) -> None:
        while not self._stop():
            targets = [s for s in self.channels.values() if s.active]
            if not targets:
                return
            targets.sort(key=lambda s: self._range_to(s))
            t = targets[0]
            if not self._home_channel(t):
                if (self.ray_earlier and t.bearings and not t.cleared
                        and not t.ray_swept):
                    if self._ray_blind_clear(t):
                        self.early_ray_hits += 1
                        continue
                t.exhausted = True

    # V1: 每走完一个环站位，给 pending 的 exhausted 源一次早期重试
    def _walk_ring(self, ring):
        remaining = self._order_ring(ring)
        since_home = 0
        while remaining and not self._stop():
            todo = [s for s in self.channels.values() if self._probe_target(s)]
            if not todo:
                break
            station = remaining.pop(0)
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop():
                    return
                self._probe_channel_at(station, state)
            since_home += 1
            ready = any(s.active for s in self.channels.values())
            if ready and (since_home >= self.config.home_batch_stations or not remaining):
                self._home_all()
                since_home = 0

            if self.retry_soon and not self._stop():
                self._early_retry(station)
        if not self._stop():
            self._home_all()

    def _early_retry(self, station) -> None:
        """对已有示向度但被判 exhausted 的源，就近再试一次归航。

        只挑离当前位置最近的一个，避免在一个站位上把时间全耗掉。
        """
        cands = [s for s in self.channels.values()
                 if s.exhausted and s.bearings and not s.cleared
                 and not s.verdict_empty
                 and self._early_retries.get(s.channel, 0) < self.max_early_retries]
        if not cands:
            return
        cands.sort(key=lambda s: self._range_to(s))
        s = cands[0]
        self._early_retries[s.channel] = self._early_retries.get(s.channel, 0) + 1
        # 复活：重置归航预算与连续沉默计数，让它能真正再走一轮
        s.exhausted = False
        s.consecutive_silent = 0
        s.homing_measures = 0
        s.clear_attempts = 0
        if self._home_channel(s):
            self.early_retry_hits += 1
        else:
            s.exhausted = True


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
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        if variant == "baseline":
            strat = DogStrategy(client, cfg)
        else:
            strat = TailFixStrategy(
                client, cfg,
                retry_soon=variant in ("retry_soon", "both"),
                ray_earlier=variant in ("ray_earlier", "both"))
        strat.run()
        if variant != "baseline":
            extra = {"early_retry_hits": strat.early_retry_hits,
                     "early_ray_hits": strat.early_ray_hits}
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


VARIANTS = ("baseline", "retry_soon", "ray_earlier", "both")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=82000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--label", default="tailfix")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        for v in variants:
            r = run_one(seed, directional=args.directional, variant=v, count=args.count)
            rows.append(r)
            print(f"seed={seed} {v:11s} cleared={r['cleared']}/{r['total']} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"virt={r['virtual_s']:.0f} move={r['move_m']:.0f} "
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
                "clears_mean": round(st.fmean(r["clears"] for r in rs), 1)}

    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["variant"]] = r

    report = {"per_variant": {v: agg(v) for v in variants}, "paired_vs_baseline": {}}
    for v in variants:
        if v == "baseline":
            continue
        d = [(p[v]["mean_clear_time_s"] - p["baseline"]["mean_clear_time_s"])
             for p in pairs.values()
             if "baseline" in p and v in p
             and p[v]["mean_clear_time_s"] and p["baseline"]["mean_clear_time_s"]]
        dc = [(p[v]["cleared"] - p["baseline"]["cleared"])
              for p in pairs.values() if "baseline" in p and v in p]
        if not d:
            continue
        n = len(d)
        m = st.fmean(d)
        se = st.stdev(d) / math.sqrt(n) if n > 1 else 0.0
        report["paired_vs_baseline"][v] = {
            "episodes": n,
            "d_mean_clear_time_s": round(m, 1),
            "d_pct": round(m / st.fmean(
                p["baseline"]["mean_clear_time_s"] for p in pairs.values()
                if "baseline" in p and p["baseline"]["mean_clear_time_s"]) * 100, 2),
            "median_s": round(st.median(d), 1),
            "ci95": [round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)] if se else None,
            "t": round(m / se, 2) if se else None,
            "better": sum(1 for x in d if x < -1),
            "worse": sum(1 for x in d if x > 1),
            "tie": sum(1 for x in d if abs(x) <= 1),
            "net_extra_cleared": sum(dc),
        }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_tail_fix_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
