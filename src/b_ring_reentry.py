"""量化"归航返回后如何重新接入环"的代价，并给出优化候选的实测对比。

现状（strategy.py `_walk_ring`）：
    remaining = self._order_ring(ring)   # 只在进入本环时排一次序
    while remaining: station = remaining.pop(0); ...; self._home_all()

问题：`_order_ring` 的切入点是按"当时的 self.pos"选的。一旦 `_home_all()` 把机器狗
带到某个源附近（离环可能 1-2 km），下一次 `remaining.pop(0)` 仍然按**当初**排好的
顺序取下一站，可能在环的另一侧。实测：每局 ring 段走 ~44,000 m，而环周长只有
7,765 m —— 相当于绕环 5.6 周。

注意这不是"最近邻 vs 顺扫"之争（那个已被证伪，见 `_order_ring` 注释：贪心会跨场
弦跳）。真正的问题是**顺扫的相位没有在返回后重新对齐**：顺扫方向是对的，但接入点
应该是"离当前位置最近的未访问站"，然后从那里继续单调顺扫。

三个候选：
  A baseline    : 现状
  B reenter     : 每次归航返回后，把剩余站位按方位角重新对齐相位（离当前最近的
                  未访问站开始，保持单调方向继续），消除跨场弦跳
  C defer_far   : 在 B 之上，归航时若目标距当前环位置 > threshold，先记账不去，
                  等顺扫到离它最近的环段再归航（把"往返"换成"顺路"）

评分指标是 mean_clear_time_s（各源被清除时刻的平均），所以早清除便宜、绕路贵。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading
import time

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-ring"


class ReentryStrategy(DogStrategy):
    """候选 B：归航返回后重新对齐顺扫相位。

    只改 `_walk_ring` 的站位取用顺序，不动任何检测/清除/预算逻辑。
    """

    def __init__(self, *a, reentry: bool = True, **kw):
        super().__init__(*a, **kw)
        self.reentry = reentry
        self.reentry_saves_m = 0.0
        self.reentry_count = 0

    def _walk_ring(self, ring: list[tuple[float, float]]) -> None:
        remaining = self._order_ring(ring)
        since_home = 0
        while remaining and not self._stop():
            todo = [s for s in self.channels.values() if self._probe_target(s)]
            if not todo:
                break

            if self.reentry and len(remaining) > 1:
                # 保持既定单调方向，但把相位对齐到"离当前位置最近的未访问站"。
                # remaining 已是单调序列；找最近站，把它转到队首，其后顺序不变
                # （循环左移），从而继续同向顺扫而不产生跨场弦跳。
                idx = min(range(len(remaining)),
                          key=lambda i: math.hypot(remaining[i][0] - self.pos[0],
                                                   remaining[i][1] - self.pos[1]))
                if idx != 0:
                    d_orig = math.hypot(remaining[0][0] - self.pos[0],
                                        remaining[0][1] - self.pos[1])
                    d_new = math.hypot(remaining[idx][0] - self.pos[0],
                                       remaining[idx][1] - self.pos[1])
                    self.reentry_saves_m += max(0.0, d_orig - d_new)
                    self.reentry_count += 1
                    remaining = remaining[idx:] + remaining[:idx]

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
        if not self._stop():
            self._home_all()


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
            strat = ReentryStrategy(client, cfg, reentry=True)
        strat.run()
        if variant != "baseline":
            extra = {"reentry_count": strat.reentry_count,
                     "reentry_saves_m": round(strat.reentry_saves_m)}
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
    ap.add_argument("--seed-start", type=int, default=90000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--label", default="ring")
    args = ap.parse_args()
    if args.count is not None and args.directional > args.count:
        ap.error("--directional 不能超过 --count")

    rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        for variant in ("baseline", "reentry"):
            r = run_one(seed, directional=args.directional, variant=variant,
                        count=args.count)
            rows.append(r)
            print(f"seed={seed} {variant:9s} cleared={r['cleared']}/{r['total']} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"move={r['move_m']:.0f} virt={r['virtual_s']:.0f} "
                  f"missed={r['missed']}", flush=True)

    def agg(v):
        rs = [r for r in rows if r["variant"] == v]
        mt = [r["mean_clear_time_s"] for r in rs if r["mean_clear_time_s"]]
        return {"variant": v, "episodes": len(rs),
                "full_clear_rate": round(sum(1 for r in rs if r["ratio"] >= 1 - 1e-9) / len(rs), 4),
                "cleared_ratio_mean": round(st.fmean(r["ratio"] for r in rs), 4),
                "total_missed": sum(len(r["missed"]) for r in rs),
                "mean_clear_time_s": round(st.fmean(mt), 1) if mt else None,
                "move_m_mean": round(st.fmean(r["move_m"] for r in rs)),
                "virtual_s_mean": round(st.fmean(r["virtual_s"] for r in rs), 1)}

    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["variant"]] = r
    full = [p for p in pairs.values() if len(p) == 2]
    dmt = [p["reentry"]["mean_clear_time_s"] - p["baseline"]["mean_clear_time_s"]
           for p in full
           if p["reentry"]["mean_clear_time_s"] and p["baseline"]["mean_clear_time_s"]]
    dmv = [p["reentry"]["move_m"] - p["baseline"]["move_m"] for p in full]
    n = len(dmt)
    se = st.stdev(dmt) / math.sqrt(n) if n > 1 else 0.0
    m = st.fmean(dmt) if dmt else 0.0
    report = {
        "baseline": agg("baseline"), "reentry": agg("reentry"),
        "paired": {
            "episodes": n,
            "d_mean_clear_time_s": round(m, 1),
            "d_mean_clear_time_median_s": round(st.median(dmt), 1) if dmt else None,
            "ci95": [round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)] if n > 1 else None,
            "t": round(m / se, 2) if se else None,
            "d_move_m": round(st.fmean(dmv)) if dmv else None,
            "better": sum(1 for x in dmt if x < -1),
            "worse": sum(1 for x in dmt if x > 1),
            "tie": sum(1 for x in dmt if abs(x) <= 1),
            "net_extra_cleared": sum(p["reentry"]["cleared"] - p["baseline"]["cleared"]
                                     for p in full),
        },
        "config": {"directional": args.directional, "count": args.count,
                   "seed_start": args.seed_start},
    }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_ring_reentry_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
