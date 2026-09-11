"""把每局的移动距离 / 虚拟时间按「策略阶段」分账，找出真正的浪费源。

方法：猴子补丁包住 `DogStrategy` 的各阶段方法，维护一个阶段栈；`_measure` /
`_clear_at` 之前记录当前位置，之后按栈顶阶段累加位移。不改源码。

同时统计环上顺扫的「跨场弦跳」：`_walk_ring` 里连续两次进入站位之间，
机器狗实际走的总位移 vs 两站位之间的弧弦距离。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading
from collections import defaultdict

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-diag"

PHASES = ["sweep_origin", "walk_ring", "home_all", "revive_nearby",
          "inner_retry", "belief_search", "blind_clear", "other"]


def instrument(strat: DogStrategy) -> dict:
    acc = {"move_m": defaultdict(float), "time_s": defaultdict(float),
           "calls": defaultdict(int)}
    stack: list[str] = ["other"]

    def wrap_phase(name: str, method):
        def inner(*a, **kw):
            stack.append(name)
            try:
                return method(*a, **kw)
            finally:
                stack.pop()
        return inner

    # 阶段包装（注意 _home_all 会被 _walk_ring 调用，栈顶归属最内层）
    strat._sweep_at = wrap_phase("sweep_origin", strat._sweep_at)          # type: ignore[method-assign]
    strat._walk_ring = wrap_phase("walk_ring", strat._walk_ring)           # type: ignore[method-assign]
    strat._home_all = wrap_phase("home_all", strat._home_all)              # type: ignore[method-assign]
    strat._revive_nearby = wrap_phase("revive_nearby", strat._revive_nearby)  # type: ignore[method-assign]
    strat._inner_retry = wrap_phase("inner_retry", strat._inner_retry)     # type: ignore[method-assign]
    strat._belief_search = wrap_phase("belief_search", strat._belief_search)  # type: ignore[method-assign]
    strat._blind_clear_sweep = wrap_phase("blind_clear", strat._blind_clear_sweep)  # type: ignore[method-assign]

    orig_measure = strat._measure
    orig_clear = strat._clear_at

    def measure(point, channel):
        p0, t0 = strat.pos, strat.client.virtual_time_s
        r = orig_measure(point, channel)
        ph = stack[-1]
        acc["move_m"][ph] += math.hypot(strat.pos[0] - p0[0], strat.pos[1] - p0[1])
        acc["time_s"][ph] += strat.client.virtual_time_s - t0
        acc["calls"][ph + ":measure"] += 1
        return r

    def clear_at(point, channel):
        p0, t0 = strat.pos, strat.client.virtual_time_s
        r = orig_clear(point, channel)
        ph = stack[-1]
        acc["move_m"][ph] += math.hypot(strat.pos[0] - p0[0], strat.pos[1] - p0[1])
        acc["time_s"][ph] += strat.client.virtual_time_s - t0
        acc["calls"][ph + ":clear"] += 1
        return r

    strat._measure = measure          # type: ignore[method-assign]
    strat._clear_at = clear_at        # type: ignore[method-assign]
    return acc


def run_one(seed: int, *, directional: int, count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strat = DogStrategy(client, StrategyConfig(directional_probe=directional > 0))
        acc = instrument(strat)
        strat.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    s = arena.stats()
    return {"seed": seed, "cleared": s["cleared"], "total": s["total_jammers"],
            "mean_clear_time_s": s["mean_clear_time_s"],
            "virtual_s": s["total_virtual_time_s"],
            "move_m": s["move_distance_m"],
            "by_phase_move_m": {k: round(v) for k, v in acc["move_m"].items()},
            "by_phase_time_s": {k: round(v) for k, v in acc["time_s"].items()},
            "calls": dict(acc["calls"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--label", default="q4")
    args = ap.parse_args()

    rows = [run_one(args.seed_start + i, directional=args.directional)
            for i in range(args.episodes)]
    for r in rows:
        print(f"seed={r['seed']} T={round(r['mean_clear_time_s'])} "
              f"virt={r['virtual_s']:.0f} move={r['move_m']:.0f}", flush=True)

    agg_move = {p: round(st.fmean(r["by_phase_move_m"].get(p, 0) for r in rows))
                for p in PHASES}
    agg_time = {p: round(st.fmean(r["by_phase_time_s"].get(p, 0) for r in rows))
                for p in PHASES}
    total_t = sum(agg_time.values())
    share = {p: round(agg_time[p] / total_t, 4) if total_t else 0.0 for p in PHASES}
    allcalls: dict[str, float] = defaultdict(float)
    for r in rows:
        for k, v in r["calls"].items():
            allcalls[k] += v / len(rows)
    report = {"episodes": len(rows),
              "mean_clear_time_s": round(st.fmean(r["mean_clear_time_s"] for r in rows), 1),
              "virtual_s_mean": round(st.fmean(r["virtual_s"] for r in rows), 1),
              "move_m_mean": round(st.fmean(r["move_m"] for r in rows)),
              "phase_move_m": agg_move, "phase_time_s": agg_time,
              "phase_time_share": share,
              "calls_mean": {k: round(v, 1) for k, v in sorted(allcalls.items())}}
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_sector_diag_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
