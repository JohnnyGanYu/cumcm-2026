"""边缘源修复：dog-leg / step_closer 的候选点越界时**裁剪**而不是放弃。

根因链（全部实测）：
  1. mean_clear_time_s 的 55% 来自最慢 3 个源；压到中位数可降 36%
  2. 尾部源第一次归航在 t≈2000-3500 s 静默失败，等 p50=6809 s 才重试，
     重试后仅 248 s 就清除成功 => 浪费的是等待，不是归航
  3. 首次失败不走任何 events 分支（events 计数为 0），是静默 return False
  4. 静默出口定位：54.5% 的 dog-leg 失败是**全部候选点越界**
     （失败时机器狗平均在 r=1624 m，区域半径 1800 m，dog-leg 基线 380 m）

即：边缘源。机器狗在外围朝区域外的目标做 dog-leg，380 m 基线把落点推到区域外，
`_dogleg` 逐个 `continue` 直到候选耗尽，返回 False，归航静默放弃。

修法（不改物理、不改预算、不改判定）：
  V1 clip     : 候选点越界时沿半径缩回到区域内（留 5 m 余量）再用，而不是丢弃。
                dog-leg 的目的是拿第二条基线，缩短基线仍然可用，只是交会角略差。
  V2 shrink   : 在 V1 之上，按可用空间自适应缩短 dog-leg 基线
                （取 min(380, 到边界的可行距离)），并在基线过短时改用更大偏转角
                以保住交会角。
指标：mean_clear_time_s 越低越好，且清除率不能降。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from .b_sim.arena import ARENA_RADIUS, Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-edge"
MARGIN_M = 5.0


def _clip_into_arena(x: float, y: float, margin: float = MARGIN_M):
    """把点沿原点方向缩回区域内，返回 None 表示无法修正（原点也越界，不可能）。"""
    r = math.hypot(x, y)
    limit = ARENA_RADIUS - margin
    if r <= limit:
        return (x, y)
    if r <= 1e-9:
        return (x, y)
    k = limit / r
    return (x * k, y * k)


class EdgeFixStrategy(DogStrategy):
    def __init__(self, *a, mode: str = "clip", **kw):
        super().__init__(*a, **kw)
        self.mode = mode
        self.clip_dogleg = 0
        self.clip_step = 0

    def _dogleg(self, state) -> bool:
        """与父类同构，但候选点越界时裁剪回区域内而不是跳过。"""
        if not state.bearings:
            return False
        last = state.bearings[-1]
        cfg = self.config
        for offset in (cfg.dogleg_offset_deg, cfg.dogleg_offset_deg * 0.5, 0.0):
            for sign in ((1.0, -1.0) if offset else (1.0,)):
                angle = math.radians(last.svd_deg + sign * offset)
                length = cfg.dogleg_length_m
                if self.mode == "shrink":
                    # 自适应：沿该方向到区域边界的最大可行距离
                    length = min(length, _max_reach(self.pos, angle))
                    if length < 40.0:
                        continue
                cand = (self.pos[0] + length * math.cos(angle),
                        self.pos[1] + length * math.sin(angle))
                if math.hypot(*cand) > ARENA_RADIUS:
                    clipped = _clip_into_arena(*cand)
                    if clipped is None:
                        continue
                    # 裁剪后若与当前位置几乎重合，这个候选没有信息量，换下一个
                    if math.dist(clipped, self.pos) < 30.0:
                        continue
                    cand = clipped
                    self.clip_dogleg += 1
                before = len(state.bearings)
                self._probe_channel_at(cand, state)
                if state.cleared:
                    return True
                if len(state.bearings) > before:
                    return True
                if self._stop() or state.homing_measures >= (
                        cfg.max_homing_measures + (6 if state.revived else 0)):
                    return False
        return False

    def _step_closer(self, state, distance: float) -> bool:
        assert state.estimate is not None
        cfg = self.config
        ex, ey = state.estimate
        dx, dy = ex - self.pos[0], ey - self.pos[1]
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        advance = max(cfg.clear_distance_m * 0.8, distance * cfg.approach_fraction)
        advance = (min(advance, distance - cfg.clear_distance_m * 0.35)
                   if distance > cfg.clear_distance_m else advance)
        lateral = min(distance * cfg.lateral_fraction, 220.0)
        for sign in (1.0, -1.0):
            cand = (self.pos[0] + ux * advance - uy * lateral * sign,
                    self.pos[1] + uy * advance + ux * lateral * sign)
            if math.hypot(*cand) > ARENA_RADIUS * 1.02:
                clipped = _clip_into_arena(*cand)
                if clipped is None or math.dist(clipped, self.pos) < 20.0:
                    continue
                cand = clipped
                self.clip_step += 1
            before = len(state.bearings)
            self._probe_channel_at(cand, state)
            if state.cleared:
                return True
            if len(state.bearings) > before:
                return True
            if state.consecutive_silent >= 2 and state.estimate is not None:
                return self._clear_at(state.estimate, state.channel) or self._spiral(state)
            return True
        return False


def _max_reach(pos, angle: float) -> float:
    """从 pos 沿 angle 方向到区域边界的距离（解 |pos + t*u| = R）。"""
    px, py = pos
    ux, uy = math.cos(angle), math.sin(angle)
    b = px * ux + py * uy
    c = px * px + py * py - (ARENA_RADIUS - MARGIN_M) ** 2
    disc = b * b - c
    if disc <= 0:
        return 0.0
    return max(0.0, -b + math.sqrt(disc))


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
            strat = EdgeFixStrategy(client, cfg, mode=variant)
        strat.run()
        if variant != "baseline":
            extra = {"clip_dogleg": strat.clip_dogleg, "clip_step": strat.clip_step}
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
    ap.add_argument("--seed-start", type=int, default=83000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,clip,shrink")
    ap.add_argument("--label", default="edge")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        for v in variants:
            r = run_one(seed, directional=args.directional, variant=v, count=args.count)
            rows.append(r)
            print(f"seed={seed} {v:9s} cleared={r['cleared']}/{r['total']} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"virt={r['virtual_s']:.0f} clip={r.get('clip_dogleg', '-')}/{r.get('clip_step', '-')} "
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
                "move_m_mean": round(st.fmean(r["move_m"] for r in rs))}

    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["variant"]] = r

    report = {"per_variant": {v: agg(v) for v in variants}, "paired_vs_baseline": {}}
    base_mean = st.fmean(p["baseline"]["mean_clear_time_s"] for p in pairs.values()
                         if "baseline" in p and p["baseline"]["mean_clear_time_s"])
    for v in variants:
        if v == "baseline":
            continue
        d = [p[v]["mean_clear_time_s"] - p["baseline"]["mean_clear_time_s"]
             for p in pairs.values()
             if "baseline" in p and v in p
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
    out = f"results/b_edge_fix_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
