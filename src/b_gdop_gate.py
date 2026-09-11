"""GDOP 门禁：交会几何太差时先改善几何，再去清除。真实闭环配对 A/B。

## 依据（诊断实测，见 results/b_gdop_probe_d10.json）

30 局 d10、521 次清除尝试：
  失败率 44.15%（230/521 次扑空，每次白烧 3 s + 往返移动）
  交会角 <5° 的尝试成功率 = 0/291 成功侧，而失败侧占 30.4%
  GDOP 中位数 成功 18,666 vs 失败 1,741,832（93 倍）
  估计误差中位数 成功 2.1 m vs 失败 56.5 m（20 m 清除半径下必然扑空）

理论：bearing-only 三角定位不确定度 U = d1·d2/|sin(交会角)|
（Isler et al. IEEE TASE 2010; Tokekar & Isler IEEE RA-L 2016）。
现状 `_refresh` 只要 >=2 条示向度就认可 estimate，从不看交会角。

## 机制

在 `_home_channel` 里，拿到 estimate 后先算交会角；若小于阈值且预算允许，
不直奔估计点，而是先做一次 **改善交会角的补测**——沿"与现有观测基线垂直"的
方向偏移一段再测，使新观测点与源的张角尽量接近 90°。
几何改善后 estimate 自然收敛，再去清除。

注意与 dogleg 的区别：`_dogleg` 用于"只有 1 条示向度、estimate 为 None"的情形，
沿示向度方向前进；这里处理"已有 >=2 条但几何很差"的情形，需要的是**横向张角**
而非更多共线观测。

## 变体

gate8/gate12/gate20 : 交会角阈值 8/12/20 度
gate8_near          : 仅当距估计点较远(>60 m)时才门禁——近距离即使几何差，
                      走过去也便宜，扑空成本低，不值得先绕路补测
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from scipy import stats as sps

from .b_sim.arena import ARENA_RADIUS, Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import ChannelState, DogStrategy, StrategyConfig

ROBOT_ID = "dog-gdop-gate"


def cross_angle_deg(state: ChannelState, at: tuple[float, float]) -> float | None:
    """以当前估计点为顶点，最近两条示向度观测点的张角。越接近 90 度越好。"""
    recent = state.bearings[-4:]
    if len(recent) < 2:
        return None
    best = 0.0
    for i in range(len(recent)):
        for j in range(i + 1, len(recent)):
            a, b = recent[i], recent[j]
            ang_a = math.atan2(a.y - at[1], a.x - at[0])
            ang_b = math.atan2(b.y - at[1], b.x - at[0])
            gap = abs(math.degrees(ang_a - ang_b)) % 360.0
            if gap > 180.0:
                gap = 360.0 - gap
            best = max(best, min(gap, 180.0 - gap))
    return best


class GdopGateStrategy(DogStrategy):
    def __init__(self, *a, min_cross_deg: float = 8.0,
                 min_distance_for_gate_m: float = 0.0,
                 max_geom_fixes: int = 2, **kw):
        super().__init__(*a, **kw)
        self.min_cross_deg = min_cross_deg
        self.min_distance_for_gate_m = min_distance_for_gate_m
        self.max_geom_fixes = max_geom_fixes
        self._fixes: dict[int, int] = {}
        self.gate_triggers = 0
        self.gate_fix_ok = 0
        self.gate_skipped_budget = 0

    def _improve_geometry(self, state: ChannelState) -> bool:
        """沿垂直于"当前位置->估计点"的方向偏移后重测，制造横向张角。"""
        est = state.estimate
        if est is None:
            return False
        used = self._fixes.get(state.channel, 0)
        if used >= self.max_geom_fixes:
            self.gate_skipped_budget += 1
            return False
        self._fixes[state.channel] = used + 1
        ex, ey = est
        d = math.hypot(ex - self.pos[0], ey - self.pos[1])
        if d < 1e-6:
            return False
        # 垂直方向单位向量
        ux, uy = (ex - self.pos[0]) / d, (ey - self.pos[1]) / d
        px, py = -uy, ux
        # 偏移量取估计距离的一半（张角约 26 度），并夹在合理范围
        step = max(120.0, min(0.5 * d, 600.0))
        for sign in (1.0, -1.0):
            cand = (self.pos[0] + px * sign * step, self.pos[1] + py * sign * step)
            if math.hypot(*cand) > ARENA_RADIUS:
                continue
            before = len(state.bearings)
            self._probe_channel_at(cand, state)
            if state.cleared:
                return True
            if len(state.bearings) > before:
                self.gate_fix_ok += 1
                return True
            if self._stop():
                return False
        return False

    def _home_channel(self, state: ChannelState) -> bool:
        """与基线同构，仅在"有 estimate 但交会角过小"时先改善几何。"""
        while not self._stop():
            if state.cleared:
                return True
            budget = self.config.max_homing_measures + (6 if state.revived else 0)
            if len(state.silent_points) >= 8:
                budget += self.config.narrow_sector_extra_measures
            if state.homing_measures >= budget:
                self.events.append(f"ch{state.channel}: measure budget exhausted")
                return self._final_attempt(state)
            if state.clear_attempts >= self.config.max_clear_attempts:
                self.events.append(f"ch{state.channel}: clear budget exhausted")
                return self._ray_blind_clear(state)
            silent_tolerance = self.config.silent_tolerance
            if state.bearings and len(state.silent_points) >= 6:
                silent_tolerance = self.config.narrow_sector_silent_tolerance
            if state.consecutive_silent >= silent_tolerance:
                self.events.append(f"ch{state.channel}: lost signal, defer to ring probe")
                if state.bearings:
                    return self._ray_blind_clear(state)
                return False

            if state.estimate is None:
                if not self._dogleg(state):
                    if state.bearings:
                        return self._ray_blind_clear(state)
                    return False
                continue

            distance = math.hypot(state.estimate[0] - self.pos[0],
                                  state.estimate[1] - self.pos[1])

            # === GDOP 门禁 ===
            # 交会角过小 => 估计的横向误差远超 20 m 清除半径，直奔过去必然扑空。
            # 先横向偏移补一条示向度改善张角，再重新判断。
            if distance >= self.min_distance_for_gate_m:
                ang = cross_angle_deg(state, state.estimate)
                if ang is not None and ang < self.min_cross_deg:
                    self.gate_triggers += 1
                    if self._improve_geometry(state):
                        continue          # 几何已改善，重新评估
            # ==================

            if distance <= self.config.clear_distance_m:
                if self._clear_at(state.estimate, state.channel):
                    return True
                if not self._spiral(state):
                    return False
                return state.cleared
            if not self._step_closer(state, distance):
                return False
        return False


VARIANTS: dict[str, dict] = {
    # 首版 gate8/gate12/gate20 在全距离生效, 8 局实测更差:
    #   prod 1290.1 / gate8 1295.2 / gate20 1345.5
    # 原因: 近距离横向绕路数百米的代价远超省下的 3 s 扑空罚时.
    # 诊断数据 (results/b_gdop_probe_d10.json, 521 次尝试) 指明阈值该怎么选:
    #   交会角 <5 度的失败尝试 est_err 中位数 165 m (清除半径仅 20 m, 必然扑空)
    #   失败尝试 d_min 中位数 371 m (p25=139), 成功尝试 d_min 中位数仅 42 m
    # 即坏几何几乎只在远距离造成损失. 故改为 "严格阈值 + 只在远处生效".
    "gate8": {"min_cross_deg": 8.0},
    "gate20": {"min_cross_deg": 20.0},
    "g5_d200": {"min_cross_deg": 5.0, "min_distance_for_gate_m": 200.0},
    "g5_d400": {"min_cross_deg": 5.0, "min_distance_for_gate_m": 400.0},
    "g3_d200": {"min_cross_deg": 3.0, "min_distance_for_gate_m": 200.0},
    "g8_d300": {"min_cross_deg": 8.0, "min_distance_for_gate_m": 300.0},
}


def run_one(seed: int, *, directional: int, variant: str,
            count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    extra: dict = {}
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        cfg = StrategyConfig(directional_probe=directional > 0)
        if variant == "prod":
            strat: DogStrategy = DogStrategy(client, cfg)
        else:
            strat = GdopGateStrategy(client, cfg, **VARIANTS[variant])
            extra = {}
        strat.run()
        s = arena.stats()
        if variant != "prod":
            extra = {"gate_triggers": strat.gate_triggers,
                     "gate_fix_ok": strat.gate_fix_ok,
                     "gate_skipped_budget": strat.gate_skipped_budget}
        return {"seed": seed, "variant": variant,
                "cleared": s["cleared"], "total": s["total_jammers"],
                "clear_ratio": s["cleared_ratio"],
                "mean_clear_time_s": s["mean_clear_time_s"],
                "move_m": s["move_distance_m"],
                "measures": s["measure_calls"],
                "clear_calls": s["clear_calls"],
                "missed": [j.channel for j in case.jammers if not j.cleared],
                **extra}
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
    return {"n_pairs": n, "baseline_mean": round(bm, 1),
            "d_mean": round(dm, 1),
            "d_pct": round(dm / bm * 100, 2) if bm else None,
            "ci95": [round(dm - half, 1), round(dm + half, 1)],
            "t": round(t_stat, 3),
            "p_ttest": float(f"{p_t:.3g}"), "p_wilcoxon": float(f"{p_w:.3g}"),
            "better": sum(1 for x in d if x < 0),
            "worse": sum(1 for x in d if x > 0),
            "tie": sum(1 for x in d if x == 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--directional", type=int, default=10)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=9000)
    ap.add_argument("--variants", default="prod,gate8")
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
            mt = r["mean_clear_time_s"]
            parts.append(f"{v}={r['cleared']}/{r['total']}"
                         + (f" T={mt:.0f}" if mt else ""))
        print("  ".join(parts), flush=True)

    def agg(rows):
        mts = [r["mean_clear_time_s"] for r in rows if r["mean_clear_time_s"] is not None]
        return {"episodes": len(rows),
                "clear_rate": round(st.fmean(r["clear_ratio"] for r in rows), 5),
                "full_clear": sum(1 for r in rows if r["cleared"] == r["total"]),
                "total_missed": sum(r["total"] - r["cleared"] for r in rows),
                "meanT": round(st.fmean(mts), 1) if mts else None,
                "move_m": round(st.fmean(r["move_m"] for r in rows), 1),
                "measures": round(st.fmean(r["measures"] for r in rows), 1),
                "clear_calls": round(st.fmean(r["clear_calls"] for r in rows), 1),
                "gate_triggers_total": sum(r.get("gate_triggers", 0) for r in rows),
                "gate_fix_ok_total": sum(r.get("gate_fix_ok", 0) for r in rows)}

    base = runs[names[0]]
    report = {"config": {"directional": a.directional, "count": a.count,
                         "seed_start": a.seed_start, "episodes": a.episodes,
                         "baseline": names[0]},
              "per_variant": {v: agg(runs[v]) for v in names},
              "paired_vs_baseline": {
                  v: {k: paired(base, runs[v], k) for k in
                      ("mean_clear_time_s", "move_m", "cleared",
                       "measures", "clear_calls")}
                  for v in names[1:]}}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_gdop_gate_{a.label}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"report": report, "runs": runs}, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
