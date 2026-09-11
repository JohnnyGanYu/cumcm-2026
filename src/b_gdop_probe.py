"""诊断：交会几何质量(GDOP)与"归航扑空"是否相关。

## 思路来源（网上检索，与既有两条思路正交）

我们此前的两条思路都在动"去哪、按什么顺序去"。文献指出还有第三个维度：
**观测的几何质量**。bearing-only 三角定位的不确定度有闭式表达
（Isler et al., *Sensor Placement for Triangulation Based Localization*, IEEE TASE 2010；
Tokekar & Isler, *Gathering Bearing Data for Target Localization*, IEEE RA-L 2016）：

    U(s1, s2, w) = d(s1,w) · d(s2,w) / |sin∠(s1 w s2)|

即 **GDOP**（几何精度稀释）。两个结论直接适用于本题：
  1. 交会角越接近 90° 越好；接近 0° 或 180°（两观测点与源近似共线）时
     |sin| -> 0，不确定度爆炸——即使两条示向度各自误差都只有 ±1°。
  2. 观测点离源越近越好（d 越小）。

对照我们的实现（strategy.py::ChannelState._refresh）：只要 `len(bearings) >= 2`
就认为"可定位"并交给 _home_channel 去清除，**从不检查交会角**。
于是一对几乎共线的示向度会产出一个方向对、距离极不可靠的 estimate，
机器狗照它走过去、发现 20 m 内没有目标（no_target_in_range，白烧 3 s），
再重测——这正是 C_homing_failed 的可能成因。

本脚本只做**测量**，不改策略：对每次"清除尝试"记录当时估计所依据的
交会几何（最小交会角、GDOP、估计误差），看失败尝试与成功尝试在几何上是否可分。
若可分，则"补测时优先补能改善交会角的频道"是有依据的下一步；若不可分，
则该思路对本题无效，应记录并停手。

用法:
  python -m src.b_gdop_probe --episodes 30 --directional 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.geometry import estimate_position
from .b_sim.server import Session, make_server
from .b_sim.strategy import ChannelState, DogStrategy, StrategyConfig

ROBOT_ID = "dog-gdop"


def pair_geometry(bearings, truth: tuple[float, float]) -> dict | None:
    """对最近若干条示向度，算最优交会对的角度与 GDOP（用真值算，仅用于诊断）。"""
    recent = list(bearings)[-4:]
    if len(recent) < 2:
        return None
    tx, ty = truth
    best = None
    for i in range(len(recent)):
        for j in range(i + 1, len(recent)):
            a, b = recent[i], recent[j]
            da = math.hypot(a.x - tx, a.y - ty)
            db = math.hypot(b.x - tx, b.y - ty)
            # 两观测点相对真源的张角
            ang_a = math.atan2(a.y - ty, a.x - tx)
            ang_b = math.atan2(b.y - ty, b.x - tx)
            gap = abs(math.degrees(ang_a - ang_b)) % 360.0
            if gap > 180.0:
                gap = 360.0 - gap
            s = abs(math.sin(math.radians(gap)))
            gdop = (da * db / s) if s > 1e-9 else float("inf")
            if best is None or gdop < best["gdop"]:
                best = {"cross_angle_deg": round(gap, 2),
                        "gdop": gdop,
                        "d_min": round(min(da, db), 1),
                        "baseline_m": round(math.hypot(a.x - b.x, a.y - b.y), 1)}
    return best


class ProbeStrategy(DogStrategy):
    """记录每次 clear 尝试当时的交会几何与估计误差。行为与基线完全一致。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.attempts: list[dict] = []
        self._truth: dict[int, tuple[float, float]] = {}

    def _clear_at(self, point, channel):
        state = self.channels[channel]
        truth = self._truth.get(channel)
        rec: dict = {"channel": channel, "n_bearings": len(state.bearings)}
        if truth is not None and state.estimate is not None:
            rec["est_err_m"] = round(math.hypot(state.estimate[0] - truth[0],
                                                state.estimate[1] - truth[1]), 1)
            rec["aim_err_m"] = round(math.hypot(point[0] - truth[0],
                                                point[1] - truth[1]), 1)
            g = pair_geometry(state.bearings, truth)
            if g:
                rec.update(g)
        ok = super()._clear_at(point, channel)
        rec["ok"] = bool(ok)
        if "cross_angle_deg" in rec:
            self.attempts.append(rec)
        return ok


def run_one(seed: int, *, directional: int, count: int | None) -> list[dict]:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        cfg = StrategyConfig(directional_probe=directional > 0)
        strat = ProbeStrategy(client, cfg)
        strat._truth = {j.channel: (j.x, j.y) for j in case.jammers}
        strat.run()
        for rec in strat.attempts:
            rec["seed"] = seed
        return strat.attempts
    finally:
        httpd.shutdown()
        httpd.server_close()


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]

    def stat(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None
                and math.isfinite(r[key])]
        if not vals:
            return None
        vals.sort()
        return {"n": len(vals), "median": round(st.median(vals), 1),
                "p10": round(vals[int(len(vals) * 0.1)], 1),
                "p90": round(vals[int(len(vals) * 0.9)], 1)}

    out = {"n_attempts": len(rows), "n_success": len(ok), "n_fail": len(bad),
           "fail_rate": round(len(bad) / len(rows), 4) if rows else None}
    for key in ("cross_angle_deg", "gdop", "est_err_m", "aim_err_m", "d_min"):
        out[f"success_{key}"] = stat(ok, key)
        out[f"fail_{key}"] = stat(bad, key)
    # 关键判据：小交会角是否显著集中在失败侧
    for thr in (5.0, 10.0, 20.0):
        so = sum(1 for r in ok if r.get("cross_angle_deg", 999) < thr)
        sb = sum(1 for r in bad if r.get("cross_angle_deg", 999) < thr)
        out[f"share_cross_lt_{int(thr)}deg"] = {
            "success": round(so / len(ok), 4) if ok else None,
            "fail": round(sb / len(bad), 4) if bad else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--directional", type=int, default=10)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=9000)
    ap.add_argument("--label", default="d10")
    a = ap.parse_args()

    rows: list[dict] = []
    for i in range(a.episodes):
        rows.extend(run_one(a.seed_start + i, directional=a.directional,
                            count=a.count))
        print(f"  seed={a.seed_start + i} 累计尝试={len(rows)}", flush=True)
    rep = summarise(rows)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    out = f"results/b_gdop_probe_{a.label}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"report": rep, "attempts": rows}, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
