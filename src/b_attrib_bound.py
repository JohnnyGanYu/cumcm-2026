"""布局可发现性的大样本蒙特卡洛：U_layout 的统计上限与不可发现源画像。

为什么需要它：60 局 × ~13 源 ≈ 780 个源，若漏失率是 1e-3 量级，样本里只有
零星几个漏失，无法刻画「什么样的源必然被漏」。这里直接对源分布抽样几十万个，
纯几何判定其能否被规划站位集合发现，给出高置信度的上限与画像。

只做几何，不跑策略，不涉及时间预算 —— 因此它给的是**布局上限的理论值**，
与 b_attrib_diag.py 的实测值互相印证。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

from .b_sim.arena import (
    ARENA_RADIUS,
    DIRECTIONAL_HALF_ANGLE_DEG,
    RECEPTION_MAX_M,
    RECEPTION_MIN_M,
    norm_deg,
)
from .b_attrib_diag import planned_stations, visible_area_fraction
from .b_sim.strategy import StrategyConfig


def sample_source(rng: random.Random, directional: bool):
    radius = ARENA_RADIUS * math.sqrt(rng.random())
    theta = rng.uniform(0.0, 2.0 * math.pi)
    x, y = radius * math.cos(theta), radius * math.sin(theta)
    r_recv = rng.uniform(RECEPTION_MIN_M, RECEPTION_MAX_M)
    ddeg = rng.uniform(0.0, 360.0) if directional else None
    return x, y, r_recv, ddeg


def visible_from(x, y, r_recv, ddeg, pts) -> bool:
    for px, py in pts:
        if math.hypot(px - x, py - y) > r_recv:
            continue
        if ddeg is None:
            return True
        if px == x and py == y:
            return True
        b = norm_deg(math.degrees(math.atan2(py - y, px - x)))
        if abs((b - ddeg + 180.0) % 360.0 - 180.0) <= DIRECTIONAL_HALF_ANGLE_DEG:
            return True
    return False


LAYOUTS = {
    # 当前实际布局
    "current_q3(origin+1250x12+inner4)": None,     # 由 planned_stations 给出
    "current_q4(origin+1250x12+1750x18+inner4)": None,
}


def build_layouts(cfg: StrategyConfig):
    out = {
        "current_q3(origin+1250x12+inner4)": planned_stations(False, cfg),
        "current_q4(origin+1250x12+1750x18+inner4)": planned_stations(True, cfg),
    }

    def ring(r, n, phase=0.0):
        return [(r * math.cos(2 * math.pi * k / n + phase),
                 r * math.sin(2 * math.pi * k / n + phase)) for k in range(n)]

    # 候选替代布局，用于回答「优化站位的收益上限是多少」
    out["alt_origin+1250x12+1750x24"] = ([(0.0, 0.0)] + ring(1250, 12)
                                        + ring(1750, 24, math.pi / 24))
    out["alt_origin+900x8+1450x16+1780x24"] = ([(0.0, 0.0)] + ring(900, 8)
                                               + ring(1450, 16, math.pi / 16)
                                               + ring(1780, 24, math.pi / 24))
    out["alt_origin+1250x12+1750x18+1795x36"] = ([(0.0, 0.0)] + ring(1250, 12)
                                                 + ring(1750, 18, math.pi / 18)
                                                 + ring(1795, 36))
    out["alt_dense_3ring_48"] = ([(0.0, 0.0)] + ring(700, 6) + ring(1250, 14)
                                 + ring(1650, 14, math.pi / 14) + ring(1795, 13))
    return out


def run(samples: int, directional: bool, seed: int) -> dict:
    rng = random.Random(seed)
    cfg = StrategyConfig()
    layouts = build_layouts(cfg)
    hits = {k: 0 for k in layouts}
    # 收集当前布局下不可发现的源，用于画像
    prof_key = ("current_q4(origin+1250x12+1750x18+inner4)" if directional
                else "current_q3(origin+1250x12+inner4)")
    blind: list[dict] = []
    seen: list[dict] = []
    for _ in range(samples):
        x, y, r_recv, ddeg = sample_source(rng, directional)
        rec = {"r_from_origin": math.hypot(x, y), "r_recv": r_recv,
               "ddeg": ddeg, "x": x, "y": y,
               # 辐射轴与「源->原点」方向的夹角：0=朝中心辐射，180=朝区域外辐射
               "aim_inward_gap_deg": (
                   None if ddeg is None else
                   abs((ddeg - norm_deg(math.degrees(math.atan2(-y, -x))) + 180.0) % 360.0 - 180.0))}
        for name, pts in layouts.items():
            if visible_from(x, y, r_recv, ddeg, pts):
                hits[name] += 1
            elif name == prof_key:
                blind.append(rec)
        if visible_from(x, y, r_recv, ddeg, layouts[prof_key]):
            seen.append(rec)

    def prof(g, name):
        if not g:
            return {"group": name, "n": 0}
        d = [s for s in g if s["ddeg"] is not None]
        out = {
            "group": name, "n": len(g),
            "share_of_samples": round(len(g) / samples, 5),
            "directional_share": round(len(d) / len(g), 3),
            "r_from_origin_mean": round(st.fmean(s["r_from_origin"] for s in g)),
            "r_from_origin_median": round(st.median([s["r_from_origin"] for s in g])),
            "r_from_origin_p10": round(sorted(s["r_from_origin"] for s in g)[len(g) // 10]),
            "r_from_origin_min": round(min(s["r_from_origin"] for s in g)),
            "r_recv_mean": round(st.fmean(s["r_recv"] for s in g)),
            "r_recv_median": round(st.median([s["r_recv"] for s in g])),
            "r_recv_p90": round(sorted(s["r_recv"] for s in g)[int(len(g) * 0.9)]),
        }
        if d:
            out["aim_inward_gap_deg_mean"] = round(st.fmean(s["aim_inward_gap_deg"] for s in d))
            out["aim_outward_share(>90deg)"] = round(
                st.fmean(1.0 if s["aim_inward_gap_deg"] > 90 else 0.0 for s in d), 3)
        # 可见区面积占比（较慢，抽样最多 400 个）
        sub = g if len(g) <= 400 else random.Random(7).sample(g, 400)
        fr = [visible_area_fraction(s["x"], s["y"], s["r_recv"], s["ddeg"]) for s in sub]
        out["visible_area_frac_mean"] = round(st.fmean(fr), 4)
        out["visible_area_frac_median"] = round(st.median(fr), 4)
        out["visible_area_frac_p90"] = round(sorted(fr)[int(len(fr) * 0.9)], 4)
        return out

    # 径向分层的可见率，回答「距中心多远开始危险」
    bands = []
    for lo in range(0, 1800, 200):
        hi = lo + 200
        g = [s for s in blind if lo <= s["r_from_origin"] < hi]
        tot = [s for s in (blind + seen) if lo <= s["r_from_origin"] < hi]
        bands.append({"band_m": f"{lo}-{hi}", "n_sampled": len(tot),
                      "n_blind": len(g),
                      "blind_rate": round(len(g) / len(tot), 5) if tot else None})

    return {
        "samples": samples,
        "directional": directional,
        "layout_visibility": {k: {"visible": v, "rate": round(v / samples, 5),
                                  "stations": len(layouts[k]),
                                  "blind_per_1000": round((samples - v) / samples * 1000, 2)}
                              for k, v in hits.items()},
        "blind_profile": prof(blind, "undiscoverable_current_layout"),
        "visible_profile_reference": prof(
            seen if len(seen) <= 3000 else random.Random(11).sample(seen, 3000), "visible_ref"),
        "blind_rate_by_radius_band": bands,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    out = {"omni": run(args.samples, False, args.seed),
           "directional": run(args.samples, True, args.seed + 1)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = Path("results/b_attrib_bound.json")
    p.write_text(json.dumps({"config": vars(args), **out}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
