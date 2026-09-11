"""L31 未覆盖源的失效模式画像（可见区大小 x 位置）。

关键区分：
- "难观测"（intrinsically hard）：可见区本身极小（<0.5% 区域面积），
  任何有限站位集都容易漏。
- "站位不巧"（unlucky geometry）：可见区很大（>5%），但 31 个离散站位
  恰好全落在扇区外——这是纯粹的布局问题，加站位就能修。

两者的比例决定了"提升布局"值不值得做。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st

from .b_observability import (
    ARENA_AREA_M2,
    covered_by,
    exact_visible_area_m2,
    pctl,
    ring,
    source_pool,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=2000)
    ap.add_argument("--seed-start", type=int, default=700000)
    ap.add_argument("--label", default="main")
    args = ap.parse_args()

    origin = [(0.0, 0.0)]
    base31 = origin + ring(1250.0, 12) + ring(1750.0, 18, math.pi / 18)
    scen = [("Q4_d5", 5, None), ("Q4_d10", 10, None), ("Q4_d16", 16, 16)]
    out = {"cases": args.cases, "scenarios": {}}
    for name, dcount, count in scen:
        pool = source_pool(args.cases, dcount, count, args.seed_start)
        unc = []
        for _, j in pool:
            if covered_by(j, base31):
                continue
            a = exact_visible_area_m2(j) / ARENA_AREA_M2 * 100.0
            unc.append({"rho": round(math.hypot(j.x, j.y), 1),
                        "area_pct": round(a, 5),
                        "r": round(j.reception_radius_m, 1)})
        n = len(pool)
        hard = [u for u in unc if u["area_pct"] < 0.5]
        lucky = [u for u in unc if u["area_pct"] >= 5.0]
        mid = [u for u in unc if 0.5 <= u["area_pct"] < 5.0]
        out["scenarios"][name] = {
            "sources": n, "uncovered": len(unc),
            "uncovered_pct": round(100.0 * len(unc) / n, 4),
            "hard_area_lt_0p5pct": {
                "count": len(hard),
                "pct_of_all_sources": round(100.0 * len(hard) / n, 4),
                "area_p50": round(pctl([u["area_pct"] for u in hard], .5), 4) if hard else None,
                "area_max": round(max(u["area_pct"] for u in hard), 4) if hard else None,
                "rho_p50": round(pctl([u["rho"] for u in hard], .5), 1) if hard else None},
            "mid_0p5_to_5pct": {"count": len(mid),
                                "pct_of_all_sources": round(100.0 * len(mid) / n, 4)},
            "unlucky_area_ge_5pct": {
                "count": len(lucky),
                "pct_of_all_sources": round(100.0 * len(lucky) / n, 4),
                "area_p50": round(pctl([u["area_pct"] for u in lucky], .5), 3) if lucky else None,
                "area_max": round(max(u["area_pct"] for u in lucky), 3) if lucky else None,
                "rho_p50": round(pctl([u["rho"] for u in lucky], .5), 1) if lucky else None,
                "rho_min": round(min(u["rho"] for u in lucky), 1) if lucky else None},
            "uncovered_area_pct_dist": {
                "p10": round(pctl([u["area_pct"] for u in unc], .1), 4),
                "p50": round(pctl([u["area_pct"] for u in unc], .5), 4),
                "p90": round(pctl([u["area_pct"] for u in unc], .9), 4),
                "max": round(max(u["area_pct"] for u in unc), 3),
                "mean": round(st.fmean(u["area_pct"] for u in unc), 3)},
            "examples_smallest": sorted(unc, key=lambda u: u["area_pct"])[:10],
            "examples_largest": sorted(unc, key=lambda u: -u["area_pct"])[:10],
        }
        s = out["scenarios"][name]
        print(f"[modes] {name}: unc={s['uncovered']} ({s['uncovered_pct']}%) "
              f"hard<0.5%={s['hard_area_lt_0p5pct']['count']} "
              f"mid={s['mid_0p5_to_5pct']['count']} "
              f"unlucky>=5%={s['unlucky_area_ge_5pct']['count']}", flush=True)
    path = os.path.join("results", f"b_observability_modes_{args.label}.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump(out, h, ensure_ascii=False, indent=2)
    print("wrote " + path)


if __name__ == "__main__":
    main()
