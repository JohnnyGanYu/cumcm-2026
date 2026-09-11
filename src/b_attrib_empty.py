"""空频道成本与「提前判空」安全边界。

诊断发现（b_attrib_q4_*.json）：Q4 每局约 6.9 个空频道吃掉 47 次检测/频道，
占全局检测的 57%-62%；而有源频道平均只要 14.7-18.5 次。空频道是最大的时间
支出项，且它与漏失无关（空频道没有源可漏）。

于是问题变成：能否更早判定「该频道无源」？风险是把真源误判为空。
本脚本用几何蒙特卡洛给出安全边界：对随机抽样的真源，按机器狗**实际访问顺序**
走规划站位，统计「首次收到信号发生在第几个站位」。
P(first_signal_index > k) 就是「连续 k 个站位沉默后判空」的误杀率。

不跑策略、不读任何单局真值 —— 纯分布层面的几何计算。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .b_sim.arena import (
    ARENA_RADIUS,
    DIRECTIONAL_HALF_ANGLE_DEG,
    RECEPTION_MAX_M,
    RECEPTION_MIN_M,
    norm_deg,
)


def visit_order(directional: bool) -> list[tuple[float, float]]:
    """复刻策略的站位访问顺序：原点 -> 内环(方位角单调) -> 外环(方位角单调)。"""
    pts = [(0.0, 0.0)]
    pts += [(1250.0 * math.cos(2 * math.pi * k / 12),
             1250.0 * math.sin(2 * math.pi * k / 12)) for k in range(12)]
    if directional:
        n = 18
        pts += [(1750.0 * math.cos(2 * math.pi * k / n + math.pi / n),
                 1750.0 * math.sin(2 * math.pi * k / n + math.pi / n)) for k in range(n)]
    return pts


def first_signal_index(x, y, r_recv, ddeg, pts) -> int | None:
    for i, (px, py) in enumerate(pts):
        if math.hypot(px - x, py - y) > r_recv:
            continue
        if ddeg is None:
            return i
        b = norm_deg(math.degrees(math.atan2(py - y, px - x)))
        if abs((b - ddeg + 180.0) % 360.0 - 180.0) <= DIRECTIONAL_HALF_ANGLE_DEG:
            return i
    return None


def run(samples: int, directional: bool, seed: int) -> dict:
    rng = random.Random(seed)
    pts = visit_order(directional)
    n = len(pts)
    idx: list[int | None] = []
    for _ in range(samples):
        radius = ARENA_RADIUS * math.sqrt(rng.random())
        th = rng.uniform(0, 2 * math.pi)
        x, y = radius * math.cos(th), radius * math.sin(th)
        r_recv = rng.uniform(RECEPTION_MIN_M, RECEPTION_MAX_M)
        ddeg = rng.uniform(0, 360) if directional else None
        idx.append(first_signal_index(x, y, r_recv, ddeg, pts))

    never = sum(1 for v in idx if v is None)
    found = [v for v in idx if v is not None]
    # 判空阈值 k：走完前 k 个站位全沉默就判空。
    # 误杀 = 真源的 first_signal_index >= k（本会在更后面的站位被发现）
    table = []
    for k in range(1, n + 1):
        kill = sum(1 for v in found if v >= k)
        table.append({
            "k_silent_stations": k,
            "false_empty_kills": kill,
            "false_empty_rate_among_real_sources": round(kill / samples, 6),
            "stations_saved_per_empty_channel": n - k,
        })
    return {
        "samples": samples,
        "directional": directional,
        "stations_in_order": n,
        "never_visible_rate": round(never / samples, 6),
        "first_signal_index_mean": round(sum(found) / len(found), 3) if found else None,
        "first_signal_index_p50": sorted(found)[len(found) // 2] if found else None,
        "first_signal_index_p90": sorted(found)[int(len(found) * 0.9)] if found else None,
        "first_signal_index_p99": sorted(found)[int(len(found) * 0.99)] if found else None,
        "first_signal_index_max": max(found) if found else None,
        "verdict_table": table,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    out = {"omni": run(args.samples, False, args.seed),
           "directional": run(args.samples, True, args.seed + 1)}
    for grp in ("omni", "directional"):
        r = out[grp]
        print(f"== {grp}: stations={r['stations_in_order']} "
              f"never_visible={r['never_visible_rate']:.5f} "
              f"idx p50={r['first_signal_index_p50']} p90={r['first_signal_index_p90']} "
              f"p99={r['first_signal_index_p99']} max={r['first_signal_index_max']}")
        for row in r["verdict_table"]:
            if row["k_silent_stations"] in (5, 8, 10, 13, 16, 20, 25, 31):
                print(f"   k={row['k_silent_stations']:3d} "
                      f"false_empty={row['false_empty_rate_among_real_sources']:.5f} "
                      f"saved={row['stations_saved_per_empty_channel']}")
    p = Path("results/b_attrib_empty.json")
    p.write_text(json.dumps({"config": vars(args), **out}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
