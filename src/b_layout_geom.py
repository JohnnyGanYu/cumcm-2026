"""站位布局的几何筛选：先算「覆盖保证 + 巡回路程 + 期望发现序号」，再上仿真。

目的
----
评分指标 mean_clear_time_s = 总虚拟时间 / 清除数。清除率已饱和（Q3 100%），
所以唯一可压的是分子。分子的大头是移动时间（速度 5 m/s）。
补扫阶段的移动时间 ≈ 巡回路程 / 5，且「越早听到源，越早归航」——
期望发现序号 E[k] 直接决定源被处理的时刻。

于是布局的三个几何指标（都不需要跑仿真）：

1) 覆盖保证：worst_gap(S) = max_{q in A} min_i |q - p_i|
   题面保证接收半径 >= RECEPTION_MIN_M = 1000 m，故 worst_gap < 1000
   <=> 任意全向源必被某站位听到（充分条件，与半径的随机性无关）。
   这给出「站位数下界」：环半径 R、n 个站位 + 原点时，
     - 弦向间隙 = 2*R*sin(pi/n)/2 = R*sin(pi/n)
     - 径向间隙 = ARENA_RADIUS - R（边界方向）
     - 原点方向间隙 = R/2（原点与环之间）
   三者取 max 必须 < 1000。

2) 巡回路程 tour_m：按 strategy._order_ring 的同一规则（方位角单调、
   从离当前位置最近的站切入）算「原点 -> 环上全部站位」的路径长度。
   注意真实轨迹还有归航绕路，这里只作候选筛选的代理量。

3) 期望发现序号 E[k]：按访问顺序，源第一次被听到的站位序号（听不到记 m+1）。
   用与 arena.generate_case 完全相同的分布做 Monte Carlo（位置 sqrt-均匀、
   半径 U[1000,1500]、定向方向 U[0,360)、半张角 90 度）。

用法：
  python -m src.b_layout_geom --mc 20000
  python -m src.b_layout_geom --mc 20000 --out results/b_layout_geom.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

from .b_sim.arena import (ARENA_RADIUS, RECEPTION_MAX_M, RECEPTION_MIN_M,
                          DIRECTIONAL_HALF_ANGLE_DEG, generate_case)

RHO = RECEPTION_MIN_M          # 1000 m，保守接收半径下限


# ---------------------------------------------------------------- 布局构造

def ring(radius: float, count: float, phase: float = 0.0) -> list[tuple[float, float]]:
    return [(radius * math.cos(2 * math.pi * k / count + phase),
             radius * math.sin(2 * math.pi * k / count + phase))
            for k in range(int(count))]


def order_ring(pts: list[tuple[float, float]],
               cur: tuple[float, float]) -> list[tuple[float, float]]:
    """复刻 strategy._order_ring：方位角单调 + 从最近站切入 + 选较短转向。"""
    if len(pts) <= 2:
        return pts[:]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    by_angle = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    start = min(range(len(by_angle)),
                key=lambda i: math.hypot(by_angle[i][0] - cur[0], by_angle[i][1] - cur[1]))
    ccw = by_angle[start:] + by_angle[:start]
    cw = [ccw[0]] + list(reversed(ccw[1:]))

    def tour(seq):
        t = math.hypot(seq[0][0] - cur[0], seq[0][1] - cur[1])
        for a, b in zip(seq, seq[1:]):
            t += math.hypot(b[0] - a[0], b[1] - a[1])
        return t

    return ccw if tour(ccw) <= tour(cw) else cw


def visit_sequence(layers: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """原点 + 各层按 _order_ring 顺序串起来的完整访问序列。"""
    seq: list[tuple[float, float]] = [(0.0, 0.0)]
    cur = (0.0, 0.0)
    for radius, count, phase in layers:
        pts = order_ring(ring(radius, count, phase), cur)
        seq.extend(pts)
        cur = pts[-1]
    return seq


def tour_length(seq: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(seq, seq[1:]))


# ------------------------------------------------------- 1) 覆盖间隙（解析+数值）

def bisector_gap(radius: float, count: int) -> float:
    """origin + ring(radius, count) 的最坏覆盖间隙（精确）。

    由对称性，最坏点必落在相邻两站位的角平分线射线上（否则可绕轴旋转到更远）。
    设平分线上距原点 r 处的点，其到原点距离 = r，到最近两个环站位距离
        sqrt(r^2 + R^2 - 2 r R cos(pi/n))
    故 gap = max_{0 <= r <= ARENA_RADIUS} min(r, sqrt(r^2+R^2-2 r R cos(pi/n)))。
    在 [0, ARENA_RADIUS] 上做细网格 1D 扫描（单变量、可任意精确）。
    """
    c = math.cos(math.pi / count)
    best = 0.0
    steps = 20000
    for i in range(steps + 1):
        r = ARENA_RADIUS * i / steps
        to_ring = math.sqrt(max(0.0, r * r + radius * radius - 2 * r * radius * c))
        best = max(best, min(r, to_ring))
    return best


def analytic_gaps(radius: float, count: int) -> dict:
    """单环 + 原点的间隙分解（诊断用）+ 精确最坏间隙。"""
    return {
        "chord_gap_m": round(radius * math.sin(math.pi / count), 1),
        "radial_gap_m": round(ARENA_RADIUS - radius, 1),
        "origin_gap_m": round(radius / 2.0, 1),
        "bisector_gap_m": round(bisector_gap(radius, count), 1),
    }


def worst_gap(stations: list[tuple[float, float]], grid: int = 500) -> tuple[float, tuple[float, float]]:
    """数值最坏覆盖间隙：区域内任一点到最近站位的最大距离。"""
    worst, arg = 0.0, (0.0, 0.0)
    step = 2 * ARENA_RADIUS / grid
    R2 = ARENA_RADIUS * ARENA_RADIUS
    for i in range(grid):
        x = -ARENA_RADIUS + step * (i + 0.5)
        for j in range(grid):
            y = -ARENA_RADIUS + step * (j + 0.5)
            if x * x + y * y > R2:
                continue
            d = min((x - px) ** 2 + (y - py) ** 2 for px, py in stations)
            if d > worst:
                worst, arg = d, (x, y)
    return math.sqrt(worst), arg


def min_stations_for_coverage(radius: float) -> int:
    """给定环半径，满足最坏覆盖间隙 < 1000 m 的最小站位数。

    加密环站位只能压 bisector_gap；`radial_gap`（边界到环）与
    `origin_gap`（原点到环）与 n 无关，任一 >= RHO 则该半径无解。
    """
    if ARENA_RADIUS - radius >= RHO or radius / 2.0 >= RHO:
        return 10 ** 6                       # 该半径无论多少站位都不达标
    for n in range(3, 200):
        if bisector_gap(radius, n) < RHO:
            return n
    return 10 ** 6


# ------------------------------------------------- 3) 覆盖概率 / 期望发现序号

def in_beam(sx: float, sy: float, phi_deg: float, px: float, py: float) -> bool:
    """观测点是否落在源的 180 度辐射扇区内（与 b_sim.directional 同义）。"""
    ang = math.degrees(math.atan2(py - sy, px - sx))
    diff = (ang - phi_deg + 180.0) % 360.0 - 180.0
    return abs(diff) <= DIRECTIONAL_HALF_ANGLE_DEG


def sample_sources(n: int, seed: int, directional: bool):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        rad = ARENA_RADIUS * math.sqrt(rng.random())
        th = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(RECEPTION_MIN_M, RECEPTION_MAX_M)
        phi = rng.uniform(0, 360.0) if directional else None
        out.append((rad * math.cos(th), rad * math.sin(th), r, phi))
    return out


def audible(sx, sy, r, phi, px, py) -> bool:
    if math.hypot(px - sx, py - sy) > r:
        return False
    return True if phi is None else in_beam(sx, sy, phi, px, py)


def discovery_stats(seq: list[tuple[float, float]], sources) -> dict:
    """按访问顺序的覆盖概率与期望发现序号，并把序号换算成累计路程代理。"""
    m = len(seq)
    cum = [0.0]
    for a, b in zip(seq, seq[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    hit = 0
    ks: list[int] = []
    dists: list[float] = []
    for sx, sy, r, phi in sources:
        found = None
        for idx, (px, py) in enumerate(seq):
            if audible(sx, sy, r, phi, px, py):
                found = idx
                break
        if found is None:
            ks.append(m + 1)
            dists.append(cum[-1])
        else:
            hit += 1
            ks.append(found + 1)
            dists.append(cum[found])
    return {
        "p_hit": round(hit / len(sources), 4),
        "mean_discover_index": round(st.fmean(ks), 3),
        "mean_discover_travel_m": round(st.fmean(dists), 0),
        "p_hit_conservative_r1000": round(st.fmean(
            1.0 if any(audible(sx, sy, RHO, phi, px, py) for px, py in seq) else 0.0
            for sx, sy, _r, phi in sources), 4),
    }


# --------------------------------------------------------------- 候选布局

def candidates() -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    """name -> {q3: layers, q4: layers}；layers = [(radius, count, phase), ...]"""
    L: dict[str, dict] = {}

    # 已调优基线（b_budget_verdict.md 的结论）
    L["TUNED_q3_1250x9"] = {"q3": [(1250.0, 9, 0.0)],
                            "q4": [(1250.0, 8, 0.0), (1750.0, 30, math.pi / 30)]}
    # 默认基线（仅供参照，不作对比基准）
    L["DEFAULT_1250x12"] = {"q3": [(1250.0, 12, 0.0)],
                            "q4": [(1250.0, 12, 0.0), (1750.0, 18, math.pi / 18)]}

    # 单环扫描：半径 x 站位数。只保留满足覆盖保证（gap<1000）的组合，
    # 外加「差一点点」的（n = n_min - 1）作为对照，看丢覆盖是否换来时间。
    for R in (1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1500, 1600, 1700):
        nmin = min_stations_for_coverage(float(R))
        if nmin > 100:
            continue
        for n in (nmin - 1, nmin, nmin + 1, nmin + 2, nmin + 3):
            if n < 3:
                continue
            key = f"r{R}x{n}"
            if key in L:
                continue
            L[key] = {"q3": [(float(R), n, 0.0)],
                      "q4": [(float(R), n, 0.0), (1750.0, 30, math.pi / 30)]}
    return L


def evaluate(name: str, layers: list[tuple[float, float, float]],
             sources_omni, sources_dir, grid: int) -> dict:
    seq = visit_sequence(layers)
    gap, arg = worst_gap(seq, grid=grid)
    row = {
        "layout": name,
        "layers": [[r, n, round(p, 4)] for r, n, p in layers],
        "stations": len(seq),
        "probe_stations": len(seq) - 1,
        "worst_gap_m": round(gap, 1),
        "guarantees_omni": bool(gap < RHO),
        "worst_point": [round(arg[0]), round(arg[1])],
        "tour_m": round(tour_length(seq), 0),
        "tour_time_s": round(tour_length(seq) / 5.0, 0),
    }
    if len(layers) == 1:
        row.update(analytic_gaps(layers[0][0], int(layers[0][1])))
    row.update({f"omni_{k}": v for k, v in discovery_stats(seq, sources_omni).items()})
    row.update({f"dir_{k}": v for k, v in discovery_stats(seq, sources_dir).items()})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=20000)
    ap.add_argument("--grid", type=int, default=500)
    ap.add_argument("--out", default="results/b_layout_geom.json")
    args = ap.parse_args()

    so = sample_sources(args.mc, seed=1301, directional=False)
    sd = sample_sources(args.mc, seed=1302, directional=True)

    # 自校验：MC 采样分布必须与 generate_case 一致（用「原点听不到的比例」对比）
    blind = tot = 0
    for seed in range(3000, 3300):
        for j in generate_case(seed).jammers:
            tot += 1
            if math.hypot(j.x, j.y) > j.reception_radius_m:
                blind += 1
    mc_blind = st.fmean(0.0 if audible(sx, sy, r, phi, 0.0, 0.0) else 1.0
                        for sx, sy, r, phi in so)
    print(f"self-check origin-blind: generate_case {blind/tot:.3%}  MC {mc_blind:.3%}")

    # 覆盖下界表
    print("\n== 单环覆盖下界（worst_gap < 1000 m 的最小站位数）==")
    print(f"{'R':>6s} {'n_min':>6s} {'chord':>7s} {'radial':>7s} {'origin':>7s} {'bisect':>7s}")
    bound_rows = []
    for R in (1000, 1100, 1150, 1200, 1250, 1300, 1400, 1500, 1600, 1700, 1800):
        n = min_stations_for_coverage(float(R))
        g = analytic_gaps(float(R), max(3, min(n, 60)))
        ok = "-" if n > 100 else str(n)
        print(f"{R:6d} {ok:>6s} {g['chord_gap_m']:7.0f} {g['radial_gap_m']:7.0f} "
              f"{g['origin_gap_m']:7.0f} {g['bisector_gap_m']:7.0f}")
        bound_rows.append({"radius_m": R, "n_min": None if n > 100 else n, **g})

    q3_rows, q4_rows = [], []
    for name, spec in candidates().items():
        q3_rows.append(evaluate(name, spec["q3"], so, sd, args.grid))
        q4_rows.append(evaluate(name, spec["q4"], so, sd, args.grid))

    def show(title, rows, pfx):
        print(f"\n== {title} ==")
        print(f"{'layout':22s} {'m':>3s} {'gap':>6s} {'ok':>3s} {'tour_km':>8s} "
              f"{'Phit':>6s} {'E[k]':>6s} {'E[trav]':>8s}")
        for r in sorted(rows, key=lambda r: r[f"{pfx}_mean_discover_travel_m"]):
            print(f"{r['layout']:22s} {r['stations']:3d} {r['worst_gap_m']:6.0f} "
                  f"{'Y' if r['guarantees_omni'] else 'n':>3s} {r['tour_m']/1000:8.2f} "
                  f"{r[f'{pfx}_p_hit']:6.3f} {r[f'{pfx}_mean_discover_index']:6.2f} "
                  f"{r[f'{pfx}_mean_discover_travel_m']:8.0f}")

    show("Q3 全向（按期望发现路程排序）", q3_rows, "omni")
    show("Q4 定向（按期望发现路程排序）", q4_rows, "dir")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": {
            "arena_radius_m": ARENA_RADIUS, "rho_conservative_m": RHO,
            "reception_range_m": [RECEPTION_MIN_M, RECEPTION_MAX_M],
            "mc_samples": args.mc, "grid": args.grid,
            "selfcheck_origin_blind_generate_case": round(blind / tot, 4),
            "selfcheck_origin_blind_mc": round(mc_blind, 4),
        },
        "coverage_bounds": bound_rows,
        "q3_layouts": q3_rows,
        "q4_layouts": q4_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
