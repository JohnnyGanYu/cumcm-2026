"""站位布局的覆盖模型与布局搜索（离线，不动 strategy.py）。

数学模型
--------
目标区域 A = {q : |q| <= 1800}。站位集合 S = {p_1..p_m}。
一个全向源 (q, r) 在站位 p 处可被听到 <=> |p-q| <= r，其中 r ~ U[1000,1500]。
定向源还要求 p 落在以源为顶点、朝向 phi、半张角 90 度的扇区内。

1) 单站有效覆盖面积（保守取 r = RECEPTION_MIN = 1000）
   eff(p) = area( D(p,1000) ∩ A ) / (pi * 1000^2)
   两圆交面积解析式（R=1800 区域圆，rho=1000 接收圆，d=|p|）：
     d <= R-rho          -> pi*rho^2                       （完全内含，eff=1）
     d >= R+rho          -> 0
     否则 lens 公式：
       rho^2*acos((d^2+rho^2-R^2)/(2 d rho))
     + R^2  *acos((d^2+R^2-rho^2)/(2 d R))
     - 0.5*sqrt((-d+rho+R)(d+rho-R)(d-rho+R)(d+rho+R))
   加权平均覆盖效率 = sum_i eff(p_i) / m（每站等权，因为每站都要付一次检测成本）。

2) 全域覆盖保证（最坏覆盖间隙）
   gap(S) = max_{q in A} min_i |q - p_i|。若 gap < 1000 则任何全向源必被某站听到。
   数值上在 A 内取等面积网格求最大值。

3) 覆盖概率（含 r 的真实分布）
   P_hit(S) = P_{q~U(A), r~U[1000,1500]} [ exists i: |p_i - q| <= r ]
   用 arena.generate_case 的同一分布做 Monte Carlo，保证与仿真器一致。

4) 发现代价：站位按访问顺序 p_1..p_m，定义
   E[k] = 期望「第一次听到该源」所需的站位序号（听不到记为 m+1）
   这是扫描阶段耗时的直接代理量：越小越好。

用法：
  python -m src.b_layout_model --mc 4000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

from .b_sim.arena import (ARENA_RADIUS, RECEPTION_MAX_M, RECEPTION_MIN_M,
                          generate_case)
from .b_sim.directional import in_beam

RHO = RECEPTION_MIN_M          # 保守接收半径下限 1000 m


# ---------- 1) 单站有效覆盖面积 ----------

def lens_area(d: float, r1: float, r2: float) -> float:
    """半径 r1、r2、圆心距 d 的两圆交面积。"""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = r1 * r1 * math.acos(max(-1.0, min(1.0, (d * d + r1 * r1 - r2 * r2) / (2 * d * r1))))
    a2 = r2 * r2 * math.acos(max(-1.0, min(1.0, (d * d + r2 * r2 - r1 * r1) / (2 * d * r2))))
    a3 = 0.5 * math.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    return a1 + a2 - a3


def station_efficiency(p: tuple[float, float], rho: float = RHO) -> float:
    """站位 p 的有效覆盖面积占完整接收圆的比例。"""
    d = math.hypot(*p)
    return lens_area(d, rho, ARENA_RADIUS) / (math.pi * rho * rho)


def weighted_efficiency(stations: list[tuple[float, float]], rho: float = RHO) -> float:
    if not stations:
        return 0.0
    return st.fmean(station_efficiency(p, rho) for p in stations)


# ---------- 2) 最坏覆盖间隙 ----------

def worst_gap(stations: list[tuple[float, float]], grid: int = 420) -> tuple[float, tuple[float, float]]:
    worst, arg = 0.0, (0.0, 0.0)
    step = 2 * ARENA_RADIUS / grid
    for i in range(grid):
        x = -ARENA_RADIUS + step * (i + 0.5)
        for j in range(grid):
            y = -ARENA_RADIUS + step * (j + 0.5)
            if x * x + y * y > ARENA_RADIUS * ARENA_RADIUS:
                continue
            d = min((x - px) ** 2 + (y - py) ** 2 for px, py in stations)
            if d > worst:
                worst, arg = d, (x, y)
    return math.sqrt(worst), arg


# ---------- 3) 覆盖概率 / 4) 发现序号 ----------

def sample_sources(n: int, seed: int = 424242, directional: bool = False):
    """按 arena 的分布采样源 (x, y, r_recv, phi|None)。"""
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
    if phi is None:
        return True
    return in_beam(sx, sy, phi, px, py)


def hit_stats(stations: list[tuple[float, float]], sources) -> dict:
    """覆盖概率 + 期望发现序号（按给定站位顺序）。"""
    m = len(stations)
    hit = 0
    ks: list[int] = []
    for sx, sy, r, phi in sources:
        k = None
        for idx, (px, py) in enumerate(stations, start=1):
            if audible(sx, sy, r, phi, px, py):
                k = idx
                break
        if k is None:
            ks.append(m + 1)
        else:
            hit += 1
            ks.append(k)
    return {
        "p_hit": hit / len(sources),
        "mean_discover_index": st.fmean(ks),
        "p_hit_conservative_r1000": st.fmean(
            1.0 if any(audible(sx, sy, RHO, phi, px, py) for px, py in stations) else 0.0
            for sx, sy, _r, phi in sources),
    }


# ---------- 布局构造 ----------

def ring(radius: float, count: int, phase: float = 0.0) -> list[tuple[float, float]]:
    return [(radius * math.cos(2 * math.pi * k / count + phase),
             radius * math.sin(2 * math.pi * k / count + phase))
            for k in range(count)]


ORIGIN = [(0.0, 0.0)]


def travel_len(stations: list[tuple[float, float]]) -> float:
    """从原点出发按给定顺序串行访问的路径长度（不含归航绕路）。"""
    total = 0.0
    cur = (0.0, 0.0)
    for p in stations:
        total += math.hypot(p[0] - cur[0], p[1] - cur[1])
        cur = p
    return total


def angular_order(stations: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted(stations, key=lambda p: math.atan2(p[1], p[0]))


def evaluate(name: str, stations: list[tuple[float, float]], sources_omni, sources_dir) -> dict:
    gap, arg = worst_gap(stations)
    probe = [p for p in stations if math.hypot(*p) > 1e-9]     # 原点是阶段1，不算补扫站位
    row = {
        "layout": name,
        "stations": len(stations),
        "probe_stations": len(probe),
        "worst_gap_m": round(gap, 1),
        "guarantees_omni": bool(gap < RHO),
        "worst_point": [round(arg[0]), round(arg[1])],
        "weighted_efficiency_pct": round(100 * weighted_efficiency(stations), 2),
        "probe_efficiency_pct": round(100 * weighted_efficiency(probe), 2) if probe else None,
        "travel_m": round(travel_len(stations), 0),
    }
    row.update({f"omni_{k}": (round(v, 4) if isinstance(v, float) else v)
                for k, v in hit_stats(stations, sources_omni).items()})
    row.update({f"dir_{k}": (round(v, 4) if isinstance(v, float) else v)
                for k, v in hit_stats(stations, sources_dir).items()})
    return row


def candidate_layouts() -> dict[str, list[tuple[float, float]]]:
    L: dict[str, list[tuple[float, float]]] = {}
    # 现状
    L["BASE_q3 origin+1250x12"] = ORIGIN + angular_order(ring(1250, 12))
    L["BASE_q4 origin+1250x12+1750x18"] = (
        ORIGIN + angular_order(ring(1250, 12))
        + angular_order(ring(1750, 18, math.pi / 18)))
    # 单环稀疏化（几何上 7 站位即可覆盖 r/R=0.5556 需要 n>=6）
    for n in (6, 7, 8, 9, 10, 12):
        for R in (1100, 1150, 1200, 1250, 1300, 1400, 1500):
            L[f"origin+{R}x{n}"] = ORIGIN + angular_order(ring(R, n))
    # 六边形 7 点覆盖（教科书最优 r/R=1/2 -> 这里 rho=1000 有余量）
    L["hex7 origin+1558x6"] = ORIGIN + angular_order(ring(1558.8, 6))
    # 双环（外环内移，替代 1750/1800 边界环）
    for Ro, no in ((1450, 10), (1500, 12), (1550, 12), (1600, 14), (1650, 16),
                   (1700, 16), (1750, 18), (1750, 14), (1500, 16), (1600, 18)):
        L[f"origin+1250x12+{Ro}x{no}"] = (
            ORIGIN + angular_order(ring(1250, 12))
            + angular_order(ring(Ro, no, math.pi / no)))
    # 稀疏内环 + 定向外环（Q4 候选）
    for Ri, ni in ((1150, 8), (1200, 9), (1250, 10)):
        for Ro, no in ((1600, 16), (1650, 18), (1700, 18), (1750, 18)):
            L[f"origin+{Ri}x{ni}+{Ro}x{no}"] = (
                ORIGIN + angular_order(ring(Ri, ni))
                + angular_order(ring(Ro, no, math.pi / no)))
    # 三环 / 螺旋（相位错开）
    L["origin+800x5+1400x11+1750x17"] = (
        ORIGIN + angular_order(ring(800, 5)) + angular_order(ring(1400, 11, 0.3))
        + angular_order(ring(1750, 17, 0.6)))
    spiral = [(1050 + 700 * k / 17, 0) for k in range(18)]
    L["spiral 1050->1750 x18"] = ORIGIN + [
        (r * math.cos(2 * math.pi * k * 0.618), r * math.sin(2 * math.pi * k * 0.618))
        for k, (r, _) in enumerate(spiral)]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=6000)
    ap.add_argument("--out", default="results/b_layout_coverage.json")
    args = ap.parse_args()

    so = sample_sources(args.mc, seed=101, directional=False)
    sd = sample_sources(args.mc, seed=202, directional=True)

    # 校验 MC 采样与 generate_case 一致：原点盲区比例
    blind = tot = 0
    for seed in range(2000, 2200):
        for j in generate_case(seed).jammers:
            tot += 1
            if math.hypot(j.x, j.y) > j.reception_radius_m:
                blind += 1
    mc_blind = st.fmean(0.0 if audible(sx, sy, r, phi, 0.0, 0.0) else 1.0
                        for sx, sy, r, phi in so)

    rows = [evaluate(name, sts, so, sd) for name, sts in candidate_layouts().items()]
    rows.sort(key=lambda r: (not r["guarantees_omni"], -r["weighted_efficiency_pct"]))

    payload = {
        "model": {
            "arena_radius_m": ARENA_RADIUS,
            "rho_conservative_m": RHO,
            "reception_range_m": [RECEPTION_MIN_M, RECEPTION_MAX_M],
            "mc_samples": args.mc,
            "validation_origin_blind_generate_case": round(blind / tot, 4),
            "validation_origin_blind_mc": round(mc_blind, 4),
        },
        "layouts": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    hdr = f"{'layout':34s} {'m':>3s} {'gap':>6s} {'grt':>4s} {'eff%':>6s} {'trav':>6s} {'Pomni':>6s} {'Ek_o':>5s} {'Pdir':>6s} {'Ek_d':>5s}"
    print(f"origin blind: generate_case {blind/tot:.1%}  MC {mc_blind:.1%}\n")
    print(hdr)
    for r in rows:
        print(f"{r['layout']:34s} {r['stations']:3d} {r['worst_gap_m']:6.0f} "
              f"{'Y' if r['guarantees_omni'] else 'n':>4s} {r['weighted_efficiency_pct']:6.1f} "
              f"{r['travel_m']/1000:6.1f} {r['omni_p_hit']:6.3f} {r['omni_mean_discover_index']:5.2f} "
              f"{r['dir_p_hit']:6.3f} {r['dir_mean_discover_index']:5.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
