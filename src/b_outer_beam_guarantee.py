"""外环半径的定向可见性保证：1870 vs 1800 到底差在哪。

## 问题

PR #8 把 Q4 固定布局的外环设为 1870 m x 12，报告只给了参数值没给论证；
我据「最坏覆盖间隙 / 巡回长度 / 60 局实测」把它改成 1800 m，并在 commit
里写了「原 PR 取 1870 无论证依据」。协作者随后指出：1800 m 在边界定向源上
会漏清。本模块检验的就是这条几何机制，不依赖任何仿真随机性。

## 判据（保守下界，不含误差场）

源 s（|s| <= 1800），朝向 phi，半角 90 度，有效接收半径 R_s ~ U[1000,1500]。
站位 p 能收到该源当且仅当

    (p - s) . u(phi) >= 0        （落在半平面内）
  且 |p - s| <= R_s

R_s 未知，取保守下限 1000 m。于是"该源在本布局下必定可见"等价于
存在站位 p 同时满足半平面条件与 |p - s| <= 1000。

最坏情形是边界源径向朝外：此时半平面只含更外侧的点，需要
    R cos(pi/n) >= |s|
对 |s| -> 1800、n = 12 即 R >= 1800 / cos(15 deg) = 1863.6 m。
**1870 正好是这条约束的最小可行值加余量，不是拍出来的数。**

用法:
    python3 -m src.b_outer_beam_guarantee --out results/b_outer_beam_guarantee.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .b_sim.arena import ARENA_RADIUS, RECEPTION_MIN_M
from .b_sim.q4_fast import Q4FastStrategy, ring_points


def layout(outer_radius: float) -> np.ndarray:
    """生产布局：原点 + 内环 1000x7 + 外环 R x12（旋转 15 度）。"""
    points = [(0.0, 0.0)]
    points += ring_points(Q4FastStrategy.inner_radius, Q4FastStrategy.inner_count)
    points += ring_points(outer_radius, Q4FastStrategy.outer_count,
                          Q4FastStrategy.outer_phase_deg)
    return np.asarray(points, dtype=float)


def source_grid(step: float) -> np.ndarray:
    axis = np.arange(-ARENA_RADIUS, ARENA_RADIUS + step / 2, step)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    return cells[np.linalg.norm(cells, axis=1) <= ARENA_RADIUS + 1e-9]


def visibility(stations: np.ndarray, sources: np.ndarray, headings: np.ndarray,
               reception: float) -> np.ndarray:
    """返回 (源, 朝向) 是否至少有一个站位保证可见的布尔矩阵。"""
    # d[i, k] = station_k - source_i
    dx = stations[None, :, 0] - sources[:, None, 0]
    dy = stations[None, :, 1] - sources[:, None, 1]
    within = (dx * dx + dy * dy) <= reception ** 2
    ux = np.cos(np.radians(headings))
    uy = np.sin(np.radians(headings))
    # in_beam[i, k, h] = (p_k - s_i) . u_h >= 0
    projection = dx[:, :, None] * ux[None, None, :] + dy[:, :, None] * uy[None, None, :]
    ok = within[:, :, None] & (projection >= 0.0)
    return ok.any(axis=1)


def analyse(outer_radius: float, sources: np.ndarray, headings: np.ndarray,
            reception: float) -> dict:
    stations = layout(outer_radius)
    seen = visibility(stations, sources, headings, reception)
    radii = np.linalg.norm(sources, axis=1)
    blind = ~seen
    blind_pairs = int(blind.sum())
    total = blind.size
    per_source_blind = blind.any(axis=1)
    rows = []
    for lo, hi in ((0, 1000), (1000, 1400), (1400, 1600), (1600, 1700),
                   (1700, 1750), (1750, 1800)):
        mask = (radii > lo) & (radii <= hi) if lo else (radii <= hi)
        if not mask.any():
            continue
        rows.append(dict(band=f"({lo},{hi}]", cells=int(mask.sum()),
                         blind_pairs=int(blind[mask].sum()),
                         blind_pair_share=float(blind[mask].mean()),
                         sources_with_blind_heading=int(per_source_blind[mask].sum())))
    tour = 2 * outer_radius * Q4FastStrategy.outer_count * math.sin(
        math.pi / Q4FastStrategy.outer_count)
    return dict(outer_radius_m=outer_radius, stations=len(stations),
                blind_pairs=blind_pairs, pairs=total,
                blind_pair_share=float(blind_pairs / total),
                sources_with_some_blind_heading=int(per_source_blind.sum()),
                outer_ring_tour_m=tour, bands=rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[1800.0, 1830.0, 1863.6, 1870.0, 1900.0])
    parser.add_argument("--grid-step", type=float, default=25.0)
    parser.add_argument("--heading-step", type=float, default=5.0)
    parser.add_argument("--reception", type=float, default=RECEPTION_MIN_M)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sources = source_grid(args.grid_step)
    headings = np.arange(0.0, 360.0, args.heading_step)
    print(f"源格点 {len(sources)}, 朝向 {len(headings)}, "
          f"保守接收半径 {args.reception:.0f} m")
    required = ARENA_RADIUS / math.cos(math.pi / Q4FastStrategy.outer_count)
    print(f"径向朝外边界源的解析下界 R >= {ARENA_RADIUS:.0f}/cos(15 deg) = {required:.1f} m")

    results = [analyse(r, sources, headings, args.reception) for r in args.radii]
    print(f"\n{'外环 R':>9} {'必盲(源,朝向)对':>16} {'占比':>9} {'含盲朝向的源':>13} {'外环巡回 m':>11}")
    for row in results:
        print(f"{row['outer_radius_m']:9.1f} {row['blind_pairs']:16d} "
              f"{row['blind_pair_share']*100:8.4f}% {row['sources_with_some_blind_heading']:13d} "
              f"{row['outer_ring_tour_m']:11.0f}")
    for row in results:
        print(f"\nR = {row['outer_radius_m']:.1f} 分环带:")
        for band in row["bands"]:
            print(f"  {band['band']:>13} 格点 {band['cells']:5d} "
                  f"必盲对 {band['blind_pairs']:6d} ({band['blind_pair_share']*100:7.3f}%) "
                  f"含盲朝向的源 {band['sources_with_blind_heading']:5d}")

    payload = dict(arena_radius_m=ARENA_RADIUS, reception_lower_bound_m=args.reception,
                   grid_step_m=args.grid_step, heading_step_deg=args.heading_step,
                   analytic_min_radius_m=required, results=results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
