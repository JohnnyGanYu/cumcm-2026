"""外环半径的边界压力测试：只改 outer_radius，比较清除率与耗时。

## 为什么要独立重做

results/b_outer_radius_1800.json 的 240 局是**随机场景**配对，两档均满清，
故当时判定 1800 优。协作者在 PR #8 讨论中指出：随机场景里贴边朝外的源太稀疏，
均值测不出漏清风险，他的 4 局压力测试中 1800 每局漏 1 个。

b_outer_beam_guarantee.py 已从几何上确认这条机制：径向朝外的边界源要求
R >= 1800/cos(pi/12) = 1863.5 m，1800 在 (1750,1800] 环带留 2.37% 必盲对。
本模块用**可复现的筛选规则**独立造压力样本，不引用对方的本地 JSON。

## 压力样本的筛选规则（不看策略，只看案例几何）

对候选种子生成案例，保留满足以下条件的：至少 `--min-hard` 个定向源同时满足
  1. |s| >= boundary_min（贴近边界）
  2. 辐射轴与径向外法线夹角 <= aim_gap（朝区域外）
  3. 在 1800 m 布局下"保证可见站位数"为 0，而在 1870 m 布局下 > 0
条件 3 直接把两档的差异点挑出来，是这次分歧的判定样本。

用法:
    python3 -m src.b_outer_radius_boundary --scan 4000 --episodes 60 \\
        --radii 1800 1870 --out results/b_outer_radius_boundary.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import statistics as st
import time

from .b_eval.dev_measure import InProcessClient
from .b_sim.arena import ARENA_RADIUS, RECEPTION_MIN_M, Arena, generate_case
from .b_sim.directional import in_beam
from .b_sim.q4_fast import Q4FastStrategy, ring_points
from .b_sim.strategy import StrategyConfig

ROOT = Path(__file__).resolve().parent.parent


def layout(outer_radius: float) -> list[tuple[float, float]]:
    points = [(0.0, 0.0)]
    points += ring_points(Q4FastStrategy.inner_radius, Q4FastStrategy.inner_count)
    points += ring_points(outer_radius, Q4FastStrategy.outer_count,
                          Q4FastStrategy.outer_phase_deg)
    return points


def guaranteed_stations(jammer, stations) -> int:
    """保守接收半径 1000 m 下，能保证收到该源的站位数。"""
    count = 0
    for px, py in stations:
        if math.dist((px, py), (jammer.x, jammer.y)) > RECEPTION_MIN_M:
            continue
        if jammer.directional_deg is None or in_beam(
            jammer.x, jammer.y, jammer.directional_deg, px, py
        ):
            count += 1
    return count


def hard_sources(case, radii, boundary_min: float, aim_gap: float) -> list[dict]:
    """返回在小半径下必盲、在大半径下可见的定向源。"""
    small, large = min(radii), max(radii)
    layouts = {r: layout(r) for r in (small, large)}
    rows = []
    for jammer in case.jammers:
        if jammer.directional_deg is None:
            continue
        radius = math.hypot(jammer.x, jammer.y)
        if radius < boundary_min:
            continue
        outward = math.degrees(math.atan2(jammer.y, jammer.x))
        gap = abs((jammer.directional_deg - outward + 180.0) % 360.0 - 180.0)
        if gap > aim_gap:
            continue
        n_small = guaranteed_stations(jammer, layouts[small])
        n_large = guaranteed_stations(jammer, layouts[large])
        if n_small == 0 and n_large > 0:
            rows.append(dict(channel=jammer.channel, radius_m=radius,
                             aim_outward_gap_deg=gap,
                             stations_small=n_small, stations_large=n_large))
    return rows


def scan(scan_seeds: range, radii, boundary_min: float, aim_gap: float,
         directional: int, count: int, min_hard: int) -> list[dict]:
    picked = []
    for seed in scan_seeds:
        case = generate_case(seed, count=count or None, directional_count=directional)
        rows = hard_sources(case, radii, boundary_min, aim_gap)
        if len(rows) >= min_hard:
            picked.append(dict(seed=seed, hard=rows))
    return picked


def run_episode(seed: int, directional: int, count: int, outer_radius: float) -> dict:
    class Variant(Q4FastStrategy):
        pass
    Variant.outer_radius = outer_radius
    arena = Arena(generate_case(seed, count=count or None, directional_count=directional))
    dog = Variant(InProcessClient(arena), StrategyConfig(directional_probe=True))
    started = time.monotonic()
    summary = dog.run()
    stats = arena.stats()
    return dict(seed=seed, outer_radius_m=outer_radius, wall_s=time.monotonic() - started,
                cleared=stats['cleared'], total=stats['total_jammers'],
                mean_clear_time_s=stats['mean_clear_time_s'],
                total_virtual_time_s=stats['total_virtual_time_s'],
                move_distance_m=stats['move_distance_m'],
                measure_calls=stats['measure_calls'],
                missed_channels=[j.channel for j in arena.case.jammers if not j.cleared],
                layout=summary.get('search_layout'))


def _task(args):
    return run_episode(*args)


def paired_stats(values_a: list[float], values_b: list[float]) -> dict:
    diffs = [b - a for a, b in zip(values_a, values_b)]
    n = len(diffs)
    if n < 2:
        return dict(n=n, mean_diff=diffs[0] if diffs else None)
    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    return dict(n=n, mean_diff=mean, sd=sd,
                t=mean / se if se else None,
                ci95=[mean - 1.96 * se, mean + 1.96 * se],
                wins=sum(d < -1e-9 for d in diffs), ties=sum(abs(d) <= 1e-9 for d in diffs),
                losses=sum(d > 1e-9 for d in diffs))


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--radii', type=float, nargs=2, default=[1800.0, 1870.0])
    ap.add_argument('--scan', type=int, default=4000, help='扫描的候选种子数')
    ap.add_argument('--seed-start', type=int, default=880000)
    ap.add_argument('--episodes', type=int, default=60, help='最多跑多少个压力局')
    ap.add_argument('--directional', type=int, default=16)
    ap.add_argument('--count', type=int, default=16)
    ap.add_argument('--boundary-min', type=float, default=1700.0)
    ap.add_argument('--aim-gap', type=float, default=25.0)
    ap.add_argument('--min-hard', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    if args.out and (ROOT / args.out).exists():
        ap.error(f'Refusing to overwrite existing evidence: {args.out}')

    seeds = range(args.seed_start, args.seed_start + args.scan)
    picked = scan(seeds, args.radii, args.boundary_min, args.aim_gap,
                  args.directional, args.count, args.min_hard)
    print(f"扫描 {args.scan} 个种子，命中压力局 {len(picked)} 个"
          f"（贴边 >= {args.boundary_min:.0f} m、朝外夹角 <= {args.aim_gap:.0f} 度、"
          f"仅大半径可见的定向源 >= {args.min_hard} 个）")
    picked = picked[:args.episodes]
    if not picked:
        raise SystemExit('未命中压力局，放宽 --boundary-min / --aim-gap 再试')
    hard_total = sum(len(p['hard']) for p in picked)
    print(f"实跑 {len(picked)} 局，其中差异源合计 {hard_total} 个\n")

    small, large = min(args.radii), max(args.radii)
    tasks = [(p['seed'], args.directional, args.count, r)
             for p in picked for r in (small, large)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, t): t for t in tasks}
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(f"seed={row['seed']} R={row['outer_radius_m']:.0f} "
                  f"cleared={row['cleared']}/{row['total']} "
                  f"T={row['mean_clear_time_s']:.2f}s miss={row['missed_channels']}",
                  flush=True)

    by_radius = {r: {rec['seed']: rec for rec in records if rec['outer_radius_m'] == r}
                 for r in (small, large)}
    common = sorted(set(by_radius[small]) & set(by_radius[large]))
    summary = {}
    for r in (small, large):
        rows = [by_radius[r][s] for s in common]
        cleared = sum(x['cleared'] for x in rows)
        total = sum(x['total'] for x in rows)
        summary[f'R{r:.0f}'] = dict(
            episodes=len(rows), cleared=cleared, total=total,
            clear_rate=cleared / total if total else None,
            full_clear_episodes=sum(x['cleared'] == x['total'] for x in rows),
            mean_T=st.mean(x['mean_clear_time_s'] for x in rows),
            pooled_T=sum(x['total_virtual_time_s'] for x in rows) / cleared if cleared else None,
            mean_move_m=st.mean(x['move_distance_m'] for x in rows))

    t_small = [by_radius[small][s]['mean_clear_time_s'] for s in common]
    t_large = [by_radius[large][s]['mean_clear_time_s'] for s in common]
    c_small = [by_radius[small][s]['cleared'] for s in common]
    c_large = [by_radius[large][s]['cleared'] for s in common]
    paired = dict(time_large_minus_small=paired_stats(t_small, t_large),
                  cleared_large_minus_small=paired_stats(
                      [float(v) for v in c_small], [float(v) for v in c_large]))

    print(f"\n{'':>10} {'局数':>5} {'清除':>12} {'满清局':>7} {'平均 T':>10} {'池化 T':>10}")
    for key, row in summary.items():
        print(f"{key:>10} {row['episodes']:5d} {row['cleared']:5d}/{row['total']:<6d} "
              f"{row['full_clear_episodes']:7d} {row['mean_T']:10.2f} {row['pooled_T']:10.2f}")
    tp = paired['time_large_minus_small']
    cp = paired['cleared_large_minus_small']
    print(f"\n配对差（大半径 - 小半径）:")
    print(f"  耗时 d={tp['mean_diff']:+.2f}s CI95[{tp['ci95'][0]:+.2f},{tp['ci95'][1]:+.2f}] "
          f"胜/平/负 {tp['wins']}/{tp['ties']}/{tp['losses']}")
    print(f"  清除数 d={cp['mean_diff']:+.3f} CI95[{cp['ci95'][0]:+.3f},{cp['ci95'][1]:+.3f}] "
          f"更多/相同/更少 {cp['losses']}/{cp['ties']}/{cp['wins']}")

    payload = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   selection=dict(scan_seeds=[seeds.start, seeds.stop],
                                  boundary_min_m=args.boundary_min,
                                  aim_gap_deg=args.aim_gap, min_hard=args.min_hard,
                                  directional=args.directional, count=args.count,
                                  matched_seeds=len(picked), hard_sources=hard_total),
                   picked=picked, summary=summary, paired=paired,
                   records=sorted(records, key=lambda r: (r['seed'], r['outer_radius_m'])))
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n写入 {out}")


if __name__ == '__main__':
    main()
