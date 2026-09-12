"""Sampled visibility and route audit; never a full-clear proof or runtime policy.

Visibility loss integrates the uncovered directional heading arc analytically
for each sampled source position. Radius 1000 is the legal worst case; the
radius-average diagnostic assumes the offline generator's uniform 1000..1500.
Grid results are samples, not conservative whole-cell guarantees.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from src.b_sim.route import order_route, path_length


def rings(inner=950, outer=1900, ni=6, no=14, phase=0):
    return [(0., 0.)] + [
        (r * math.cos(2 * math.pi * k / n + a),
         r * math.sin(2 * math.pi * k / n + a))
        for r, n, a in ((inner, ni, 0), (outer, no, math.radians(phase)))
        for k in range(n)
    ]


def grid(step):
    xy = np.arange(-1800, 1800 + step / 2, step)
    xx, yy = np.meshgrid(xy, xy)
    cells = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    return cells[np.linalg.norm(cells, axis=1) <= 1800]


def visibility(points, cells, reception):
    v = np.array(points)[None, :, :] - cells[:, None, :]
    d = np.linalg.norm(v, axis=-1)
    valid = d <= reception
    angles = np.arctan2(v[:, :, 1], v[:, :, 0]) % (2 * np.pi)
    angles[~valid] = 10 * np.pi
    angles.sort(axis=1)
    counts = valid.sum(axis=1)
    gaps = np.where(np.arange(len(points) - 1)[None, :] < counts[:, None] - 1,
                    np.diff(angles, axis=1), 0).max(axis=1, initial=0)
    last = angles[np.arange(len(cells)), np.maximum(0, counts - 1)]
    gap = np.maximum(gaps, angles[:, 0] + 2 * np.pi - last)
    lost = np.maximum(0., gap - np.pi) / (2 * np.pi)
    lost[counts == 0] = 1.
    # At an exact source coordinate Jammer.covers is true for all headings.
    lost[(d == 0).any(axis=1)] = 0.
    return dict(uncovered_source_fraction=float((lost > 1e-12).mean()),
                invisible_heading_fraction=float(lost.mean()),
                worst_invisible_heading_fraction=float(lost.max()),
                mean_reachable_stations=float(counts.mean()))


def audit(points, step=75):
    cells = grid(step)
    worst = visibility(points, cells, 1000.)
    route = order_route((0., 0.), points[1:])
    return dict(stations=len(points), sampled_positions=len(cells),
                sampled_min_radius=worst,
                open_route_m=path_length((0., 0.), route))


def refine():
    cells = grid(20)
    angles = np.linspace(0, 2*np.pi, 1440, endpoint=False)
    edge = np.array([(r*np.cos(t), r*np.sin(t))
                     for r in (1780, 1790, 1800) for t in angles])
    configs = [(6,950,1900,14,0), (6,1000,1850,12,15),
               (6,1000,1870,12,15), (6,1000,1870,12,7.5),
               (6,1000,1870,12,0), (6,975,1870,12,15),
               (6,975,1870,12,7.5), (6,1000,1880,12,15),
               (6,1000,1900,12,15), (7,1000,1870,12,15),
               (7,1000,1870,12,7.5), (7,1000,1900,13,0),
               (6,1000,1850,14,15)]
    rows = []
    for ni, inner, outer, no, phase in configs:
        points = rings(inner, outer, ni, no, phase)
        row = dict(ni=ni,inner=inner,outer=outer,no=no,phase=phase,
                   **audit(points,75))
        row['dense_min_radius'] = visibility(points,cells,1000)
        row['edge_min_radius'] = visibility(points,edge,1000)
        row['uniform_radius_heading_loss'] = sum(
            visibility(points,cells,r)['invisible_heading_fraction']
            for r in np.linspace(1000,1500,11)) / 11
        rows.append(row)
        print(json.dumps(row),flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--step', type=float, default=75)
    ap.add_argument('--out', default='results/q4_410_layout_design.json')
    ap.add_argument('--refine', action='store_true',
                    help='20 m grid and boundary audit of short-listed layouts')
    args = ap.parse_args()
    if args.refine:
        Path(args.out).write_text(json.dumps(refine(),indent=2),encoding='utf-8')
        return
    rows = []
    configs = [(950, 1900, 6, 14, 0)]
    configs += list(itertools.product((900, 1000, 1050, 1100, 1150),
                                     (1850, 1900, 1950), (5, 6, 7), (12, 13, 14), (0, 7.5, 15)))
    for inner, outer, ni, no, phase in configs:
        points = rings(inner, outer, ni, no, phase)
        result = audit(points, args.step)
        rows.append(dict(inner=inner, outer=outer, ni=ni, no=no, phase=phase, **result))
    rows.sort(key=lambda r: (r['sampled_min_radius']['invisible_heading_fraction'], r['open_route_m']))
    baseline = next(r for r in rows if r['inner']==950 and r['outer']==1900)
    candidates = [r for r in rows if r['sampled_min_radius']['invisible_heading_fraction'] <=
                  baseline['sampled_min_radius']['invisible_heading_fraction'] + 1e-9]
    candidates.sort(key=lambda r:r['open_route_m'])
    for row in candidates[:20]:
        points = rings(*(row[k] for k in ('inner','outer','ni','no','phase')))
        dense = grid(25)
        row['dense_min_radius'] = visibility(points, dense, 1000)
        row['uniform_radius_heading_loss'] = sum(visibility(points, dense, r)['invisible_heading_fraction']
                                                 for r in np.linspace(1000,1500,11)) / 11
    result = dict(disclaimer='Sampled visibility only, not a coverage or full-clear proof. Radius average assumes local generator uniform reception radius.',
                  step=args.step, baseline=baseline, recommended_by_route=candidates[:20], rows=rows)
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(baseline=baseline,candidates=candidates[:12]),indent=2))


if __name__ == '__main__':
    main()
