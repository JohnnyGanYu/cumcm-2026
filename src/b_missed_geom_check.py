"""判定指定漏失案例是"几何不可达"还是"策略欠账"。

判据：把 Q4 实际使用的全部规划站位（内环 + 定向加密外环 + 原点）逐个检查，
看该源在其中任何一个站位上是否可被接收——需同时满足
  (1) 距离 <= 该源的有效接收半径
  (2) 定向源还须落在其 180° 覆盖扇区内（全向源无此约束）
可见站位数为 0 => 站在任何规划点上都收不到信号，属信息不可达；
可见站位数 >= 1 => 信息充足，漏失是策略/时序问题。

用法: python -m src.b_missed_geom_check 10:9103:13 10:9158:13 ...
"""

from __future__ import annotations

import math
import sys

from .b_sim.arena import ARENA_RADIUS, generate_case
from .b_sim.strategy import DogStrategy, StrategyConfig


def stations(directional: int) -> list[tuple[float, float]]:
    cfg = StrategyConfig(directional_probe=directional > 0)
    strat = DogStrategy.__new__(DogStrategy)
    strat.config = cfg
    pts = [(0.0, 0.0)]
    for ring in strat._probe_stations():
        pts.extend(ring)
    return pts


def visible(jam, pt: tuple[float, float]) -> bool:
    d = math.hypot(jam.x - pt[0], jam.y - pt[1])
    if d > jam.reception_radius_m:
        return False
    if jam.directional_deg is None:
        return True
    # 站位相对源的方位，需落在定向方向两侧各 90 度内
    brg = math.degrees(math.atan2(pt[1] - jam.y, pt[0] - jam.x)) % 360.0
    off = abs((brg - jam.directional_deg + 180.0) % 360.0 - 180.0)
    return off <= 90.0


def main() -> None:
    specs = sys.argv[1:]
    if not specs:
        print(__doc__)
        return
    for spec in specs:
        dr, seed, ch = (int(x) for x in spec.split(":"))
        case = generate_case(seed, count=None, directional_count=dr)
        jam = next(j for j in case.jammers if j.channel == ch)
        pts = stations(dr)
        vis = [p for p in pts if visible(jam, p)]
        r = math.hypot(jam.x, jam.y)
        kind = "全向" if jam.directional_deg is None else f"定向{jam.directional_deg:.0f}°"
        extra = ""
        if jam.directional_deg is not None:
            inward = math.degrees(math.atan2(-jam.y, -jam.x)) % 360.0
            off = abs((jam.directional_deg - inward + 180.0) % 360.0 - 180.0)
            extra = f" 辐射轴偏离朝内{off:.1f}°"
        verdict = "信息不可达(几何)" if not vis else f"信息充足({len(vis)}站可见)->策略欠账"
        print(f"d{dr} seed={seed} ch={ch}: r={r:.0f} 到边界={ARENA_RADIUS - r:.0f}m "
              f"接收半径={jam.reception_radius_m:.0f} {kind}{extra}")
        print(f"    共{len(pts)}个规划站位, 可见={len(vis)}  => {verdict}")
        if vis:
            ds = sorted(round(math.hypot(jam.x - p[0], jam.y - p[1])) for p in vis)
            print(f"    可见站位距离: {ds}")


if __name__ == "__main__":
    main()
