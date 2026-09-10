#!/usr/bin/env python3
"""诊断 seed 7001 ch11 为什么归航失败。不猜参数，直接看每步状态。"""
import math

from src.b_sim.arena import Arena, generate_case, DIRECTIONAL_HALF_ANGLE_DEG

SEED, CH = 7001, 11
case = generate_case(seed=SEED, directional_count=5)
j = [x for x in case.jammers if x.channel == CH][0]
print(f"真值 ch{CH}: ({j.x:.1f},{j.y:.1f}) R_recv={j.reception_radius_m:.0f} "
      f"朝向={j.directional_deg:.1f}°  距原点={math.hypot(j.x,j.y):.0f}")


def visible(px, py):
    d = math.hypot(px - j.x, py - j.y)
    if d > j.reception_radius_m:
        return False, d, None
    ang = math.degrees(math.atan2(py - j.y, px - j.x)) % 360
    diff = abs((ang - j.directional_deg + 180) % 360 - 180)
    return diff <= DIRECTIONAL_HALF_ANGLE_DEG, d, diff


# 可见区的几何范围：沿朝向前方的半圆与接收圆的交集
print("\n可见区边界探查（以源为中心，沿各方位取最远可见点）:")
for ang_deg in range(0, 360, 30):
    a = math.radians(ang_deg)
    # 该方位是否在扇区内
    diff = abs((ang_deg - j.directional_deg + 180) % 360 - 180)
    if diff > DIRECTIONAL_HALF_ANGLE_DEG:
        continue
    # 沿该方位向外走，找仍在目标区域内的最远可见点
    best = None
    for r in range(20, int(j.reception_radius_m) + 1, 20):
        px, py = j.x + r * math.cos(a), j.y + r * math.sin(a)
        if math.hypot(px, py) > 1800:
            break
        vis, _, _ = visible(px, py)
        if vis:
            best = (r, px, py)
    if best:
        r, px, py = best
        print(f"  方位{ang_deg:3d}° 最远可见 r={r:4d}m at ({px:8.1f},{py:8.1f}) "
              f"距原点{math.hypot(px,py):6.0f}m")
    else:
        print(f"  方位{ang_deg:3d}° 区域内无可见点")

# 关键：可见且在目标区域内的点，距源多近？
print("\n可见区内距源最近的可达点（决定能否进入 20m 清除半径）:")
best_d = 1e9
best_pt = None
for gx in range(-1800, 1801, 10):
    for gy in range(-1800, 1801, 10):
        if gx * gx + gy * gy > 1800 ** 2:
            continue
        vis, d, _ = visible(gx, gy)
        if vis and d < best_d:
            best_d, best_pt = d, (gx, gy)
print(f"  最近可见点 {best_pt} 距源 {best_d:.1f} m")
print(f"  清除半径 20 m -> {'可以在可见状态下清除' if best_d <= 20 else '必须盲清除(走进20m时已看不见)'}")

# 源本身位置是否在目标区域内
print(f"\n源是否在目标区域内: {math.hypot(j.x, j.y) <= 1800}")
print(f"源正下方(即源位置)是否可见: {visible(j.x + 1, j.y)[0]}")

# 从最近可见点出发，朝源走，什么时候丢信号
print("\n从最近可见点朝源直线逼近，逐步检查可见性:")
px, py = best_pt
vx, vy = j.x - px, j.y - py
L = math.hypot(vx, vy)
for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99):
    qx, qy = px + vx * frac, py + vy * frac
    vis, d, diff = visible(qx, qy)
    inside = math.hypot(qx, qy) <= 1800
    print(f"  frac={frac:4.2f} ({qx:8.1f},{qy:8.1f}) 距源{d:7.1f}m "
          f"角差{diff if diff is not None else -1:6.1f}° 可见={vis} 区域内={inside}")

print("""
结论解读：
  若"最近可见点距源"远大于 20 m，说明策略必须在丢失信号后靠推算盲走进 20m。
  若朝源逼近过程中很快丢信号，则 dogleg/step_closer 会连续沉默 -> 触发放弃。
  这类源的正解是：用已有示向度直接盲清除候选点，而不是要求"看得见才清除"。
""")
