#!/usr/bin/env python3
"""B题问题1：定位区域直径算法 + "以直径为直径的圆能否覆盖定位区域"的严格反例。

结论：不能。给出高精度可复核反例。
"""
import itertools
import mpmath as mp

mp.mp.dps = 60


def wedge_halfplanes(S, th_deg, eps_deg=1):
    """检测点S、示向度th_deg、误差±eps_deg 决定的楔形角域，返回两个半平面 (a,b,c)：ax+by<=c。

    楔形 = {P : 从S看P的方位角 ∈ [th-eps, th+eps]}
    等价于两条边界射线各自定义的半平面之交（当 2*eps < 180° 时）。
    """
    sx, sy = mp.mpf(S[0]), mp.mpf(S[1])
    lo = mp.mpf(th_deg) - eps_deg
    hi = mp.mpf(th_deg) + eps_deg
    d_lo = (mp.cos(mp.radians(lo)), mp.sin(mp.radians(lo)))
    d_hi = (mp.cos(mp.radians(hi)), mp.sin(mp.radians(hi)))
    # P 在 d_lo 左侧: cross(d_lo, P-S) >= 0  ->  -d_lo_y*(x-sx) + d_lo_x*(y-sy) >= 0
    a1, b1 = d_lo[1], -d_lo[0]
    c1 = a1 * sx + b1 * sy
    # P 在 d_hi 右侧: cross(d_hi, P-S) <= 0
    a2, b2 = -d_hi[1], d_hi[0]
    c2 = a2 * sx + b2 * sy
    return [(a1, b1, c1), (a2, b2, c2)]


def polygon_from_halfplanes(hps, tol=mp.mpf("1e-30")):
    """半平面交 -> 顶点集（枚举两两交点，保留满足全部约束的点）。"""
    verts = []
    n = len(hps)
    for i, j in itertools.combinations(range(n), 2):
        a1, b1, c1 = hps[i]
        a2, b2, c2 = hps[j]
        det = a1 * b2 - a2 * b1
        if abs(det) < tol:
            continue
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        if all(a * x + b * y <= c + mp.mpf("1e-18") for a, b, c in hps):
            if not any(mp.sqrt((x - u) ** 2 + (y - v) ** 2) < mp.mpf("1e-9")
                       for u, v in verts):
                verts.append((x, y))
    return verts


def diameter(verts):
    """定位区域直径 = 顶点集中最远点对距离（凸多边形的直径必在顶点上取得）。"""
    best, pair = mp.mpf(0), None
    for i, j in itertools.combinations(range(len(verts)), 2):
        d = mp.sqrt((verts[i][0] - verts[j][0]) ** 2 + (verts[i][1] - verts[j][1]) ** 2)
        if d > best:
            best, pair = d, (i, j)
    return best, pair


def circum_center_3(p, q, r):
    """三点外接圆心。"""
    ax, ay = p; bx, by = q; cx, cy = r
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < mp.mpf("1e-40"):
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / d
    return (ux, uy)


def min_enclosing_circle(verts):
    """最小外接圆：枚举所有 2-点（直径圆）与 3-点（外接圆）候选，取能覆盖全部点的最小者。"""
    def covers(ctr, rad):
        return all(mp.sqrt((x - ctr[0])**2 + (y - ctr[1])**2) <= rad + mp.mpf("1e-25")
                   for x, y in verts)

    best = None
    # 2点候选
    for i, j in itertools.combinations(range(len(verts)), 2):
        ctr = ((verts[i][0] + verts[j][0]) / 2, (verts[i][1] + verts[j][1]) / 2)
        rad = mp.sqrt((verts[i][0] - ctr[0])**2 + (verts[i][1] - ctr[1])**2)
        if covers(ctr, rad) and (best is None or rad < best[0]):
            best = (rad, ctr, ("2pt", i, j))
    # 3点候选
    for i, j, k in itertools.combinations(range(len(verts)), 3):
        ctr = circum_center_3(verts[i], verts[j], verts[k])
        if ctr is None:
            continue
        rad = mp.sqrt((verts[i][0] - ctr[0])**2 + (verts[i][1] - ctr[1])**2)
        if covers(ctr, rad) and (best is None or rad < best[0]):
            best = (rad, ctr, ("3pt", i, j, k))
    return best


def analyze(points_thetas, label=""):
    hps = []
    for S, th in points_thetas:
        hps += wedge_halfplanes(S, th)
    verts = polygon_from_halfplanes(hps)
    if len(verts) < 3:
        print(f"{label}: 区域无界或退化，顶点数={len(verts)}")
        return None
    d, dpair = diameter(verts)
    R, ctr, how = min_enclosing_circle(verts)
    ratio = R / (d / 2)
    # 直径圆是否覆盖
    i, j = dpair
    dctr = ((verts[i][0] + verts[j][0]) / 2, (verts[i][1] + verts[j][1]) / 2)
    far = max(mp.sqrt((x - dctr[0])**2 + (y - dctr[1])**2) for x, y in verts)
    excess = far - d / 2

    print(f"\n{'='*72}\n{label}")
    print(f"顶点数 {len(verts)}")
    for x, y in verts:
        print(f"   ({mp.nstr(x, 12)}, {mp.nstr(y, 12)})")
    print(f"定位区域直径 d = {mp.nstr(d, 12)} m   (由顶点 {i},{j} 决定)")
    print(f"最小外接圆 R = {mp.nstr(R, 12)} m   由 {how[0]}")
    print(f"R / (d/2) = {mp.nstr(ratio, 15)}")
    print(f"以直径为直径的圆: 最远顶点距圆心 {mp.nstr(far, 12)} m, 半径 d/2 = {mp.nstr(d/2, 12)} m")
    if excess > mp.mpf("1e-9"):
        print(f"  -> 不覆盖！超出 {mp.nstr(excess, 8)} m   ★反例★")
    else:
        print(f"  -> 覆盖 (余量 {mp.nstr(-excess, 8)} m)")
    # 顶点三角形最大角，解释为何 MEC 不由直径决定
    print("  顶点三角形最大角检查:")
    for tri in itertools.combinations(range(len(verts)), 3):
        pts = [verts[t] for t in tri]
        angs = []
        for a in range(3):
            p0 = pts[a]; p1 = pts[(a+1) % 3]; p2 = pts[(a+2) % 3]
            v1 = (p1[0]-p0[0], p1[1]-p0[1]); v2 = (p2[0]-p0[0], p2[1]-p0[1])
            ca = (v1[0]*v2[0]+v1[1]*v2[1]) / (mp.sqrt(v1[0]**2+v1[1]**2)*mp.sqrt(v2[0]**2+v2[1]**2))
            angs.append(mp.degrees(mp.acos(max(-1, min(1, ca)))))
        mx = max(angs)
        kind = "锐角三角形(MEC由三点定)" if mx < 90 else "非锐角"
        print(f"     {tri} 最大角 {mp.nstr(mx, 8)}°  {kind}")
    return {"d": d, "R": R, "ratio": ratio, "excess": excess, "verts": verts}


if __name__ == "__main__":
    print("B题问题1：定位区域直径算法 + 直径圆覆盖性判定")

    # 教科书情形：两个检测点，交会角接近 90°
    analyze([((0, 0), 45.0), ((1000, 0), 135.0)],
            label="情形1: 两检测点，交会角≈90°（题目图2的典型四边形）")

    # 反：四个检测点，交会出的四边形含锐角三角形
    CE = [
        ((-915.74,   72.39), None),
        (( 874.69,  729.83), None),
        (( 990.01, 1232.70), None),
        ((1071.64,  -65.76), None),
    ]
    TRUE_SRC = (600.0, 500.0)
    pts = []
    for (sx, sy), _ in CE:
        th = mp.degrees(mp.atan2(TRUE_SRC[1] - sy, TRUE_SRC[0] - sx)) % 360
        pts.append(((sx, sy), th))
    res = analyze(pts, label="情形2: 四检测点（真实源(600,500)，示向度取真值）★关键反例★")

    print(f"\n{'='*72}")
    print("结论")
    print("  问题1第二问答案：一般情况下【不能】覆盖。")
    print("  原因：定位区域是凸多边形，其最小外接圆(MEC)并不总由直径两端点决定。")
    print("  当顶点集中存在锐角三角形时，MEC 由该三点的外接圆决定，半径 R > d/2，")
    print("  此时以直径为直径的圆无法覆盖整个定位区域。")
    print("  上界：由 Jung 定理，平面点集恒有 R <= d/sqrt(3) = 1.1547*(d/2)，该上界在正三角形取到。")
    print(f"  Jung 上界 R/(d/2) = {mp.nstr(2/mp.sqrt(3), 12)}")
