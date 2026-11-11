"""CUMCM2026-B 侦测/定位/路径规划 —— 候选架构的量化预算测算.

只依赖 numpy。所有数字都是本脚本实跑输出，不是引用。
题设参数（来自任务背景描述）：
  区域 R=1800m 圆, 10~16 个源, 每源占 1~20 中不同频道
  机器狗 v=5 m/s, 原点出发
  检测 5s 返回示向度(有误差), 清除半径 ~20m, <=5m 且在覆盖角内返回 near
  有效接收半径 1000~1500m 未知
"""
import numpy as np
import itertools, math, json

R_AREA = 1800.0
V = 5.0
T_DETECT = 5.0
R_CLEAR = 20.0
RMIN_RX, RMAX_RX = 1000.0, 1500.0

rng = np.random.default_rng(20260910)


# ---------------------------------------------------------------- 1. 交会几何
def tri_mc(p1, p2, src, sig_deg=1.0, n=40000):
    """两条含噪示向度射线交会的定位误差 MC。返回 (P50,P90), P(err<=20m)。"""
    sig = np.deg2rad(sig_deg)
    b1 = math.atan2(*(src - p1)[::-1])
    b2 = math.atan2(*(src - p2)[::-1])
    z1 = b1 + rng.normal(0, sig, n)
    z2 = b2 + rng.normal(0, sig, n)
    d1 = np.stack([np.cos(z1), np.sin(z1)])
    d2 = np.stack([np.cos(z2), np.sin(z2)])
    A = p2 - p1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    t = (A[0] * d2[1] - A[1] * d2[0]) / den
    est = p1[:, None] + d1 * t
    e = np.linalg.norm(est - src[:, None], axis=0)
    e = e[np.isfinite(e)]
    ang = abs(math.degrees(b1 - b2)) % 180
    return ang, np.percentile(e, [50, 90]), (e <= R_CLEAR).mean()


def crlb_two_bearing(p1, p2, src, sig_deg=1.0):
    """两次测向的 CRLB: trace(inv(FIM)) 的平方根 (RMSE 下界)。"""
    sig = np.deg2rad(sig_deg)
    J = np.zeros((2, 2))
    for p in (p1, p2):
        d = src - p
        r2 = d @ d
        r = math.sqrt(r2)
        # dtheta/dx = -dy/r^2 ; dtheta/dy = dx/r^2
        g = np.array([-d[1] / r2, d[0] / r2])
        J += np.outer(g, g) / sig ** 2
    if abs(np.linalg.det(J)) < 1e-18:
        return float("inf"), r      # 共线/同向 -> 不可观测
    C = np.linalg.inv(J)
    return math.sqrt(np.trace(C)), r


# ---------------------------------------------------------------- 2. 圆覆盖
def ring_cover(radius_list, counts, r_sense):
    """给定环半径/每环站点数，估计目标圆盘内被 r_sense 覆盖的面积比（MC）。"""
    pts = [np.zeros(2)]
    for rad, k in zip(radius_list, counts):
        for j in range(k):
            a = 2 * math.pi * j / k
            pts.append(np.array([rad * math.cos(a), rad * math.sin(a)]))
    P = np.array(pts)
    # MC 面积覆盖率
    n = 200000
    rr = R_AREA * np.sqrt(rng.random(n))
    tt = 2 * math.pi * rng.random(n)
    Q = np.stack([rr * np.cos(tt), rr * np.sin(tt)], 1)
    d = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2)
    cov = (d.min(1) <= r_sense).mean()
    return len(P), cov, P


def tour_len(P):
    """最近邻 + 2-opt 粗略巡回长度（从原点出发不闭合）。"""
    n = len(P)
    unv = set(range(1, n))
    order = [0]
    cur = 0
    while unv:
        nxt = min(unv, key=lambda j: np.linalg.norm(P[j] - P[cur]))
        order.append(nxt)
        unv.discard(nxt)
        cur = nxt
    # 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c = order[j]
                d_ = order[j + 1] if j + 1 < n else None
                old = np.linalg.norm(P[a] - P[b]) + (np.linalg.norm(P[c] - P[d_]) if d_ is not None else 0)
                new = np.linalg.norm(P[a] - P[c]) + (np.linalg.norm(P[b] - P[d_]) if d_ is not None else 0)
                if new < old - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    L = sum(np.linalg.norm(P[order[k + 1]] - P[order[k]]) for k in range(n - 1))
    return L


# ---------------------------------------------------------------- 3. 射线盲扫
def ray_blind_scan(dist_true, step=32.0, r_max=1500.0):
    """方向已知、距离未知：沿射线按 step 铺点清除。返回 (走行距离, 清除调用数)。"""
    k = int(math.ceil(dist_true / step))
    probe = k  # 命中所需的清除尝试次数（上界，实际可提前命中）
    travel = dist_true
    return travel, probe


def main():
    out = {}
    print("=" * 74)
    print("[1] 两站交会：交会角 / 距离 对定位精度的影响 (示向度误差 sigma=1deg)")
    print("=" * 74)
    src = np.array([0.0, 1200.0])
    cases = [
        ("原点+沿路600m侧移", np.array([0.0, 0.0]), np.array([600.0, 0.0])),
        ("原点+侧移1200m", np.array([0.0, 0.0]), np.array([1200.0, 0.0])),
        ("正交交会 r=850", np.array([0.0, 350.0]), np.array([850.0, 1200.0])),
        ("正交交会 r=400", np.array([0.0, 800.0]), np.array([400.0, 1200.0])),
        ("小角交会(同向前进)", np.array([0.0, 0.0]), np.array([0.0, 600.0])),
    ]
    rows = []
    for nm, p1, p2 in cases:
        ang, q, hit = tri_mc(p1, p2, src)
        crlb, _ = crlb_two_bearing(p1, p2, src)
        print(f"  {nm:20s} 交会角={ang:6.1f}deg  P50={q[0]:8.1f}m  P90={q[1]:9.1f}m  "
              f"P(err<=20m)={hit:5.3f}  CRLB_RMSE={crlb:8.1f}m")
        rows.append(dict(case=nm, angle=ang, p50=q[0], p90=q[1], hit20=hit, crlb=crlb))
    out["triangulation"] = rows

    print()
    print("=" * 74)
    print("[2] 结论性推论：20m 清除半径 vs 交会精度 —— 需要多少次测量才能直接命中")
    print("=" * 74)
    # 用最好几何(正交, 中等距离)看 1/2/3/4 次测量后的 P(err<=20)
    for nmeas in (2, 3, 4, 6):
        sig = np.deg2rad(1.0)
        # 理想: nmeas 次均匀分布在源周围 r=600m 处
        J = np.zeros((2, 2))
        for j in range(nmeas):
            a = math.pi * j / nmeas
            p = src + 600 * np.array([math.cos(a), math.sin(a)])
            d = src - p
            r2 = d @ d
            g = np.array([-d[1] / r2, d[0] / r2])
            J += np.outer(g, g) / sig ** 2
        C = np.linalg.inv(J)
        rmse = math.sqrt(np.trace(C))
        print(f"  n={nmeas} 次理想布站(r=600m) CRLB_RMSE={rmse:7.1f}m  "
              f"-> {'可直接落到清除半径内' if rmse < R_CLEAR else '仍需盲扫/贴近补测'}")

    print()
    print("=" * 74)
    print("[3] 覆盖站位设计：几个环 / 多少站 才能保证任一源被 '听到'")
    print("=" * 74)
    configs = [
        ([0], [1], "只在原点"),
        ([1000.0], [6], "单环1000x6"),
        ([1250.0], [12], "单环1250x12"),
        ([900.0, 1560.0], [8, 14], "双环900x8+1560x14"),
        ([800.0, 1450.0], [7, 13], "双环800x7+1450x13"),
    ]
    for rl, ct, nm in configs:
        for r_sense in (1000.0, 1500.0):
            npts, cov, P = ring_cover(rl, ct, r_sense)
            L = tour_len(P)
            t_move = L / V
            t_det_all = npts * 20 * T_DETECT      # 每站扫全部 20 频道
            t_det_live = npts * 16 * T_DETECT     # 只扫尚未清除的活跃频道(上界16)
            print(f"  {nm:22s} r_sense={r_sense:6.0f}m 站数={npts:3d} 覆盖率={cov:6.3f} "
                  f"巡回={L:7.0f}m 行进={t_move:6.0f}s 全频扫描={t_det_all:6.0f}s "
                  f"(活跃频道扫描={t_det_live:6.0f}s)")

    print()
    print("=" * 74)
    print("[4] 时间预算拆解：检测 5s 是主要成本还是行进是主要成本？")
    print("=" * 74)
    print(f"  移动 100m 耗时 = {100/V:.0f}s = {100/V/T_DETECT:.0f} 次检测")
    print(f"  一次检测 5s 等价于移动 {T_DETECT*V:.0f}m")
    print(f"  扫全 20 频道 = {20*T_DETECT:.0f}s 等价于移动 {20*T_DETECT*V:.0f}m")
    print(f"  跨区域直径 3600m 行进 = {3600/V:.0f}s = {3600/V/T_DETECT:.0f} 次检测")
    print("  => 单站多频道扫描比长距离移动便宜得多：站位应少而精，每站尽量榨干所有活跃频道")

    print()
    print("=" * 74)
    print("[5] 射线盲扫 vs 交会补测 的成本对比 (方向已知,距离未知)")
    print("=" * 74)
    for dist in (300.0, 800.0, 1400.0):
        for step in (32.0, 40.0):
            travel, probe = ray_blind_scan(dist, step)
            # 横向偏差: 1deg 测向误差在 dist 处
            lat = math.radians(1.0) * dist
            need_side = lat > R_CLEAR
            t = travel / V
            print(f"  真实距离={dist:6.0f}m step={step:4.0f}m: 行进{travel:6.0f}m={t:5.0f}s "
                  f"最多{probe:3d}次清除尝试; 1deg横向偏差={lat:5.1f}m "
                  f"{'需两侧补点' if need_side else '单线足够'}")

    print()
    print("=" * 74)
    print("[6] 定向源(180度扇区)可见性：为什么外环半径必须够大")
    print("=" * 74)
    n = 4000
    for src_r_frac in (0.4, 0.7, 0.9):
        vis_by_ring = {}
        for ring_r in (1000.0, 1250.0, 1560.0, 1750.0):
            ok = 0
            for _ in range(n):
                a = 2 * math.pi * rng.random()
                s = src_r_frac * R_AREA * np.array([math.cos(a), math.sin(a)])
                heading = 2 * math.pi * rng.random()          # 扇区中心朝向
                rx = rng.uniform(RMIN_RX, RMAX_RX)            # 未知有效半径
                # 该环上 16 个站里有几个能收到
                hit = False
                for j in range(16):
                    b = 2 * math.pi * j / 16
                    p = ring_r * np.array([math.cos(b), math.sin(b)])
                    d = p - s
                    if np.linalg.norm(d) > rx:
                        continue
                    ang = abs((math.atan2(d[1], d[0]) - heading + math.pi) % (2 * math.pi) - math.pi)
                    if ang <= math.pi / 2:                     # 180deg 扇区
                        hit = True
                        break
                ok += hit
            vis_by_ring[ring_r] = ok / n
        print(f"  源位于 {src_r_frac*R_AREA:6.0f}m 处: " +
              "  ".join(f"环{int(k)}m可见率={v:5.3f}" for k, v in vis_by_ring.items()))

    print()
    print("=" * 74)
    print("[7] 上界参考：如果每个源平均需要 k 次检测 + 一次奔赴，总时长量级")
    print("=" * 74)
    for nsrc in (10, 13, 16):
        for k in (3, 4, 6):
            t_det = nsrc * k * T_DETECT
            t_scan = 20 * T_DETECT * 3           # 3 个全频扫描站
            t_move = nsrc * 900 / V              # 每源平均奔赴 900m
            tot = t_det + t_scan + t_move
            print(f"  源数={nsrc} 每源{k}次测向: 检测{t_det:5.0f}s + 全频扫{t_scan:4.0f}s "
                  f"+ 奔赴{t_move:6.0f}s = {tot:6.0f}s ({tot/60:5.1f}min)")


if __name__ == "__main__":
    main()
