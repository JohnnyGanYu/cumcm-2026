#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUMCM 2026 B题 问题2：第二检测点选择策略的解析推导 + 蒙特卡洛验证。

坐标系：目标区域为以原点为心、半径 R_AREA=1800 m 的圆域，x 正东、y 正北。
测向机示向度误差 e ~ U[-EPS_DEG, +EPS_DEG]，EPS_DEG = 1 度（同一地点固定，不同地点独立）。

本脚本内容：
  Part A  交会定位区域的精确几何（半平面裁剪）与解析近似的对照
  Part B  等距族 (d2 = d1) 下，定位区域直径/面积随交会角 gamma 的蒙特卡洛表格
  Part C  gamma -> 0 的退化（定位区域无界）实验
  Part D  实际可操作参数 (beta, L) 的扫描：偏移角与步行距离的影响、精度—耗时权衡
  Part E  第二检测点候选区域的网格计算（含“干扰源距离未知”的稳健处理）+ 图形输出

所有输出数字均由本脚本实际运行产生，结果写入 results/ 与 figures/。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ----------------------------------------------------------------------------- 常量
R_AREA = 1800.0          # 目标圆域半径 (m)
EPS_DEG = 1.0            # 示向度误差上界 (度)
EPS = math.radians(EPS_DEG)
R_EFF_LO, R_EFF_HI = 1000.0, 1500.0   # 有效接收半径未知区间
SPEED = 5.0              # 机器狗行进速度 m/s
SEED = 20260910

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


# ----------------------------------------------------------------- 精确几何：半平面裁剪
def _clip(poly, nx, ny, c):
    """用半平面 nx*x + ny*y <= c 裁剪凸多边形 poly（顶点列表，逆时针或顺时针均可）。"""
    if not poly:
        return []
    out = []
    m = len(poly)
    for i in range(m):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % m]
        d1 = nx * x1 + ny * y1 - c
        d2 = nx * x2 + ny * y2 - c
        if d1 <= 0.0:
            out.append((x1, y1))
        if (d1 < 0.0 < d2) or (d2 < 0.0 < d1):
            t = d1 / (d1 - d2)
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return out


def wedge_planes(S, theta_rad, eps=EPS):
    """顶点 S、中心方位 theta、半张角 eps 的楔形 = 两个半平面 (nx, ny, c)。"""
    sx, sy = S
    a = theta_rad - eps
    b = theta_rad + eps
    ax, ay = math.cos(a), math.sin(a)
    bx, by = math.cos(b), math.sin(b)
    # cross(dir_a, x-S) >= 0  ->  ( ay, -ax) . x <=  ( ay, -ax) . S
    p1 = (ay, -ax, ay * sx - ax * sy)
    # cross(dir_b, x-S) <= 0  ->  (-by,  bx) . x <= (-by,  bx) . S
    p2 = (-by, bx, -by * sx + bx * sy)
    return p1, p2


def locate_region(S1, th1, S2, th2, eps=EPS, box=1.0e7, center=(0.0, 0.0)):
    """两个楔形的交集。返回 (顶点列表, 是否无界)。"""
    cx, cy = center
    poly = [(cx - box, cy - box), (cx + box, cy - box), (cx + box, cy + box), (cx - box, cy + box)]
    for pl in (*wedge_planes(S1, th1, eps), *wedge_planes(S2, th2, eps)):
        poly = _clip(poly, *pl)
        if not poly:
            return [], False
    unbounded = any(abs(x - cx) > 0.5 * box or abs(y - cy) > 0.5 * box for x, y in poly)
    return poly, unbounded


def poly_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def poly_diameter(poly):
    """凸多边形直径 = 顶点间最大距离。"""
    best = 0.0
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(poly[i][0] - poly[j][0], poly[i][1] - poly[j][1])
            if d > best:
                best = d
    return best


def point_in_poly(poly, p, tol=1e-6):
    """凸多边形（顶点按裁剪顺序）内部判定：对每条边同号。"""
    if len(poly) < 3:
        return False
    px, py = p
    sign = 0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cr = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cr) < tol:
            continue
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


# ------------------------------------------------------------------------ 解析近似公式
def analytic_area(d1, d2, gamma_rad, eps=EPS):
    """条带近似：面积 ≈ (2 eps d1)(2 eps d2)/sin gamma。"""
    return (2 * eps * d1) * (2 * eps * d2) / abs(math.sin(gamma_rad))


def analytic_diameter(d1, d2, gamma_rad, eps=EPS):
    """平行四边形长对角线：sqrt(w1^2 + w2^2 + 2 w1 w2 |cos g|)/sin g。"""
    w1, w2 = 2 * eps * d1, 2 * eps * d2
    return math.sqrt(w1 * w1 + w2 * w2 + 2 * w1 * w2 * abs(math.cos(gamma_rad))) / abs(math.sin(gamma_rad))


# --------------------------------------------------------------------------- 几何工具
def true_bearing(S, G):
    return math.atan2(G[1] - S[1], G[0] - S[0])


def crossing_angle(S1, S2, G):
    """真实交会角 gamma：G 处 GS1 与 GS2 的夹角（= 两条真实示向线方向 u1,u2 的夹角）。"""
    u1 = np.array([G[0] - S1[0], G[1] - S1[1]], float)
    u2 = np.array([G[0] - S2[0], G[1] - S2[1]], float)
    n1, n2 = np.linalg.norm(u1), np.linalg.norm(u2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = float(np.clip(np.dot(u1, u2) / (n1 * n2), -1.0, 1.0))
    return math.acos(c)


def s2_isoceles(d1, gamma_deg, side=+1):
    """在 S1=(0,0)、G=(d1,0) 的局部系中，构造使 |GS2| = |GS1| = d1 且 G 处夹角 = gamma 的 S2。"""
    g = math.radians(gamma_deg)
    ang = math.pi - side * g          # G->S2 的方位（G->S1 方位为 pi）
    return (d1 + d1 * math.cos(ang), d1 * math.sin(ang))


# --------------------------------------------------------------------- 蒙特卡洛核心例程
def mc_region_stats(S1, S2, G, n, rng, eps=EPS):
    """对 (S1,S2,G) 采样 n 次 ±1° 独立均匀误差，返回定位区域统计。"""
    t1 = true_bearing(S1, G)
    t2 = true_bearing(S2, G)
    e1 = rng.uniform(-eps, eps, n)
    e2 = rng.uniform(-eps, eps, n)
    diam = np.empty(n)
    area = np.empty(n)
    unb = 0
    contained = 0
    ctr = (0.5 * (S1[0] + S2[0]), 0.5 * (S1[1] + S2[1]))
    for k in range(n):
        poly, u = locate_region(S1, t1 + e1[k], S2, t2 + e2[k], eps, center=ctr)
        if u or len(poly) < 3:
            unb += 1
            diam[k] = np.inf
            area[k] = np.inf
            continue
        diam[k] = poly_diameter(poly)
        area[k] = poly_area(poly)
        if point_in_poly(poly, G):
            contained += 1
    fin = np.isfinite(diam)
    out = {
        "n": n,
        "unbounded_frac": unb / n,
        "contain_frac": contained / max(1, int(fin.sum())),
    }
    if fin.any():
        d, a = diam[fin], area[fin]
        out.update(
            diam_mean=float(d.mean()), diam_median=float(np.median(d)),
            diam_p90=float(np.percentile(d, 90)), diam_max=float(d.max()),
            area_mean=float(a.mean()), area_median=float(np.median(a)),
            area_p90=float(np.percentile(a, 90)), area_max=float(a.max()),
        )
    else:
        out.update(diam_mean=np.inf, diam_median=np.inf, diam_p90=np.inf, diam_max=np.inf,
                   area_mean=np.inf, area_median=np.inf, area_p90=np.inf, area_max=np.inf)
    return out


# ============================================================================ Part A
def part_a_validation(rng):
    """精确裁剪几何 vs 四交点凸包 vs 解析近似：确认实现正确。"""
    print("=" * 78)
    print("Part A  几何实现校验（精确半平面裁剪 vs 解析条带近似）")
    print("=" * 78)
    rows = []
    for d1, gdeg in [(500, 90), (1000, 90), (1400, 90), (1000, 30), (1000, 150), (1000, 60)]:
        S1 = (0.0, 0.0)
        G = (float(d1), 0.0)
        S2 = s2_isoceles(d1, gdeg)
        g_true = crossing_angle(S1, S2, G)
        d2 = math.hypot(G[0] - S2[0], G[1] - S2[1])
        poly, u = locate_region(S1, true_bearing(S1, G), S2, true_bearing(S2, G), center=G)
        rows.append(dict(d1=d1, gamma_deg=gdeg, gamma_check_deg=math.degrees(g_true), d2=d2,
                         nverts=len(poly), unbounded=u,
                         area_exact=poly_area(poly), area_approx=analytic_area(d1, d2, g_true),
                         diam_exact=poly_diameter(poly), diam_approx=analytic_diameter(d1, d2, g_true),
                         G_inside=point_in_poly(poly, G)))
    print(f"{'d1':>6} {'gamma':>6} {'gamma_chk':>10} {'d2':>8} {'nv':>3} {'area_exact':>12} "
          f"{'area_appr':>12} {'diam_exact':>11} {'diam_appr':>10} {'G_in':>5}")
    for r in rows:
        print(f"{r['d1']:>6.0f} {r['gamma_deg']:>6.0f} {r['gamma_check_deg']:>10.4f} {r['d2']:>8.1f} "
              f"{r['nverts']:>3d} {r['area_exact']:>12.2f} {r['area_approx']:>12.2f} "
              f"{r['diam_exact']:>11.3f} {r['diam_approx']:>10.3f} {str(r['G_inside']):>5}")
    return rows


# ============================================================================ Part B
def part_b_gamma_table(rng, n_mc=4000):
    """等距族 d2 = d1：直径/面积随 gamma (10..170, 步长10) 的蒙特卡洛表。"""
    print()
    print("=" * 78)
    print(f"Part B  等距族 (|GS2| = |GS1| = d1) 定位区域 vs 交会角 gamma，每格 {n_mc} 次采样")
    print("=" * 78)
    table = {}
    for d1 in (500.0, 1000.0, 1400.0):
        rows = []
        for gdeg in range(10, 180, 10):
            S1 = (0.0, 0.0)
            G = (d1, 0.0)
            S2 = s2_isoceles(d1, gdeg)
            L = math.hypot(*S2)
            st = mc_region_stats(S1, S2, G, n_mc, rng)
            st.update(gamma_deg=gdeg, d1=d1, d2=d1, L_S1S2=L, walk_s=L / SPEED,
                      S2=[S2[0], S2[1]], S2_in_area=bool(L <= R_AREA),
                      analytic_area=analytic_area(d1, d1, math.radians(gdeg)),
                      analytic_diam=analytic_diameter(d1, d1, math.radians(gdeg)))
            rows.append(st)
        table[str(int(d1))] = rows

        best_d = min(rows, key=lambda r: r["diam_mean"])
        best_a = min(rows, key=lambda r: r["area_mean"])
        print(f"\n--- d1 = {d1:.0f} m ---")
        print(f"{'gamma':>6} {'|S1S2|':>8} {'walk_s':>7} {'diam_mean':>10} {'diam_p90':>9} "
              f"{'diam_max':>9} {'diam_ana':>9} {'area_mean':>11} {'area_ana':>11} "
              f"{'D/Dmin':>7} {'A/Amin':>7} {'inArea':>7} {'G_in':>6}")
        for r in rows:
            print(f"{r['gamma_deg']:>6d} {r['L_S1S2']:>8.1f} {r['walk_s']:>7.1f} "
                  f"{r['diam_mean']:>10.2f} {r['diam_p90']:>9.2f} {r['diam_max']:>9.2f} "
                  f"{r['analytic_diam']:>9.2f} {r['area_mean']:>11.1f} {r['analytic_area']:>11.1f} "
                  f"{r['diam_mean']/best_d['diam_mean']:>7.3f} {r['area_mean']/best_a['area_mean']:>7.3f} "
                  f"{str(r['S2_in_area']):>7} {r['contain_frac']:>6.3f}")
        print(f"  实测最优：直径最小 gamma = {best_d['gamma_deg']}° (mean diam {best_d['diam_mean']:.2f} m)，"
              f"面积最小 gamma = {best_a['gamma_deg']}° (mean area {best_a['area_mean']:.1f} m^2)")
        acc_d = [r["gamma_deg"] for r in rows if r["diam_mean"] <= 2 * best_d["diam_mean"]]
        acc_a = [r["gamma_deg"] for r in rows if r["area_mean"] <= 2 * best_a["area_mean"]]
        print(f"  直径 <= 2x 最优的 gamma 网格点：{acc_d}")
        print(f"  面积 <= 2x 最优的 gamma 网格点：{acc_a}")
    return table


def s2_fixed_d2(d1, gamma_deg, d2, side=+1):
    """S1=(0,0), G=(d1,0)：S2 落在从 G 出发、与 GS1 夹角 gamma 的射线上，|GS2| = d2。"""
    g = math.radians(gamma_deg)
    ang = math.pi - side * g
    return (d1 + d2 * math.cos(ang), d2 * math.sin(ang))


def part_b2_fixed_d2(rng, d2=800.0, n_mc=2000):
    """固定 d2 = 800 m 族：gamma 10..170 步长 10 的直径/面积表（题目指定口径）。"""
    print()
    print("=" * 78)
    print(f"Part B2  固定 |GS2| = d2 = {d2:.0f} m，gamma 10°~170° 步长 10°，每格 {n_mc} 次采样")
    print("=" * 78)
    table = {}
    for d1 in (500.0, 1000.0, 1400.0):
        rows = []
        for gdeg in range(10, 180, 10):
            S1, G = (0.0, 0.0), (d1, 0.0)
            S2 = s2_fixed_d2(d1, gdeg, d2)
            L = math.hypot(*S2)
            st = mc_region_stats(S1, S2, G, n_mc, rng)
            st.update(gamma_deg=gdeg, d1=d1, d2=d2, L_S1S2=L, walk_s=L / SPEED,
                      S2=[S2[0], S2[1]], S2_in_area=bool(L <= R_AREA),
                      analytic_area=analytic_area(d1, d2, math.radians(gdeg)),
                      analytic_diam=analytic_diameter(d1, d2, math.radians(gdeg)))
            rows.append(st)
        table[str(int(d1))] = rows
        bd = min(rows, key=lambda r: r["diam_mean"])
        ba = min(rows, key=lambda r: r["area_mean"])
        print(f"\n--- d1 = {d1:.0f} m, d2 = {d2:.0f} m ---")
        print(f"{'gamma':>6} {'S2x':>9} {'S2y':>9} {'|S1S2|':>8} {'walk_s':>7} {'diam_mean':>10} "
              f"{'diam_ana':>9} {'area_mean':>11} {'area_ana':>11} {'D/Dmin':>7} {'A/Amin':>7} {'inArea':>7}")
        for r in rows:
            print(f"{r['gamma_deg']:>6d} {r['S2'][0]:>9.1f} {r['S2'][1]:>9.1f} {r['L_S1S2']:>8.1f} "
                  f"{r['walk_s']:>7.1f} {r['diam_mean']:>10.2f} {r['analytic_diam']:>9.2f} "
                  f"{r['area_mean']:>11.1f} {r['analytic_area']:>11.1f} "
                  f"{r['diam_mean']/bd['diam_mean']:>7.3f} {r['area_mean']/ba['area_mean']:>7.3f} "
                  f"{str(r['S2_in_area']):>7}")
        print(f"  实测最优：直径 gamma = {bd['gamma_deg']}° ({bd['diam_mean']:.2f} m)，"
              f"面积 gamma = {ba['gamma_deg']}° ({ba['area_mean']:.1f} m^2)")
        print(f"  直径<=2x 最优的 gamma：{[r['gamma_deg'] for r in rows if r['diam_mean'] <= 2*bd['diam_mean']]}")
        print(f"  面积<=2x 最优的 gamma：{[r['gamma_deg'] for r in rows if r['area_mean'] <= 2*ba['area_mean']]}")
    return table


def part_b_fine_thresholds(rng, d1=1000.0, n_mc=2500):
    """细分 gamma 网格，定出 “<=2x 最优” 的连续可接受区间端点（实测）。"""
    print()
    print("=" * 78)
    print(f"Part B'  d1 = {d1:.0f} m 细网格 (2°) 求 2 倍阈值的 gamma 边界，每格 {n_mc} 次采样")
    print("=" * 78)
    gs = list(range(4, 180, 2))
    diam, area = [], []
    for gdeg in gs:
        S1, G = (0.0, 0.0), (d1, 0.0)
        S2 = s2_isoceles(d1, gdeg)
        st = mc_region_stats(S1, S2, G, n_mc, rng)
        diam.append(st["diam_mean"])
        area.append(st["area_mean"])
    diam = np.array(diam); area = np.array(area); gs = np.array(gs, float)
    dmin, amin = float(np.nanmin(diam)), float(np.nanmin(area))
    gd = gs[diam <= 2 * dmin]
    ga = gs[area <= 2 * amin]
    print(f"  直径最小值 {dmin:.2f} m 出现在 gamma = {gs[int(np.argmin(diam))]:.0f}°")
    print(f"  面积最小值 {amin:.1f} m^2 出现在 gamma = {gs[int(np.argmin(area))]:.0f}°")
    print(f"  直径 <= 2x 最优区间（实测网格）：[{gd.min():.0f}°, {gd.max():.0f}°]")
    print(f"  面积 <= 2x 最优区间（实测网格）：[{ga.min():.0f}°, {ga.max():.0f}°]")
    return dict(gamma_grid=gs.tolist(), diam_mean=diam.tolist(), area_mean=area.tolist(),
                diam_min=dmin, area_min=amin,
                diam_band=[float(gd.min()), float(gd.max())],
                area_band=[float(ga.min()), float(ga.max())],
                gamma_star_diam=float(gs[int(np.argmin(diam))]),
                gamma_star_area=float(gs[int(np.argmin(area))]))


# ============================================================================ Part C
def part_c_degenerate(rng, d1=1000.0, n_mc=2000):
    """gamma -> 0（近共线，S2 在 S1 与 G 连线附近）：定位区域无界的概率。"""
    print()
    print("=" * 78)
    print(f"Part C  小交会角退化实验（d1 = {d1:.0f} m，每格 {n_mc} 次采样）")
    print("=" * 78)
    rows = []
    for gdeg in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0):
        S1, G = (0.0, 0.0), (d1, 0.0)
        S2 = s2_isoceles(d1, gdeg)
        st = mc_region_stats(S1, S2, G, n_mc, rng)
        st["gamma_deg"] = gdeg
        rows.append(st)
        print(f"  gamma = {gdeg:>4.1f}°  P(无界) = {st['unbounded_frac']:>6.3f}   "
              f"有界样本 diam_mean = {st['diam_mean']:>10.1f} m   area_mean = {st['area_mean']:>12.1f} m^2")
    print("  说明：|theta1 - theta2| < 2*eps = 2° 时两楔形方向锥重叠，交集沿公共方向无界，")
    print("        定位区域退化为无界域（直径无限），交会定位失效。")
    return rows


# ============================================================================ Part D
def part_d_beta_L(rng, n_mc=2500):
    """实际可控参数：偏移角 beta（相对测得示向度）与步行距离 L。"""
    print()
    print("=" * 78)
    print(f"Part D  可操作参数扫描：S2 = S1 + L*(方位 theta1 + beta)，每格 {n_mc} 次采样")
    print("=" * 78)
    out = {"beta_sweep": {}, "L_sweep": {}}

    print("\nD1  固定 L，扫描 beta（局部系：S1=(0,0)，示向度沿 +x，G=(d1,0)）")
    for d1 in (500.0, 1000.0, 1400.0):
        rows = []
        for L in (0.5 * d1, 1.0 * d1):
            for bdeg in range(15, 180, 15):
                b = math.radians(bdeg)
                S1, G = (0.0, 0.0), (d1, 0.0)
                S2 = (L * math.cos(b), L * math.sin(b))
                g = crossing_angle(S1, S2, G)
                d2 = math.hypot(G[0] - S2[0], G[1] - S2[1])
                st = mc_region_stats(S1, S2, G, n_mc, rng)
                st.update(d1=d1, L=L, beta_deg=bdeg, gamma_deg=math.degrees(g), d2=d2,
                          walk_s=L / SPEED, S2=[S2[0], S2[1]])
                rows.append(st)
        out["beta_sweep"][str(int(d1))] = rows
        print(f"\n--- d1 = {d1:.0f} m ---")
        print(f"{'L':>7} {'beta':>5} {'gamma':>7} {'d2':>8} {'diam_mean':>10} {'area_mean':>11} {'walk_s':>7}")
        for r in rows:
            print(f"{r['L']:>7.0f} {r['beta_deg']:>5d} {r['gamma_deg']:>7.2f} {r['d2']:>8.1f} "
                  f"{r['diam_mean']:>10.2f} {r['area_mean']:>11.1f} {r['walk_s']:>7.1f}")
        for L in (0.5 * d1, 1.0 * d1):
            sub = [r for r in rows if abs(r["L"] - L) < 1e-9]
            bb = min(sub, key=lambda r: r["diam_mean"])
            print(f"  L = {L:.0f} m 时直径最小的 beta = {bb['beta_deg']}° "
                  f"(gamma = {bb['gamma_deg']:.1f}°, diam {bb['diam_mean']:.2f} m)")

    print("\nD2  固定 beta = 90°（垂直示向度方向前进），扫描 L —— 精度与耗时权衡")
    for d1 in (500.0, 1000.0, 1400.0):
        rows = []
        for ratio in (0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            L = ratio * d1
            S1, G = (0.0, 0.0), (d1, 0.0)
            S2 = (0.0, L)
            g = crossing_angle(S1, S2, G)
            d2 = math.hypot(G[0] - S2[0], G[1] - S2[1])
            st = mc_region_stats(S1, S2, G, n_mc, rng)
            st.update(d1=d1, L=L, ratio=ratio, beta_deg=90, gamma_deg=math.degrees(g), d2=d2,
                      walk_s=L / SPEED, S2_in_area=bool(L <= R_AREA),
                      recv_guaranteed_1000=bool(d2 <= R_EFF_LO), recv_if_Reff_ge_d1=bool(d2 <= d1))
            rows.append(st)
        out["L_sweep"][str(int(d1))] = rows
        best = min(rows, key=lambda r: r["diam_mean"])
        print(f"\n--- d1 = {d1:.0f} m （beta = 90°） ---")
        print(f"{'L/d1':>6} {'L':>7} {'walk_s':>7} {'gamma':>7} {'d2':>8} {'diam_mean':>10} "
              f"{'area_mean':>11} {'D/Dmin':>7} {'d2<=1000':>9}")
        for r in rows:
            print(f"{r['ratio']:>6.2f} {r['L']:>7.0f} {r['walk_s']:>7.1f} {r['gamma_deg']:>7.2f} "
                  f"{r['d2']:>8.1f} {r['diam_mean']:>10.2f} {r['area_mean']:>11.1f} "
                  f"{r['diam_mean']/best['diam_mean']:>7.3f} {str(r['recv_guaranteed_1000']):>9}")
        print(f"  实测最优 L/d1 = {best['ratio']:.2f} (L = {best['L']:.0f} m, "
              f"gamma = {best['gamma_deg']:.1f}°, diam {best['diam_mean']:.2f} m, "
              f"步行 {best['walk_s']:.0f} s)")
        ok = [r["ratio"] for r in rows if r["diam_mean"] <= 1.25 * best["diam_mean"]]
        print(f"  直径 <= 1.25x 最优的 L/d1 网格点：{ok}")
    return out


# ============================================================================ Part E
@dataclass
class Scenario:
    name: str
    S1: tuple
    theta1_deg: float


def ray_circle_exit(S, u, R=R_AREA):
    """从 S 沿单位方向 u 射线到半径 R 圆边界的距离。"""
    b = S[0] * u[0] + S[1] * u[1]
    c = S[0] ** 2 + S[1] ** 2 - R ** 2
    disc = b * b - c
    if disc <= 0:
        return 0.0
    return max(0.0, -b + math.sqrt(disc))


def feasible_G_set(S1, theta1_deg, n_r=41, n_a=9):
    """
    与观测一致的干扰源可能位置集合 F：
      顶点 S1、中心方位 theta1、半张角 1° 的扇形，径向范围 [d_lo, Dmax]，
      Dmax = min(R_EFF_HI, 射线出界距离)（因为 S1 收到信号 => d1 <= R_eff <= 1500）。
    返回 (采样点数组, Dmax, d_lo)
    """
    d_lo = 5.0     # 距离 <=5 m 时无示向度读数，可直接光学定位，故不在讨论范围
    pts = []
    Dmax_dirs = []
    for adeg in np.linspace(theta1_deg - EPS_DEG, theta1_deg + EPS_DEG, n_a):
        a = math.radians(adeg)
        u = (math.cos(a), math.sin(a))
        Dmax_dirs.append(min(R_EFF_HI, ray_circle_exit(S1, u)))
    Dmax = float(min(Dmax_dirs))
    for adeg, dm in zip(np.linspace(theta1_deg - EPS_DEG, theta1_deg + EPS_DEG, n_a), Dmax_dirs):
        a = math.radians(adeg)
        for r in np.linspace(d_lo, dm, n_r):
            pts.append((S1[0] + r * math.cos(a), S1[1] + r * math.sin(a)))
    return np.array(pts), Dmax, d_lo


def part_e_candidate_region(scen: Scenario, gamma_lo, gamma_hi, step=20.0):
    """
    在 1800 m 圆域内网格搜索 S2 候选区域，三个条件：
      (a) |S2| <= R_AREA
      (b) 保证在 S2 能收到信号（对所有与观测一致的 G 都成立）
          b1 稳健版：max_{G in F} |S2 - G| <= R_EFF_LO = 1000（因为 R_eff >= 1000 必成立）
          b2 更弱的充分条件：|S2 - G| <= |S1 - G| 对所有 G in F 成立（S1 收到 => R_eff >= d1）
      (c) 交会角稳健：对所有 G in F，gamma(S1,S2,G) in [gamma_lo, gamma_hi]
    """
    S1 = scen.S1
    F, Dmax, d_lo = feasible_G_set(S1, scen.theta1_deg)
    xs = np.arange(-R_AREA, R_AREA + step, step)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)

    in_area = (P[:, 0] ** 2 + P[:, 1] ** 2) <= R_AREA ** 2

    # 到所有可行 G 的距离
    dif = P[:, None, :] - F[None, :, :]
    dist = np.sqrt((dif ** 2).sum(axis=2))              # (nP, nF)
    dmax_to_F = dist.max(axis=1)
    recv_robust = dmax_to_F <= R_EFF_LO                  # (b1)

    d1_arr = np.linalg.norm(F - np.array(S1)[None, :], axis=1)   # |S1 G|
    recv_closer = (dist <= d1_arr[None, :] + 1e-9).all(axis=1)   # (b2)

    # 交会角（向量化）：u1 = G - S1, u2 = G - S2
    u1 = F - np.array(S1)[None, :]
    u1n = u1 / np.linalg.norm(u1, axis=1, keepdims=True)
    u2 = -dif                                            # G - S2
    with np.errstate(invalid="ignore", divide="ignore"):
        u2n = u2 / dist[:, :, None]
    cosg = np.einsum("pfk,fk->pf", u2n, u1n)
    cosg = np.clip(cosg, -1.0, 1.0)
    gam = np.degrees(np.arccos(cosg))
    gmin = np.nanmin(gam, axis=1)
    gmax = np.nanmax(gam, axis=1)
    gamma_ok = (gmin >= gamma_lo) & (gmax <= gamma_hi)

    masks = {
        "in_area": in_area,
        "recv_robust": recv_robust & in_area,
        "recv_closer": recv_closer & in_area,
        "gamma_ok": gamma_ok & in_area,
        "candidate_strict": in_area & recv_robust & gamma_ok,
        "candidate_relaxed": in_area & (recv_robust | recv_closer) & gamma_ok,
    }
    cell = step * step
    info = {"scenario": scen.name, "S1": list(S1), "theta1_deg": scen.theta1_deg,
            "Dmax_d1": Dmax, "d_lo": d_lo, "grid_step": step,
            "gamma_band": [gamma_lo, gamma_hi],
            "areas_m2": {k: float(v.sum() * cell) for k, v in masks.items()}}

    cand = P[masks["candidate_strict"]]
    if len(cand):
        u = np.array([math.cos(math.radians(scen.theta1_deg)), math.sin(math.radians(scen.theta1_deg))])
        n = np.array([-u[1], u[0]])
        rel = cand - np.array(S1)[None, :]
        along = rel @ u
        perp = rel @ n
        L = np.linalg.norm(rel, axis=1)
        info["candidate_strict_extent"] = {
            "n_cells": int(len(cand)),
            "along_bearing_m": [float(along.min()), float(along.max())],
            "abs_perp_m": [float(np.abs(perp).min()), float(np.abs(perp).max())],
            "L_from_S1_m": [float(L.min()), float(L.max())],
            "walk_time_s": [float(L.min() / SPEED), float(L.max() / SPEED)],
        }
        # 推荐点：在 strict 候选区中最小化 max_{G} 定位区域面积近似，同时兼顾步行时间
        cand_idx = np.where(masks["candidate_strict"])[0]
        w1 = 2 * EPS * d1_arr[None, :]
        w2 = 2 * EPS * dist[cand_idx, :]
        sing = np.sin(np.radians(gam[cand_idx, :]))
        approx_area = w1 * w2 / np.maximum(sing, 1e-9)
        worst = approx_area.max(axis=1)
        med = np.median(approx_area, axis=1)
        best_worst = int(np.argmin(worst))
        best_med = int(np.argmin(med))
        info["recommended"] = {
            "minimax_area": {"S2": P[cand_idx[best_worst]].tolist(),
                             "worst_area_approx_m2": float(worst[best_worst]),
                             "median_area_approx_m2": float(med[best_worst]),
                             "L_m": float(np.linalg.norm(P[cand_idx[best_worst]] - np.array(S1))),
                             "walk_s": float(np.linalg.norm(P[cand_idx[best_worst]] - np.array(S1)) / SPEED)},
            "min_median_area": {"S2": P[cand_idx[best_med]].tolist(),
                                "worst_area_approx_m2": float(worst[best_med]),
                                "median_area_approx_m2": float(med[best_med]),
                                "L_m": float(np.linalg.norm(P[cand_idx[best_med]] - np.array(S1))),
                                "walk_s": float(np.linalg.norm(P[cand_idx[best_med]] - np.array(S1)) / SPEED)},
        }
    else:
        info["candidate_strict_extent"] = None
        info["recommended"] = None
    return info, P, masks, F, Dmax


def part_f_d1_bracket(scen: Scenario, gamma_lo, gamma_hi, brackets, step=20.0):
    """
    Part E 的 strict 规则要求 S2 对 d1 ∈ [5, 1500] 的**每个**候选 G 都在 1000 m 内，
    过于保守。实用做法：先用场强/两点粗测把 d1 收窄到 [dlo, dhi]，再算候选区。
    这里定量给出候选区面积与推荐 S2 随 d1 区间收窄的变化。
    """
    S1 = scen.S1
    xs = np.arange(-R_AREA, R_AREA + step, step)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    in_area = (P[:, 0] ** 2 + P[:, 1] ** 2) <= R_AREA ** 2
    cell = step * step
    out = []
    masks_by_bracket = {}
    for dlo, dhi in brackets:
        F = []
        for adeg in np.linspace(scen.theta1_deg - EPS_DEG, scen.theta1_deg + EPS_DEG, 9):
            a = math.radians(adeg)
            u = (math.cos(a), math.sin(a))
            dm = min(dhi, R_EFF_HI, ray_circle_exit(S1, u))
            if dm < dlo:
                continue
            for r in np.linspace(dlo, dm, 41):
                F.append((S1[0] + r * u[0], S1[1] + r * u[1]))
        F = np.array(F)
        dif = P[:, None, :] - F[None, :, :]
        dist = np.sqrt((dif ** 2).sum(axis=2))
        recv = dist.max(axis=1) <= R_EFF_LO
        d1_arr = np.linalg.norm(F - np.array(S1)[None, :], axis=1)
        closer = (dist <= d1_arr[None, :] + 1e-9).all(axis=1)
        u1 = F - np.array(S1)[None, :]
        u1n = u1 / np.linalg.norm(u1, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            u2n = (-dif) / dist[:, :, None]
        gam = np.degrees(np.arccos(np.clip(np.einsum("pfk,fk->pf", u2n, u1n), -1, 1)))
        gamma_ok = (np.nanmin(gam, axis=1) >= gamma_lo) & (np.nanmax(gam, axis=1) <= gamma_hi)
        cand = in_area & recv & gamma_ok
        rec = None
        if cand.any():
            idx = np.where(cand)[0]
            w1 = 2 * EPS * d1_arr[None, :]
            w2 = 2 * EPS * dist[idx, :]
            approx = w1 * w2 / np.maximum(np.sin(np.radians(gam[idx, :])), 1e-9)
            worst = approx.max(axis=1)
            k = int(np.argmin(worst))
            L = float(np.linalg.norm(P[idx[k]] - np.array(S1)))
            rec = dict(S2=P[idx[k]].tolist(), worst_area_approx_m2=float(worst[k]),
                       L_m=L, walk_s=L / SPEED)
        row = dict(d1_bracket=[dlo, dhi],
                   area_recv_km2=float(recv.sum() * cell) / 1e6,
                   area_closer_km2=float((closer & in_area).sum() * cell) / 1e6,
                   area_gamma_km2=float(gamma_ok.sum() * cell) / 1e6,
                   area_candidate_km2=float(cand.sum() * cell) / 1e6,
                   n_cells=int(cand.sum()), recommended=rec)
        if cand.any():
            rel = P[cand] - np.array(S1)[None, :]
            L = np.linalg.norm(rel, axis=1)
            row["L_range_m"] = [float(L.min()), float(L.max())]
        out.append(row)
        masks_by_bracket[f"{dlo:.0f}-{dhi:.0f}"] = cand
    return out, P, masks_by_bracket


def plot_candidate(scenarios_data, fname):
    ncol = len(scenarios_data)
    fig, axes = plt.subplots(1, ncol, figsize=(7.2 * ncol, 7.0))
    if ncol == 1:
        axes = [axes]
    for ax, (info, P, masks, F, Dmax, scen) in zip(axes, scenarios_data):
        ax.add_patch(Circle((0, 0), R_AREA, fill=False, ec="k", lw=1.4))
        sub = P[masks["gamma_ok"]]
        ax.scatter(sub[:, 0], sub[:, 1], s=3, c="#cfe6ff", label="robust crossing angle OK")
        sub = P[masks["recv_robust"]]
        ax.scatter(sub[:, 0], sub[:, 1], s=3, c="#ffe0b3", label="guaranteed reception (d<=1000 m)")
        sub = P[masks["candidate_strict"]]
        ax.scatter(sub[:, 0], sub[:, 1], s=6, c="#d62728", label="candidate region for $S_2$")
        ax.plot(F[:, 0], F[:, 1], color="#2ca02c", lw=1.0, alpha=0.35)
        u = np.array([math.cos(math.radians(scen.theta1_deg)), math.sin(math.radians(scen.theta1_deg))])
        ax.plot([scen.S1[0], scen.S1[0] + Dmax * u[0]], [scen.S1[1], scen.S1[1] + Dmax * u[1]],
                "g--", lw=1.6, label=r"measured bearing $\theta_1$ (source somewhere on it)")
        ax.plot(*scen.S1, "k*", ms=15, label="$S_1$")
        rec = info.get("recommended")
        if rec:
            s2 = rec["minimax_area"]["S2"]
            ax.plot(s2[0], s2[1], "bP", ms=13, label="recommended $S_2$ (minimax area)")
        ax.set_aspect("equal")
        ax.set_xlim(-1950, 1950)
        ax.set_ylim(-1950, 1950)
        ax.set_title(f"{scen.name}\n$S_1$={scen.S1}, "
                     r"$\theta_1$=" f"{scen.theta1_deg:.0f}$^\\circ$, "
                     f"$\\gamma\\in[{info['gamma_band'][0]:.0f}^\\circ,{info['gamma_band'][1]:.0f}^\\circ]$",
                     fontsize=11)
        ax.set_xlabel("x / m (East)")
        ax.set_ylabel("y / m (North)")
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"  图已保存: {fname}")


def plot_gamma_curves(table, fine, fname):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    colors = {"500": "#1f77b4", "1000": "#ff7f0e", "1400": "#2ca02c"}
    ax = axes[0]
    for k, rows in table.items():
        g = [r["gamma_deg"] for r in rows]
        ax.plot(g, [r["diam_mean"] for r in rows], "o-", color=colors[k], label=f"MC mean, $d_1$={k} m")
        ax.plot(g, [r["analytic_diam"] for r in rows], "--", color=colors[k], alpha=0.6,
                label=f"analytic, $d_1$={k} m")
    ax.set_xlabel(r"crossing angle $\gamma$ / deg")
    ax.set_ylabel("localization-region diameter / m")
    ax.set_yscale("log")
    ax.set_title(r"Diameter vs $\gamma$  ($|GS_2|=|GS_1|=d_1$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    for k, rows in table.items():
        g = [r["gamma_deg"] for r in rows]
        ax.plot(g, [r["area_mean"] for r in rows], "o-", color=colors[k], label=f"MC mean, $d_1$={k} m")
        ax.plot(g, [r["analytic_area"] for r in rows], "--", color=colors[k], alpha=0.6)
    ax.set_xlabel(r"crossing angle $\gamma$ / deg")
    ax.set_ylabel("localization-region area / m$^2$")
    ax.set_yscale("log")
    ax.set_title(r"Area vs $\gamma$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    g = np.array(fine["gamma_grid"])
    d = np.array(fine["diam_mean"]) / fine["diam_min"]
    a = np.array(fine["area_mean"]) / fine["area_min"]
    ax.plot(g, d, "-", label="diameter / min")
    ax.plot(g, a, "-", label="area / min")
    ax.axhline(2.0, color="r", ls=":", label="2x optimum")
    ax.axvspan(fine["diam_band"][0], fine["diam_band"][1], color="#1f77b4", alpha=0.12,
               label=f"diam band {fine['diam_band'][0]:.0f}-{fine['diam_band'][1]:.0f}$^\\circ$")
    ax.axvspan(fine["area_band"][0], fine["area_band"][1], color="#ff7f0e", alpha=0.10,
               label=f"area band {fine['area_band'][0]:.0f}-{fine['area_band'][1]:.0f}$^\\circ$")
    ax.set_ylim(0.8, 8)
    ax.set_xlabel(r"crossing angle $\gamma$ / deg")
    ax.set_ylabel("ratio to optimum")
    ax.set_title(r"Acceptable $\gamma$ band ($d_1$=1000 m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"  图已保存: {fname}")


def plot_tradeoff(dsweep, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    for k, rows in dsweep["L_sweep"].items():
        ax.plot([r["walk_s"] for r in rows], [r["diam_mean"] for r in rows], "o-", label=f"$d_1$={k} m")
        best = min(rows, key=lambda r: r["diam_mean"])
        ax.plot(best["walk_s"], best["diam_mean"], "k*", ms=14)
    ax.set_xlabel("walking time $L/5$ / s")
    ax.set_ylabel("mean diameter / m")
    ax.set_title(r"Accuracy vs travel cost ($\beta=90^\circ$; stars = optimum)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax = axes[1]
    for k, rows in dsweep["L_sweep"].items():
        ax.plot([r["ratio"] for r in rows], [r["gamma_deg"] for r in rows], "o-", label=f"$d_1$={k} m")
    ax.axhline(45, color="r", ls=":", label=r"$\gamma=45^\circ$ at $L=d_1$")
    ax.set_xlabel(r"$L/d_1$")
    ax.set_ylabel(r"crossing angle $\gamma$ / deg")
    ax.set_title(r"$\gamma=\arctan(L/d_1)$ for perpendicular move")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"  图已保存: {fname}")


# ============================================================================ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print(f"CUMCM2026 B题 问题2 蒙特卡洛验证   seed={SEED}  eps=±{EPS_DEG}°  R={R_AREA} m")

    a_rows = part_a_validation(rng)
    table = part_b_gamma_table(rng, n_mc=4000)
    table_d2 = part_b2_fixed_d2(rng, d2=800.0, n_mc=2000)
    fine = part_b_fine_thresholds(rng, d1=1000.0, n_mc=2500)
    c_rows = part_c_degenerate(rng, n_mc=2000)
    dsweep = part_d_beta_L(rng, n_mc=2500)

    print()
    print("=" * 78)
    print("Part E  第二检测点候选区域（网格计算）")
    print("=" * 78)
    gl, gh = fine["area_band"]
    gl, gh = float(max(20.0, gl)), float(min(160.0, gh))
    scens = [Scenario("S1 at area centre", (0.0, 0.0), 30.0),
             Scenario("S1 near boundary", (1200.0, 600.0), 200.0)]
    e_data, e_info = [], []
    for scen in scens:
        info, P, masks, F, Dmax = part_e_candidate_region(scen, gl, gh, step=20.0)
        e_info.append(info)
        e_data.append((info, P, masks, F, Dmax, scen))
        print(f"\n--- {scen.name}: S1={scen.S1}, theta1={scen.theta1_deg}°, "
              f"d1 可行上界 Dmax={info['Dmax_d1']:.1f} m ---")
        for k, v in info["areas_m2"].items():
            print(f"  面积[{k:>17}] = {v/1e6:>8.3f} km^2")
        ext = info["candidate_strict_extent"]
        if ext:
            print(f"  strict 候选区: {ext['n_cells']} 个 20m 网格, 沿示向度方向 "
                  f"[{ext['along_bearing_m'][0]:.0f}, {ext['along_bearing_m'][1]:.0f}] m, "
                  f"|垂直偏移| 最大 {ext['abs_perp_m'][1]:.0f} m, "
                  f"|S1S2| ∈ [{ext['L_from_S1_m'][0]:.0f}, {ext['L_from_S1_m'][1]:.0f}] m, "
                  f"步行 {ext['walk_time_s'][0]:.0f}~{ext['walk_time_s'][1]:.0f} s")
            rec = info["recommended"]["minimax_area"]
            print(f"  推荐 S2 (minimax 面积): {np.round(rec['S2'],1).tolist()}, "
                  f"L={rec['L_m']:.0f} m, 步行 {rec['walk_s']:.0f} s, "
                  f"最坏近似面积 {rec['worst_area_approx_m2']:.0f} m^2")
        else:
            print("  strict 候选区为空")

    # 用真实 MC 校验推荐点：在若干可能 d1 上评估
    print()
    print("=" * 78)
    print("Part E'  推荐 S2 的蒙特卡洛校验（对不同真实 d1，每格 2000 次采样）")
    print("=" * 78)
    verify = []
    for info, scen in zip(e_info, scens):
        rec = info.get("recommended")
        if not rec:
            continue
        S2 = tuple(rec["minimax_area"]["S2"])
        u = (math.cos(math.radians(scen.theta1_deg)), math.sin(math.radians(scen.theta1_deg)))
        print(f"\n--- {scen.name}: 推荐 S2 = {np.round(S2,1).tolist()} ---")
        print(f"{'d1':>7} {'gamma':>7} {'d2':>8} {'diam_mean':>10} {'diam_p90':>9} {'area_mean':>11} "
              f"{'d2<=1000':>9} {'G_in':>6}")
        for d1 in (200.0, 500.0, 800.0, 1000.0, 1200.0, info["Dmax_d1"]):
            G = (scen.S1[0] + d1 * u[0], scen.S1[1] + d1 * u[1])
            g = math.degrees(crossing_angle(scen.S1, S2, G))
            d2 = math.hypot(G[0] - S2[0], G[1] - S2[1])
            st = mc_region_stats(scen.S1, S2, G, 2000, rng)
            st.update(scenario=scen.name, d1=d1, gamma_deg=g, d2=d2, S2=list(S2))
            verify.append(st)
            print(f"{d1:>7.0f} {g:>7.2f} {d2:>8.1f} {st['diam_mean']:>10.2f} {st['diam_p90']:>9.2f} "
                  f"{st['area_mean']:>11.1f} {str(d2<=R_EFF_LO):>9} {st['contain_frac']:>6.3f}")

    print()
    print("=" * 78)
    print("Part F  d1 未知性的处理：把 d1 收窄到区间后候选区如何变化")
    print("=" * 78)
    brackets = [(5.0, 1500.0), (5.0, 1000.0), (300.0, 800.0), (600.0, 1000.0),
                (800.0, 1200.0), (900.0, 1100.0), (1200.0, 1500.0)]
    f_rows, PF, maskF = part_f_d1_bracket(scens[0], gl, gh, brackets, step=20.0)
    print(f"  情形: S1={scens[0].S1}, theta1={scens[0].theta1_deg}°, "
          f"gamma 稳健带 [{gl:.0f}°, {gh:.0f}°]")
    print(f"{'d1 bracket':>16} {'A_recv/km2':>11} {'A_gamma/km2':>12} {'A_cand/km2':>11} "
          f"{'cells':>6} {'L range/m':>16} {'rec S2':>18} {'walk_s':>7}")
    for r in f_rows:
        rec = r["recommended"]
        lr = r.get("L_range_m")
        s_lr = "[%.0f,%.0f]" % (lr[0], lr[1]) if lr else "-"
        s_s2 = str([int(v) for v in rec["S2"]]) if rec else "-"
        s_walk = ("%.0f" % rec["walk_s"]) if rec else "-"
        s_br = "[%.0f, %.0f]" % (r["d1_bracket"][0], r["d1_bracket"][1])
        print(f"{s_br:>16} {r['area_recv_km2']:>11.3f} {r['area_gamma_km2']:>12.3f} "
              f"{r['area_candidate_km2']:>11.3f} {r['n_cells']:>6d} "
              f"{s_lr:>16} {s_s2:>18} {s_walk:>7}")
    print("  注：'recv' 列用的是最坏情况规则 max_G |S2-G| <= 1000 m（R_eff 下界），")
    print("      即无论 R_eff 取 [1000,1500] 中哪个值、G 在可行集何处，S2 都必然收到信号。")

    plot_gamma_curves(table, fine, os.path.join(FIGURES, "b_q2_gamma_effect.png"))
    plot_tradeoff(dsweep, os.path.join(FIGURES, "b_q2_accuracy_cost_tradeoff.png"))
    plot_candidate(e_data, os.path.join(FIGURES, "b_q2_candidate_region.png"))

    payload = dict(meta=dict(seed=SEED, eps_deg=EPS_DEG, R_area=R_AREA,
                             R_eff_lo=R_EFF_LO, R_eff_hi=R_EFF_HI, speed=SPEED,
                             runtime_s=time.time() - t0),
                   part_a_validation=a_rows, part_b_gamma_table=table,
                   part_b2_fixed_d2_table=table_d2,
                   part_b_fine=fine, part_c_degenerate=c_rows,
                   part_d_sweeps=dsweep, part_e_regions=e_info, part_e_verify=verify,
                   part_f_d1_brackets=f_rows)
    outp = os.path.join(RESULTS, "b_q2_results.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n结果已保存: {outp}")

    # CSV：主表
    csvp = os.path.join(RESULTS, "b_q2_gamma_table.csv")
    with open(csvp, "w", encoding="utf-8") as f:
        f.write("d1_m,gamma_deg,L_S1S2_m,walk_s,diam_mean_m,diam_median_m,diam_p90_m,diam_max_m,"
                "diam_analytic_m,area_mean_m2,area_median_m2,area_analytic_m2,"
                "S2_in_area,contain_frac,unbounded_frac\n")
        for k, rows in table.items():
            for r in rows:
                f.write(f"{r['d1']:.0f},{r['gamma_deg']},{r['L_S1S2']:.2f},{r['walk_s']:.2f},"
                        f"{r['diam_mean']:.4f},{r['diam_median']:.4f},{r['diam_p90']:.4f},"
                        f"{r['diam_max']:.4f},{r['analytic_diam']:.4f},{r['area_mean']:.3f},"
                        f"{r['area_median']:.3f},{r['analytic_area']:.3f},"
                        f"{int(r['S2_in_area'])},{r['contain_frac']:.4f},{r['unbounded_frac']:.4f}\n")
    print(f"结果已保存: {csvp}")
    print(f"总耗时 {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
