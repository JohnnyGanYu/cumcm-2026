#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUMCM 2026 B题 问题2 论文配图（设计口径对齐版）。

本脚本重绘 Q2 的三张论文配图，使其与 design.md §3.2 / §4.4 与
`b_sim.geometry.second_point_candidates` 的**实现口径**严格一致：

    * 偏移角集合 offsets = {32°, 16°, 0°}（design §3.2/§4.4，R2.1/R2.4）；
    * 前进长度 length = 380 m（±1 m 容差，R2.1）；
    * 交会角良态带 α ∈ [60°, 120°]（sin α ≥ √3/2，design §3.2，R2.2）；
    * 接收半径下限 d2 ≤ 1000 m、越界约束 ‖p2‖ ≤ 1800 m（R2.2/R2.7）；
    * 候选区域 = θ1 ± 32° 前向楔形 ∩ α∈[60,120] ∩ d2≤1000 ∩ ‖p2‖≤1800 的
      环状扇形带（R2.7），由 `geometry.in_second_point_region` 判定。

与 `src/b_q2.py`（蒙特卡洛解析验证，保留不动）的区别：本脚本**不做 MC**，
而是直接调用 `b_sim` 的产品级几何函数，把"策略实际生成的候选序列"叠加到
候选区域上，确保论文图与提交给模拟器的机器狗程序口径一致。

输出（覆盖 figures/ 下同名文件）：
    figures/b_q2_candidate_region.png
    figures/b_q2_gamma_effect.png
    figures/b_q2_accuracy_cost_tradeoff.png

坐标系：目标区域为以原点为心、半径 1800 m 的圆域，x 正东、y 正北，
角度以度计（atan2(dy,dx) 口径，逆时针为正）。示向度误差 ε = 1°。

可从仓库根目录运行：
    python -m src.b_q2_figures     或     python src/b_q2_figures.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无显示环境下的后端。
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

# --- 允许 `python src/b_q2_figures.py` 直接运行时找到 b_sim 包 -----------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from b_sim.geometry import (  # noqa: E402
    ARENA_RADIUS,
    CROSSING_ANGLE_MAX_DEG,
    CROSSING_ANGLE_MIN_DEG,
    RECEP_LOWER_M,
    SECOND_POINT_WEDGE_HALF_DEG,
    crossing_angle_deg,
    in_second_point_region,
    second_point_candidates,
)
from b_sim.strategy import lateral_error_at  # noqa: E402

# ------------------------------------------------------------------ 常量（设计口径）
EPS_DEG = 1.0                      # 示向度误差绝对上界（度）。
EPS = math.radians(EPS_DEG)
OFFSETS = (32.0, 16.0, 0.0)        # 偏移角集合 δ（design §3.2/§4.4，R2.1）。
LENGTH_M = 380.0                   # 前进长度 L（m）（R2.1）。
SPEED = 5.0                        # 机器狗速度 m/s。
ALPHA_MIN = CROSSING_ANGLE_MIN_DEG  # 60°
ALPHA_MAX = CROSSING_ANGLE_MAX_DEG  # 120°
ALPHA_STAR = 90.0                  # 交会角最优值。

FIGURES = os.path.join(_ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)


# --------------------------------------------------------------------- 通用几何工具
def analytic_diameter(d1: float, d2: float, alpha_rad: float, eps: float = EPS) -> float:
    """一阶误差传播下定位区域直径（平行四边形长对角线）。

    两条 ±ε 楔形的窄条宽度 w1 = 2εd1、w2 = 2εd2，交会角 α：
        diam ≈ sqrt(w1² + w2² + 2 w1 w2 |cos α|) / sin α。
    """
    w1, w2 = 2 * eps * d1, 2 * eps * d2
    s = abs(math.sin(alpha_rad))
    return math.sqrt(w1 * w1 + w2 * w2 + 2 * w1 * w2 * abs(math.cos(alpha_rad))) / max(s, 1e-12)


def spread_width(d1: float, d2: float, alpha_rad: float, eps: float = EPS) -> float:
    """沿平分线方向的展宽 ≈ (d1 + d2)·ε / sin α（design §3.2 的一阶传播式）。"""
    return (d1 + d2) * eps / max(abs(math.sin(alpha_rad)), 1e-12)


# ============================================================ 图 1：候选区域 + 候选序列
def _candidate_region_mask(S1, theta1_deg, source, xs, ys):
    """在网格上逐点调用 `in_second_point_region`，返回布尔掩码（设计口径 R2.7）。"""
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            mask[iy, ix] = in_second_point_region(
                (xv, yv), S1, theta1_deg, source
            )
    return mask


def _panel_candidate(ax, S1, theta1_deg, d1, title):
    """单个子图：楔形、候选区域、second_point_candidates 序列、逐点有效性。"""
    theta1 = math.radians(theta1_deg)
    # 估计源：沿示向度前进 d1（论文里 d1 为“对源距离的当前估计”）。
    source = (S1[0] + d1 * math.cos(theta1), S1[1] + d1 * math.sin(theta1))

    # 竞技场圆边界。
    ax.add_patch(Circle((0, 0), ARENA_RADIUS, fill=False, ec="k", lw=1.4, zorder=1))

    # θ1 ± 32° 前向楔形（候选区域的角边界，R2.7）。
    ax.add_patch(
        Wedge(
            S1,
            LENGTH_M * 1.35,
            theta1_deg - SECOND_POINT_WEDGE_HALF_DEG,
            theta1_deg + SECOND_POINT_WEDGE_HALF_DEG,
            width=None,
            facecolor="#e8f0ff",
            edgecolor="#6f9bd8",
            lw=1.0,
            alpha=0.55,
            zorder=1.5,
            label=r"forward wedge $\theta_1\pm32^\circ$",
        )
    )

    # 候选区域掩码（网格判定 in_second_point_region，R2.7）。
    span = LENGTH_M + 260.0
    xs = np.arange(S1[0] - span, S1[0] + span, 8.0)
    ys = np.arange(S1[1] - span, S1[1] + span, 8.0)
    mask = _candidate_region_mask(S1, theta1_deg, source, xs, ys)
    X, Y = np.meshgrid(xs, ys)
    ax.scatter(
        X[mask], Y[mask], s=3.0, c="#d62728", alpha=0.5, zorder=2,
        label=r"candidate region ($\alpha\in[60,120]$, $d_2\leq1000$)",
    )

    # 示向度射线（源在此射线附近；到有效接收上界 1500 m 与出界距离取小）。
    ray_len = min(1500.0, d1 + 400.0)
    ax.plot(
        [S1[0], S1[0] + ray_len * math.cos(theta1)],
        [S1[1], S1[1] + ray_len * math.sin(theta1)],
        "g--", lw=1.6, zorder=2.5,
        label=r"measured bearing $\theta_1\pm1^\circ$",
    )
    # ±1° 误差扇（源可行方向锥）。
    for s in (-1.0, +1.0):
        a = theta1 + s * EPS
        ax.plot(
            [S1[0], S1[0] + ray_len * math.cos(a)],
            [S1[1], S1[1] + ray_len * math.sin(a)],
            color="#2ca02c", lw=0.7, alpha=0.6, zorder=2.3,
        )

    # second_point_candidates 实际生成的候选序列（offsets={32,16,0}, L=380）。
    cands = second_point_candidates(S1, theta1_deg, offsets=OFFSETS, length=LENGTH_M)
    signed = []
    for delta in sorted(OFFSETS, reverse=True):
        if delta > 1e-9:
            signed.extend((delta, -delta))
        else:
            signed.append(0.0)
    first_valid = True
    first_invalid = True
    for (p2, s_off) in zip(cands, signed):
        inside = in_second_point_region(p2, S1, theta1_deg, source)
        alpha = crossing_angle_deg(source, S1, p2)
        if inside:
            lbl = "candidate $p_2$ (valid)" if first_valid else None
            first_valid = False
            ax.plot(p2[0], p2[1], "o", ms=9, mfc="#1f77b4", mec="k", mew=0.8, zorder=4, label=lbl)
        else:
            lbl = "candidate $p_2$ (rejected)" if first_invalid else None
            first_invalid = False
            ax.plot(p2[0], p2[1], "X", ms=9, mfc="#bbbbbb", mec="k", mew=0.8, zorder=4, label=lbl)
        ax.annotate(
            f"$\\delta$={s_off:+.0f}°\n$\\alpha$={alpha:.0f}°",
            (p2[0], p2[1]), textcoords="offset points", xytext=(6, 6),
            fontsize=7.0, zorder=5,
        )
        ax.plot([S1[0], p2[0]], [S1[1], p2[1]], color="#1f77b4", lw=0.6, alpha=0.5, zorder=3)

    ax.plot(*S1, "k*", ms=16, zorder=6, label="$p_1$ (first point)")
    ax.plot(*source, "^", ms=10, mfc="#ff7f0e", mec="k", zorder=6,
            label=f"estimated source ($d_1$={d1:.0f} m)")

    ax.set_aspect("equal")
    m = 260.0
    ax.set_xlim(S1[0] - LENGTH_M - m, S1[0] + LENGTH_M + m)
    ax.set_ylim(S1[1] - LENGTH_M - m, S1[1] + LENGTH_M + m)
    ax.set_title(
        f"{title}\n$p_1$=({S1[0]:.0f},{S1[1]:.0f}), "
        r"$\theta_1$=" f"{theta1_deg:.0f}$^\\circ$, "
        r"offsets $\{32,16,0\}^\circ$, $L$=380 m",
        fontsize=10.5,
    )
    ax.set_xlabel("x / m (East)")
    ax.set_ylabel("y / m (North)")
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.92)
    ax.grid(alpha=0.25)


def plot_candidate_region(fname):
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.4))
    # d1≈350 m 使 δ=32°、δ=16° 候选的交会角均落入 [60,120]（L=380 m 下的良态区间）。
    _panel_candidate(axes[0], (0.0, 0.0), 30.0, 350.0, "S1 at area centre")
    _panel_candidate(axes[1], (900.0, 500.0), 205.0, 350.0, "S1 near boundary")
    fig.suptitle(
        r"Q2 second detection point: candidate region ($\theta_1\pm32^\circ$ wedge "
        r"$\cap\ \alpha\in[60^\circ,120^\circ]\cap d_2\leq1000\,$m$\cap\ \|p_2\|\leq1800\,$m)"
        r" and the $\{32,16,0\}^\circ$/380 m candidate sequence",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"  图已保存: {fname}")


# ============================================================ 图 2：交会角对定位精度的影响
def plot_gamma_effect(fname):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    alphas = np.arange(5.0, 176.0, 1.0)
    colors = {500.0: "#1f77b4", 1000.0: "#ff7f0e", 1400.0: "#2ca02c"}

    # --- 子图 1：定位区域直径 vs α（等距族 d2 = d1） ---
    ax = axes[0]
    for d1 in (500.0, 1000.0, 1400.0):
        diam = [analytic_diameter(d1, d1, math.radians(a)) for a in alphas]
        ax.plot(alphas, diam, "-", color=colors[d1], lw=1.8, label=f"$d_1=d_2$={d1:.0f} m")
    ax.axvspan(ALPHA_MIN, ALPHA_MAX, color="#2ca02c", alpha=0.14,
               label=r"design band $\alpha\in[60,120]$")
    ax.axvline(ALPHA_STAR, color="k", ls=":", lw=1.2, label=r"$\alpha^\star=90^\circ$")
    ax.set_yscale("log")
    ax.set_xlabel(r"crossing angle $\alpha$ / deg")
    ax.set_ylabel("localization-region diameter / m")
    ax.set_title(r"Diameter vs $\alpha$  ($d_2=d_1$, $\varepsilon=1^\circ$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # --- 子图 2：沿平分线展宽 (d1+d2)ε/sinα vs α ---
    ax = axes[1]
    for d1 in (500.0, 1000.0, 1400.0):
        w = [spread_width(d1, d1, math.radians(a)) for a in alphas]
        ax.plot(alphas, w, "-", color=colors[d1], lw=1.8, label=f"$d_1=d_2$={d1:.0f} m")
    ax.axvspan(ALPHA_MIN, ALPHA_MAX, color="#2ca02c", alpha=0.14,
               label=r"design band $\alpha\in[60,120]$")
    ax.axvline(ALPHA_STAR, color="k", ls=":", lw=1.2, label=r"$\alpha^\star=90^\circ$")
    ax.set_yscale("log")
    ax.set_xlabel(r"crossing angle $\alpha$ / deg")
    ax.set_ylabel(r"bisector spread $(d_1{+}d_2)\varepsilon/\sin\alpha$ / m")
    ax.set_title(r"Error propagation $\propto 1/\sin\alpha$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # --- 子图 3：相对最优的放大倍数 1/sinα，标出 α∈[60,120] 的封顶 ---
    ax = axes[2]
    amp = 1.0 / np.sin(np.radians(alphas))
    ax.plot(alphas, amp, "-", color="#d62728", lw=2.0, label=r"$1/\sin\alpha$ (rel. to $\alpha^\star$)")
    cap = 1.0 / math.sin(math.radians(ALPHA_MIN))  # = 2/√3 ≈ 1.1547
    ax.axhline(cap, color="b", ls="--", lw=1.2,
               label=r"cap at band edge $=2/\sqrt{3}\approx1.155$")
    ax.axvspan(ALPHA_MIN, ALPHA_MAX, color="#2ca02c", alpha=0.16,
               label=r"design band $\alpha\in[60,120]$")
    ax.axvline(ALPHA_STAR, color="k", ls=":", lw=1.2, label=r"$\alpha^\star=90^\circ$")
    ax.set_ylim(0.9, 4.0)
    ax.set_xlabel(r"crossing angle $\alpha$ / deg")
    ax.set_ylabel(r"error amplification $1/\sin\alpha$")
    ax.set_title(r"Why $[60,120]$: amplification $\leq 1.155\times$ optimum")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"  图已保存: {fname}")


# ============================================================ 图 3：精度—耗时权衡（含 L=380）
def plot_accuracy_cost_tradeoff(fname):
    """
    局部系：p1=(0,0)、示向度沿 +x、估计源 G=(d1,0)。
    第二检测点沿 δ 偏移前进 L（不同 L/d1）。绘制：
      左：平均直径 vs 步行时间 L/5，标出设计 L=380 m 工作点；
      右：交会角 α 随 L（固定偏移 δ）变化，标出 α∈[60,120] 带与 L=380。
    这里 δ 取实现口径的最大偏移 32°（第二检测点候选序列的首选）。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    colors = {500.0: "#1f77b4", 1000.0: "#ff7f0e", 1400.0: "#2ca02c"}
    delta = 32.0  # 首选偏移角（design/实现口径）。
    Ls = np.arange(60.0, 1200.0, 20.0)

    def geom_for(d1, L, ddeg):
        """给定 d1、步长 L、偏移 δ：返回 (α度, d2, diam)。"""
        phi = math.radians(ddeg)
        p1 = (0.0, 0.0)
        G = (d1, 0.0)
        p2 = (L * math.cos(phi), L * math.sin(phi))
        a = crossing_angle_deg(G, p1, p2)
        d2 = math.hypot(p2[0] - G[0], p2[1] - G[1])
        diam = analytic_diameter(d1, d2, math.radians(a))
        return a, d2, diam

    # --- 左：精度 vs 步行时间 ---
    ax = axes[0]
    for d1 in (500.0, 1000.0, 1400.0):
        walk = Ls / SPEED
        diam = [geom_for(d1, L, delta)[2] for L in Ls]
        ax.plot(walk, diam, "-", color=colors[d1], lw=1.6, label=f"$d_1$={d1:.0f} m")
        # 设计工作点 L = 380 m。
        a380, d2_380, diam380 = geom_for(d1, LENGTH_M, delta)
        ax.plot(LENGTH_M / SPEED, diam380, "P", ms=13, mfc="#d62728", mec="k",
                zorder=5, label=None)
    ax.axvline(LENGTH_M / SPEED, color="#d62728", ls="--", lw=1.3,
               label=r"design $L=380$ m ($t=76$ s)")
    ax.set_xlabel("walking time $L/5$ / s")
    ax.set_ylabel("localization-region diameter / m")
    ax.set_yscale("log")
    ax.set_title(r"Accuracy vs travel cost ($\delta=32^\circ$; marker = design $L$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # --- 右：交会角 α vs L（固定 δ=32°），α∈[60,120] 带 + L=380 ---
    ax = axes[1]
    for d1 in (500.0, 1000.0, 1400.0):
        a = [geom_for(d1, L, delta)[0] for L in Ls]
        ax.plot(Ls, a, "-", color=colors[d1], lw=1.6, label=f"$d_1$={d1:.0f} m")
        a380 = geom_for(d1, LENGTH_M, delta)[0]
        ax.plot(LENGTH_M, a380, "P", ms=13, mfc="#d62728", mec="k", zorder=5)
    ax.axhspan(ALPHA_MIN, ALPHA_MAX, color="#2ca02c", alpha=0.14,
               label=r"good band $\alpha\in[60,120]$")
    ax.axvline(LENGTH_M, color="#d62728", ls="--", lw=1.3, label=r"design $L=380$ m")
    ax.set_xlabel("forward length $L$ / m")
    ax.set_ylabel(r"crossing angle $\alpha$ / deg")
    ax.set_title(r"Crossing angle grows with $L$ (fixed $\delta=32^\circ$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"  图已保存: {fname}")


def main():
    print("CUMCM2026 B题 问题2 论文配图（设计口径对齐版）")
    print(f"  offsets={OFFSETS}  length={LENGTH_M} m  α∈[{ALPHA_MIN:.0f},{ALPHA_MAX:.0f}]  "
          f"d2≤{RECEP_LOWER_M:.0f} m  ‖p2‖≤{ARENA_RADIUS:.0f} m  ε=±{EPS_DEG:.0f}°")
    # 打印一份候选序列取值样例，便于论文正文引用（数字由本脚本实际产生）。
    S1, theta1 = (0.0, 0.0), 30.0
    d1 = 350.0
    source = (S1[0] + d1 * math.cos(math.radians(theta1)),
              S1[1] + d1 * math.sin(math.radians(theta1)))
    cands = second_point_candidates(S1, theta1, offsets=OFFSETS, length=LENGTH_M)
    signed = []
    for de in sorted(OFFSETS, reverse=True):
        if de > 1e-9:
            signed.extend((de, -de))
        else:
            signed.append(0.0)
    print(f"  样例：p1={S1}, θ1={theta1}°, 估计源距 d1={d1:.0f} m")
    for (p2, s_off) in zip(cands, signed):
        a = crossing_angle_deg(source, S1, p2)
        d2 = math.hypot(p2[0] - source[0], p2[1] - source[1])
        inside = in_second_point_region(p2, S1, theta1, source)
        print(f"    δ={s_off:+5.0f}°  p2=({p2[0]:8.1f},{p2[1]:8.1f})  "
              f"α={a:6.1f}°  d2={d2:7.1f} m  in_region={inside}")
    print(f"  逼近横向误差样例：e(1000)={lateral_error_at(1000.0):.2f} m, "
          f"e(500)={lateral_error_at(500.0):.2f} m, e(100)={lateral_error_at(100.0):.2f} m")

    plot_candidate_region(os.path.join(FIGURES, "b_q2_candidate_region.png"))
    plot_gamma_effect(os.path.join(FIGURES, "b_q2_gamma_effect.png"))
    plot_accuracy_cost_tradeoff(os.path.join(FIGURES, "b_q2_accuracy_cost_tradeoff.png"))
    print("完成。")


if __name__ == "__main__":
    main()
