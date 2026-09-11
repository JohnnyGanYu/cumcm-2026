#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B 题问题 1 论文配图：定位多边形 + 直径 AB + 外接圆（以直径为直径）覆盖/不覆盖两例。

对应设计 §3.1 / §4.2 与需求 R1.5 / R1.7 / R1.8：

  * 定位区域 = 各示向度楔形（±1° 半平面交）之交所得凸多边形。
  * 区域直径 D = 顶点对最大欧氏距离，端点 A、B；中点 M=(A+B)/2。
  * 外接圆 = 以 AB 为直径、M 为圆心、半径 D/2 的圆。
  * 覆盖裕度 margin = D/2 − max_{v} ‖v − M‖；
        covers ⟺ margin ≥ −1e-7（等价 max‖v−M‖ ≤ D/2 + 1e-7）。
    直径端点 A、B 本身恒在 D/2 处，故 margin ≤ 0：
        - margin ≈ 0  → 覆盖（最远顶点即直径端点）；
        - margin < 0  → **不覆盖**（存在顶点落在圆外，如正三角形型区域）。

生成两张图到 figures/：
  * b_q1_localization_diameter_covered.png —— 由两条示向度楔形交会得到的凸定位
    多边形（钝角三角形型），直径圆恰好覆盖（margin ≈ 0，最远顶点即 A/B）。
  * b_q1_covering_counterexample.png —— 正三角形型区域，第三顶点距中点
    ≈0.866L > L/2 落在直径圆外（margin < 0，不覆盖）；标注 Jung 上界 r*≤D/√3。

图形标注使用英文/LaTeX（与既有 figures/b_q2_*.png 一致，规避 CJK 字体缺失）。
以非交互后端 Agg 输出。可从仓库根目录以 `python -m src.b_q1_figures` 运行。
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")  # 非交互后端：无显示环境下也能落盘 PNG。
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from src.b_sim.geometry import (
    Bearing,
    circle_on_diameter_covers,
    localization_polygon,
    polygon_diameter,
    wedge_half_planes,
)

# figures/ 目录（相对仓库根：src 的上一级下的 figures）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_REPO_ROOT, "figures")

# ε = 1° 的示向度误差绝对上界（楔形半张角）。
ERROR_DEG = 1.0


def _wedge_edges(bearing: Bearing, length: float) -> tuple[list[float], list[float], list[float], list[float]]:
    """返回一条观测楔形两条边界射线的端点，供绘制楔形边（示意 ±1° 张角）。

    返回 ``(lo_x, lo_y, hi_x, hi_y)``，各为 ``[起点, 终点]`` 两点坐标列表。
    """
    lo = math.radians(bearing.svd_deg - ERROR_DEG)
    hi = math.radians(bearing.svd_deg + ERROR_DEG)
    lo_x = [bearing.x, bearing.x + length * math.cos(lo)]
    lo_y = [bearing.y, bearing.y + length * math.sin(lo)]
    hi_x = [bearing.x, bearing.x + length * math.cos(hi)]
    hi_y = [bearing.y, bearing.y + length * math.sin(hi)]
    return lo_x, lo_y, hi_x, hi_y


def _draw_region(
    ax,
    polygon: list[tuple[float, float]],
    *,
    title: str,
    covered_color: str,
    wedges: list[tuple[Bearing, float]] | None = None,
    jung: bool = False,
) -> tuple[float, bool, float]:
    """在坐标轴 ``ax`` 上绘制定位多边形、直径 AB、以直径为直径的外接圆及标注。

    返回 ``(D, covers, margin)``。
    """
    diameter, a, b = polygon_diameter(polygon)
    covers, margin = circle_on_diameter_covers(polygon)
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    radius = diameter / 2.0
    worst = max(math.hypot(x - mid[0], y - mid[1]) for x, y in polygon)

    # 可选：观测楔形边界射线（示意 ±1° 交会）。
    if wedges:
        for i, (bearing, length) in enumerate(wedges):
            lo_x, lo_y, hi_x, hi_y = _wedge_edges(bearing, length)
            lbl = "wedge $\\pm 1^\\circ$" if i == 0 else None
            ax.plot(lo_x, lo_y, color="0.6", lw=0.8, ls="--", zorder=1)
            ax.plot(hi_x, hi_y, color="0.6", lw=0.8, ls="--", zorder=1, label=lbl)
            ax.plot([bearing.x], [bearing.y], marker="^", color="0.4", ms=7, zorder=3)
            ax.annotate(
                f"$p_{i + 1}$", (bearing.x, bearing.y),
                textcoords="offset points", xytext=(6, 6), fontsize=9, color="0.3",
            )

    # 定位多边形（凸）。
    patch = Polygon(
        polygon, closed=True, facecolor=covered_color, edgecolor="navy",
        alpha=0.35, lw=1.6, zorder=2, label="localization region $P$",
    )
    ax.add_patch(patch)

    # 外接圆（以直径 AB 为直径）。
    circ = Circle(
        mid, radius, fill=False, edgecolor="crimson", lw=1.8, ls="-",
        zorder=4, label="circle on diameter ($r=D/2$)",
    )
    ax.add_patch(circ)

    # 直径 AB。
    ax.plot([a[0], b[0]], [a[1], b[1]], color="darkgreen", lw=2.0, zorder=5,
            label=f"diameter $AB$, $D={diameter:.2f}$ m")
    ax.plot([a[0], b[0]], [a[1], b[1]], "o", color="darkgreen", ms=6, zorder=6)
    ax.annotate("$A$", a, textcoords="offset points", xytext=(6, 6), fontsize=11, color="darkgreen")
    ax.annotate("$B$", b, textcoords="offset points", xytext=(6, 6), fontsize=11, color="darkgreen")
    ax.plot([mid[0]], [mid[1]], "x", color="crimson", ms=9, mew=2, zorder=6)
    ax.annotate("$M$", mid, textcoords="offset points", xytext=(6, -12), fontsize=11, color="crimson")

    # 标出"最远顶点"（决定覆盖与否的顶点）。
    fx, fy = max(polygon, key=lambda v: math.hypot(v[0] - mid[0], v[1] - mid[1]))
    ax.plot([mid[0], fx], [mid[1], fy], color="crimson", lw=1.0, ls=":", zorder=5)
    ax.plot([fx], [fy], "s", color="crimson", ms=7, zorder=7)
    # 偏移方向：顶点在中点右侧则标注向右，否则向左，避免与标题/边界冲突。
    dx_off = 10 if fx >= mid[0] else -90
    ax.annotate(
        f"farthest vertex\n$\\|v-M\\|={worst:.2f}$ m",
        (fx, fy), textcoords="offset points", xytext=(dx_off, -28), fontsize=8.5, color="crimson",
    )

    verdict = "COVERS" if covers else "does NOT cover"
    txt = (
        f"$D={diameter:.2f}$ m,  $r=D/2={radius:.2f}$ m\n"
        f"$\\max\\|v-M\\|={worst:.2f}$ m\n"
        f"margin $=D/2-\\max\\|v-M\\|={margin:.2f}$ m\n"
        f"$\\Rightarrow$ diameter-circle {verdict}"
    )
    if jung:
        txt += f"\nJung bound: $r^*\\leq D/\\sqrt{{3}}={diameter / math.sqrt(3):.2f}$ m"
    box_color = "#e8f5e9" if covers else "#ffebee"
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor=box_color, edgecolor="0.6", alpha=0.95),
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x / m (East)")
    ax.set_ylabel("y / m (North)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    return diameter, covers, margin


def make_covered_figure(path: str) -> tuple[float, bool, float]:
    """由两条示向度楔形交会得到的凸定位多边形，直径圆恰好覆盖（margin ≈ 0）。

    构造：估计源 ``s≈(500,300)``，两个检测点使视线交会角落在良态区间；两条 ±1° 楔形
    之交为一个狭长凸多边形。其直径端点即最远顶点，故 ``max‖v−M‖=D/2``、margin≈0，
    直径圆覆盖整个区域（凸区域宽度方向被 D/2 半径充分包住）。
    """
    src = (500.0, 300.0)

    def bearing_to_source(px: float, py: float) -> float:
        return math.degrees(math.atan2(src[1] - py, src[0] - px))

    p1 = (-800.0, 300.0)   # 视线大致朝东
    p2 = (100.0, -900.0)   # 视线大致朝北偏东 → 与视线 1 形成良态交会角
    bearings = [
        Bearing(p1[0], p1[1], bearing_to_source(*p1)),
        Bearing(p2[0], p2[1], bearing_to_source(*p2)),
    ]
    polygon = localization_polygon(bearings, error_deg=ERROR_DEG)
    if len(polygon) < 3:
        raise RuntimeError("covered 例的定位多边形退化，无法绘制")

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    # 楔形边只画到定位区域附近（近处放大观察），故只在小图里另作示意；主图聚焦区域。
    result = _draw_region(
        ax, polygon,
        title="Q1 (covered): localization polygon & its diameter-circle\n"
              "two $\\pm1^\\circ$ bearing wedges $\\Rightarrow$ convex region, "
              "diameter-circle covers ($margin\\approx0$)",
        covered_color="#66bb6a",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return result


def make_counterexample_figure(path: str) -> tuple[float, bool, float]:
    """正三角形型定位区域：第三顶点距直径中点 ≈0.866L > L/2，落在直径圆外（不覆盖）。

    直接以边长 ``L`` 的正三角形顶点作为定位区域（凸多边形），说明"以区域直径为直径的
    圆不一定覆盖该区域"这一结论（设计 §3.1 反例族）。数值：``D = L``，
    ``max‖v−M‖ = L·√3/2 ≈ 0.866L``，margin ``= L/2 − 0.866L = -0.366L < 0``。
    Jung 平面上界 ``r* ≤ D/√3 ≈ 0.577D``（正三角形取到）为最紧上界。
    """
    L = 100.0
    polygon = [
        (0.0, 0.0),
        (L, 0.0),
        (L / 2.0, L * math.sqrt(3.0) / 2.0),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    result = _draw_region(
        ax, polygon,
        title="Q1 (counter-example): equilateral region\n"
              "third vertex at $0.866L > L/2$ lies OUTSIDE the diameter-circle "
              "($margin<0$)",
        covered_color="#ef5350",
        jung=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return result


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    covered_path = os.path.join(FIGURES_DIR, "b_q1_localization_diameter_covered.png")
    counter_path = os.path.join(FIGURES_DIR, "b_q1_covering_counterexample.png")

    d1, cov1, m1 = make_covered_figure(covered_path)
    print(
        f"[covered]        {covered_path}\n"
        f"                 D={d1:.4f} m, covers={cov1}, margin={m1:.4f} m"
    )
    d2, cov2, m2 = make_counterexample_figure(counter_path)
    print(
        f"[counter-example]{counter_path}\n"
        f"                 D={d2:.4f} m, covers={cov2}, margin={m2:.4f} m "
        f"(=-0.366L, third vertex at 0.866L)"
    )
    print("\n两张 Q1 配图已写入 figures/。")


if __name__ == "__main__":
    main()
