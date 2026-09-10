"""覆盖站位设计配图（Task 10.3）。

由 ``coverage_design`` 的几何内核生成论文用图，写入 ``figures/``：

* ``figures/b_coverage_stations.png`` —— 主图：原点 + 1250 m × 12（13 站位）
  布局散点、每站 1000 m 接收下限覆盖圆、竞技场边界，并高亮最坏覆盖间隙点。
* ``figures/b_coverage_worstgap.png`` —— 辅图：各候选布局最坏覆盖间隙条形图，
  标注 1000 m 保证发现下限线。

数据口径与 ``results/b_coverage_design.json`` 完全一致（同一函数计算）。
从仓库根运行::

    python -m src.b_coverage_figures
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境，必须在 pyplot 之前设定
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .b_sim.arena import ARENA_RADIUS, RECEPTION_MIN_M
from .b_sim.coverage_design import ring, worst_coverage_gap

# 若环境有 CJK 字体则优先使用；否则图内文字回退为英文，避免出现缺字方块。
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def _has_cjk_font() -> bool:
    """检测是否存在可渲染中文的字体，据此决定图内标签语言。"""
    from matplotlib.font_manager import findSystemFonts, FontProperties

    wanted = {"noto sans cjk", "wenquanyi", "simhei", "microsoft yahei",
              "source han", "droid sans fallback"}
    for path in findSystemFonts():
        try:
            name = FontProperties(fname=path).get_name().lower()
        except Exception:
            continue
        if any(w in name for w in wanted):
            return True
    return False


_CJK = _has_cjk_font()


def _t(zh: str, en: str) -> str:
    """按字体可用性选择中文或英文标签（数值与几何完全一致）。"""
    return zh if _CJK else en

FIG_DIR = Path("figures")
RESULTS = Path("results/b_coverage_design.json")


def _chosen_stations() -> list[tuple[float, float]]:
    """论文选定布局：原点 + 1250 m 环 12 站位（13 站位）。"""
    return [(0.0, 0.0)] + ring(1250.0, 12)


def plot_coverage_stations(out: Path) -> Path:
    """主图：选定布局散点 + 1000 m 覆盖圆 + 最坏间隙点。"""
    stations = _chosen_stations()
    gap, argmax = worst_coverage_gap(stations)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # 竞技场边界（半径 1800 m 圆域）
    ax.add_patch(Circle((0, 0), ARENA_RADIUS, fill=False, ec="#333333", lw=1.6, ls="--",
                        label=_t(f"竞技场边界 R={ARENA_RADIUS:.0f} m",
                                 f"Arena boundary R={ARENA_RADIUS:.0f} m")))

    # 每站 1000 m（接收下限）覆盖圆
    for k, (px, py) in enumerate(stations):
        ax.add_patch(Circle((px, py), RECEPTION_MIN_M, fill=True, ec="#1f77b4",
                            fc="#1f77b4", alpha=0.06, lw=0.8,
                            label=_t("覆盖圆 (r=1000 m)", "Coverage disk (r=1000 m)")
                            if k == 0 else None))

    # 站位散点：原点与环站位分色
    ox, oy = stations[0]
    ring_pts = stations[1:]
    ax.scatter([ox], [oy], c="#d62728", s=90, marker="*", zorder=5,
               label=_t("原点站位 (0,0)", "Origin station (0,0)"))
    ax.scatter([p[0] for p in ring_pts], [p[1] for p in ring_pts],
               c="#1f77b4", s=48, marker="o", zorder=5,
               label=_t("1250 m 环站位 × 12", "Ring stations @1250 m x12"))

    # 最坏覆盖间隙点及其到最近站位的连线
    wx, wy = argmax
    nearest = min(stations, key=lambda s: math.hypot(wx - s[0], wy - s[1]))
    ax.plot([wx, nearest[0]], [wy, nearest[1]], color="#ff7f0e", lw=2.0, zorder=6)
    ax.scatter([wx], [wy], c="#ff7f0e", s=110, marker="X", zorder=7,
               label=_t(f"最坏覆盖间隙点 g={gap:.1f} m",
                        f"Worst-gap point g={gap:.1f} m"))

    ax.set_aspect("equal")
    lim = ARENA_RADIUS * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(_t(
        "覆盖站位布局：原点 + 1250 m × 12（13 站位）\n"
        f"最坏覆盖间隙 g = {gap:.1f} m < 1000 m ⇒ 保证发现任一全向源",
        "Coverage layout: origin + 1250 m ring x12 (13 stations)\n"
        f"worst gap g = {gap:.1f} m < 1000 m => detection guaranteed"))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_worst_gap_bars(out: Path) -> Path:
    """辅图：各候选布局最坏覆盖间隙条形图（数据取自 JSON）。"""
    if not RESULTS.exists():
        raise FileNotFoundError(
            f"{RESULTS} 不存在，请先运行 `python -m src.b_sim.coverage_design`。"
        )
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = payload["coverage"]
    # 布局名 ASCII 化，避免无 CJK 字体时出现缺字方块（数值不变）。
    def _ascii_layout(name: str) -> str:
        return name.replace("仅原点", "origin-only").replace("原点", "origin")
    labels = [f"{(r['layout'] if _CJK else _ascii_layout(r['layout']))}\n"
              f"({r['stations']})" for r in rows]
    gaps = [r["worst_gap_m"] for r in rows]
    colors = ["#2ca02c" if r["guarantees_detection"] else "#d62728" for r in rows]
    chosen = "原点+1250x12"

    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar(range(len(rows)), gaps, color=colors, alpha=0.85)
    for i, r in enumerate(rows):
        if r["layout"] == chosen:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2.2)
    ax.axhline(RECEPTION_MIN_M, color="#1f77b4", ls="--", lw=1.6,
               label=_t("接收半径下限 1000 m（保证发现门限）",
                        "Reception lower bound 1000 m (detection threshold)"))
    for i, g in enumerate(gaps):
        ax.text(i, g + 20, f"{g:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel(_t("最坏覆盖间隙 g / m", "Worst coverage gap g / m"))
    ax.set_title(_t(
        "候选布局最坏覆盖间隙对比（绿=保证发现，红=否；黑框=选定布局）",
        "Worst coverage gap by layout (green=guaranteed, red=not; black edge=chosen)"))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p1 = plot_coverage_stations(FIG_DIR / "b_coverage_stations.png")
    print(f"wrote {p1}")
    p2 = plot_worst_gap_bars(FIG_DIR / "b_coverage_worstgap.png")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
