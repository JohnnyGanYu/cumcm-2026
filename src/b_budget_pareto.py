"""汇总检测预算扫描结果，画 Pareto 曲线（投入 vs 评分指标 / 清除率）。

读取 results/b_budget_*.json，输出：
  figures/b_budget_pareto.png
  results/b_budget_pareto_summary.json  （含推荐参数与盈亏平衡分析）
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

VIRTUAL_BUDGET_S = 360000.0


def load(name: str) -> dict | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def rows_of(name: str) -> list[dict]:
    d = load(name)
    return d["rows"] if d else []


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    summary: dict = {"virtual_budget_s": VIRTUAL_BUDGET_S, "sets": {}}

    sets = {
        "q3_final": ("b_budget_final_q3.json", "Q3 omni  (300 eps, seed 9000+)"),
        "q4_final": ("b_budget_final_q4.json", "Q4 directional (300 eps, seed 9000+)"),
        "q4_multiplier": ("b_budget_mult_q4.json", "Q4 budget 2x/3x/5x (40 eps)"),
        "q4_expand": ("b_budget_expand_q4.json", "Q4 expansion knobs (120 eps, seed 7000+)"),
        "q3_holdout": ("b_budget_holdout_q3.json", "Q3 holdout (200 eps, seed 7000+)"),
        "q4_holdout": ("b_budget_holdout_q4.json", "Q4 holdout (200 eps, seed 7000+)"),
        "q3_lean": ("b_budget_lean_q3.json", "Q3 lean scan (40 eps)"),
    }

    for key, (fname, label) in sets.items():
        rs = [r for r in rows_of(fname) if r.get("metric_mean_s") is not None]
        if not rs:
            continue
        base = next((r for r in rs if r["name"] in ("baseline", "x1_baseline")), rs[0])
        entries = []
        for r in rs:
            entries.append({
                "name": r["name"],
                "overrides": r["overrides"],
                "metric_mean_s": r["metric_mean_s"],
                "metric_delta_pct": round(100.0 * (r["metric_mean_s"] - base["metric_mean_s"])
                                          / base["metric_mean_s"], 2),
                "cleared_ratio_mean": r["cleared_ratio_mean"],
                "cleared_total": r["cleared_total"],
                "jammers_total": r["jammers_total"],
                "full_clear_rate": r["full_clear_rate"],
                "measure_calls_mean": r["measure_calls_mean"],
                "virtual_time_mean_s": r["virtual_time_mean_s"],
                "virtual_budget_used_pct": round(100.0 * r["virtual_time_mean_s"] / VIRTUAL_BUDGET_S, 2),
                "wall_s_max": r["wall_s_max"],
            })
        entries.sort(key=lambda e: e["metric_mean_s"])
        summary["sets"][key] = {
            "label": label,
            "baseline_metric_s": base["metric_mean_s"],
            "baseline_break_even_s": round(base["virtual_time_mean_s"] / base["cleared_per_episode"], 1),
            "best": entries[0],
            "entries": entries,
        }

    # ---- 图：投入(检测次数) vs 评分指标，点大小=清除率缺口 ----
    panels = [k for k in ("q3_final", "q4_final", "q4_multiplier") if k in summary["sets"]]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.0))
        if len(panels) == 1:
            axes = [axes]
        for ax, key in zip(axes, panels):
            s = summary["sets"][key]
            es = s["entries"]
            x = [e["measure_calls_mean"] for e in es]
            y = [e["metric_mean_s"] for e in es]
            miss = [(1.0 - e["cleared_ratio_mean"]) * 100 for e in es]
            sc = ax.scatter(x, y, c=miss, cmap="coolwarm", s=110,
                            edgecolors="black", linewidths=0.6, zorder=3, vmin=0)
            for e in es:
                ax.annotate(e["name"].replace("lean_", "").replace("_", " "),
                            (e["measure_calls_mean"], e["metric_mean_s"]),
                            fontsize=7, xytext=(4, 4), textcoords="offset points")
            bx = next((e for e in es if e["name"] in ("baseline", "x1_baseline")), None)
            if bx:
                ax.axhline(bx["metric_mean_s"], ls="--", c="gray", lw=1,
                           label=f"baseline {bx['metric_mean_s']:.0f}s")
                ax.legend(fontsize=8, loc="best")
            ax.set_xlabel("measure calls per episode  (search investment)")
            ax.set_ylabel("score metric  mean_clear_time_s  (lower = better)")
            ax.set_title(s["label"], fontsize=10)
            ax.grid(alpha=0.3, zorder=0)
            plt.colorbar(sc, ax=ax, label="miss rate (%)")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        fig.suptitle("Detection budget vs score metric: expansion does NOT help "
                     "(clear rate already saturated); optimum lies toward LESS search",
                     fontsize=12)
        fig.tight_layout()
        out = FIGURES / "b_budget_pareto.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")

    (RESULTS / "b_budget_pareto_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS / 'b_budget_pareto_summary.json'}")

    for key, s in summary["sets"].items():
        print(f"\n=== {s['label']} ===")
        print(f"  baseline={s['baseline_metric_s']}s  盈亏平衡(每多清1源可花)={s['baseline_break_even_s']}s")
        for e in s["entries"][:6]:
            print(f"  {e['name']:<22} metric={e['metric_mean_s']:>8.1f} ({e['metric_delta_pct']:+.2f}%) "
                  f"clr={e['cleared_total']}/{e['jammers_total']} "
                  f"meas={e['measure_calls_mean']:>6.1f} used={e['virtual_budget_used_pct']}%")


if __name__ == "__main__":
    main()
