"""官方日志时间预算分解：定位优化杠杆在哪。

**已被 src/b_tsp_bound.py 取代，占比数字不要再引用。** 本脚本的成本模型有两处错：
  1. 漏了切频道的 +1 s（四局各 538–591 次，约 560 s/局 = 虚拟时间的 3.4%）；
  2. 成功清除按 3 s 计，实为 5 s。
因此它报的 移动 81.0% / 检测 18.3% 偏了，正确值为 移动 77.4% / 检测 21.7%。
成本模型已由 b_tsp_bound.py 用逐调用残差反解并对账（误差 < 3.2e-6 s）。
保留本文件仅为历史可追溯；新分析请用：
    python -m src.b_tsp_bound

用法（不推荐）：
    python -m src.b_time_budget official_runs/q4_practice_run*.sanitized.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

MEASURE_COST_S = 5.0
CLEAR_COST_S = 3.0


def analyze(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    log = d.get("protocol_log", [])
    s = d.get("summary", {})

    n_measure = sum(1 for e in log if e.get("path") == "/measure")
    clears = [e for e in log if e.get("path") == "/clear"]
    n_clear = len(clears)

    done: set[int] = set()
    miss_before = miss_after = success = 0
    for e in clears:
        ch = (e.get("request") or {}).get("channel")
        res = (e.get("response") or {}).get("clear_result")
        if res == "success":
            success += 1
            done.add(ch)
        elif ch in done:
            miss_after += 1
        else:
            miss_before += 1

    virt = float(s.get("virtual_time_s", 0.0))
    t_measure = n_measure * MEASURE_COST_S
    t_clear = n_clear * CLEAR_COST_S
    t_move = virt - t_measure - t_clear
    cleared = s.get("cleared_count", 0) or 1

    return {
        "file": path.name,
        "cleared_count": s.get("cleared_count"),
        "virtual_time_s": round(virt, 1),
        "mean_clear_time_s": round(virt / cleared, 1),
        "measure_calls": n_measure,
        "clear_calls": n_clear,
        "clear_success": success,
        "clear_miss_before_success": miss_before,
        "clear_miss_after_success": miss_after,
        "t_measure_s": round(t_measure, 1),
        "t_clear_s": round(t_clear, 1),
        "t_move_s": round(t_move, 1),
        "pct_measure": round(100 * t_measure / virt, 1) if virt else None,
        "pct_clear": round(100 * t_clear / virt, 1) if virt else None,
        "pct_move": round(100 * t_move / virt, 1) if virt else None,
        "saving_if_no_waste_s": round(miss_after * CLEAR_COST_S, 1),
        "mean_after_fix_s": round((virt - miss_after * CLEAR_COST_S) / cleared, 1),
    }


def main(argv: list[str]) -> int:
    patterns = argv[1:] or ["official_runs/*.sanitized.json"]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in sorted(glob.glob(pat)))
    if not paths:
        print("no input files")
        return 1

    rows = [analyze(p) for p in paths]
    hdr = f"{'file':34s} {'src':>4s} {'meas%':>6s} {'clr%':>5s} {'move%':>6s} {'mean_t':>8s} {'fixed':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['file']:34s} {r['cleared_count']:4d} {r['pct_measure']:6.1f} "
              f"{r['pct_clear']:5.1f} {r['pct_move']:6.1f} "
              f"{r['mean_clear_time_s']:8.1f} {r['mean_after_fix_s']:8.1f}")

    n = len(rows)
    print(f"\n均值: 检测 {sum(r['pct_measure'] for r in rows)/n:.1f}%  "
          f"清除 {sum(r['pct_clear'] for r in rows)/n:.1f}%  "
          f"移动 {sum(r['pct_move'] for r in rows)/n:.1f}%")
    print(f"消除成功后浪费清除的收益: {sum(r['mean_clear_time_s'] for r in rows)/n:.1f} -> "
          f"{sum(r['mean_after_fix_s'] for r in rows)/n:.1f} s "
          f"({100*(sum(r['mean_after_fix_s'] for r in rows)-sum(r['mean_clear_time_s'] for r in rows))/sum(r['mean_clear_time_s'] for r in rows):+.1f}%)")

    out = Path("results/b_time_budget.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
