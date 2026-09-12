"""从脱敏官方日志生成论文表 1（题目规定的四列口径）。

赛日 Q3/Q4 各 3 局，人工抄六行数字进论文极易出错，而且抄完无法追溯到日志。
本脚本直接读 official_runs/*.sanitized.json 的 paper_table1_row 字段生成表格，
并同时算三次均值与池化指标（池化 = 总虚拟时间 / 总清除数，与 arena 评分口径一致）。

用法：
    python -m src.build_table1                 # 默认只取 formal
    python -m src.build_table1 --kind practice # 用演练局验证格式
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "official_runs"

DASH = "—"


def load(kind: str) -> dict[str, list[dict]]:
    """按题号分组读取脱敏日志。directional=True 归 Q4，否则 Q3。"""
    groups: dict[str, list[dict]] = {"Q3": [], "Q4": []}
    for path in sorted(RUNS.glob("*.sanitized.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if doc.get("test_kind") != kind:
            continue
        row = doc.get("paper_table1_row")
        if not row:
            continue
        # 作废局不入表：/enter 被拒（accepted=false）的那局清除数与虚拟时间都是 0，
        # 计进均值会把成绩拉低。9-12 演练第一局队号年份打成 2025 就属这种。
        if not row.get("清除干扰源个数") and not (doc.get("summary") or {}).get("virtual_time_s"):
            continue
        summary = doc.get("summary", {})
        groups["Q4" if doc.get("directional") else "Q3"].append({
            "file": path.name,
            "case": row.get("测试案例编码") or DASH,
            "cleared": row.get("清除干扰源个数"),
            "total": row.get("干扰源总数_人工抄录"),
            "ratio": row.get("被清除干扰源个数的比例"),
            "tbar": row.get("平均定位清除时间_s"),
            "wall": row.get("程序运行时间_s"),
            "virtual": summary.get("virtual_time_s"),
            "profile": summary.get("strategy_profile"),
            "outer": (summary.get("search_layout") or {}).get("outer_radius_m"),
        })
    return groups


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return DASH
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(title: str, rows: list[dict]) -> list[str]:
    """一题一张表，四列严格按题目表 1 口径，附三次均值与池化指标。"""
    out = [f"### {title}", ""]
    out.append("| 测试案例编码 | 清除干扰源个数 | 平均定位清除时间 $\\bar T$ / s | 程序运行时间 / s（上限 1200） |")
    out.append("|---|---|---|---|")
    if not rows:
        for i in (1, 2, 3):
            out.append(f"| 正式测试 {i}（待填） | {DASH} | {DASH} | {DASH} |")
        out.append(f"| **三次均值** | {DASH} | {DASH} | {DASH} |")
        out += ["", "> 尚无该类型日志，表为骨架。"]
        return out
    for row in rows:
        out.append(
            f"| {row['case']} | {fmt(row['cleared'], 0)} "
            f"| {fmt(row['tbar'], 2)} | {fmt(row['wall'], 2)} |"
        )
    cleared = [r["cleared"] for r in rows if r["cleared"] is not None]
    tbars = [r["tbar"] for r in rows if r["tbar"] is not None]
    walls = [r["wall"] for r in rows if r["wall"] is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    out.append(
        f"| **{len(rows)} 次均值** | {fmt(mean(cleared), 1)} "
        f"| {fmt(mean(tbars), 2)} | {fmt(mean(walls), 2)} |"
    )
    return out


def pooled(rows: list[dict]) -> str:
    """池化指标：总虚拟时间 / 总清除数，与 arena 评分口径一致（不是均值的均值）。"""
    vs = [r["virtual"] for r in rows if r["virtual"] is not None]
    cs = [r["cleared"] for r in rows if r["cleared"] is not None]
    if not vs or not cs or sum(cs) == 0:
        return DASH
    return f"{sum(vs) / sum(cs):.2f}"


def audit(rows: list[dict]) -> list[str]:
    """把每行溯源到日志文件，并核对策略版本字段，避免把旧版本成绩当本版。"""
    if not rows:
        return []
    out = ["", "**数据溯源与版本核对**", ""]
    out.append("| 日志文件 | strategy_profile | outer_radius_m | 清除 | 虚拟时间 / s |")
    out.append("|---|---|---|---|---|")
    for row in rows:
        out.append(
            f"| `{row['file']}` | {row['profile'] or DASH} "
            f"| {fmt(row['outer'], 1)} | {fmt(row['cleared'], 0)} "
            f"| {fmt(row['virtual'], 1)} |"
        )
    profiles = {r["profile"] for r in rows if r["profile"]}
    if len(profiles) > 1:
        out += ["", f"> ⚠ 这些局横跨多个策略版本 {sorted(profiles)}，"
                    "不能合并报同一成绩。"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="从脱敏官方日志生成论文表 1")
    ap.add_argument("--kind", choices=["formal", "practice"], default="formal",
                    help="取哪种测试的日志，默认 formal")
    ap.add_argument("--out", default=None,
                    help="写入文件；省略则打印到标准输出")
    args = ap.parse_args()

    groups = load(args.kind)
    lines = [f"# 官方测试结果表（表 1 口径，来源 test_kind={args.kind}）", ""]
    lines += ["评分指标 $\\bar T=$ 总虚拟时间 / 已清除个数（s，越小越好）。", ""]
    for key, title in (("Q3", "表 1-Q3　问题 3（全向源）"),
                       ("Q4", "表 1-Q4　问题 4（含定向源）")):
        rows = groups[key]
        lines += table(title, rows)
        if rows:
            lines += ["", f"池化 $\\bar T$ = **{pooled(rows)}** s"
                      f"（总虚拟时间 / 总清除数，{len(rows)} 局）"]
            lines += audit(rows)
        lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out:
        dest = ROOT / args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        n3, n4 = len(groups["Q3"]), len(groups["Q4"])
        print(f"已写入 {dest.relative_to(ROOT)}（Q3 {n3} 局 / Q4 {n4} 局）")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
