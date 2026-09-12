"""论文引用的文件必须存在，7.2 节成绩必须能从逐局记录复算。

竞赛规则：支撑材料与论文不符可能被取消评奖资格。引用一个不存在的
结果文件、或写一个无法从记录复算的池化值，都属于这一类风险。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "main.md"
RESULTS = ROOT / "results"


def _md() -> str:
    return PAPER.read_text(encoding="utf-8")


def test_all_cited_files_exist():
    """反引号里的仓库路径与 markdown 链接必须都能落地。"""
    md = _md()
    cands = set(re.findall(r"`([A-Za-z0-9_./\-]+\.(?:json|py|md|csv|zip|txt))`", md))
    cands |= set(re.findall(r"\]\(([A-Za-z0-9_./\-]+\.md)\)", md))
    assert cands, "没抽到任何文件引用，正则可能失效"
    missing = [c for c in sorted(cands)
               if not (ROOT / c).exists() and not (ROOT / "paper" / c).exists()]
    assert not missing, f"论文引用了不存在的文件: {missing}"


def _pooled(records, variant):
    t = c = 0.0
    for r in records:
        if r.get("variant") != variant:
            continue
        t += r.get("total_virtual_time_s") or 0.0
        c += r.get("cleared") or 0
    return t / c if c else None


def test_holdout_summary_recomputes_from_records():
    """summary 的池化值必须能从逐局记录独立复算出来。"""
    for name in ("q3_200_holdout_random.json",
                 "q4_410_holdout_fixed.json",
                 "q4_410_holdout_random.json"):
        d = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        recs = d["records"]
        for arm in ("baseline", "candidate"):
            got = _pooled(recs, arm)
            want = d["summary"][arm]["pooled_s"]
            assert got is not None, f"{name}:{arm} 复算不出池化值"
            assert abs(got - want) < 0.01, (
                f"{name}:{arm} 池化值与逐局记录不一致: 复算 {got:.4f} vs summary {want:.4f}"
            )


def test_worst_case_claims_match_records():
    """7.2 节引用的最差单局必须等于记录里的最大值。"""
    md = _md()
    for name, arm, want in (("q4_410_holdout_fixed.json", "baseline", 832.33),
                            ("q4_410_holdout_fixed.json", "candidate", 672.69),
                            ("q4_410_holdout_random.json", "baseline", 753.60),
                            ("q4_410_holdout_random.json", "candidate", 604.55)):
        d = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        vals = [r["mean_clear_time_s"] for r in d["records"]
                if r.get("variant") == arm and r.get("mean_clear_time_s")]
        assert abs(max(vals) - want) < 0.01, (
            f"{name}:{arm} 最差单局为 {max(vals):.2f}，论文写 {want}"
        )
        assert f"{want:.2f}" in md, f"论文未引用 {name}:{arm} 的最差单局 {want}"


def test_outer_radius_stress_numbers():
    """外环 1800 vs 1870 压力样本的关键数字必须与结果文件一致。"""
    d = json.loads((RESULTS / "b_outer_radius_boundary.json").read_text(encoding="utf-8"))
    s = d["summary"]
    r1800 = s["R1800"]
    assert r1800["cleared"] == 899 and r1800["total"] == 960, "1800 档清除数变了"
    assert r1800["full_clear_episodes"] == 0, "1800 档不该有满清局"
    assert abs(r1800["mean_T"] - 417.53) < 0.01, "1800 档平均 T 变了"
    md = _md()
    assert "899/960" in md, "论文未写 1800 档的漏清事实"
    assert "417.53" in md, "论文未引用 1800 档平均 T"
