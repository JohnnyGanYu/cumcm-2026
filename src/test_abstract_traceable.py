"""摘要里的成绩数字必须能在结果文件中溯源。

摘要是评审第一眼看的地方，写了无出处的数字风险最大。曾出现两次：
一是把 7.1 节的历史对照（324.7/1070.9/1142.5 s）当成最终配置成绩，
且比真实留出集差 2.4 倍、清除率还低报；二是引用一轮结果文件未入库的
120 局配对（694.77 -> 489.74 s）。两处都已按 holdout 文件改写。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "main.md"
RESULTS = ROOT / "results"

HOLDOUT = {
    "q3_200_holdout_random.json": (241.24, 237.47),
    "q4_410_holdout_fixed.json": (500.48, 454.49),
    "q4_410_holdout_random.json": (497.78, 454.93),
}


def _abstract() -> str:
    t = PAPER.read_text(encoding="utf-8")
    i, j = t.find("## 摘要"), t.find("## 1 问题重述")
    assert i >= 0 and j > i, "找不到摘要段"
    return t[i:j]


def test_holdout_numbers_in_abstract_match_files():
    """摘要引用的留出集数字必须等于 JSON 里的池化值（两位小数）。"""
    abstract = _abstract()
    for name, (base, cand) in HOLDOUT.items():
        d = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        s = d["summary"]
        assert round(s["baseline"]["pooled_s"], 2) == base, f"{name} 基线池化变了"
        assert round(s["candidate"]["pooled_s"], 2) == cand, f"{name} 候选池化变了"
        assert f"{cand:.2f}" in abstract, (
            f"摘要未引用 {name} 的候选池化 {cand:.2f} —— 摘要与留出集数据脱节"
        )


def test_abstract_has_no_untraceable_legacy_numbers():
    """三个已确认无出处的数字不得再出现在摘要里。"""
    abstract = _abstract()
    for bad in ("324.7", "1070.9", "1142.5", "694.77", "489.74", "1541/1543"):
        assert bad not in abstract, (
            f"摘要出现无法溯源的旧数字 {bad}；正文历史对照可保留，摘要不得引用"
        )


def test_zero_miss_claim_matches_files():
    """摘要声称的零漏失必须与三个 holdout 文件一致。"""
    total = 0
    for name in HOLDOUT:
        d = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        for arm in ("baseline", "candidate"):
            a = d["summary"][arm]
            assert a["cleared"] == a["total"], f"{name}:{arm} 并非零漏失"
        total += d["summary"]["candidate"]["cleared"]
    assert total == 3347, f"三组候选臂清除总数应为 3347，实为 {total}"
    assert "3347" in _abstract(), "摘要未写出可核验的源总数 3347"


def test_untraceable_table_is_annotated():
    """未入库的 120 局配对表必须带可核验状态说明。"""
    t = PAPER.read_text(encoding="utf-8")
    if "694.77" not in t:
        return
    i = t.find("694.77")
    tail = t[i:i + 2600]
    assert "本表的可核验状态" in tail, "120 局配对表缺可核验状态标注"
    assert "以 7.2 节为准" in tail, "未指明引用提速幅度应以 7.2 节为准"
