"""论文必须逐条回答题目的每个子问。

检查单有「Every subproblem is answered」这一条，但此前无人验证。实际漏了两处：
问题 1 的「以定位区域直径为直径的圆能否覆盖此定位区域」只在图注里带过一句，
问题 2 的「第二个检测点的候选区域」被写成了具体参数（偏转 32 度、基线 380 m）。
两者都是评审逐条打分时会判未答的。

这里按题目原文的关键词锁住覆盖，避免改稿时又被漏掉。
"""
from __future__ import annotations

from pathlib import Path

PAPER = Path("paper/main.md")

# (子问标识, 论文中必须出现的关键词之一)
SUBQUESTIONS = (
    ("Q1 直径算法", ("凸包极点", "直径等于")),
    ("Q1 直径圆覆盖性", ("以定位区域直径为直径的圆不能覆盖", "直径圆")),
    ("Q2 选点策略", ("选点准则", "dog-leg")),
    ("Q2 检测点候选区域", ("第二个检测点的候选区域", "候选区")),
    ("Q3 全向源策略", ("Q3FastStrategy", "q3-fast")),
    ("Q4 定向源策略", ("Q4FastStrategy", "q4-fast")),
)


def _paper() -> str:
    return PAPER.read_text(encoding="utf-8")


def test_every_subquestion_is_addressed():
    text = _paper()
    missing = [
        label for label, keys in SUBQUESTIONS
        if not any(k in text for k in keys)
    ]
    assert not missing, f"论文未见对这些子问的回答: {missing}"


def test_diameter_circle_answer_is_in_body_not_only_caption():
    """覆盖性结论必须在正文小节里，只写在图注里评审可能判未答。"""
    text = _paper()
    assert "### 2.2 以定位区域直径为直径的圆不能覆盖该区域" in text, (
        "缺少专门回答直径圆覆盖性的小节"
    )
    section = text.split("### 2.2")[1].split("### 2.3")[0]
    assert "不能" in section and "Jung" in section, (
        "该节须给出否定结论并引用 Jung 定理的紧上界"
    )
    assert "0.3660" in section or "0.366" in section, (
        "须给出正三角形反例的定量缺口"
    )


def test_second_point_region_is_a_region_not_a_parameter():
    """题目要的是区域；只给偏转角与基线长度等于没答。"""
    text = _paper()
    assert "### 3.3 第二个检测点的候选区域" in text
    section = text.split("### 3.3")[1].split("### 3.4")[0]
    for token in ("m²", "格点", "候选区"):
        assert token in section, f"候选区域一节缺少 {token}（须量化面积与范围）"
    assert "in_second_point_region" in section, (
        "须指向生产实现，避免论文口径与代码脱节"
    )
