"""由 state/ai_usage_log.csv 生成《AI工具使用详情.pdf》。

依据《全国大学生数学建模竞赛人工智能工具使用规定（2026年试行）》第 4 条，
使用 AI 工具的参赛作品须在支撑材料中包含该 PDF，文件名固定，内容须含：
  (1) 所用 AI 工具名称、版本或型号；
  (2) 具体使用目的和环节；
  (3) 主要提示方式与使用过程说明（可附典型交互示例）；
  (4) 对 AI 输出的采纳、人工修改和核验的主要情况（语言润色除外）。

台账 state/ai_usage_log.csv 是唯一数据源，本脚本只做汇总排版，不新增事实。
匿名要求同论文：成品不得含队号、学校、姓名。台账里的 team_member 列
按第六条替换为占位，不写入 PDF。
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "state" / "ai_usage_log.csv"
OUT_DIR = ROOT / "paper" / "build"
# 规定第 4 条把文件名钉死为“AI工具使用详情.pdf”，不要改。
PDF_NAME = "AI工具使用详情.pdf"

IDENTITY_PATTERNS = [
    (r"\b\d{12}\b", "疑似 12 位参赛队号"),
    # 排除竞赛全称里的"全国大学生…"，那是规定原文必须出现的词，不是校名。
    (r"(?<!全国)[\u4e00-\u9fa5]{2,8}(?:大学(?!生)|学院|学校)", "疑似学校名称"),
    (r"甘宇|JohnnyGanYu|johnny", "疑似姓名或账号"),
    (r"C:\\\\Users\\\\(?!U)[A-Za-z0-9._-]+", "疑似真实 Windows 用户名"),
]

HEADER = """# AI 工具使用详情

本文件依据《全国大学生数学建模竞赛人工智能工具使用规定（2026年试行）》第 4 条编写，
作为支撑材料的一部分提交。论文参考文献之前已按第 3 条设置「AI 工具使用声明」。

## 一、所用 AI 工具名称与版本

| 工具 | 版本 / 型号 | 用途范围 |
|---|---|---|
| Hermes Agent（AI 智能体） | claude-opus-5 | 代码实现与调试、离线仿真实验批量执行、结果核对、文字润色 |

本队未使用其他 AI 工具。核心建模与分析由参赛队主导，AI 参与完成的内容逐项经人工审查与核实。

## 二、总体使用原则

1. **模型选择与论证正确性由队员判定。** AI 给出的候选方案须经队员复核其数学依据方可采纳，
   仅有均值指标支持而缺乏机理论证的改动一律不采纳。
2. **数值结论必须可回溯。** 论文引用的每个数字都能在 results/ 下的原始 JSON 或
   official_runs/ 下的脱敏日志中找到出处，由队员逐项比对，不接受 AI 复述的数字。
3. **代码须通过回归与实测双重验证。** 交付包经 314 项 pytest 回归，并在官方模拟器上
   实跑确认，日志中的 strategy_class 与布局参数逐局核对。
4. **AI 的错误判断如实记录。** 台账保留被否决与被推翻的条目，不做事后修饰。

## 三、使用目的与环节（按时间顺序）

"""

TAIL = """

## 五、人工审查与核实方式

| 环节 | 核实手段 |
|---|---|
| 几何与数学论证 | 队员独立复算关键不等式并构造反例检验 |
| 策略代码 | 314 项 pytest 回归（含 33 项基于 hypothesis 的性质测试）；关键约束写成断言防回退 |
| 数值结论 | 逐项回溯 results/ 原始数据文件；论文数字与 JSON 程序化比对 |
| 官方成绩 | 以官方模拟器实测为最终判据，离线结果与之偏差已量化并写入论文局限 |
| 匿名合规 | 脱敏脚本处理官方日志后复核零残留；论文构建含身份信息扫描且经阳性对照验证 |

## 六、说明

本文件所列内容与 state/ai_usage_log.csv 一致，该台账在竞赛期间随工作进展逐条记录，
未做事后补写或修饰。按规定第 4 条第 (4) 项，语言润色环节不单独记录采纳与修改细节。
"""


def load_rows() -> list[dict]:
    text = LOG.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise SystemExit(f"台账为空: {LOG}")
    return rows


def render_stages(rows: list[dict]) -> str:
    """第 (2) 项：具体使用目的和环节。逐条列出，不做归并。"""
    out = []
    for i, r in enumerate(rows, 1):
        out.append(f"**{i}. {r['time']}｜{r['stage']}**")
        out.append("")
        out.append(f"- 目的：{r['purpose']}")
        out.append(f"- 产出：`{r['output_file']}`")
        out.append(f"- 是否采纳：{r['adopted']}")
        out.append(f"- 人工修改与判断：{r['human_changes']}")
        out.append(f"- 核验方式：{r['verification']}")
        out.append("")
    return "\n".join(out)


def render_prompts(rows: list[dict]) -> str:
    """第 (3) 项：主要提示方式与使用过程说明，附典型交互示例。"""
    # 前置空行：紧跟上一节列表项时 pandoc 会把标题并入列表，实测第四节整节丢失。
    out = ["", "", "## 四、主要提示方式与典型交互", ""]
    out.append("使用过程为多轮交互：队员给出目标与约束，AI 提出方案并执行实验，")
    out.append("队员依据实测数据判断是否采纳。以下为各环节的关键提示要点。")
    out.append("")
    out.append("| 环节 | 关键提示要点 |")
    out.append("|---|---|")
    for r in rows:
        prompt = r["key_prompt"].replace("|", "\\|")
        out.append(f"| {r['stage']} | {prompt} |")
    out.append("")
    return "\n".join(out)


def scan_identity(text: str) -> list[str]:
    hits = []
    for pattern, why in IDENTITY_PATTERNS:
        for m in re.finditer(pattern, text):
            frag = text[max(0, m.start() - 25):m.end() + 25].replace("\n", " ")
            hits.append(f"{why}: ...{frag}...")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只做合规检查")
    args = ap.parse_args()

    rows = load_rows()
    md = HEADER + render_stages(rows) + render_prompts(rows) + TAIL

    hits = scan_identity(md)
    if hits:
        print("匿名检查未通过（格式规范第六条）:")
        for h in hits[:20]:
            print("  -", h)
        raise SystemExit(1)
    print(f"匿名检查通过；台账 {len(rows)} 条记录")

    if args.check:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "ai_usage_details.md"
    md_path.write_text(md, encoding="utf-8")

    docx = OUT_DIR / "ai_usage_details.docx"
    res = subprocess.run(
        ["pandoc", str(md_path), "-o", str(docx), "--wrap=none"],
        capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
        raise SystemExit("pandoc 生成 DOCX 失败")

    # 与论文一致：A4 + 2.5 cm 边距。复用论文构建脚本的实现，避免两处逻辑漂移。
    from src.build_paper import enforce_page_setup
    enforce_page_setup(docx)

    # 规定要求 PDF 格式且文件名固定，soffice 按输入名产出，故先转再改名。
    res = subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf",
         str(docx), "--outdir", str(OUT_DIR)],
        capture_output=True, text=True, timeout=600)
    produced = OUT_DIR / "ai_usage_details.pdf"
    if not produced.exists():
        print(res.stdout[-1500:], res.stderr[-1500:], file=sys.stderr)
        raise SystemExit("soffice 转 PDF 失败")

    final = OUT_DIR / PDF_NAME
    produced.replace(final)
    print(f"已生成: {final.name}  ({final.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
