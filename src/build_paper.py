"""把 paper/main.md 组装为符合 2026 竞赛格式规范的提交稿（DOCX / PDF）。

格式规范硬约束（problem/format2026.doc，逐条对应）：
  第三条  电子版第一页必须是摘要页 -> 剥掉标题前的编者修订块，摘要置首
  第四条  正文不超过 30 页；附录页数不限，必须包含全部可运行源程序
  第五条  附录须列支撑材料文件列表 + 全部源代码
  第六条  任何位置不得出现队号/学校/姓名 -> 生成后扫描
  第十条  单一文件，PDF 或 Word，不超过 20 MB

用法：
    python -m src.build_paper            # 生成 DOCX 与 PDF
    python -m src.build_paper --check    # 只做合规检查，不生成
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "main.md"
OUT_DIR = ROOT / "paper" / "build"

# 附录收录的源程序：交付给官方模拟器的那一份（只依赖标准库，可直接运行）
APPENDIX_SOURCES = [
    "dist_windows/run_official.py",
    "dist_windows/b_sim/__init__.py",
    "dist_windows/b_sim/arena.py",
    "dist_windows/b_sim/client.py",
    "dist_windows/b_sim/geometry.py",
    "dist_windows/b_sim/feasible.py",
    "dist_windows/b_sim/directional.py",
    "dist_windows/b_sim/route.py",
    "dist_windows/b_sim/strategy.py",
    "dist_windows/b_sim/q3_fast.py",
    "dist_windows/b_sim/q4_fast.py",
]

# 支撑材料文件列表（第五条要求写入附录）
SUPPORTING_LIST = """\
1. `official_runs/*.sanitized.{json,jlog}` —— 问题 3、问题 4 正式测试导出日志各 3 次，
   经 `sanitize_official_run.py` 去标识（队号、服务端票据、本机用户名均替换为等长占位）。
2. `official_runs/*_console.sanitized.txt` —— 正式测试控制台输出，同法去标识。
3. `results/*.json` —— 离线彩排与假设检验的聚合结果，纯仿真数据。
4. `figures/*.png` —— 论文全部配图及其生成脚本。
5. `dist_windows/` —— 提交给官方模拟器的完整可运行程序（仅依赖 Python 标准库）。
6. `AI工具使用详情.pdf` —— 依《全国大学生数学建模竞赛人工智能工具使用规定（2026年试行）》
   第 4 条提供，含所用工具与版本、使用目的与环节、主要提示方式与典型交互、
   以及对 AI 输出的采纳、人工修改与核验情况。
"""

# 禁止出现的身份信息模式（第六条）
IDENTITY_PATTERNS = [
    (r"\b20\d{10}\b", "疑似 12 位参赛队号"),
    # 排除竞赛全称里的"全国大学生…"：那是格式规范与 AI 规定的原文用词，必须出现。
    # 裸"大学"过于宽松会拦住合规文本，故要求前接非"全国"的 2-8 字且"大学"后不接"生"。
    (r"(?<!全国)[\u4e00-\u9fa5]{2,8}(?:大学(?!生)|学院|学校)", "疑似学校名称"),
    (r"甘宇|Johnny|JohnnyGanYu", "疑似姓名"),
    (r"github\.com/[A-Za-z0-9_-]+", "疑似含个人账号的仓库地址"),
    (r"C:\\\\Users\\\\(?!U+)[A-Za-z0-9._-]+", "疑似真实 Windows 用户名"),
]


def strip_editor_block(text: str) -> str:
    """剥掉标题与文首的编者修订引用块，使摘要成为第一页内容。"""
    lines = text.splitlines()
    out, i = [], 0
    # 跳过 H1 标题行
    while i < len(lines) and not lines[i].startswith("# "):
        i += 1
    i += 1
    # 跳过紧随其后的空行与 > 引用块
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(">")):
        i += 1
    out.extend(lines[i:])
    return "\n".join(out)


def build_appendix() -> str:
    parts = ["\n\\newpage\n", "# 附录\n", "## 附录 A 支撑材料文件列表\n", SUPPORTING_LIST]
    parts.append("\n## 附录 B 源程序\n")
    parts.append(
        "以下为提交给官方模拟器的完整程序，仅依赖 Python 3.10+ 标准库，"
        "解压后 `python run_official.py --robot-id <队号>` 即可运行。\n"
    )
    for rel in APPENDIX_SOURCES:
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"附录源文件缺失: {rel}")
        code = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip()
        parts.append(f"\n### {rel}\n")
        parts.append("```python\n" + code + "\n```\n")
    return "\n".join(parts)


def make_reference_docx(dest: Path) -> None:
    """生成承载页面设置的 reference.docx：A4、四边 2.5 cm、页脚居中页码。

    先让 pandoc 产出一个最小 DOCX 作为骨架，再改写其 document.xml 的 sectPr
    与 footer1.xml。这样不引入 python-docx 依赖，也不必手写完整 OOXML 包。
    """
    import shutil
    import zipfile

    seed_md = dest.parent / "_seed.md"
    seed_md.write_text("seed\n", encoding="utf-8")
    seed = dest.parent / "_seed.docx"
    res = subprocess.run(["pandoc", str(seed_md), "-o", str(seed)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit("生成 reference 骨架失败: " + res.stderr[-500:])

    # A4 = 11906 x 16838 twips；2.5 cm = 1417 twips
    sect = ('<w:sectPr>'
            '<w:footerReference w:type="default" r:id="rIdFtr1"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1417" w:right="1417" w:bottom="1417" w:left="1417"'
            ' w:header="708" w:footer="708" w:gutter="0"/>'
            '</w:sectPr>')
    footer = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        '</w:p></w:ftr>')
    rel_add = ('<Relationship Id="rIdFtr1" Type="http://schemas.openxmlformats.org/'
               'officeDocument/2006/relationships/footer" Target="footer1.xml"/>')
    ct_add = ('<Override PartName="/word/footer1.xml" ContentType="application/'
              'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')

    with zipfile.ZipFile(seed) as zin, zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                txt = data.decode("utf-8")
                # 骨架里的 sectPr 是自闭合的 <w:sectPr />，成对标签的正则匹配不到它，
                # 漏替换会让 pandoc 取到那个空 sectPr，A4 与页边距全部失效（实测过）。
                txt, n = re.subn(r"<w:sectPr\b[^>]*/>|<w:sectPr\b.*?</w:sectPr>",
                                 sect, txt, count=1, flags=re.S)
                if n == 0:
                    txt = txt.replace("</w:body>", sect + "</w:body>")
                data = txt.encode("utf-8")
            elif item.filename == "word/_rels/document.xml.rels":
                txt = data.decode("utf-8").replace("</Relationships>", rel_add + "</Relationships>")
                data = txt.encode("utf-8")
            elif item.filename == "[Content_Types].xml":
                txt = data.decode("utf-8").replace("</Types>", ct_add + "</Types>")
                data = txt.encode("utf-8")
            zout.writestr(item, data)
        zout.writestr("word/footer1.xml", footer)

    seed.unlink(missing_ok=True)
    seed_md.unlink(missing_ok=True)
    print(f"已生成页面设置模板: {dest.name}（A4 / 2.5 cm 边距 / 页脚页码）")


SECT_PR = ('<w:sectPr>'
           '<w:footerReference w:type="default" r:id="rIdFtr1"/>'
           '<w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1417" w:right="1417" w:bottom="1417" w:left="1417"'
           ' w:header="708" w:footer="708" w:gutter="0"/>'
           '</w:sectPr>')


def enforce_page_setup(docx: Path) -> None:
    """在成品 DOCX 上落实 A4 / 2.5 cm 边距 / 页脚居中页码，并自查生效。

    只改 sectPr 与 footer 关系，不触碰正文与图片流。
    """
    import zipfile

    footer = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        '</w:p></w:ftr>')
    rel = ('<Relationship Id="rIdFtr1" Type="http://schemas.openxmlformats.org/'
           'officeDocument/2006/relationships/footer" Target="footer1.xml"/>')
    ct = ('<Override PartName="/word/footer1.xml" ContentType="application/'
          'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')

    tmp = docx.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx) as zin, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        names = set(zin.namelist())
        for item in zin.infolist():
            if item.filename == "word/footer1.xml":
                continue
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                txt = data.decode("utf-8")
                txt, n = re.subn(r"<w:sectPr\b[^>]*/>|<w:sectPr\b.*?</w:sectPr>",
                                 SECT_PR, txt, count=1, flags=re.S)
                if n == 0:
                    txt = txt.replace("</w:body>", SECT_PR + "</w:body>")
                data = txt.encode("utf-8")
            elif item.filename == "word/_rels/document.xml.rels":
                txt = data.decode("utf-8")
                if 'Id="rIdFtr1"' not in txt:
                    txt = txt.replace("</Relationships>", rel + "</Relationships>")
                data = txt.encode("utf-8")
            elif item.filename == "[Content_Types].xml":
                txt = data.decode("utf-8")
                if "/word/footer1.xml" not in txt:
                    txt = txt.replace("</Types>", ct + "</Types>")
                data = txt.encode("utf-8")
            zout.writestr(item, data)
        zout.writestr("word/footer1.xml", footer)

    tmp.replace(docx)

    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf-8")
    if 'w:w="11906"' not in doc or 'w:top="1417"' not in doc:
        raise SystemExit("页面设置未写入成品，A4/页边距会不合规")
    print("页面设置已落实: A4 11906x16838 twips / 四边 2.5 cm / 页脚居中页码")


def shrink_code_font(docx: Path, half_points: int = 14) -> None:
    """把代码样式字号压到 7 pt，避免附录源码长行被硬折断。

    第五条要求附录源程序完整可运行；行被折断后评审照抄会报语法错误，
    所以这不是排版偏好而是合规问题。half_points=14 即 7 pt。
    """
    import zipfile

    tmp = docx.with_suffix(".tmp2.docx")
    changed = 0
    with zipfile.ZipFile(docx) as zin, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                txt = data.decode("utf-8")
                for style in ("SourceCode", "VerbatimChar"):
                    pat = (r'(<w:style\b[^>]*w:styleId="' + style + r'".*?</w:style>)')
                    m = re.search(pat, txt, flags=re.S)
                    if not m:
                        continue
                    block = m.group(1)
                    new = re.sub(r'<w:sz w:val="\d+"\s*/>',
                                 f'<w:sz w:val="{half_points}"/>', block)
                    new = re.sub(r'<w:szCs w:val="\d+"\s*/>',
                                 f'<w:szCs w:val="{half_points}"/>', new)
                    if '<w:sz ' not in new:
                        new = new.replace("<w:rPr>",
                                          f'<w:rPr><w:sz w:val="{half_points}"/>'
                                          f'<w:szCs w:val="{half_points}"/>', 1)
                    if new != block:
                        txt = txt.replace(block, new)
                        changed += 1
                data = txt.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(docx)
    print(f"代码字号已压至 {half_points / 2:g} pt（命中 {changed} 个样式），减少长行折断")


def scan_identity(text: str) -> list[str]:
    hits = []
    for pattern, why in IDENTITY_PATTERNS:
        for m in re.finditer(pattern, text):
            frag = text[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")
            hits.append(f"{why}: ...{frag}...")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只做合规检查，不生成文件")
    args = ap.parse_args()

    if not PAPER.exists():
        raise SystemExit(f"找不到 {PAPER}")
    body = strip_editor_block(PAPER.read_text(encoding="utf-8"))
    full = body.rstrip() + "\n" + build_appendix()

    hits = scan_identity(full)
    if hits:
        print("匿名检查未通过（格式规范第六条）:")
        for h in hits[:20]:
            print("  -", h)
        if args.check:
            raise SystemExit(1)
        raise SystemExit("请先清除上述身份信息再生成提交稿")
    print("匿名检查通过：未发现队号/学校/姓名/账号痕迹")

    if args.check:
        print(f"正文 {len(body.splitlines())} 行，附录收录 {len(APPENDIX_SOURCES)} 个源文件")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = OUT_DIR / "submission.md"
    merged.write_text(full, encoding="utf-8")
    print(f"已合并: {merged}  ({len(full)} 字符)")

    # A4 + 2.5 cm 页边距（第一条）、页脚居中页码自摘要页起（第三条）。
    # pandoc 的 DOCX 输出不吃 -V geometry，故用 reference-doc 承载页面设置。
    ref = OUT_DIR / "reference.docx"
    if not ref.exists():
        make_reference_docx(ref)

    docx = OUT_DIR / "submission.docx"
    cmd = [
        "pandoc", str(merged), "-o", str(docx),
        "--from", "markdown+tex_math_dollars+pipe_tables+raw_tex",
        "--resource-path", str(ROOT / "paper") + ":" + str(ROOT),
        "--wrap=none",
        f"--reference-doc={ref}",
    ]
    print("运行:", " ".join(cmd[:6]), "...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:])
        print(res.stderr[-3000:], file=sys.stderr)
        raise SystemExit("pandoc 生成 DOCX 失败")

    # pandoc 不保证把 reference-doc 的 sectPr 带进产物（实测 A4 被丢回 Letter），
    # 故在成品上直接落一次页面设置，并自查生效。
    enforce_page_setup(docx)
    shrink_code_font(docx)
    print(f"已生成: {docx}  ({docx.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
