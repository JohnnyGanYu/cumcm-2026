"""交付 zip 必须字节可复现。

赛日第一阶段交的是 MD5，第二阶段才上传文件。中间若因任何原因重打一次包，
字节不可复现就意味着已交的 MD5 立刻失效 —— 论文 PDF 就是这种情况（实测过），
所以 zip 这边靠 ZipInfo(date_time=固定值) 保证可复现，这条性质必须锁住。
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "CUMCM2026B_robot_windows.zip"
BUILDER = ROOT / "src" / "build_windows_zip.py"


def test_builder_pins_timestamps():
    """打包必须显式指定 ZipInfo 时间戳；用默认值会写入当前时间。"""
    text = BUILDER.read_text(encoding="utf-8")
    assert "ZipInfo(" in text, "未使用 ZipInfo，无法固定元数据"
    assert "date_time=" in text, (
        "ZipInfo 未固定 date_time —— 每次打包写入当前时间，MD5 不可复现"
    )


def test_archive_entries_share_one_fixed_timestamp():
    """包内每个条目的时间戳都应是同一个固定值。"""
    assert ZIP.exists(), "交付 zip 不存在，先跑 python -m src.build_windows_zip"
    with zipfile.ZipFile(ZIP) as z:
        stamps = {info.date_time for info in z.infolist()}
    assert len(stamps) == 1, f"条目时间戳不统一: {sorted(stamps)}"
    stamp = stamps.pop()
    assert stamp[0] >= 1980, f"时间戳 {stamp} 非法（zip 格式最早 1980）"


def test_md5_is_stable_across_rebuilds(tmp_path):
    """按打包脚本自己的逻辑重压一遍，MD5 必须与仓库里的包一致。

    复用 build_windows_zip 的 payload_files 与元数据设置，而不是自己另写一套 ——
    否则测的是我的压缩方式而非脚本的可复现性（第一版就栽在漏了 external_attr）。
    不调用 build() 是为了避免覆盖仓库里的交付物。
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from src import build_windows_zip as builder

    rebuilt = tmp_path / "again.zip"
    files = builder.payload_files()
    assert files, "dist_windows 下没有可打包文件"
    with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src in files:
            rel = src.relative_to(builder.DIST).as_posix()
            info = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, src.read_bytes())

    original = hashlib.md5(ZIP.read_bytes()).hexdigest()
    again = hashlib.md5(rebuilt.read_bytes()).hexdigest()
    assert original == again, (
        f"同内容重压后 MD5 变了（{original} -> {again}）；"
        "赛日交完 MD5 若需重打包，哈希会失效"
    )
