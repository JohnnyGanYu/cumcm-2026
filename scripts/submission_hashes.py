"""冻结并校验提交文件的哈希，防止第一阶段交 MD5 后文件被意外改动。

用法：
    python scripts/submission_hashes.py freeze <文件...>   # 记录基线
    python scripts/submission_hashes.py verify             # 比对当前状态

重要：正式 MD5 必须由竞赛客户端生成、由队长账号上传。本脚本只是本地旁路记录，
用来回答「我交完 MD5 之后，这个文件到底动过没有」——客户端不提供这个能力。
第一阶段提交后不要打开/编辑/保存/另存任一提交文件，Word 与 PDF 阅读器的
自动保存都会改字节，MD5 随之失效。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "state" / "submission_hashes.json"


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "md5": hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def do_freeze(targets: list[str]) -> int:
    if not targets:
        print("freeze 需要至少一个文件", file=sys.stderr)
        return 2
    entries = {}
    for raw in targets:
        path = Path(raw)
        if not path.is_file():
            print(f"找不到文件: {raw}", file=sys.stderr)
            return 2
        rel = str(path.resolve())
        entries[rel] = digest(path)
        d = entries[rel]
        print(f"  {path.name}")
        print(f"    md5    {d['md5']}")
        print(f"    sha256 {d['sha256'][:32]}...")
        print(f"    bytes  {d['bytes']}")
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps({
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n基线已写入 {BASELINE.relative_to(ROOT)}（{len(entries)} 个文件）")
    print("此后不要再打开/保存这些文件。")
    return 0


def do_verify() -> int:
    if not BASELINE.is_file():
        print("还没有基线，先跑 freeze", file=sys.stderr)
        return 2
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(f"基线时间 {base['frozen_at']}")
    bad = 0
    for rel, want in base["files"].items():
        path = Path(rel)
        name = path.name
        if not path.is_file():
            print(f"  ✗ {name}: 文件已不存在")
            bad += 1
            continue
        got = digest(path)
        if got["sha256"] == want["sha256"]:
            print(f"  ✓ {name}: 未改动（md5 {got['md5']}）")
            continue
        bad += 1
        print(f"  ✗ {name}: 已改动")
        print(f"      基线 md5 {want['md5']}  {want['bytes']} 字节")
        print(f"      当前 md5 {got['md5']}  {got['bytes']} 字节")
    if bad:
        print(f"\n{bad} 个文件与基线不符。已提交的 MD5 不再匹配 ——")
        print("要么恢复原文件，要么在客户端重新走第一阶段。")
        return 1
    print("\n全部与基线一致，可进入第二阶段上传。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="提交文件哈希冻结与校验")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="记录提交文件的哈希基线")
    f.add_argument("files", nargs="*", help="论文 PDF 与支撑材料 RAR")
    sub.add_parser("verify", help="比对当前文件与基线")
    args = ap.parse_args()
    if args.cmd == "freeze":
        return do_freeze(args.files)
    return do_verify()


if __name__ == "__main__":
    raise SystemExit(main())
