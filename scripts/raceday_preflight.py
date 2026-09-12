"""赛日总闸：一条命令跑完全部提交前检查，任一不过即非零退出。

赛日凌晨逐条敲五个脚本容易漏，漏掉的恰恰是最贵的那类错（交付包没同步、
文档 MD5 过期、图停在旧参数）。本脚本把它们串起来并打印一张结果表。

用法：
    python scripts/raceday_preflight.py            # 全部检查
    python scripts/raceday_preflight.py --quick    # 跳过 8 分钟全量回归
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(label: str, args: list[str], why: str) -> dict:
    """跑一个闸门。why 说明它挡的是什么错，失败时打印出来。"""
    start = time.time()
    proc = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    return {
        "label": label,
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "secs": time.time() - start,
        "why": why,
        "tail": (proc.stdout + proc.stderr).strip().splitlines()[-6:],
    }


GATES = [
    (
        "交付包同步 (src/b_sim -> dist_windows)",
        ["-m", "src.pack_dist_windows", "--check"],
        "只改源码不同步，正式测试跑的是旧策略",
    ),
    (
        "交付 zip 同步 (dist_windows -> zip)",
        ["-m", "src.build_windows_zip", "--check"],
        "解压出来的可能是几天前的包，实测过慢 24% 的先例",
    ),
    (
        "论文合规自检 (匿名/附录覆盖)",
        ["-m", "src.build_paper", "--check"],
        "身份信息入稿或附录漏收源文件，都属取消评奖情形",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="赛日提交前总闸")
    ap.add_argument("--quick", action="store_true",
                    help="跳过 8 分钟全量回归，只跑三个快速闸门与一致性断言")
    args = ap.parse_args()

    gates = list(GATES)
    gates.append((
        "一致性断言 (图/文档/表1/入口护栏)",
        ["-m", "pytest", "-q",
         "src/test_figures_match_strategy.py",
         "src/test_docs_match_code.py",
         "src/test_build_table1.py",
         "src/test_run_official_inputs.py"],
        "参数迭代后文档/配图/表格停在旧值，评审拿到自相矛盾的材料",
    ))
    if not args.quick:
        gates.append((
            "全量回归 (pytest src)",
            ["-m", "pytest", "src", "-q", "--no-header"],
            "策略改动引入回归，约 8 分钟",
        ))

    print(f"赛日总闸：{len(gates)} 项  解释器 {PY}")
    print("=" * 72)
    results = []
    for label, argv, why in gates:
        print(f"[跑] {label} ...", flush=True)
        res = run(label, argv, why)
        results.append(res)
        mark = "通过" if res["ok"] else "失败"
        print(f"     {mark}  {res['secs']:.1f}s")
        if not res["ok"]:
            print(f"     它挡的是：{res['why']}")
            for line in res["tail"]:
                print(f"     | {line}")

    print("=" * 72)
    failed = [r for r in results if not r["ok"]]
    total = sum(r["secs"] for r in results)
    for res in results:
        mark = "  OK  " if res["ok"] else " FAIL "
        print(f"[{mark}] {res['label']}")
    print(f"\n{len(results) - len(failed)}/{len(results)} 通过，"
          f"共 {total:.1f}s")
    if failed:
        print("\n以下闸门未通过，不要提交：")
        for res in failed:
            print(f"  - {res['label']}（rc={res['rc']}）")
        return 1
    print("\n全部通过。接下来的人工步骤：")
    print("  1. 数据回填后生成 PDF，且此后不再重跑构建（PDF 无法字节复现）")
    print("  2. scripts/submission_hashes.py freeze 冻结基线")
    print("  3. 客户端交 MD5，上传前再 verify 一次（须 rc=0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
