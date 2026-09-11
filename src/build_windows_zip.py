"""把 dist_windows/ 打成正式测试用的交付 zip，并校验其内容与源码一致。

## 为什么需要这个脚本

仓库根目录的 CUMCM2026B_robot_windows.zip 是真正拿去正式测试的那一份。它此前是
手工打的，结果与 dist_windows/ 悄悄漂移：实测发现根目录 zip 停留在 1250x12 的
旧版本，既无 route.py（TSP 路径规划），射线盲扫调用点也只有 3 处而非 4 处。
若当天直接用它，会跑到比当前 main 慢约 24% 的策略，并缺一处清除率兜底。

pack_dist_windows.py 只保证 src/b_sim -> dist_windows/b_sim 同步；本脚本再往下
一层，保证 dist_windows -> zip 也同步，并把 --check 做成可在提交前跑的自检。

## 用法

    python -m src.build_windows_zip           # 重新打包
    python -m src.build_windows_zip --check   # 只校验 zip 是否与 dist_windows 一致

排除 __pycache__ 与 .pyc：它们随本机 Python 版本而变，且解压后可能与目标机的
解释器版本不符。交付包只放源码。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / 'dist_windows'
ZIP = ROOT / 'CUMCM2026B_robot_windows.zip'
SKIP_DIRS = {'__pycache__'}
SKIP_SUFFIX = {'.pyc', '.pyo'}


def payload_files() -> list[Path]:
    """dist_windows 下应当入包的文件，按相对路径排序（保证打包可复现）。"""
    out = []
    for p in sorted(DIST.rglob('*')):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in SKIP_SUFFIX:
            continue
        out.append(p)
    return out


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def build() -> int:
    files = payload_files()
    if not files:
        print('dist_windows 下没有可打包的文件', file=sys.stderr)
        return 1
    # 固定时间戳，使同样的源码打出字节相同的 zip（便于 MD5 复核）
    with zipfile.ZipFile(ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = p.relative_to(DIST).as_posix()
            info = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    print(f'已打包 {len(files)} 个文件 -> {ZIP.name} ({ZIP.stat().st_size} bytes)')
    print(f'  MD5: {hashlib.md5(ZIP.read_bytes()).hexdigest()}')
    for p in files:
        print(f'  {p.relative_to(DIST).as_posix()}')
    return 0


def check() -> int:
    if not ZIP.exists():
        print(f'{ZIP.name} 不存在，请运行：python -m src.build_windows_zip', file=sys.stderr)
        return 1
    files = {p.relative_to(DIST).as_posix(): p.read_bytes() for p in payload_files()}
    problems = []
    with zipfile.ZipFile(ZIP) as z:
        inzip = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
    for name in sorted(set(files) | set(inzip)):
        if name not in inzip:
            problems.append(f'  zip 缺少 {name}')
        elif name not in files:
            problems.append(f'  zip 多出 {name}（dist_windows 里没有）')
        elif files[name] != inzip[name]:
            problems.append(f'  内容不一致 {name}: dist={digest(files[name])} zip={digest(inzip[name])}')
    if problems:
        print(f'{ZIP.name} 与 dist_windows 不一致：', file=sys.stderr)
        for line in problems:
            print(line, file=sys.stderr)
        print('请运行：python -m src.build_windows_zip', file=sys.stderr)
        return 2
    print(f'{ZIP.name} 与 dist_windows 一致（{len(files)} 个文件）')
    print(f'  MD5: {hashlib.md5(ZIP.read_bytes()).hexdigest()}')
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='只校验，不重新打包；不一致时退出码 2')
    args = ap.parse_args()
    raise SystemExit(check() if args.check else build())


if __name__ == '__main__':
    main()
