"""把 `src/b_sim/` 的机器狗运行时同步到 Windows 交付包 `dist_windows/b_sim/`。

## 为什么要有这个脚本

`dist_windows/` 是真正拿去连官方模拟器的那份代码。它此前是**手工抄过去**的，结果与
`src/b_sim/` 悄悄漂移了：`client.py` 少了发送前坐标越界拦截与 400/409 处理，`geometry.py`
少了三分之二（第二检测点候选区、覆盖判定等），`strategy.py` 也落后一个版本。这种漂移最坏的
后果是"离线调好的策略"和"正式测试真正跑的策略"根本不是同一份，成绩无从复现。

因此改为脚本化同步 + 校验：

    python -m src.pack_dist_windows            # 同步
    python -m src.pack_dist_windows --check    # 只校验是否已同步（可用于提交前自检）

## 只同步运行时，不同步离线工具

交付包里只放机器狗本体依赖的模块（`RUNTIME_MODULES`）。仿真器（`arena`/`server`）、彩排与
调参工具（`rehearse`/`tune`/`timeline`/`phase_probe`/`coverage_audit`/`coverage_design`）和测试
一律不入包：它们只在离线用，且 `tune`/`timeline` 依赖 `arena`，带进去反而增加正式测试当天的
出错面。机器狗本体只用 Python 标准库。

导入形式无需改写：`b_sim` 内部一律用相对导入（`from .arena import ...`），交付包把 `b_sim`
放在 `run_official.py` 同级目录即可，二者共用同一套源码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 机器狗运行时依赖闭包：strategy → {arena, client, directional, feasible, geometry, route}
RUNTIME_MODULES = (
    "__init__.py",
    "arena.py",
    "client.py",
    "directional.py",
    "feasible.py",
    "geometry.py",
    "route.py",
    "strategy.py",
    "q4_fast.py",
)

SOURCE_DIR = Path("src/b_sim")
TARGET_DIR = Path("dist_windows/b_sim")


def sync(*, check_only: bool = False) -> int:
    """同步（或校验）交付包。返回需要更新的文件数。"""
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"找不到源目录 {SOURCE_DIR}（请在仓库根目录运行）")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    stale = 0
    for name in RUNTIME_MODULES:
        source = SOURCE_DIR / name
        target = TARGET_DIR / name
        if not source.is_file():
            raise SystemExit(f"源文件缺失：{source}")
        wanted = source.read_text(encoding="utf-8")
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        if current == wanted:
            print(f"  = {name}")
            continue
        stale += 1
        if check_only:
            state = "内容不一致" if current is not None else "交付包缺失"
            print(f"  ! {name}（{state}）")
        else:
            target.write_text(wanted, encoding="utf-8")
            print(f"  > {name}（已更新）")

    # The actual entry point is part of the delivery contract as well.
    entry_source = SOURCE_DIR.parent / "run_official.py"
    entry_target = TARGET_DIR.parent / "run_official.py"
    wanted = entry_source.read_text(encoding="utf-8")
    current = entry_target.read_text(encoding="utf-8") if entry_target.exists() else None
    if wanted != current:
        stale += 1
        if check_only:
            print("  ! run_official.py（入口未同步）")
        else:
            entry_target.write_text(wanted, encoding="utf-8")
            print("  > run_official.py（入口已更新）")

    # 交付包里不应残留 src/ 已经删掉或不属于运行时的模块
    for path in sorted(TARGET_DIR.glob("*.py")):
        if path.name not in RUNTIME_MODULES:
            stale += 1
            if check_only:
                print(f"  ! {path.name}（不属于运行时依赖，应删除）")
            else:
                path.unlink()
                print(f"  x {path.name}（已删除：不属于运行时依赖）")
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description="同步 Windows 交付包中的机器狗运行时")
    parser.add_argument("--check", action="store_true",
                        help="只校验是否已同步，不写入；有差异时以退出码 1 结束")
    args = parser.parse_args()

    print(f"{'校验' if args.check else '同步'} {SOURCE_DIR} -> {TARGET_DIR}")
    stale = sync(check_only=args.check)
    if args.check and stale:
        print(f"\n交付包有 {stale} 处未同步，请运行：python -m src.pack_dist_windows")
        sys.exit(1)
    print(f"\n{'交付包已是最新' if not stale else f'已同步 {stale} 个文件'}")


if __name__ == "__main__":
    main()
