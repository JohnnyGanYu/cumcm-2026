#!/usr/bin/env python3
"""官方模拟器实战入口。在跑着模拟器的那台 Windows 机器上运行。

用法（先在模拟器界面启动"演练测试"，看到倒计时结束、接口开放后再运行）：

    python run_official.py --robot-id 你的参赛队号
    python run_official.py --robot-id 你的参赛队号 --directional     # 问题4
    python run_official.py --robot-id 你的参赛队号 --dry-run         # 只探接口连通性

会在当前目录写下 official_log_<时间戳>.json，包含完整指令与响应序列
（模拟器本身不提供该功能，支撑材料需要它）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from b_sim.client import ProtocolError, RobotClient, TestEnded
from b_sim.strategy import DogStrategy, StrategyConfig


def probe(base_url: str, robot_id: str) -> bool:
    """探测接口是否开放。

    警告：这会真的调用一次 /enter。官方模拟器一次测试只允许 enter 一次，
    探测后必须重新开始一次测试才能正式跑，否则 /measure 会返回
    accepted=false（本地 mock 实测已复现）。因此 --dry-run 只用于
    "确认模拟器装好了、端口对了"，不要在正式测试窗口里用。
    """
    print("警告: --dry-run 会消耗本次测试的 /enter。探测后请重新启动一次测试再正式跑。")
    c = RobotClient(robot_id=robot_id, base_url=base_url)
    try:
        c.enter()
    except TestEnded as e:
        print(f"[探测] 接口未开放或测试已结束: {e}")
        return False
    except ProtocolError as e:
        print(f"[探测] 协议被拒: {e}")
        print("  常见原因: robot_id 与登录参赛队号不一致 / 重复 enter / arena_id 错误")
        return False
    except Exception as e:
        print(f"[探测] 连接失败: {type(e).__name__}: {e}")
        print(f"  确认模拟器已启动、测试窗口已开启、端口正确 ({base_url})")
        return False
    print(f"[探测] OK  remaining_real_duration_s={c.remaining_real_s}")
    try:
        c.exit()
    except Exception:
        pass
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-id", required=True, help="参赛队号，必须与模拟器登录账号完全一致")
    ap.add_argument("--base-url", default="http://127.0.0.1:2026")
    ap.add_argument("--directional", action="store_true",
                    help="问题4：启用定向源推理与盲清除")
    ap.add_argument("--dry-run", action="store_true", help="只探测接口连通性")
    ap.add_argument("--timeout", type=float, default=5.0, help="单请求超时秒数")
    args = ap.parse_args()

    if args.dry_run:
        sys.exit(0 if probe(args.base_url, args.robot_id) else 1)

    client = RobotClient(robot_id=args.robot_id, base_url=args.base_url,
                         timeout_s=args.timeout)
    config = StrategyConfig(directional_probe=args.directional)
    strategy = DogStrategy(client, config)

    print(f"启动: robot_id={args.robot_id} 模式={'问题4(含定向)' if args.directional else '问题3(全向)'}")
    t0 = time.time()
    try:
        summary = strategy.run()
    except KeyboardInterrupt:
        print("\n用户中断，尝试正常退出...")
        try:
            client.exit()
        except Exception:
            pass
        summary = strategy.summary()
    wall = time.time() - t0

    print("\n===== 结果 =====")
    print(f"已清除频道: {summary['cleared_channels']}")
    print(f"清除个数:   {summary['cleared_count']}")
    print(f"虚拟时间:   {summary['virtual_time_s']:.1f} s  (上限 360000)")
    print(f"平均定位清除时间: {summary['mean_clear_time_s']} s   <-- 评分指标")
    print(f"检测次数:   {summary['measure_calls']}   清除调用: {summary['clear_calls']}")
    print(f"程序运行时间: {wall:.1f} s  (上限 1200)")

    # 干扰源总数官方接口不返回（附件2 明确"不会通过接口直接返回"），
    # 只能从认知状态推断，用于判断本局是否真的全清。
    cleared = set(summary["cleared_channels"])
    suspect = [ch for ch, st in strategy.channels.items()
               if not st.cleared and (st.bearings or st.silent_points == [])]
    silent_only = [ch for ch, st in strategy.channels.items()
                   if not st.cleared and not st.bearings and st.silent_points]
    print(f"\n--- 全清判定 ---")
    print(f"已清除 {len(cleared)} 个: {sorted(cleared)}")
    if suspect:
        print(f"⚠ 疑似漏失 {len(suspect)} 个频道(曾收到信号却未清除): {sorted(suspect)}")
        print(f"  本局很可能【未全清】")
    else:
        print(f"✓ 无\"曾收到信号却未清除\"的频道，本局很可能已全清")
    print(f"全程沉默频道 {len(silent_only)} 个（推定无源）: {sorted(silent_only)}")
    print(f"推定干扰源总数 = {len(cleared)}（题面范围 10-16）")
    if not (10 <= len(cleared) <= 16):
        print(f"  ⚠ 清除数 {len(cleared)} 落在 10-16 之外，说明确有漏失")
    if summary.get("events"):
        print("\n事件:")
        for e in summary["events"]:
            print("  ", e)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"official_log_{stamp}.json")
    out.write_text(json.dumps({
        "robot_id": args.robot_id,
        "directional": args.directional,
        "summary": summary,
        "program_wall_s": round(wall, 2),
        "full_clear_assessment": {
            "cleared_channels": sorted(cleared),
            "cleared_count": len(cleared),
            "suspect_missed_channels": sorted(suspect),
            "silent_only_channels": sorted(silent_only),
            "likely_full_clear": not suspect and 10 <= len(cleared) <= 16,
            "note": "干扰源总数官方接口不返回；suspect=曾收到信号却未清除，非空即很可能漏失",
        },
        "per_channel_state": {
            str(ch): {
                "cleared": st.cleared,
                "bearing_count": len(st.bearings),
                "silent_count": len(st.silent_points),
                "measure_count": st.measure_count,
                "clear_attempts": st.clear_attempts,
            } for ch, st in sorted(strategy.channels.items())
        },
        "strategy_config": vars(config),
        "protocol_log": client.log,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整协议日志已写入: {out.resolve()}")
    print("请把这个文件回传，用于校准离线仿真器。")


if __name__ == "__main__":
    main()
