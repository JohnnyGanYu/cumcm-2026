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
    ap.add_argument("--total-jammers", type=int, default=None, metavar="N",
                    help="演练测试结束时模拟器界面告示的干扰源总数（人工抄录）。"
                         "官方接口不返回该值，填入后才能算出题目要求的"
                         "\"被清除干扰源个数的比例\"。正式测试不告示，留空即可。")
    ap.add_argument("--test-kind", choices=["practice", "formal"], default="practice",
                    help="practice=演练测试, formal=正式测试（写入日志便于归档区分）")
    ap.add_argument("--case-code", default=None,
                    help="测试案例编码（正式测试须记录，论文表1要求该列）")
    ap.add_argument("--note", default=None, help="本次测试备注，写入日志")
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
    suspect = [ch for ch, st in strategy.channels.items() if not st.cleared and st.bearings]
    silent_only = [ch for ch, st in strategy.channels.items()
                   if not st.cleared and not st.bearings and st.silent_points]
    basis = summary.get("completion_basis", "unresolved")
    certified = summary.get("certified_empty_channels", [])
    unresolved = summary.get("unresolved_channels", [])
    complete = basis in {"source_count_upper_bound", "continuous_coverage"}
    print(f"\n--- 全清判定 ---")
    print(f"已清除 {len(cleared)} 个: {sorted(cleared)}")
    if suspect:
        print(f"⚠ 疑似漏失 {len(suspect)} 个频道(曾收到信号却未清除): {sorted(suspect)}")
        print(f"  本局很可能【未全清】")
    if basis == "source_count_upper_bound":
        print("✓ 成功清除 16 个，已达到题面源总数上界")
    elif basis == "continuous_coverage":
        print("✓ 已发现源全部清除，其余频道的实际无信号观测满足连续覆盖证明")
    else:
        print(f"尚未证明全清，未解决频道: {unresolved}")
    print(f"覆盖证明为空的频道: {certified}")
    print(f"全程沉默频道（沉默本身不等于无源）: {sorted(silent_only)}")
    if args.total_jammers is not None:
        ratio = len(cleared) / args.total_jammers
        print(f"\n--- 题目要求的统计值（按人工抄录的总数 {args.total_jammers}）---")
        print(f"被清除干扰源个数的比例 = {len(cleared)}/{args.total_jammers} = {ratio:.4f}")
        print(f"平均定位清除时间       = {summary['mean_clear_time_s']} s")
        if len(cleared) > args.total_jammers:
            print("  ⚠ 清除数大于抄录总数，请核对界面数字")
        elif ratio < 1.0:
            print(f"  ⚠ 未全清，漏失 {args.total_jammers - len(cleared)} 个")
        else:
            print("  ✓ 全部清除")
    else:
        print("  注: 官方接口不返回总数。演练测试结束时模拟器界面会告示，")
        print("      请记录本局界面总数；不同局的总数不能沿用。")
    if not (10 <= len(cleared) <= 16):
        print(f"  ⚠ 清除数 {len(cleared)} 落在 10-16 之外，说明确有漏失")
    if summary.get("events"):
        print("\n事件:")
        for e in summary["events"]:
            print("  ", e)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    kind = "formal" if args.test_kind == "formal" else "practice"
    prob = "q4" if args.directional else "q3"
    out = Path(f"official_log_{prob}_{kind}_{stamp}.json")
    ratio_val = (len(cleared) / args.total_jammers) if args.total_jammers else None
    out.write_text(json.dumps({
        "robot_id": args.robot_id,
        "directional": args.directional,
        "test_kind": args.test_kind,
        "case_code": args.case_code,
        "note": args.note,
        "reported_total_jammers": args.total_jammers,
        "paper_table1_row": {
            "测试案例编码": args.case_code,
            "清除干扰源个数": len(cleared),
            "干扰源总数_人工抄录": args.total_jammers,
            "被清除干扰源个数的比例": (round(ratio_val, 4) if ratio_val is not None else None),
            "平均定位清除时间_s": (round(summary["mean_clear_time_s"], 4)
                                      if summary["mean_clear_time_s"] is not None else None),
            "程序运行时间_s": round(wall, 2),
        },
        "summary": summary,
        "program_wall_s": round(wall, 2),
        "full_clear_assessment": {
            "cleared_channels": sorted(cleared),
            "cleared_count": len(cleared),
            "suspect_missed_channels": sorted(suspect),
            "silent_only_channels": sorted(silent_only),
            "likely_full_clear": complete,
            "completion_basis": basis,
            "certified_empty_channels": certified,
            "unresolved_channels": unresolved,
            "abandoned_channels": summary.get("abandoned_channels", []),
            "note": "仅成功回执达到16或实际无信号观测的连续覆盖证书支持全清判定；依赖题面模型",
        },
        "per_channel_state": {
            str(ch): {
                "cleared": st.cleared,
                "bearing_count": len(st.bearings),
                "silent_count": len(st.silent_points),
                "measure_count": st.measure_count,
                "clear_attempts": st.clear_attempts,
                "certified_empty": st.certified_empty,
                "abandoned": st.abandoned,
            } for ch, st in sorted(strategy.channels.items())
        },
        "strategy_config": vars(config),
        "protocol_log": client.log,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整协议日志已写入: {out.resolve()}")
    print("请把这个文件回传。")
    if args.total_jammers is None and args.test_kind == "practice":
        print("提示: 演练测试结束时界面会告示干扰源总数，抄下来后可直接改写日志中的")
        print("      本局总数及比例，并保留原始日志；不要将上一局总数用于下一局。")
    if args.test_kind == "formal" and not args.case_code:
        print("提示: 正式测试请用 --case-code 记录案例编码，论文表1需要该列。")


if __name__ == "__main__":
    main()
