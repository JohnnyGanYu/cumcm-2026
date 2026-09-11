"""参数扫描与消融实验：并行跑一批离线演练，比较不同 `StrategyConfig` 的评分指标。

用法：

    # 单点评估（默认配置，Q3 40 局）
    python -m src.b_sim.tune --episodes 40

    # 一维扫描：搭载检测的覆盖增益门槛
    python -m src.b_sim.tune --episodes 40 --sweep piggyback_min_gain_cells=10,30,60,120

    # 消融：逐项关掉本轮的四条机制，看各自贡献
    python -m src.b_sim.tune --episodes 40 --ablation

    # 固定若干参数再扫描
    python -m src.b_sim.tune --set approach_fraction=0.5 --sweep lateral_fraction=0.1,0.16,0.24

输出按评分指标 T̄（平均定位清除时间）排序，并给出清除率——**清除率不足 100% 的配置
一律不可接受**（题面："在确保所有干扰源被清除的前提下，完成任务的时间越短越好"）。
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

from .strategy import StrategyConfig
from .timeline import profile_episode, summarize

# 消融实验：每项把一条机制退回本轮优化之前的行为
ABLATIONS: dict[str, dict[str, object]] = {
    "完整策略": {},
    "去掉搭载检测": {"piggyback": False},
    "去掉可行域自适应站位": {"adaptive_cover": False},
    "去掉巡回局部改进(2-opt/Or-opt)": {"route_improve": False},
    "去掉直径直清捷径": {"direct_clear_diameter_m": 0.0},
    "全部关闭(≈基线)": {"piggyback": False, "adaptive_cover": False,
                        "route_improve": False, "direct_clear_diameter_m": 0.0},
}


def _coerce(name: str, raw: str):
    """按 StrategyConfig 的字段类型解析命令行字符串。"""
    types = {f.name: f.type for f in fields(StrategyConfig)}
    if name not in types:
        raise SystemExit(f"StrategyConfig 没有字段 {name!r}")
    hint = str(types[name])
    if "bool" in hint:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if "int" in hint:
        return int(raw)
    if "float" in hint:
        return float(raw)
    return raw


def build_config(directional: int, overrides: dict) -> StrategyConfig:
    config = StrategyConfig(directional_probe=directional > 0)
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise SystemExit(f"StrategyConfig 没有字段 {key!r}")
        setattr(config, key, value)
    return config


def _run_one(args) -> dict:
    seed, directional, count, overrides = args
    config = build_config(directional, overrides)
    row = profile_episode(seed, directional=directional, count=count, config=config)
    return {
        "seed": row.seed, "jammers": row.jammers, "cleared": row.cleared,
        "total_s": row.total_s, "mean_clear_time_s": row.mean_clear_time_s,
        "move_s": row.move_s, "measure_s": row.measure_s, "switch_s": row.switch_s,
        "clear_success_s": row.clear_success_s, "clear_fail_s": row.clear_fail_s,
        "move_distance_m": row.move_distance_m, "nn_tour_m": row.nn_tour_m,
        "detour_ratio": row.detour_ratio, "measures": row.measures,
        "measures_direction": row.measures_direction,
        "measures_near": row.measures_near, "measures_silent": row.measures_silent,
        "clears": row.clears, "clears_success": row.clears_success,
        "events": row.events,
    }


def evaluate(overrides: dict, *, episodes: int, seed_start: int, directional: int,
             count: int | None = None, workers: int = 8) -> dict:
    """跑一组配置，返回汇总。用进程池并行（每局各起一个独立 mock 服务，互不影响）。"""
    jobs = [(seed_start + i, directional, count, overrides) for i in range(episodes)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_run_one, jobs))
    from .timeline import TimeBreakdown
    breakdowns = [TimeBreakdown(**{k: v for k, v in row.items()}) for row in rows]
    summary = summarize(breakdowns)
    summary["missed_episodes"] = sum(1 for r in rows if r["cleared"] < r["jammers"])
    summary["worst_seed"] = max(
        (r for r in rows if r["mean_clear_time_s"] is not None),
        key=lambda r: r["mean_clear_time_s"], default={"seed": None})["seed"]
    summary["missed_seeds"] = [r["seed"] for r in rows if r["cleared"] < r["jammers"]]
    summary["mean_clear_time_s_sd"] = round(statistics.pstdev(
        [r["mean_clear_time_s"] for r in rows if r["mean_clear_time_s"] is not None]) or 0.0, 2)
    return summary


def _print_header() -> None:
    print(f"{'配置':46s} {'T̄':>8s} {'σ':>7s} {'最坏':>8s} {'清除率':>8s} "
          f"{'漏局':>5s} {'移动m':>8s} {'绕路':>6s} {'检测':>6s}")
    print("-" * 116)


def _print_row(label: str, summary: dict) -> None:
    print(f"{label:46s} {summary['mean_clear_time_s']:8.1f} "
          f"{summary['mean_clear_time_s_sd']:7.1f} {summary['mean_clear_time_s_max']:8.1f} "
          f"{summary['clear_rate']:8.4f} {summary['missed_episodes']:5d} "
          f"{summary['move_distance_m']:8.0f} {summary['detour_ratio']:6.2f} "
          f"{summary['measures']:6.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B 题策略参数扫描 / 消融实验")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--directional", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE", help="固定一个 StrategyConfig 字段")
    parser.add_argument("--sweep", default=None, metavar="KEY=V1,V2,...",
                        help="对一个字段做一维扫描")
    parser.add_argument("--ablation", action="store_true", help="跑消融实验")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    seed_start = args.seed_start if args.seed_start is not None else (
        7001 if args.directional else 5000)
    base: dict = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        base[key.strip()] = _coerce(key.strip(), raw)

    cases: list[tuple[str, dict]] = []
    if args.ablation:
        for label, delta in ABLATIONS.items():
            cases.append((label, {**base, **delta}))
    elif args.sweep:
        key, _, values = args.sweep.partition("=")
        key = key.strip()
        for raw in values.split(","):
            cases.append((f"{key}={raw}", {**base, key: _coerce(key, raw)}))
    else:
        cases.append(("默认配置" if not base else json.dumps(base, ensure_ascii=False),
                      dict(base)))

    print(f"Q{4 if args.directional else 3}  局数={args.episodes} "
          f"种子={seed_start}..{seed_start + args.episodes - 1} "
          f"定向源={args.directional}\n")
    _print_header()
    results = {}
    for label, overrides in cases:
        summary = evaluate(overrides, episodes=args.episodes, seed_start=seed_start,
                           directional=args.directional, count=args.count,
                           workers=args.workers)
        results[label] = {"overrides": {k: v for k, v in overrides.items()}, "summary": summary}
        _print_row(label, summary)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
