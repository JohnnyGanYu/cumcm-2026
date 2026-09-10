"""检测预算扩张的交换比量化：搜索投入 vs 评分指标 的 Pareto 扫描。

背景：虚拟时间预算 360000 s 只用了约 5%（官方四局 15879-17237 s），
而评分指标 = 总时间 / 清除个数 的**分母也是变量**。因此「多花时间多清一个源」
在边际上可能净赚：当前指标 M 时，额外花 < M 秒多清 1 个源即改善指标。

本脚本系统扫描 StrategyConfig 中所有控制「搜索投入量」的参数，
对每组配置记录：平均定位清除时间、清除个数、清除率、全清局数、
虚拟时间、检测次数、真实墙钟时间。

不修改 src/b_sim/strategy.py：通过 StrategyConfig 参数注入，
外加 BudgetStrategy 子类把 strategy.py 里硬编码的两个收尾参数
(_belief_search 的 max_rounds、_blind_clear_sweep 的 max_targets) 也开放出来。

用法：
  python -m src.b_budget_sweep --plan baseline --episodes 40 --directional 0
  python -m src.b_budget_sweep --plan rings   --episodes 40 --directional 5
  python -m src.b_budget_sweep --list
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "budget"
RESULTS = Path(__file__).resolve().parent.parent / "results"


class BudgetStrategy(DogStrategy):
    """把 strategy.py 里硬编码的收尾参数开放为可注入配置。

    _belief_search(max_rounds=60) 与 _blind_clear_sweep(max_targets=24)
    是问题 4 尾部的两个投入上限，属于「搜索投入量」但没有暴露在
    StrategyConfig 里。这里用 getattr 读取注入值，缺省时保持原行为。
    """

    def _belief_search(self, max_rounds: int | None = None) -> None:
        rounds = getattr(self.config, "belief_search_rounds", None)
        return super()._belief_search(max_rounds if max_rounds is not None
                                      else (rounds if rounds is not None else 60))

    def _blind_clear_sweep(self, max_targets: int | None = None) -> None:
        targets = getattr(self.config, "blind_clear_max_targets", None)
        return super()._blind_clear_sweep(max_targets if max_targets is not None
                                         else (targets if targets is not None else 24))

    def _inner_retry(self) -> None:
        """可选加密内圈重测（默认走父类原行为）。"""
        extra = getattr(self.config, "inner_retry_scale", None)
        if not extra or abs(extra - 1.0) < 1e-9:
            return super()._inner_retry()
        targets = [st for st in self.channels.values() if not st.cleared and st.bearings]
        if not targets:
            return
        base = [(0.0, 0.0), (300.0, 0.0), (-150.0, 260.0), (-150.0, -260.0)]
        inner = list(base)
        if extra >= 2.0:
            # 加一圈半径 600 m 的六边形
            inner += [(600.0 * math.cos(2 * math.pi * k / 6), 600.0 * math.sin(2 * math.pi * k / 6))
                      for k in range(6)]
        if extra >= 3.0:
            inner += [(900.0 * math.cos(2 * math.pi * k / 8 + 0.4),
                       900.0 * math.sin(2 * math.pi * k / 8 + 0.4)) for k in range(8)]
        for station in inner:
            if self._stop():
                return
            todo = [st for st in self.channels.values() if not st.cleared and st.bearings]
            if not todo:
                return
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop():
                    return
                state.exhausted = False
                self._probe_channel_at(station, state)
            self._home_all()


_CFG_FIELDS = {f.name for f in fields(StrategyConfig)}
# 允许注入的非 StrategyConfig 字段（由 BudgetStrategy 读取）
_EXTRA_FIELDS = {"belief_search_rounds", "blind_clear_max_targets", "inner_retry_scale"}


def build_config(directional: int, overrides: dict) -> StrategyConfig:
    cfg = StrategyConfig(directional_probe=directional > 0)
    for key, value in overrides.items():
        if key in _CFG_FIELDS:
            setattr(cfg, key, value)
        elif key in _EXTRA_FIELDS:
            object.__setattr__(cfg, key, value)
        else:
            raise KeyError(f"unknown config key: {key}")
    return cfg


def run_episode(seed: int, directional: int, overrides: dict,
                real_limit_s: float = 1200.0) -> dict:
    case = generate_case(seed, count=None, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=real_limit_s)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strategy = BudgetStrategy(client, build_config(directional, overrides))
        strategy.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    wall_s = time.monotonic() - started
    stats = arena.stats()
    truth = case.truth_table()
    return {
        "seed": seed,
        "total_jammers": stats["total_jammers"],
        "cleared": stats["cleared"],
        "cleared_ratio": stats["cleared_ratio"],
        "total_virtual_time_s": round(stats["total_virtual_time_s"], 3),
        "mean_clear_time_s": (None if stats["mean_clear_time_s"] is None
                              else round(stats["mean_clear_time_s"], 3)),
        "measure_calls": stats["measure_calls"],
        "clear_calls": stats["clear_calls"],
        "move_distance_m": round(stats["move_distance_m"], 1),
        "program_wall_s": round(wall_s, 2),
        "missed_channels": [r["channel"] for r in truth if not r["cleared"]],
    }


def _worker(args):
    seed, directional, overrides, real_limit = args
    try:
        return run_episode(seed, directional, overrides, real_limit)
    except Exception as exc:  # noqa: BLE001
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}


def run_config(name: str, overrides: dict, *, directional: int, episodes: int,
               seed_start: int, workers: int = 8, real_limit: float = 1200.0) -> dict:
    tasks = [(seed_start + i, directional, overrides, real_limit) for i in range(episodes)]
    out: list[dict] = []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        for f in as_completed(futs):
            out.append(f.result())
    elapsed = time.monotonic() - t0
    errs = [e for e in out if "error" in e]
    ok = [e for e in out if "error" not in e]
    if not ok:
        return {"name": name, "overrides": overrides, "directional": directional,
                "episodes": 0, "errors": errs[:5], "error_count": len(errs)}
    ok.sort(key=lambda e: e["seed"])
    means = [e["mean_clear_time_s"] for e in ok if e["mean_clear_time_s"] is not None]
    ratios = [e["cleared_ratio"] for e in ok]
    virt = [e["total_virtual_time_s"] for e in ok]
    wall = [e["program_wall_s"] for e in ok]
    meas = [e["measure_calls"] for e in ok]
    clears = [e["clear_calls"] for e in ok]
    total_jam = sum(e["total_jammers"] for e in ok)
    total_cleared = sum(e["cleared"] for e in ok)
    full = sum(1 for e in ok if e["cleared_ratio"] >= 1.0 - 1e-9)
    return {
        "name": name,
        "overrides": overrides,
        "directional": directional,
        "episodes": len(ok),
        "error_count": len(errs),
        "errors": errs[:5],
        # 评分指标：每局 mean_clear_time_s 的均值（与官方按局评分一致）
        "metric_mean_s": round(statistics.fmean(means), 2) if means else None,
        "metric_median_s": round(statistics.median(means), 2) if means else None,
        "metric_max_s": round(max(means), 2) if means else None,
        # 聚合指标：总时间/总清除数（对分母加权，看整体交换比）
        "pooled_metric_s": round(sum(virt) / total_cleared, 2) if total_cleared else None,
        "cleared_ratio_mean": round(statistics.fmean(ratios), 4),
        "cleared_ratio_min": round(min(ratios), 4),
        "cleared_total": total_cleared,
        "jammers_total": total_jam,
        "cleared_per_episode": round(total_cleared / len(ok), 3),
        "full_clear_episodes": full,
        "full_clear_rate": round(full / len(ok), 4),
        "virtual_time_mean_s": round(statistics.fmean(virt), 1),
        "virtual_time_max_s": round(max(virt), 1),
        "virtual_budget_used_pct": round(100.0 * max(virt) / 360000.0, 2),
        "measure_calls_mean": round(statistics.fmean(meas), 1),
        "clear_calls_mean": round(statistics.fmean(clears), 1),
        "wall_s_mean": round(statistics.fmean(wall), 2),
        "wall_s_max": round(max(wall), 2),
        "wall_s_p95": round(sorted(wall)[max(0, int(0.95 * len(wall)) - 1)], 2),
        "sweep_elapsed_s": round(elapsed, 1),
        "episodes_detail": ok,
    }


# ---------------- 扫描计划 ----------------
# 每个 plan 是 (名称, overrides) 列表。overrides 为空 = 当前默认配置（基线）。

def plan_baseline() -> list[tuple[str, dict]]:
    return [("baseline", {})]


def plan_rings() -> list[tuple[str, dict]]:
    """内环密度 / 半径扫描：最直接的「搜索投入量」旋钮。"""
    out = [("baseline", {})]
    for n in (8, 12, 16, 20, 24, 30, 36):
        out.append((f"inner_n{n}", {"probe_ring_stations": n}))
    for r in (1000.0, 1100.0, 1250.0, 1400.0, 1550.0):
        out.append((f"inner_r{int(r)}", {"probe_ring_radius": r}))
    return out


def plan_outer() -> list[tuple[str, dict]]:
    """外环启用与密度（Q3 默认关闭；Q4 用 directional_ring_stations）。"""
    out = [("baseline", {})]
    for n in (8, 14, 18, 24, 30):
        out.append((f"outer_on_n{n}", {"use_outer_ring": True, "outer_ring_stations": n}))
    for n in (12, 18, 24, 30, 36):
        out.append((f"dirring_n{n}", {"directional_ring_stations": n}))
    return out


def plan_multiplier() -> list[tuple[str, dict]]:
    """任务要求的 2x / 3x / 5x 搜索投入验证。

    投入量以「扫描站位总数」定义。基线 Q3 = 原点 + 内环12 = 13 站位。
    2x ≈ 26 站位、3x ≈ 39、5x ≈ 65。用「内环加密 + 启用外环 + 外环加密」组合达成。
    """
    return [
        ("x1_baseline", {}),
        # 2x：内环 12->24（或内环12+外环14）
        ("x2_inner24", {"probe_ring_stations": 24}),
        ("x2_inner12_outer14", {"probe_ring_stations": 12, "use_outer_ring": True,
                                "outer_ring_stations": 14}),
        # 3x：约 39 站位
        ("x3_inner38", {"probe_ring_stations": 38}),
        ("x3_inner18_outer20", {"probe_ring_stations": 18, "use_outer_ring": True,
                                "outer_ring_stations": 20}),
        # 5x：约 65 站位
        ("x5_inner64", {"probe_ring_stations": 64}),
        ("x5_inner24_outer40", {"probe_ring_stations": 24, "use_outer_ring": True,
                                "outer_ring_stations": 40}),
        ("x5_tri_ring", {"probe_ring_stations": 24, "probe_ring_radius": 900.0,
                         "use_outer_ring": True, "outer_ring_radius": 1500.0,
                         "outer_ring_stations": 40}),
    ]


def plan_homing() -> list[tuple[str, dict]]:
    """归航/放弃阈值：花更多检测把已发现的源啃下来。"""
    out = [("baseline", {})]
    for m in (14, 24, 40, 80):
        out.append((f"homing{m}", {"max_homing_measures": m}))
    for s in (3, 6, 12):
        out.append((f"silent_tol{s}", {"silent_tolerance": s}))
    for e in (22, 44, 88):
        out.append((f"narrow_extra{e}", {"narrow_sector_extra_measures": e}))
    for p in (6, 12, 20):
        out.append((f"spiral{p}", {"spiral_points": p}))
    for c in (9, 20, 40):
        out.append((f"clear_att{c}", {"max_clear_attempts": c}))
    return out


def plan_tail() -> list[tuple[str, dict]]:
    """问题 4 尾部收尾投入：belief 搜索轮数、盲清除候选数、射线盲清除额度。"""
    out = [("baseline", {})]
    for r in (60, 150, 400):
        out.append((f"belief_rounds{r}", {"belief_search_rounds": r}))
    for t in (24, 60, 150):
        out.append((f"blind_targets{t}", {"blind_clear_max_targets": t}))
    for a in (120, 300, 800):
        out.append((f"ray_att{a}", {"ray_clear_max_attempts": a}))
    for rng in (1600.0, 1800.0):
        out.append((f"ray_range{int(rng)}", {"ray_clear_max_range_m": rng}))
    for s in (1.0, 2.0, 3.0):
        out.append((f"inner_scale{s}", {"inner_retry_scale": s}))
    return out


def plan_lean() -> list[tuple[str, dict]]:
    """反向扫描：既然清除率已近 100%，减少搜索投入能否直接压低分子？

    这是「交换比」的另一半——如果分母已饱和，最优方向是往回收缩投入，
    直到清除率开始掉，那个拐点才是真正的最优工作点。
    """
    out = [("baseline", {})]
    for n in (4, 6, 8, 10, 12):
        out.append((f"lean_inner_n{n}", {"probe_ring_stations": n}))
    for r in (900.0, 1000.0, 1100.0, 1250.0):
        out.append((f"lean_inner_r{int(r)}", {"probe_ring_radius": r}))
    for m in (6, 10, 14):
        out.append((f"lean_homing{m}", {"max_homing_measures": m}))
    for e in (0, 10, 22):
        out.append((f"lean_narrow{e}", {"narrow_sector_extra_measures": e}))
    for p in (2, 4, 6):
        out.append((f"lean_spiral{p}", {"spiral_points": p}))
    for n in (12, 15, 18):
        out.append((f"lean_dirring{n}", {"directional_ring_stations": n}))
    return out


PLANS = {
    "baseline": plan_baseline,
    "lean": plan_lean,
    "rings": plan_rings,
    "outer": plan_outer,
    "multiplier": plan_multiplier,
    "homing": plan_homing,
    "tail": plan_tail,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="baseline", choices=sorted(PLANS) + ["custom"])
    ap.add_argument("--custom", default=None, help="JSON: [[name, overrides], ...]")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--real-limit", type=float, default=1200.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, fn in sorted(PLANS.items()):
            print(f"{name}: {len(fn())} configs")
        return

    if args.plan == "custom":
        configs = [(n, o) for n, o in json.loads(args.custom)]
    else:
        configs = PLANS[args.plan]()

    rows = []
    t0 = time.monotonic()
    for name, overrides in configs:
        res = run_config(name, overrides, directional=args.directional,
                         episodes=args.episodes, seed_start=args.seed_start,
                         workers=args.workers, real_limit=args.real_limit)
        rows.append(res)
        print(f"[{name}] metric={res.get('metric_mean_s')} "
              f"pooled={res.get('pooled_metric_s')} "
              f"ratio={res.get('cleared_ratio_mean')} "
              f"cleared={res.get('cleared_total')}/{res.get('jammers_total')} "
              f"full={res.get('full_clear_episodes')}/{res.get('episodes')} "
              f"virt={res.get('virtual_time_mean_s')} "
              f"meas={res.get('measure_calls_mean')} "
              f"wall_mean={res.get('wall_s_mean')} wall_max={res.get('wall_s_max')} "
              f"({res.get('sweep_elapsed_s')}s)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "plan": args.plan,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "directional": args.directional,
        "total_elapsed_s": round(time.monotonic() - t0, 1),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
