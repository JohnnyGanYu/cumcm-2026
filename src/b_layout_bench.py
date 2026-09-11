"""站位布局的配对种子实测台：同一批 seed、同一案例，只换补扫站位布局。

不修改 src/b_sim/* —— 通过子类化 DogStrategy 覆盖 `_probe_stations()` 注入布局。
布局用「层」描述：layers = [(radius, count, phase_rad), ...]，原点由阶段 1 负责。

指标口径与评分同构：
  pooled = sum(各局总虚拟时间) / sum(各局清除数)
单局 mean_clear_time_s 的算术平均（mean_of_means）同时报出，作为交叉检查。
清除率一律给「已清除数/总源数」原始分数。

用法：
  python -m src.b_layout_bench --episodes 200 --seed-start 5000 --directional 0 \
      --arm TUNED --arm r1150x6 --workers 4 --out results/b_layout_tune_q3.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics as st
import threading
import time
from pathlib import Path

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "layout_bench"

Layer = tuple[float, int, float]


def _ring(radius: float, count: int, phase: float = 0.0) -> Layer:
    return (radius, count, phase)


def _auto_phase(count: int) -> float:
    """外环相位错开半格，与 strategy._probe_stations 的 pi/n 约定一致。"""
    return math.pi / max(1, count)


# ---- 布局库 -----------------------------------------------------------------
# 每个布局给 q3（单环，全向）与 q4（内环 + 定向外环）两套层定义。
# 基准组：已调优基线（results/b_budget_verdict.md）
#   Q3 probe_ring_stations=9  -> 1250x9
#   Q4 probe_ring_stations=8 + directional_ring_stations=30 -> 1250x8 + 1750x30

OUTER_Q4: Layer = (1750.0, 30, _auto_phase(30))

LAYOUTS: dict[str, dict[str, list[Layer]]] = {
    # ===== 对照基准：已调优基线，不是默认的 12/18 =====
    "TUNED": {"q3": [_ring(1250.0, 9)],
              "q4": [_ring(1250.0, 8), OUTER_Q4]},
    # 默认出厂配置，仅作参照
    "DEFAULT": {"q3": [_ring(1250.0, 12)],
                "q4": [_ring(1250.0, 12), _ring(1750.0, 18, _auto_phase(18))]},

    # ===== 几何筛选胜出者：小半径 + 覆盖下界站位数 =====
    # 依据 src/b_layout_geom.py 的精确 bisector_gap：
    #   R=1150 n=6 -> gap 988 m < 1000 m（覆盖保证成立，仅 6 个补扫站位）
    #   R=1000 n=7 -> gap 995 m；R=1200 n=6 -> gap 969 m
    "r1150x6": {"q3": [_ring(1150.0, 6)], "q4": [_ring(1150.0, 6), OUTER_Q4]},
    "r1150x7": {"q3": [_ring(1150.0, 7)], "q4": [_ring(1150.0, 7), OUTER_Q4]},
    "r1200x6": {"q3": [_ring(1200.0, 6)], "q4": [_ring(1200.0, 6), OUTER_Q4]},
    "r1250x6": {"q3": [_ring(1250.0, 6)], "q4": [_ring(1250.0, 6), OUTER_Q4]},
    "r1000x7": {"q3": [_ring(1000.0, 7)], "q4": [_ring(1000.0, 7), OUTER_Q4]},
    "r1100x7": {"q3": [_ring(1100.0, 7)], "q4": [_ring(1100.0, 7), OUTER_Q4]},
    "r1100x6": {"q3": [_ring(1100.0, 6)], "q4": [_ring(1100.0, 6), OUTER_Q4]},
    "r1300x6": {"q3": [_ring(1300.0, 6)], "q4": [_ring(1300.0, 6), OUTER_Q4]},
    "r1050x7": {"q3": [_ring(1050.0, 7)], "q4": [_ring(1050.0, 7), OUTER_Q4]},
    "r1150x8": {"q3": [_ring(1150.0, 8)], "q4": [_ring(1150.0, 8), OUTER_Q4]},
    # 相位对照：环整体旋转半格，检验结果对相位是否敏感
    "r1150x6_ph": {"q3": [_ring(1150.0, 6, math.pi / 6)],
                   "q4": [_ring(1150.0, 6, math.pi / 6), OUTER_Q4]},

    # ===== Q4 外环变体（内环固定为 Q3 胜出者）=====
    "q4_out1750x24": {"q3": [_ring(1150.0, 6)],
                      "q4": [_ring(1150.0, 6), _ring(1750.0, 24, _auto_phase(24))]},
    "q4_out1750x36": {"q3": [_ring(1150.0, 6)],
                      "q4": [_ring(1150.0, 6), _ring(1750.0, 36, _auto_phase(36))]},
    "q4_out1700x30": {"q3": [_ring(1150.0, 6)],
                      "q4": [_ring(1150.0, 6), _ring(1700.0, 30, _auto_phase(30))]},
    "q4_out1800x30": {"q3": [_ring(1150.0, 6)],
                      "q4": [_ring(1150.0, 6), _ring(1800.0, 30, _auto_phase(30))]},
}


def layers_to_rings(layers: list[Layer]) -> list[list[tuple[float, float]]]:
    return [[(radius * math.cos(2 * math.pi * k / count + phase),
              radius * math.sin(2 * math.pi * k / count + phase))
             for k in range(count)]
            for radius, count, phase in layers]


class LayoutStrategy(DogStrategy):
    """只覆盖 `_probe_stations`，其余行为与主策略逐字节一致。"""

    layers: list[Layer] = []

    def _probe_stations(self):            # type: ignore[override]
        return layers_to_rings(self.layers)


def make_strategy_cls(layers: list[Layer]):
    return type("LayoutStrategyBound", (LayoutStrategy,), {"layers": layers})


def run_episode(seed: int, directional: int, layers: list[Layer],
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
        cfg = StrategyConfig(directional_probe=directional > 0)
        summary = make_strategy_cls(layers)(client, cfg).run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    s = arena.stats()
    truth = case.truth_table()
    return {
        "seed": seed,
        "total_jammers": s["total_jammers"],
        "cleared": s["cleared"],
        "cleared_ratio": s["cleared_ratio"],
        "total_virtual_time_s": round(s["total_virtual_time_s"], 3),
        "mean_clear_time_s": None if s["mean_clear_time_s"] is None else round(s["mean_clear_time_s"], 3),
        "measure_calls": s["measure_calls"],
        "clear_calls": s["clear_calls"],
        "move_distance_m": round(s["move_distance_m"], 1),
        "program_wall_s": round(time.monotonic() - started, 2),
        "missed_channels": [r["channel"] for r in truth if not r["cleared"]],
        "events": summary["events"],
    }


def _worker(job):
    seed, directional, layers = job
    try:
        return run_episode(seed, directional, layers)
    except Exception as exc:                                  # noqa: BLE001
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}",
                "cleared_ratio": 0.0, "mean_clear_time_s": None,
                "total_jammers": 0, "cleared": 0, "measure_calls": 0,
                "clear_calls": 0, "move_distance_m": 0.0, "program_wall_s": 0.0,
                "total_virtual_time_s": 0.0, "missed_channels": [], "events": []}


def aggregate(eps: list[dict]) -> dict:
    ok = [e for e in eps if "error" not in e]
    means = [e["mean_clear_time_s"] for e in ok if e["mean_clear_time_s"] is not None]
    tot_src = sum(e["total_jammers"] for e in ok)
    tot_cl = sum(e["cleared"] for e in ok)
    tot_t = sum(e["total_virtual_time_s"] for e in ok)
    full = sum(1 for e in ok if e["cleared_ratio"] >= 1 - 1e-9)
    return {
        "episodes": len(eps),
        "errors": len(eps) - len(ok),
        "sources_total": tot_src,
        "sources_cleared": tot_cl,
        "clear_rate_fraction": f"{tot_cl}/{tot_src}",
        "pooled_clear_rate": round(tot_cl / tot_src, 5) if tot_src else 0.0,
        "full_clear_episodes": full,
        # 评分同构指标
        "pooled_mean_clear_time_s": round(tot_t / tot_cl, 2) if tot_cl else None,
        "mean_of_means_s": round(st.fmean(means), 2) if means else None,
        "median_of_means_s": round(st.median(means), 2) if means else None,
        "measure_calls_mean": round(st.fmean([e["measure_calls"] for e in ok]), 1) if ok else None,
        "move_distance_m_mean": round(st.fmean([e["move_distance_m"] for e in ok]), 0) if ok else None,
    }


def paired_delta(base_eps: list[dict], arm_eps: list[dict]) -> dict:
    """配对差：同 seed 逐局比较单局 mean_clear_time_s。"""
    bmap = {e["seed"]: e for e in base_eps if "error" not in e}
    diffs = []
    for e in arm_eps:
        if "error" in e or e["seed"] not in bmap:
            continue
        b = bmap[e["seed"]]
        if e["mean_clear_time_s"] is None or b["mean_clear_time_s"] is None:
            continue
        diffs.append(e["mean_clear_time_s"] - b["mean_clear_time_s"])
    if not diffs:
        return {}
    n = len(diffs)
    mean_d = st.fmean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {
        "paired_n": n,
        "paired_mean_diff_s": round(mean_d, 2),
        "paired_se_s": round(se, 2),
        # 正态近似 95% CI；差值分布右偏，故同时给出胜局比例
        "paired_ci95_s": [round(mean_d - 1.96 * se, 2), round(mean_d + 1.96 * se, 2)],
        "paired_win_frac": round(sum(1 for d in diffs if d < 0) / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--baseline", default="TUNED", help="对照基准布局名")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    key = "q4" if args.directional > 0 else "q3"
    seeds = [args.seed_start + i for i in range(args.episodes)]
    arms = list(dict.fromkeys([args.baseline] + args.arm))   # 基准置首、去重
    results: dict[str, dict] = {}

    for arm in arms:
        if arm not in LAYOUTS:
            raise SystemExit(f"unknown arm {arm}; have {sorted(LAYOUTS)}")
        layers = LAYOUTS[arm][key]
        t0 = time.monotonic()
        with mp.Pool(args.workers) as pool:
            eps = pool.map(_worker, [(s, args.directional, layers) for s in seeds])
        agg = aggregate(eps)
        results[arm] = {"layers": [list(l) for l in layers], "summary": agg,
                        "episodes": eps, "wall_s": round(time.monotonic() - t0, 1)}
        print(f"[{arm:16s}] {key} n={args.episodes} pooled={agg['pooled_mean_clear_time_s']} "
              f"mom={agg['mean_of_means_s']} clear={agg['clear_rate_fraction']} "
              f"full={agg['full_clear_episodes']}/{args.episodes} "
              f"meas={agg['measure_calls_mean']} err={agg['errors']} ({results[arm]['wall_s']}s)",
              flush=True)

    ref = results.get(args.baseline)
    if ref and ref["summary"]["pooled_mean_clear_time_s"]:
        rp = ref["summary"]["pooled_mean_clear_time_s"]
        print(f"\n== vs {args.baseline} (pooled = sum T / sum cleared) ==")
        print(f"{'arm':17s} {'pooled':>9s} {'delta%':>8s} {'paired_d':>9s} "
              f"{'ci95':>18s} {'win':>6s} {'clear':>12s}")
        for arm, r in results.items():
            s = r["summary"]
            v = s["pooled_mean_clear_time_s"]
            if not v:
                continue
            pd = paired_delta(ref["episodes"], r["episodes"])
            results[arm]["paired_vs_baseline"] = pd
            ci = pd.get("paired_ci95_s")
            print(f"{arm:17s} {v:9.2f} {100*(v-rp)/rp:+7.2f}% "
                  f"{pd.get('paired_mean_diff_s', 0):+9.2f} "
                  f"{f'[{ci[0]:+.1f},{ci[1]:+.1f}]' if ci else '':>18s} "
                  f"{pd.get('paired_win_frac', 0):6.2f} {s['clear_rate_fraction']:>12s}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"episodes": args.episodes, "seed_start": args.seed_start,
                   "directional": args.directional, "arms": arms,
                   "baseline": args.baseline, "workers": args.workers},
        "arms": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
