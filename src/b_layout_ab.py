"""站位布局的隔离 A/B 实测：同一批 seed、同一案例，只换补扫站位布局。

不修改 src/b_sim/strategy.py —— 通过 subclass 覆盖 `_probe_stations()` 注入布局
（与 src/b_tail_diag.py 同样的做法）。布局用 "层" 列表描述：
    layers = [(radius, count, phase_rad), ...]
半径 0 的层只放一个原点（阶段 1 已扫过，通常不写）。

用法：
  python -m src.b_layout_ab --episodes 40 --seed-start 5000 --directional 0 \
      --arm base --arm 1150x8 --out results/b_layout_ab_q3.json
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

ROBOT_ID = "layout_ab"

Layer = tuple[float, int, float]

# ---- 布局库：name -> (q3_layers, q4_layers) ----------------------------------
# q4 布局单独给，因为定向源需要更密的方位采样 + 外环。


def _L(*layers: Layer) -> list[Layer]:
    return list(layers)


PH = math.pi   # 便于写相位


LAYOUTS: dict[str, dict[str, list[Layer]]] = {
    # 现状（对照组）
    "base": {
        "q3": _L((1250.0, 12, 0.0)),
        "q4": _L((1250.0, 12, 0.0), (1750.0, 18, math.pi / 18)),
    },
    # 单环稀疏化：站位数 12 -> 8，半径 1250 -> 1150（覆盖效率 80.3% -> 86.3%）
    "r1150x8": {
        "q3": _L((1150.0, 8, 0.0)),
        "q4": _L((1150.0, 8, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1150x7": {
        "q3": _L((1150.0, 7, 0.0)),
        "q4": _L((1150.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1100x7": {
        "q3": _L((1100.0, 7, 0.0)),
        "q4": _L((1100.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1100x8": {
        "q3": _L((1100.0, 8, 0.0)),
        "q4": _L((1100.0, 8, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1200x9": {
        "q3": _L((1200.0, 9, 0.0)),
        "q4": _L((1200.0, 9, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1250x9": {
        "q3": _L((1250.0, 9, 0.0)),
        "q4": _L((1250.0, 9, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1250x10": {
        "q3": _L((1250.0, 10, 0.0)),
        "q4": _L((1250.0, 10, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1150x6": {
        "q3": _L((1150.0, 6, 0.0)),
        "q4": _L((1150.0, 6, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1200x6": {
        "q3": _L((1200.0, 6, 0.0)),
        "q4": _L((1200.0, 6, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1200x7": {
        "q3": _L((1200.0, 7, 0.0)),
        "q4": _L((1200.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1250x7": {
        "q3": _L((1250.0, 7, 0.0)),
        "q4": _L((1250.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1250x8": {
        "q3": _L((1250.0, 8, 0.0)),
        "q4": _L((1250.0, 8, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1300x7": {
        "q3": _L((1300.0, 7, 0.0)),
        "q4": _L((1300.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1050x7": {
        "q3": _L((1050.0, 7, 0.0)),
        "q4": _L((1050.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1000x7": {
        "q3": _L((1000.0, 7, 0.0)),
        "q4": _L((1000.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    # 三环阶梯（内密外疏，每环半径都保持高覆盖效率）
    "tri_900x5_1250x9": {
        "q3": _L((900.0, 5, 0.0), (1250.0, 9, math.pi / 9)),
        "q4": _L((900.0, 5, 0.0), (1250.0, 9, math.pi / 9), (1750.0, 18, math.pi / 18)),
    },
    "r1130x7": {
        "q3": _L((1130.0, 7, 0.0)),
        "q4": _L((1130.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    "r1180x7": {
        "q3": _L((1180.0, 7, 0.0)),
        "q4": _L((1180.0, 7, 0.0), (1750.0, 18, math.pi / 18)),
    },
    # === 推荐布局：Q3 单环 1150x7；Q4 同内环 + 1750x20 外环 ===
    "REC": {
        "q3": _L((1150.0, 7, 0.0)),
        "q4": _L((1150.0, 7, 0.0), (1750.0, 20, math.pi / 20)),
    },
    "q4_in1150x7_out1750x16": {
        "q3": _L((1150.0, 7, 0.0)),
        "q4": _L((1150.0, 7, 0.0), (1750.0, 16, math.pi / 16)),
    },
    "q4_in1150x7_out1750x22": {
        "q3": _L((1150.0, 7, 0.0)),
        "q4": _L((1150.0, 7, 0.0), (1750.0, 22, math.pi / 22)),
    },
    "q4_in1150x7_out1700x20": {
        "q3": _L((1150.0, 7, 0.0)),
        "q4": _L((1150.0, 7, 0.0), (1700.0, 20, math.pi / 20)),
    },
    "q4_in1150x7_noouter": {
        "q3": _L((1150.0, 7, 0.0)),
        "q4": _L((1150.0, 7, 0.0)),
    },
    # Q4 专用：外环内移到 1650 / 1600
    "q4_outer1650": {
        "q3": _L((1150.0, 8, 0.0)),
        "q4": _L((1150.0, 8, 0.0), (1650.0, 18, math.pi / 18)),
    },
    "q4_outer1700": {
        "q3": _L((1150.0, 8, 0.0)),
        "q4": _L((1150.0, 8, 0.0), (1700.0, 18, math.pi / 18)),
    },
    # Q4：内环稀疏 + 外环加密到 20
    "q4_in1150x8_out1750x20": {
        "q3": _L((1150.0, 8, 0.0)),
        "q4": _L((1150.0, 8, 0.0), (1750.0, 20, math.pi / 20)),
    },
    # Q4：中环 1450 插入（三环）
    "q4_tri_1100x7_1450x12_1750x18": {
        "q3": _L((1100.0, 7, 0.0)),
        "q4": _L((1100.0, 7, 0.0), (1450.0, 12, math.pi / 12), (1750.0, 18, math.pi / 18)),
    },
}


def layers_to_rings(layers: list[Layer]) -> list[list[tuple[float, float]]]:
    rings = []
    for radius, count, phase in layers:
        rings.append([(radius * math.cos(2 * math.pi * k / count + phase),
                       radius * math.sin(2 * math.pi * k / count + phase))
                      for k in range(count)])
    return rings


class LayoutStrategy(DogStrategy):
    """只覆盖 `_probe_stations`，其余行为与主策略完全一致。"""

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
        strategy = make_strategy_cls(layers)(client, cfg)
        summary = strategy.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    wall = time.monotonic() - started
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
        "program_wall_s": round(wall, 2),
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
    ratios = [e["cleared_ratio"] for e in ok]
    full = sum(1 for e in ok if e["cleared_ratio"] >= 1 - 1e-9)
    tot_src = sum(e["total_jammers"] for e in ok)
    tot_cl = sum(e["cleared"] for e in ok)
    tot_t = sum(e["total_virtual_time_s"] for e in ok)
    return {
        "episodes": len(eps),
        "errors": len(eps) - len(ok),
        "full_clear_episodes": full,
        "full_clear_rate": round(full / len(ok), 4) if ok else 0.0,
        "cleared_ratio_mean": round(st.fmean(ratios), 4) if ratios else 0.0,
        "cleared_ratio_min": round(min(ratios), 4) if ratios else 0.0,
        "sources_total": tot_src,
        "sources_cleared": tot_cl,
        "pooled_clear_rate": round(tot_cl / tot_src, 4) if tot_src else 0.0,
        # 池化指标 = 全部局的总时间 / 全部局清除总数，与评分定义同构且不受单局分母噪声影响
        "pooled_mean_clear_time_s": round(tot_t / tot_cl, 2) if tot_cl else None,
        "mean_clear_time_s_mean": round(st.fmean(means), 2) if means else None,
        "mean_clear_time_s_median": round(st.median(means), 2) if means else None,
        "mean_clear_time_s_max": round(max(means), 2) if means else None,
        "total_virtual_time_s_mean": round(st.fmean([e["total_virtual_time_s"] for e in ok]), 1) if ok else None,
        "measure_calls_mean": round(st.fmean([e["measure_calls"] for e in ok]), 1) if ok else None,
        "clear_calls_mean": round(st.fmean([e["clear_calls"] for e in ok]), 1) if ok else None,
        "move_distance_m_mean": round(st.fmean([e["move_distance_m"] for e in ok]), 0) if ok else None,
        "program_wall_s_max": round(max(e["program_wall_s"] for e in ok), 2) if ok else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--arm", action="append", required=True,
                    help="布局名（LAYOUTS 的 key），可多次给出")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    key = "q4" if args.directional > 0 else "q3"
    seeds = [args.seed_start + i for i in range(args.episodes)]
    results: dict[str, dict] = {}

    for arm in args.arm:
        if arm not in LAYOUTS:
            raise SystemExit(f"unknown arm {arm}; have {sorted(LAYOUTS)}")
        layers = LAYOUTS[arm][key]
        jobs = [(s, args.directional, layers) for s in seeds]
        t0 = time.monotonic()
        with mp.Pool(args.jobs) as pool:
            eps = pool.map(_worker, jobs)
        agg = aggregate(eps)
        results[arm] = {"layers": layers, "summary": agg, "episodes": eps,
                        "wall_s": round(time.monotonic() - t0, 1)}
        print(f"[{arm}] {key} n={args.episodes}  pooled_t={agg['pooled_mean_clear_time_s']}  "
              f"mean_t={agg['mean_clear_time_s_mean']}  full={agg['full_clear_episodes']}/{args.episodes}  "
              f"pooled_clear={agg['pooled_clear_rate']}  meas={agg['measure_calls_mean']}  "
              f"move={agg['move_distance_m_mean']}  ({results[arm]['wall_s']}s)", flush=True)

    base = results.get("base", {}).get("summary")
    if base and base["pooled_mean_clear_time_s"]:
        print("\n== vs base (pooled mean clear time) ==")
        for arm, r in results.items():
            v = r["summary"]["pooled_mean_clear_time_s"]
            if v:
                d = 100 * (v - base["pooled_mean_clear_time_s"]) / base["pooled_mean_clear_time_s"]
                print(f"  {arm:32s} {v:9.2f}s  {d:+6.2f}%  full={r['summary']['full_clear_episodes']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"episodes": args.episodes, "seed_start": args.seed_start,
                   "directional": args.directional, "arms": args.arm},
        "arms": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
