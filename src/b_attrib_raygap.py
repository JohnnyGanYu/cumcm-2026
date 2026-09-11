"""验证并量化「拿到示向度却零次清除」的漏失通道（ray blind clear 未触发）。

归因诊断发现：9 个漏失源中 6 个（67%）满足 bearings>=1 且 clear_calls==0，
即机器狗**已经知道目标方向**，却一次 /clear 都没发过。

代码路径假设（strategy.py:549-552）：
    if state.estimate is None:
        if not self._dogleg(state):
            return False          # <-- 直接返回，未走 _ray_blind_clear
只有一条示向度时 estimate 为 None（_refresh 要求 >=2 条）。若 _dogleg 的
候选点全部沉默或越界，`_home_channel` 直接 return False，`_home_all` 把频道
标成 exhausted。而 `_ray_blind_clear` 只在「lost signal」与「预算耗尽」两条
分支里被调用，dogleg 失败这条路绕过了它。

本脚本两件事：
A) exit_path 计数：hook `_dogleg` / `_ray_blind_clear` / `_home_channel`，
   记录每个漏失频道实际走的退出路径 —— 只观察，不改行为。
B) 修复验证：`RayGapStrategy` 子类只改一处 —— dogleg 失败且已有示向度时，
   改走 `_ray_blind_clear` 兜底。对比同种子的清除数与平均耗时。

严格不碰 src/b_sim/strategy.py；不使用真值驱动策略（真值只用于事后判定）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading
from collections import Counter
from pathlib import Path

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig
from .b_attrib_diag import planned_stations, receivable

ROBOT = "dog-raygap"


class PathProbe(DogStrategy):
    """只读观察者：记录每个频道的退出路径。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.path: dict[int, list[str]] = {}
        self.clear_calls: dict[int, int] = {}

    def _note(self, ch, tag):
        self.path.setdefault(ch, []).append(tag)

    def _dogleg(self, state):
        r = super()._dogleg(state)
        self._note(state.channel, f"dogleg={'ok' if r else 'FAIL'}")
        return r

    def _ray_blind_clear(self, state):
        r = super()._ray_blind_clear(state)
        self._note(state.channel, f"ray={'ok' if r else 'fail'}")
        return r

    def _home_channel(self, state):
        r = super()._home_channel(state)
        self._note(state.channel, f"home={'ok' if r else 'GIVEUP'}")
        return r

    def _clear_at(self, point, channel):
        ok = super()._clear_at(point, channel)
        self.clear_calls[channel] = self.clear_calls.get(channel, 0) + 1
        return ok


class RayGapStrategy(DogStrategy):
    """候选修复：dogleg 失败但已有示向度时，用射线盲清除兜底而不是直接放弃。

    只覆盖 _home_channel 中 `estimate is None` 且 dogleg 失败的那一步。
    做法：包装 _dogleg —— 失败且已有示向度且还没扫过射线时，就地触发
    _ray_blind_clear；若清除成功则让 _home_channel 看到 state.cleared。
    """

    def _dogleg(self, state):
        r = super()._dogleg(state)
        if not r and state.bearings and not state.ray_swept and not state.cleared:
            self._ray_blind_clear(state)
        return r


def one(seed: int, directional: int, count: int | None, cls) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    httpd = make_server(Session(ROBOT, arena, real_limit_s=1200.0), port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        strat = cls(RobotClient(ROBOT, base_url=f"http://127.0.0.1:{port}"),
                    StrategyConfig(directional_probe=directional > 0))
        strat.run()
    finally:
        httpd.shutdown(); httpd.server_close()
    s = arena.stats()
    out = {"seed": seed, "cleared": s["cleared"], "total": s["total_jammers"],
           "mean_clear_time_s": (None if s["mean_clear_time_s"] is None
                                 else round(s["mean_clear_time_s"], 1)),
           "virtual_s": round(s["total_virtual_time_s"], 1),
           "measures": s["measure_calls"], "clears": s["clear_calls"]}
    if isinstance(strat, PathProbe):
        cfg = StrategyConfig(directional_probe=directional > 0)
        stations = planned_stations(directional > 0, cfg)
        missed = []
        for row in case.truth_table():
            if row["cleared"]:
                continue
            ch = row["channel"]
            u_layout = any(receivable(row["x"], row["y"], row["reception_radius_m"],
                                      row["directional_deg"], px, py) for px, py in stations)
            missed.append({"channel": ch, "u_layout": u_layout,
                           "bearings": len(strat.channels[ch].bearings),
                           "clear_calls": strat.clear_calls.get(ch, 0),
                           "path": strat.path.get(ch, [])[-6:]})
        out["missed_detail"] = missed
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-start", type=int, default=350000)
    ap.add_argument("--directional", type=int, default=14)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--label", default="d14")
    args = ap.parse_args()

    res = {}
    for name, cls in (("probe_baseline", PathProbe), ("raygap_fix", RayGapStrategy)):
        rows = [one(args.seed_start + i, args.directional, args.count, cls)
                for i in range(args.episodes)]
        cleared = sum(r["cleared"] for r in rows)
        total = sum(r["total"] for r in rows)
        res[name] = {
            "episodes": len(rows), "cleared": cleared, "sources": total,
            "clear_rate": round(cleared / total, 4),
            "full_clear_episodes": sum(1 for r in rows if r["cleared"] == r["total"]),
            "mean_clear_time_s": round(st.fmean(r["mean_clear_time_s"] for r in rows
                                                if r["mean_clear_time_s"] is not None), 1),
            "measures_mean": round(st.fmean(r["measures"] for r in rows), 1),
            "clears_mean": round(st.fmean(r["clears"] for r in rows), 1),
            "rows": rows,
        }
        print(f"{name}: cleared={cleared}/{total} ({cleared/total:.4f}) "
              f"full={res[name]['full_clear_episodes']}/{len(rows)} "
              f"meanT={res[name]['mean_clear_time_s']}", flush=True)

    # 退出路径统计
    paths = Counter()
    for r in res["probe_baseline"]["rows"]:
        for m in r.get("missed_detail", []):
            key = ("u_layout" if m["u_layout"] else "not_u_layout")
            paths[f"{key} | bearings={min(m['bearings'],2)} | clears={'0' if m['clear_calls']==0 else '>0'}"] += 1
    print("\nmissed exit-path buckets:", json.dumps(dict(paths), ensure_ascii=False, indent=2))

    # 配对差异
    bl = {r["seed"]: r for r in res["probe_baseline"]["rows"]}
    fx = {r["seed"]: r for r in res["raygap_fix"]["rows"]}
    d_cleared = sum(fx[s]["cleared"] - bl[s]["cleared"] for s in bl)
    dt = [fx[s]["mean_clear_time_s"] - bl[s]["mean_clear_time_s"] for s in bl
          if fx[s]["mean_clear_time_s"] and bl[s]["mean_clear_time_s"]]
    paired = {"d_cleared_sources": d_cleared,
              "d_mean_clear_time_s": round(st.fmean(dt), 1),
              "d_pct": round(st.fmean(dt) / res["probe_baseline"]["mean_clear_time_s"] * 100, 2),
              "better": sum(1 for v in dt if v < -1), "worse": sum(1 for v in dt if v > 1)}
    print("paired:", json.dumps(paired, ensure_ascii=False))

    p = Path(f"results/b_attrib_raygap_{args.label}.json")
    p.write_text(json.dumps({"config": vars(args),
                             "summary": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                         for k, v in res.items()},
                             "missed_exit_paths": dict(paths), "paired": paired,
                             "detail": res}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
