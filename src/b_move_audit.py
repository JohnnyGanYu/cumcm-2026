"""移动成本审计：81% 的虚拟时间花在移动上，这里定位它到底花在哪。

评分指标是 mean_clear_time_s（每个源被清除的平均时刻），不是总时长。
=> 早清除的源便宜，晚清除的源贵；关键是**顺序**和**绕路**。

分解每一步移动，归因到：
  ring       : 环形站位之间的巡逻
  divert     : 从环上离开去归航（ring -> 目标附近）
  homing     : 归航过程内部（dogleg/step_closer 的逐步逼近）
  return     : 归航结束回到环上下一个站位
  spiral     : 清除失败后的补点
  blind      : 射线盲清除铺点

并回答三个优化问题：
  Q-A 归航往返（divert+return）占多少？若改成"扫完整环再按最优顺序归航"能省多少？
       （注：strategy.py 里 home_batch_stations 曾被证伪，但那是按"站位数"攒批，
        不是按"全局最优顺序"重排；这里给出理论下界供判断值不值得再试。）
  Q-B 清除顺序对 mean_clear_time_s 的影响：当前顺序 vs 最优顺序（TSP 下界）。
  Q-C 早清除的收益：把最容易的源先清掉能压低多少 mean_clear_time_s。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading
import time

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-audit"


class AuditStrategy(DogStrategy):
    """记录每一步移动的归因标签，不改变任何决策。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.moves: list[dict] = []
        self._phase = "init"
        self.clear_events: list[dict] = []
        self.homing_spans: list[dict] = []

    def _measure(self, point, channel):
        prev = self.pos
        r = super()._measure(point, channel)
        if prev != self.pos:
            self.moves.append({
                "phase": self._phase,
                "d": math.dist(prev, self.pos),
                "vt": self.client.virtual_time_s,
            })
        return r

    def _clear_at(self, point, channel):
        prev = self.pos
        ok = super()._clear_at(point, channel)
        if prev != self.pos:
            self.moves.append({
                "phase": self._phase,
                "d": math.dist(prev, self.pos),
                "vt": self.client.virtual_time_s,
            })
        if ok:
            self.clear_events.append({
                "channel": channel, "vt": self.client.virtual_time_s,
                "pos": self.pos,
            })
        return ok

    # 给各阶段打标签
    def _walk_ring(self, ring):
        self._phase = "ring"
        return super()._walk_ring(ring)

    def _home_channel(self, state):
        # 归航开始点（通常在环上）与结束点（源附近），用于算 divert+return 的净往返
        start = self.pos
        self._phase = "homing"
        self.homing_spans.append({"start": start, "start_vt": self.client.virtual_time_s})
        r = super()._home_channel(state)
        self.homing_spans[-1].update({"end": self.pos,
                                      "end_vt": self.client.virtual_time_s,
                                      "channel": state.channel, "ok": bool(r)})
        self._phase = "ring"
        return r

    def _walk_ring_station(self, point):
        """环上站位之间的纯巡逻移动（由 _walk_ring 逐站调用时打点）。"""
        return None

    def _dogleg(self, state):
        self._phase = "homing"
        return super()._dogleg(state)

    def _step_closer(self, state, distance):
        self._phase = "homing"
        return super()._step_closer(state, distance)

    def _spiral(self, state):
        self._phase = "spiral"
        return super()._spiral(state)

    def _ray_blind_clear(self, state):
        self._phase = "blind"
        return super()._ray_blind_clear(state)


def run_one(seed: int, *, directional: int, count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strat = AuditStrategy(client, StrategyConfig(directional_probe=directional > 0))
        strat.run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    s = arena.stats()
    by = {}
    for m in strat.moves:
        b = by.setdefault(m["phase"], {"d": 0.0, "n": 0})
        b["d"] += m["d"]
        b["n"] += 1
    total_d = sum(v["d"] for v in by.values())

    # 清除时刻序列 -> mean_clear_time_s 与"最优顺序"下界
    truth = {r["channel"]: r for r in case.truth_table()}
    ev = sorted(strat.clear_events, key=lambda e: e["vt"])
    times = [e["vt"] for e in ev]

    # 下界1：同样的清除位置集合，按最近邻+2-opt 排序，只算移动时间(5 m/s)
    pts = [e["pos"] for e in ev]
    lb_mean = _tsp_mean_completion(pts, speed=5.0) if pts else None

    # 归航往返开销：每次 _home_channel 的净位移 vs 实际走的路
    spans = [sp for sp in strat.homing_spans if "end" in sp]
    homing_net = sum(math.dist(sp["start"], sp["end"]) for sp in spans)
    homing_actual = by.get("homing", {"d": 0.0})["d"]
    # 归航结束后回到环上的代价（下一次 ring 移动的起点是源附近，不是环上）
    detour = sum(math.dist(sp["end"], sp["start"]) for sp in spans)  # 回程估计

    return {
        "seed": seed,
        "n_homing_excursions": len(spans),
        "homing_net_m": round(homing_net),
        "homing_actual_m": round(homing_actual),
        "return_to_ring_est_m": round(detour),
        "total": s["total_jammers"],
        "cleared": s["cleared"],
        "virtual_s": s["total_virtual_time_s"],
        "mean_clear_time_s": s["mean_clear_time_s"],
        "move_m_total": round(total_d),
        "by_phase": {k: {"m": round(v["d"]), "n": v["n"],
                         "pct": round(v["d"] / total_d * 100, 1) if total_d else 0}
                     for k, v in sorted(by.items(), key=lambda kv: -kv[1]["d"])},
        "clear_times": [round(t) for t in times],
        "tsp_lower_bound_mean_clear_s": None if lb_mean is None else round(lb_mean),
    }


def _tsp_mean_completion(points, speed: float) -> float:
    """最近邻 + 2-opt，返回从原点出发依次访问所有点的**平均完成时刻**下界。

    这是 mean_clear_time_s 的乐观下界：假设已经知道所有源的位置（免费），
    只付移动的钱。它回答"顺序优化最多能带来多少"。
    """
    n = len(points)
    if n == 0:
        return 0.0
    cur = (0.0, 0.0)
    remaining = list(points)
    order = []
    while remaining:
        nxt = min(remaining, key=lambda p: math.dist(cur, p))
        remaining.remove(nxt)
        order.append(nxt)
        cur = nxt
    # 2-opt 改善总长（对平均完成时刻，短前缀更重要，但先用总长近似）
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                a = order[i - 1] if i else (0.0, 0.0)
                b, c = order[i], order[j]
                d = order[j + 1] if j + 1 < len(order) else None
                before = math.dist(a, b) + (math.dist(c, d) if d else 0)
                after = math.dist(a, c) + (math.dist(b, d) if d else 0)
                if after < before - 1e-9:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
    cur = (0.0, 0.0)
    t = 0.0
    times = []
    for p in order:
        t += math.dist(cur, p) / speed + 3.0  # 移动 + 清除 3 s
        times.append(t)
        cur = p
    return st.fmean(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=80000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--label", default="audit")
    args = ap.parse_args()

    rows = []
    for i in range(args.episodes):
        r = run_one(args.seed_start + i, directional=args.directional, count=args.count)
        rows.append(r)
        print(f"seed={r['seed']} cleared={r['cleared']}/{r['total']} "
              f"meanT={r['mean_clear_time_s']} lb={r['tsp_lower_bound_mean_clear_s']} "
              f"move={r['move_m_total']} " + " ".join(
                  f"{k}={v['pct']}%" for k, v in r["by_phase"].items()), flush=True)

    agg = {}
    for r in rows:
        for k, v in r["by_phase"].items():
            a = agg.setdefault(k, [])
            a.append(v["pct"])
    meanT = st.fmean(r["mean_clear_time_s"] for r in rows if r["mean_clear_time_s"])
    lb = [r["tsp_lower_bound_mean_clear_s"] for r in rows
          if r["tsp_lower_bound_mean_clear_s"]]
    report = {
        "episodes": len(rows),
        "move_share_by_phase_pct": {k: round(st.fmean(v), 1)
                                    for k, v in sorted(agg.items(),
                                                       key=lambda kv: -st.fmean(kv[1]))},
        "mean_move_m": round(st.fmean(r["move_m_total"] for r in rows)),
        "mean_clear_time_s_observed": round(meanT, 1),
        "mean_clear_time_s_tsp_lower_bound": round(st.fmean(lb), 1) if lb else None,
        "headroom_s": round(meanT - st.fmean(lb), 1) if lb else None,
        "headroom_pct": round((meanT - st.fmean(lb)) / meanT * 100, 1) if lb else None,
        "note": "TSP 下界假设所有源位置已知且免费，只付移动+清除；是顺序优化的乐观上限",
    }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_move_audit_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
