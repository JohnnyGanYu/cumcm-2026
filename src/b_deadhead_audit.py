"""量化"归航与扫描交织"造成的段间空驶（deadhead）。

背景：`DogStrategy._walk_ring` 的调度是**交织式**的——每扫完一个环站位就立刻
`_home_all()` 跑去清除，然后回来 `remaining.pop(0)` 取**当初排好的**下一站。
于是每一次归航都在环的巡逻路线上撕开一个口子：出发点是环上站位 A，结束点 B
在源附近（可能离环 1-2 km），而下一段却要从 B 走到"计划中的下一站"。

本脚本把总移动拆成可归因的段，并给出三个互补的空驶度量：

D1 段间接入空驶（reentry deadhead）
    每次前往下一个环站位时，比较"实际接入距离 d(pos, station_planned)"与
    "若允许改接入最近未访问站 d(pos, station_nearest)"。差值即**相位未对齐**
    造成的空驶。只在从归航返回（pos 不在环上）时才可能为正。

D2 归航往返冗余（excursion overhead）
    每次归航段的实际路径长度 path 与其净位移 |AB| 之差，加上返回环的接入距离。
    回答"往返式处理一个源比顺路处理贵多少"。

D3 交织总超额（interleaving excess）
    实际总移动 L_actual 与"已知全部工作点后一次性最优巡回" L_tsp 之差。
    L_tsp = 最近邻+2-opt 巡回，点集 = 原点 ∪ 实际访问过的环站位 ∪ 成功清除点。
    这是信息免费假设下的乐观下界，回答"调度最多还能省多少移动"。

另外报告环段绕圈数 = ring 段移动 / 环周长，用于验证 README 中"绕环 5.6 周"的说法。

自检：本审计子类必须与 `DogStrategy` 逐局**完全一致**（同种子下 cleared /
virtual_s / move_m 三项相同），否则归因数据不可信。`--selfcheck` 打开该校验。

用法：
  python -m src.b_deadhead_audit --episodes 30 --directional 0 --label q3
  python -m src.b_deadhead_audit --episodes 30 --directional 5 --label q4d5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-deadhead"
SPEED_MPS = 5.0


class DeadheadAudit(DogStrategy):
    """只记录，不改决策。`_walk_ring` 是对基线逐行复制 + 打点。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.moves: list[dict] = []
        self._phase = "init"
        self._path_total = 0.0
        self.ring_circumference = 0.0
        self.ring_station_count = 0
        self.stations_visited: list[tuple[float, float]] = []
        self.links: list[dict] = []          # 每次前往下一环站位的接入段
        self.excursions: list[dict] = []     # 每次离环归航
        self.clear_points: list[tuple[float, float]] = []
        self.clear_times: list[float] = []
        self.switches = 0

    # ---- 移动打点 ----

    def _note_move(self, prev: tuple[float, float]) -> None:
        if prev == self.pos:
            return
        d = math.dist(prev, self.pos)
        self._path_total += d
        self.moves.append({"phase": self._phase, "d": d,
                           "vt": self.client.virtual_time_s})

    def _measure(self, point, channel):
        prev = self.pos
        r = super()._measure(point, channel)
        self._note_move(prev)
        return r

    def _clear_at(self, point, channel):
        prev = self.pos
        ok = super()._clear_at(point, channel)
        self._note_move(prev)
        if ok:
            self.clear_points.append(self.pos)
            self.clear_times.append(self.client.virtual_time_s)
        return ok

    # `run()` 里 `_walk_ring` 之外还有几处 `_home_all`（原点普扫后、_revive_exhausted 后、
    # _inner_retry / belief 阶段）。把归航按**调用它的上下文**分桶，否则原点普扫后那一大段
    # 归航会被记进 "init"，看不出交织结构。_tracked_excursion 走 _path_total 差值，不受影响。
    _HOMING_BUCKET = {
        "init": "homing_origin",      # 原点普扫后的第一轮归航（尚未进环）
        "ring": "homing_ring",        # 环上交织归航（本任务关注的段间空驶来源）
        "inner_retry": "homing_inner",
        "belief": "homing_belief",
        "blind_sweep": "homing_blind_sweep",
    }

    def _home_channel(self, state):
        keep = self._phase
        self._phase = self._HOMING_BUCKET.get(keep, keep)
        try:
            return super()._home_channel(state)
        finally:
            self._phase = keep

    def _spiral(self, state):
        keep, self._phase = self._phase, "spiral"
        try:
            return super()._spiral(state)
        finally:
            self._phase = keep

    def _ray_blind_clear(self, state):
        keep, self._phase = self._phase, "blind"
        try:
            return super()._ray_blind_clear(state)
        finally:
            self._phase = keep

    def _inner_retry(self):
        keep, self._phase = self._phase, "inner_retry"
        try:
            return super()._inner_retry()
        finally:
            self._phase = keep

    def _belief_search(self, *a, **kw):
        keep, self._phase = self._phase, "belief"
        try:
            return super()._belief_search(*a, **kw)
        finally:
            self._phase = keep

    def _blind_clear_sweep(self, *a, **kw):
        keep, self._phase = self._phase, "blind_sweep"
        try:
            return super()._blind_clear_sweep(*a, **kw)
        finally:
            self._phase = keep

    # ---- 归航段打点 ----

    def _tracked_excursion(self, fn, kind: str) -> bool:
        """执行一次离环归航并记账。返回本次是否真的移动了（=> 下一段是"返回接入"）。

        不在这里改 `_phase`：让 `_home_channel` 按调用上下文分桶（ring -> homing_ring），
        否则环上交织归航与原点归航会混进同一个 "homing" 桶。
        """
        A = self.pos
        before = self._path_total
        cleared_before = len(self.clear_points)
        fn()
        path = self._path_total - before
        gained = len(self.clear_points) - cleared_before
        if path <= 0.0:
            return False
        self.excursions.append({
            "kind": kind, "A": A, "B": self.pos, "path": path,
            "net": math.dist(A, self.pos), "cleared": gained,
        })
        self.switches += 1
        return True

    # ---- 逐行复制基线 _walk_ring，插入打点 ----

    def _walk_ring(self, ring):
        remaining = self._order_ring(ring)
        n = len(remaining)
        self.ring_station_count += n
        if n > 1:
            self.ring_circumference += sum(
                math.dist(remaining[i], remaining[(i + 1) % n]) for i in range(n))
        since_home = 0
        off_ring = False      # 上一步是否刚从归航返回（=> 本次接入是"重新接入环"）
        while remaining and not self._stop():
            todo = [st for st in self.channels.values() if self._probe_target(st)]
            if not todo:
                break
            station = remaining.pop(0)
            # === D1 打点：接入下一站的空驶 ===
            came_from_excursion = off_ring
            d_actual = math.dist(self.pos, station)
            pool = [station] + remaining
            d_best = min(math.dist(self.pos, s) for s in pool)
            self.links.append({
                "d_actual": d_actual, "d_best": d_best,
                "off_ring": came_from_excursion,
                "n_unvisited": len(pool),
            })
            self.stations_visited.append(station)
            self._phase = "ring"
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop():
                    return
                self._probe_channel_at(station, state)
            since_home += 1
            off_ring = False
            ready = any(st.active for st in self.channels.values())
            if ready and (since_home >= self.config.home_batch_stations or not remaining):
                off_ring |= self._tracked_excursion(self._home_all, "home_all")
                since_home = 0
            if self._stop():
                return
            off_ring |= self._tracked_excursion(self._revive_nearby, "revive_nearby")
        if not self._stop():
            self._tracked_excursion(self._home_all, "home_all_tail")


# ---- TSP 下界 ----

def _nn_2opt(points: list[tuple[float, float]],
             start: tuple[float, float] = (0.0, 0.0)) -> tuple[list, float]:
    """最近邻 + 2-opt，返回 (顺序, 总长)。开放路径（不回起点）。"""
    remaining = list(points)
    order: list[tuple[float, float]] = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda p: math.dist(cur, p))
        remaining.remove(nxt)
        order.append(nxt)
        cur = nxt
    improved = True
    guard = 0
    while improved and guard < 200:
        improved = False
        guard += 1
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                a = order[i - 1] if i else start
                b, c = order[i], order[j]
                d = order[j + 1] if j + 1 < len(order) else None
                before = math.dist(a, b) + (math.dist(c, d) if d else 0.0)
                after = math.dist(a, c) + (math.dist(b, d) if d else 0.0)
                if after < before - 1e-9:
                    order[i:j + 1] = list(reversed(order[i:j + 1]))
                    improved = True
    total = 0.0
    cur = start
    for p in order:
        total += math.dist(cur, p)
        cur = p
    return order, total


def _dedupe(points: list[tuple[float, float]], tol: float = 1.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points:
        if all(math.dist(p, q) > tol for q in out):
            out.append(p)
    return out


def run_one(seed: int, *, directional: int, count: int | None = None,
            selfcheck: bool = False) -> dict:
    """跑一局审计；selfcheck 时再用基线跑同一局并核对三项指标一致。"""
    def _run(strategy_cls):
        case = generate_case(seed, count=count, directional_count=directional)
        arena = Arena(case)
        session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
        httpd = make_server(session, port=0)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
            strat = strategy_cls(client, StrategyConfig(directional_probe=directional > 0))
            strat.run()
        finally:
            httpd.shutdown()
            httpd.server_close()
        return strat, arena, case

    strat, arena, case = _run(DeadheadAudit)
    s = arena.stats()

    mismatch = None
    if selfcheck:
        base, base_arena, _ = _run(DogStrategy)
        b = base_arena.stats()
        same = (abs(b["total_virtual_time_s"] - s["total_virtual_time_s"]) < 1e-6
                and b["cleared"] == s["cleared"]
                and abs(b["move_distance_m"] - s["move_distance_m"]) < 1e-6)
        if not same:
            mismatch = {"baseline": {k: b[k] for k in
                                    ("cleared", "total_virtual_time_s", "move_distance_m")},
                        "audit": {k: s[k] for k in
                                  ("cleared", "total_virtual_time_s", "move_distance_m")}}

    # 相位归因
    by: dict[str, dict] = {}
    for m in strat.moves:
        b = by.setdefault(m["phase"], {"d": 0.0, "n": 0})
        b["d"] += m["d"]
        b["n"] += 1
    total_d = sum(v["d"] for v in by.values())

    # D1 段间接入空驶
    links = strat.links
    d1_total = sum(max(0.0, l["d_actual"] - l["d_best"]) for l in links)
    d1_offring = sum(max(0.0, l["d_actual"] - l["d_best"]) for l in links if l["off_ring"])
    link_actual = sum(l["d_actual"] for l in links)
    link_best = sum(l["d_best"] for l in links)
    offring_links = [l for l in links if l["off_ring"]]

    # D2 归航往返冗余
    exc = strat.excursions
    exc_path = sum(e["path"] for e in exc)
    exc_net = sum(e["net"] for e in exc)
    # D4 无果归航：一次离环归航走了路却没清掉任何源。这是无歧义的浪费
    #（不像"顺序不优"那样只是相对次优），且无需重排即可攻击。
    barren = [e for e in exc if e["cleared"] == 0]
    barren_path = sum(e["path"] for e in barren)
    # 返回环的接入段（紧跟在 excursion 之后的那次 link）
    reentry_actual = sum(l["d_actual"] for l in offring_links)

    # D3 交织总超额
    work_points = _dedupe(strat.stations_visited + strat.clear_points)
    _, tsp_len = _nn_2opt(work_points)
    # 仅扫描的巡回（不含清除点），用于分离"扫描本身"与"归航插入"
    _, tsp_scan = _nn_2opt(_dedupe(strat.stations_visited))

    laps = (by.get("ring", {"d": 0.0})["d"] / strat.ring_circumference
            if strat.ring_circumference > 0 else None)

    return {
        "seed": seed,
        "total": s["total_jammers"],
        "cleared": s["cleared"],
        "ratio": s["cleared_ratio"],
        "virtual_s": s["total_virtual_time_s"],
        "mean_clear_time_s": s["mean_clear_time_s"],
        "move_m": s["move_distance_m"],
        "move_m_traced": round(total_d, 1),
        "by_phase": {k: {"m": round(v["d"], 1), "n": v["n"],
                         "pct": round(v["d"] / total_d * 100, 1) if total_d else 0.0}
                     for k, v in sorted(by.items(), key=lambda kv: -kv[1]["d"])},
        "ring_circumference_m": round(strat.ring_circumference, 1),
        "ring_stations_planned": strat.ring_station_count,
        "ring_stations_visited": len(strat.stations_visited),
        "ring_laps": None if laps is None else round(laps, 2),
        "switches_ring_homing": strat.switches,
        # D1
        "d1_reentry_deadhead_m": round(d1_total, 1),
        "d1_offring_share_m": round(d1_offring, 1),
        "link_actual_m": round(link_actual, 1),
        "link_best_m": round(link_best, 1),
        "n_links": len(links),
        "n_offring_links": len(offring_links),
        "reentry_actual_m": round(reentry_actual, 1),
        # D2
        "n_excursions": len(exc),
        "excursion_path_m": round(exc_path, 1),
        "excursion_net_m": round(exc_net, 1),
        "d2_excursion_overhead_m": round(exc_path - exc_net, 1),
        # D4
        "n_barren_excursions": len(barren),
        "d4_barren_path_m": round(barren_path, 1),
        "d4_barren_pct_of_move": round(barren_path / s["move_distance_m"] * 100, 1)
        if s["move_distance_m"] else None,
        # D3
        "tsp_all_m": round(tsp_len, 1),
        "tsp_scan_only_m": round(tsp_scan, 1),
        "d3_interleaving_excess_m": round(s["move_distance_m"] - tsp_len, 1),
        "d3_excess_pct": round((s["move_distance_m"] - tsp_len) / s["move_distance_m"] * 100, 1)
        if s["move_distance_m"] else None,
        "selfcheck_mismatch": mismatch,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=310000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--label", default="q3")
    args = ap.parse_args()
    if args.count is not None and args.directional > args.count:
        ap.error("--directional 不能超过 --count")

    rows = []
    for i in range(args.episodes):
        r = run_one(args.seed_start + i, directional=args.directional,
                    count=args.count, selfcheck=args.selfcheck)
        rows.append(r)
        print(f"seed={r['seed']} cleared={r['cleared']}/{r['total']} "
              f"move={r['move_m']:.0f} laps={r['ring_laps']} sw={r['switches_ring_homing']} "
              f"D1={r['d1_reentry_deadhead_m']:.0f} D2={r['d2_excursion_overhead_m']:.0f} "
              f"D3={r['d3_interleaving_excess_m']:.0f}({r['d3_excess_pct']}%) "
              f"mismatch={'YES' if r['selfcheck_mismatch'] else 'no'}", flush=True)

    def mean(key) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return round(st.fmean(vals), 1) if vals else 0.0

    def pct_of(key, denom: float) -> float | None:
        return round(mean(key) / denom * 100, 1) if denom else None

    phase_pct: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r["by_phase"].items():
            phase_pct.setdefault(k, []).append(v["pct"])

    mismatches = [r["seed"] for r in rows if r["selfcheck_mismatch"]]
    move_mean = mean("move_m")
    report = {
        "episodes": len(rows),
        "config": {"directional": args.directional, "count": args.count,
                   "seed_start": args.seed_start},
        "selfcheck_run": bool(args.selfcheck),
        "selfcheck_mismatched_seeds": mismatches,
        "clear_rate": round(sum(r["cleared"] for r in rows) / sum(r["total"] for r in rows), 4),
        "full_clear_rate": round(sum(1 for r in rows if r["ratio"] >= 1 - 1e-9) / len(rows), 4),
        "move_m_mean": move_mean,
        "virtual_s_mean": mean("virtual_s"),
        "mean_clear_time_s_mean": mean("mean_clear_time_s"),
        "move_share_by_phase_pct": {k: round(st.fmean(v), 1) for k, v in
                                    sorted(phase_pct.items(), key=lambda kv: -st.fmean(kv[1]))},
        "ring_circumference_m_mean": mean("ring_circumference_m"),
        "ring_laps_mean": mean("ring_laps"),
        "switches_ring_homing_mean": mean("switches_ring_homing"),
        "D1_reentry_deadhead_m_mean": mean("d1_reentry_deadhead_m"),
        "D1_pct_of_total_move": pct_of("d1_reentry_deadhead_m", move_mean),
        "D1_offring_share_m_mean": mean("d1_offring_share_m"),
        "link_actual_m_mean": mean("link_actual_m"),
        "link_best_m_mean": mean("link_best_m"),
        "n_links_mean": mean("n_links"),
        "n_offring_links_mean": mean("n_offring_links"),
        "reentry_actual_m_mean": mean("reentry_actual_m"),
        "D2_excursion_overhead_m_mean": mean("d2_excursion_overhead_m"),
        "D2_pct_of_total_move": pct_of("d2_excursion_overhead_m", move_mean),
        "n_excursions_mean": mean("n_excursions"),
        "excursion_path_m_mean": mean("excursion_path_m"),
        "excursion_net_m_mean": mean("excursion_net_m"),
        "n_barren_excursions_mean": mean("n_barren_excursions"),
        "D4_barren_path_m_mean": mean("d4_barren_path_m"),
        "D4_pct_of_total_move": pct_of("d4_barren_path_m", move_mean),
        "tsp_all_m_mean": mean("tsp_all_m"),
        "tsp_scan_only_m_mean": mean("tsp_scan_only_m"),
        "D3_interleaving_excess_m_mean": mean("d3_interleaving_excess_m"),
        "D3_excess_pct_mean": mean("d3_excess_pct"),
        "note": "D1/D2/D3 是互相重叠的三种视角，不可相加。D3 为乐观下界差值。",
    }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_deadhead_audit_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
