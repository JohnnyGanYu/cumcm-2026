"""结构性方案对比：先勘察后清除（survey-then-clear）vs 现状（边走边清）。

已确立的账：
  纯环形勘察成本 = 进环 250 s + 绕环 1553 s + 13 站 x 20 频道 x 5 s 检测 1300 s
                 ≈ 3103 s（不含任何归航）
  实测首次示向度 p50 = 2079 s，尾部源 t_clear 高达 10000 s
  按源平均口径 mean_clear_time = 3807 s，理论下界（发现后 248 s 清除）= 2414 s

现状（`_walk_ring`）：每个站位检测完就 `_home_all()` 归航清除，归航把机器狗带离环，
回来接着走。好处是"就近处理"，坏处是**发现节奏被拖慢**——环走得慢，后半段的源
迟迟得不到第一条示向度。

候选 survey_first：
  阶段 A 只勘察不清除：走完原点 + 整个内环（必要时外环），对每个频道累积示向度，
         期间**不归航**（除了 measure 返回 near 时的顺手清除，那是免费的）。
  阶段 B 一次性规划清除顺序：对所有已定位的源，按"最近邻 + 2-opt"路线依次归航清除。
         这直接优化 mean_clear_time_s（早清除的源便宜）。

预期权衡：
  阶段 A 结束时刻 ≈ 3100 s，此后所有源在一条优化路线上被依次清除。
  若源数 14、路线总长约 30 km（TSP 下界实测），则 mean_clear_time
  ≈ 3100 + (平均完成时刻 ≈ 半程) ≈ 3100 + 1500 = 4600 s —— 可能**比现状更差**，
  因为现状的前几个源在 400-1000 s 就清掉了。
  真正的对比必须实测。这也是为什么要跑这个实验而不是纸上推断。

变体：
  survey_first : 严格两阶段
  hybrid       : 原点阶段先把"原点即可定位"的源清掉（它们便宜且近），
                 然后再进入"只勘察"模式走环，最后统一清除。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from .b_sim.arena import ARENA_RADIUS, Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig

ROBOT_ID = "dog-sf"


class SurveyFirstStrategy(DogStrategy):
    """两阶段：先勘察（不归航），再按优化顺序统一清除。"""

    def __init__(self, *a, clear_origin_first: bool = False, **kw):
        super().__init__(*a, **kw)
        self.survey_mode = False
        self.clear_origin_first = clear_origin_first
        self.phase_a_end_vt = None

    # 勘察阶段禁止归航
    def _home_all(self) -> None:
        if self.survey_mode:
            return
        super()._home_all()

    def run(self) -> dict:
        cfg = self.config
        self.client.enter()
        try:
            # 原点扫描
            if cfg.sweep_origin:
                self._sweep_at((0.0, 0.0), range(1, 21))
            if self.clear_origin_first:
                # 原点即可定位的源：就近清掉（便宜）
                self._home_all()

            # 阶段 A：只勘察，走完所有站位
            self.survey_mode = True
            self._probe_ring()
            self.survey_mode = False
            self.phase_a_end_vt = self.client.virtual_time_s

            # 阶段 B：按优化顺序统一清除
            self._clear_all_in_route_order()

            # 兜底：仍未清除的走原有路径。
            # 实测教训（30 局 val_d5）：只跑一遍兜底会掉 7 个源（清除率 96.7%->73.3%）。
            # 漏失源的接收半径都贴近下限 1000 m、距原点 800-1700 m，属于"环上站位
            # 勉强能听到、但一次归航打不掉"的类型 —— 现状策略靠 `_probe_ring` 与
            # `_home_all` 的**交替往返**反复给它们新几何，而严格两阶段把这个循环拆掉了。
            # 修法：兜底阶段恢复"环形补扫 <-> 归航"的交替，直到无进展为止。
            for _ in range(3):
                before = sum(1 for s in self.channels.values() if s.cleared)
                self._revive_exhausted()
                self._home_all()
                self._inner_retry()
                if cfg.directional_probe:
                    self._belief_search()
                    self._blind_clear_sweep()
                # 仍有未清除且已定位的 -> 再走一轮环补新观测，然后重试
                if self._stop():
                    break
                # 实测教训 2（fix1，12 局）：让 `_probe_ring` 重走整环把耗时抬高 43%
                # （1320->1891 s），因为整环是 13 个站位 x 20 频道的全量重扫。
                # 卡住的通常只有 1-2 个频道，只需给**这些频道**补新几何即可。
                # 修法：对卡住的频道，在其估计位置周围的环上取最近的 3 个站位重测。
                stuck = [s for s in self.channels.values()
                         if not s.cleared and not s.verdict_empty and s.bearings]
                if stuck:
                    self._retarget_stuck(stuck)
                    self._clear_all_in_route_order()
                after = sum(1 for s in self.channels.values() if s.cleared)
                if after == before:
                    break
        finally:
            try:
                self.client.exit()
            except Exception:
                pass
        return self.summary()

    def _retarget_stuck(self, stuck: list) -> None:
        """只给卡住的频道补新观测：绕其估计位置取几个不同方位的近距离观测点。

        比重走整环便宜得多（每频道 3-4 次检测 + 短距移动，而非 13 站 x 20 频道）。
        观测点取在估计位置周围 R=450 m 的 4 个方位上（够长的基线改善交会角，
        又在接收半径下限 1000 m 之内），并裁剪到目标区域内。
        """
        for state in stuck:
            if self._stop():
                return
            if state.estimate is None:
                continue
            ex, ey = state.estimate
            state.exhausted = False
            state.consecutive_silent = 0
            state.ray_swept = False
            pts = []
            for k in range(4):
                ang = math.radians(90.0 * k + 45.0)
                px, py = ex + 450.0 * math.cos(ang), ey + 450.0 * math.sin(ang)
                r = math.hypot(px, py)
                if r > ARENA_RADIUS:
                    px, py = px / r * ARENA_RADIUS, py / r * ARENA_RADIUS
                pts.append((px, py))
            pts.sort(key=lambda q: math.hypot(q[0] - self.pos[0], q[1] - self.pos[1]))
            for p in pts:
                if self._stop() or state.cleared:
                    break
                self._probe_channel_at(p, state)

    def _clear_all_in_route_order(self) -> None:
        """对所有已定位但未清除的频道，按最近邻+2-opt 路线依次归航。"""
        while not self._stop():
            targets = [s for s in self.channels.values()
                       if s.bearings and not s.cleared and not s.exhausted
                       and s.estimate is not None]
            if not targets:
                return
            # 每次取当前位置最近的（等价于最近邻，且随位置更新自然重排）
            targets.sort(key=lambda s: math.hypot(s.estimate[0] - self.pos[0],
                                                  s.estimate[1] - self.pos[1]))
            t = targets[0]
            if not self._home_channel(t):
                t.exhausted = True


def run_one(seed: int, *, directional: int, variant: str,
            count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional > 0)
    extra = {}
    try:
        client = RobotClient(ROBOT_ID, base_url="http://127.0.0.1:" + str(port))
        if variant == "baseline":
            strat = DogStrategy(client, cfg)
        else:
            strat = SurveyFirstStrategy(client, cfg,
                                        clear_origin_first=variant == "hybrid")
            
        strat.run()
        if variant != "baseline":
            extra = {"phase_a_end_vt": None if strat.phase_a_end_vt is None
                     else round(strat.phase_a_end_vt)}
    finally:
        httpd.shutdown()
        httpd.server_close()
    s = arena.stats()
    return {"seed": seed, "variant": variant, "total": s["total_jammers"],
            "cleared": s["cleared"], "ratio": s["cleared_ratio"],
            "virtual_s": s["total_virtual_time_s"],
            "mean_clear_time_s": s["mean_clear_time_s"],
            "measures": s["measure_calls"], "clears": s["clear_calls"],
            "move_m": s["move_distance_m"],
            "missed": [r["channel"] for r in case.truth_table() if not r["cleared"]],
            **extra}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=81000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="baseline,survey_first,hybrid")
    ap.add_argument("--label", default="sf")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        for v in variants:
            r = run_one(seed, directional=args.directional, variant=v, count=args.count)
            rows.append(r)
            print(f"seed={seed} {v:13s} cleared={r['cleared']}/{r['total']} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'])} "
                  f"virt={r['virtual_s']:.0f} phaseA={r.get('phase_a_end_vt','-')} "
                  f"missed={r['missed']}", flush=True)

    def agg(v):
        rs = [r for r in rows if r["variant"] == v]
        mt = [r["mean_clear_time_s"] for r in rs if r["mean_clear_time_s"]]
        return {"variant": v, "episodes": len(rs),
                "full_clear_rate": round(sum(1 for r in rs if r["ratio"] >= 1 - 1e-9) / len(rs), 4),
                "cleared_ratio_mean": round(st.fmean(r["ratio"] for r in rs), 4),
                "total_missed": sum(len(r["missed"]) for r in rs),
                "mean_clear_time_s": round(st.fmean(mt), 1) if mt else None,
                "virtual_s_mean": round(st.fmean(r["virtual_s"] for r in rs)),
                "move_m_mean": round(st.fmean(r["move_m"] for r in rs))}

    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["variant"]] = r
    base_mean = st.fmean(p["baseline"]["mean_clear_time_s"] for p in pairs.values()
                         if "baseline" in p and p["baseline"]["mean_clear_time_s"])
    report = {"per_variant": {v: agg(v) for v in variants},
              "baseline_mean_clear_time_s": round(base_mean, 1),
              "paired_vs_baseline": {}}
    for v in variants:
        if v == "baseline":
            continue
        d = [p[v]["mean_clear_time_s"] - p["baseline"]["mean_clear_time_s"]
             for p in pairs.values() if "baseline" in p and v in p
             and p[v]["mean_clear_time_s"] and p["baseline"]["mean_clear_time_s"]]
        dc = [p[v]["cleared"] - p["baseline"]["cleared"] for p in pairs.values()
              if "baseline" in p and v in p]
        if not d:
            continue
        n = len(d)
        m = st.fmean(d)
        se = st.stdev(d) / math.sqrt(n) if n > 1 else 0.0
        report["paired_vs_baseline"][v] = {
            "episodes": n, "d_mean_clear_time_s": round(m, 1),
            "d_pct_of_baseline": round(m / base_mean * 100, 2),
            "median_s": round(st.median(d), 1),
            "ci95": [round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)] if se else None,
            "t": round(m / se, 2) if se else None,
            "better": sum(1 for x in d if x < -1),
            "worse": sum(1 for x in d if x > 1),
            "tie": sum(1 for x in d if abs(x) <= 1),
            "net_extra_cleared": sum(dc),
        }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_survey_first_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
