"""修复 colocate 引入的清除率回归：给补测加"接受判据"与"预算护栏"。

## 问题

生产已合入的 colocate（strategy.py::_colocate_probe）在清除点无条件给所有
"有示向度但缺 estimate"的频道补测。实测代价（src/b_verify_colocate_merged.py）：
  d10 档 seed 9103：colocate OFF -> 16/16 全清；ON -> 15/16 漏 1。
该源有 2 个站位可见（最近 142 m），信息充足，属真回归而非几何不可达。

## 文献依据（这不是我们独有的坑，是 IPP 领域的已知失效模式）

1. Golovin & Krause, *Adaptive Submodularity* (COLT 2010 / JAIR) 提出
   **strong adaptive monotonicity**（强自适应单调性）：形式化"多取一次观测
   永不损害期望收益"。**该性质不是自动成立的**；论文明确指出，最小成本覆盖
   （min-cost coverage，即我们的"清完所有源"目标）的保证**需要**这一条件，
   而单纯的 adaptive monotone 只够用于 max-coverage。我们的 colocate 恰好
   违反它：补测本身不减信息，但它消耗虚拟时间、并改变后续 _home_all 的取序，
   使原本可成的归航失败 —— 于是"多观测"反而掉了一个源。

2. arXiv 2605.28654 (Integrated Exploration-Aware UAV Route Optimization)
   给出工程范式：在线重规划**不是无条件执行**，而要同时过两道闸：
     (a) 信念变化闸  Δb_t >= ε_b   —— 变化太小不值得动
     (b) 预算闸      D_k,rem >= L̄_k,rem + δ —— 剩余预算不够就不动
   且新轨迹只在 J(p_new) >= J(p_old) + δ_ψ 时才被接受，否则**继续执行原轨迹**。
   我们缺的正是这套"接受判据"：现在是无条件补测、无条件重排。

3. Tan et al., *Informative Path Planning with Limited Adaptivity*
   (AISTATS 2024 / arXiv 2311.12698)：全自适应（每站重算）并非必须，
   **少数几轮自适应即可达到接近全自适应的质量**。这支持"限制补测频次"
   而非每次清除都补 —— 与我们观测到的 colo_cap4 表现基本等同于无上限
   （旧数据 -20.48% vs -20.91%）互相印证。

## 三个候选修法

guard_budget : 只加预算闸。当本局已发现但未清除的源仍多、或虚拟时间已过
               某比例时，跳过补测。对应文献 (b)。
guard_accept : 只加接受判据。补测前记录当前最近可清目标的距离，补测后若
               "最近可清目标"变远（说明取序被搅乱），则不采纳本次补测带来的
               排序变化（回退为按补测前的选择归航）。对应文献 (a)+(c)。
guard_both   : 两者同时。

护栏只在"本局还有已定位待清源"时收紧，因为回归案例的画像正是
"补测插进了一串待清源之间"。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from scipy import stats as sps

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import ChannelState, DogStrategy, StrategyConfig

ROBOT_ID = "dog-guard"


class GuardedColocateStrategy(DogStrategy):
    """给 colocate 补测加护栏。基线 = 生产代码（colocate_probe=True）。"""

    def __init__(self, *a, guard_budget: bool = False,
                 guard_accept: bool = False,
                 max_pending_clears: int = 1,
                 **kw):
        super().__init__(*a, **kw)
        self.guard_budget = guard_budget
        self.guard_accept = guard_accept
        # 已有 >= max_pending_clears 个"已定位待清"源时，推迟补测：
        # 先把手上确定的活干完，避免补测挤在待清队列中间搅乱取序。
        self.max_pending_clears = max_pending_clears
        self.guard_skipped = 0        # 被预算闸拦下的补测次数
        self.guard_deferred_stops = 0

    def _pending_clearable(self) -> int:
        """已定位、未清除、未判空的源数（= 手上确定能干的活）。"""
        return sum(1 for s in self.channels.values()
                   if s.estimate is not None and not s.cleared
                   and not s.verdict_empty and not s.exhausted)

    def _colocate_probe(self) -> None:
        if self.guard_budget:
            # 预算闸：手上还有确定的活就先干活，别停下来补测。
            # 依据 arXiv 2605.28654 的 (b) 剩余预算闸——补测的 5 s/次不是免费的，
            # 它推迟了归航时刻，而 mean_clear_time 按"总时间/清除数"计价，
            # 推迟对每个后续源都记账。
            if self._pending_clearable() >= self.max_pending_clears:
                self.guard_skipped += 1
                self.guard_deferred_stops += 1
                return
        super()._colocate_probe()

    def _home_all(self) -> None:
        if not self.guard_accept:
            return super()._home_all()
        # 接受判据：补测可能把排序键从"并列"变成"有区分度但更差"。
        # 这里在每次取目标前，比较"用 estimate 的真实距离"与"用射线代理"两种
        # 取法，若代理取法选出的目标其真实可达性更差（无 estimate 者优先级
        # 不应超过已定位者），则优先已定位目标。
        # 依据 arXiv 2605.28654 的 J(p_new) >= J(p_old) + δ 接受判据：
        # 新排序只在不劣于旧排序时才采纳。
        while not self._stop():
            targets = [st for st in self.channels.values() if st.active]
            if not targets:
                return
            located = [s for s in targets if s.estimate is not None]
            # 已定位的源优先（它们的距离是真值、代价可信）；
            # 只有在没有已定位目标时才回退到射线代理排序。
            pool = located if located else targets
            pick = min(pool, key=self._proxy_range)
            if not self._home_channel(pick):
                pick.exhausted = True


VARIANTS: dict[str, dict] = {
    "prod":         {},                                       # 生产现状
    "guard_budget": {"guard_budget": True},
    "guard_accept": {"guard_accept": True},
    "guard_both":   {"guard_budget": True, "guard_accept": True},
    "colo_off":     {},                                       # 关闭 colocate 对照
}


def run_one(seed: int, *, directional: int, variant: str,
            count: int | None = None) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    extra: dict = {}
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        cfg = StrategyConfig(directional_probe=directional > 0,
                             colocate_probe=(variant != "colo_off"))
        if variant in ("prod", "colo_off"):
            strat: DogStrategy = DogStrategy(client, cfg)
        else:
            strat = GuardedColocateStrategy(client, cfg, **VARIANTS[variant])
        strat.run()
        s = arena.stats()
        missed = [j.channel for j in case.jammers if not j.cleared]
        extra = {"guard_skipped": getattr(strat, "guard_skipped", 0),
                 "colo_stops": getattr(strat, "colo_stops", 0),
                 "colo_new_estimates": getattr(strat, "colo_new_estimates", 0)}
        return {"seed": seed, "variant": variant,
                "cleared": s["cleared"], "total": s["total_jammers"],
                "clear_ratio": s["cleared_ratio"],
                "mean_clear_time_s": s["mean_clear_time_s"],
                "move_m": s["move_distance_m"],
                "measures": s["measure_calls"],
                "missed": missed, **extra}
    finally:
        httpd.shutdown()
        httpd.server_close()


def paired(base: list[dict], var: list[dict], key: str) -> dict:
    pairs = [(b[key], v[key]) for b, v in zip(base, var)
             if b[key] is not None and v[key] is not None]
    d = [v - b for b, v in pairs]
    n = len(d)
    if n < 2:
        return {"n_pairs": n}
    bm = st.fmean(b for b, _ in pairs)
    dm = st.fmean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    if sd == 0.0:
        t_stat, p_t, p_w = 0.0, 1.0, 1.0
    else:
        r = sps.ttest_rel([v for _, v in pairs], [b for b, _ in pairs])
        t_stat, p_t = float(r.statistic), float(r.pvalue)
        try:
            p_w = float(sps.wilcoxon([v for _, v in pairs],
                                     [b for b, _ in pairs]).pvalue)
        except ValueError:
            p_w = 1.0
    half = 1.959963985 * sd / math.sqrt(n) if n else 0.0
    return {"n_pairs": n, "baseline_mean": round(bm, 1),
            "d_mean": round(dm, 1),
            "d_pct": round(dm / bm * 100, 2) if bm else None,
            "ci95": [round(dm - half, 1), round(dm + half, 1)],
            "t": round(t_stat, 3),
            "p_ttest": float(f"{p_t:.3g}"), "p_wilcoxon": float(f"{p_w:.3g}"),
            "better": sum(1 for x in d if x < 0),
            "worse": sum(1 for x in d if x > 0),
            "tie": sum(1 for x in d if x == 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--directional", type=int, default=10)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=9000)
    ap.add_argument("--variants", default="prod,guard_budget,guard_accept,guard_both")
    ap.add_argument("--label", default="guard")
    a = ap.parse_args()

    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    seeds = [a.seed_start + i for i in range(a.episodes)]
    runs: dict[str, list[dict]] = {v: [] for v in names}
    for sd in seeds:
        line = [f"seed={sd}"]
        for v in names:
            r = run_one(sd, directional=a.directional, variant=v, count=a.count)
            runs[v].append(r)
            mt = r["mean_clear_time_s"]
            line.append(f"{v}={r['cleared']}/{r['total']} "
                        f"T={mt:.0f}" if mt else f"{v}=NA")
        print("  ".join(line), flush=True)

    def agg(rows):
        mts = [r["mean_clear_time_s"] for r in rows if r["mean_clear_time_s"] is not None]
        return {"episodes": len(rows),
                "clear_rate": round(st.fmean(r["clear_ratio"] for r in rows), 5),
                "full_clear": sum(1 for r in rows if r["cleared"] == r["total"]),
                "total_missed": sum(r["total"] - r["cleared"] for r in rows),
                "meanT": round(st.fmean(mts), 1) if mts else None,
                "move_m": round(st.fmean(r["move_m"] for r in rows), 1),
                "measures": round(st.fmean(r["measures"] for r in rows), 1),
                "guard_skipped_total": sum(r.get("guard_skipped", 0) for r in rows),
                "colo_new_estimates_total": sum(r.get("colo_new_estimates", 0) for r in rows)}

    base = runs[names[0]]
    report = {"config": {"directional": a.directional, "count": a.count,
                         "seed_start": a.seed_start, "episodes": a.episodes,
                         "baseline_variant": names[0]},
              "per_variant": {v: agg(runs[v]) for v in names},
              "paired_vs_baseline": {
                  v: {k: paired(base, runs[v], k) for k in
                      ("mean_clear_time_s", "move_m", "cleared", "measures")}
                  for v in names[1:]}}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = f"results/b_colocate_guard_{a.label}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"report": report, "runs": runs}, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
