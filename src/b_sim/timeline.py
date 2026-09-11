"""虚拟时间构成分解：把一局（或一批）离线演练的耗时拆成移动/检测/切频/清除四项。

为什么需要它：评分指标是"平均定位清除时间 T̄ = 总虚拟时间 / 已清除个数"，而总虚拟
时间由四类成本相加而成（附件1 计时规则）：

    移动 = 距离 / 5 m·s⁻¹      检测 = 5 s/次      切频 = 1 s/次      清除 = 成功 5 s / 失败 3 s

只看 T̄ 无法判断该往哪优化。本模块给出：

1. 四项占比 —— 指出优化的主战场（实测移动占 ~80%，故路径规划是主要抓手）；
2. **最近邻巡回下界** —— 机器狗最终必须走到每个源的 20 m 内，因此"从原点出发访问
   全部真值点的最近邻巡回长度 / 5"是移动时间的一个可计算参考下界。实际移动时间与
   它的比值（"绕路倍数"）是路径规划质量的直接度量，且与案例难度无关，可跨配置比较；
3. 检测有效率 —— 拿到示向度的检测占比，衡量有多少检测被浪费在沉默点上。

用法：

    python -m src.b_sim.timeline                        # 问题 3，默认 12 局
    python -m src.b_sim.timeline --directional 5        # 问题 4
    python -m src.b_sim.timeline --episodes 20 --seed-start 5000 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
from dataclasses import asdict, dataclass, field

from .arena import (
    CHANNEL_SWITCH_SECONDS,
    CLEAR_FAIL_SECONDS,
    CLEAR_SUCCESS_SECONDS,
    MEASURE_SECONDS,
    SPEED_MPS,
    Arena,
    Case,
    generate_case,
)
from .client import RobotClient
from .server import Session, make_server
from .strategy import DogStrategy, StrategyConfig


@dataclass
class TimeBreakdown:
    """一局的时间构成（秒）与计数。"""

    seed: int = 0
    jammers: int = 0
    cleared: int = 0
    total_s: float = 0.0
    mean_clear_time_s: float | None = None
    move_s: float = 0.0
    measure_s: float = 0.0
    switch_s: float = 0.0
    clear_success_s: float = 0.0
    clear_fail_s: float = 0.0
    move_distance_m: float = 0.0
    nn_tour_m: float = 0.0
    detour_ratio: float = 0.0          # 实际移动 / 最近邻巡回，路径规划质量度量
    measures: int = 0
    measures_direction: int = 0
    measures_near: int = 0
    measures_silent: int = 0
    clears: int = 0
    clears_success: int = 0
    events: list[str] = field(default_factory=list)

    @property
    def measure_yield(self) -> float:
        """检测有效率：返回示向度或 near 的检测占比。"""
        if not self.measures:
            return 0.0
        return (self.measures_direction + self.measures_near) / self.measures


def nearest_neighbour_tour_m(case: Case, start: tuple[float, float] = (0.0, 0.0)) -> float:
    """从 start 出发、用最近邻启发式访问所有干扰源真值点的巡回长度（米，不回原点）。

    这是移动距离的参考基准而非严格下界（最近邻通常比最优 TSP 长 10-25%），但它
    只依赖案例本身、与策略无关，因此适合作为跨配置比较的分母。
    """
    remaining = [(j.x, j.y) for j in case.jammers]
    cur = start
    total = 0.0
    while remaining:
        remaining.sort(key=lambda p: math.hypot(p[0] - cur[0], p[1] - cur[1]))
        nxt = remaining.pop(0)
        total += math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
        cur = nxt
    return total


def decompose(arena: Arena, case: Case, seed: int, events: list[str] | None = None) -> TimeBreakdown:
    """按 arena 轨迹重放计时规则，得到四项成本。

    重放而非埋点：这样分解结果与 `Arena` 的实际计费同源，二者之和必然等于
    `robot.virtual_time_s`（由 `verify()` 断言）。
    """
    out = TimeBreakdown(seed=seed, jammers=case.total, events=list(events or []))
    x, y, channel = 0.0, 0.0, 1
    for ev in arena.robot.trace:
        out.move_s += math.hypot(ev["x"] - x, ev["y"] - y) / SPEED_MPS
        x, y = ev["x"], ev["y"]
        if ev["action"] == "measure":
            out.measure_s += MEASURE_SECONDS
            if ev["channel"] != channel:
                out.switch_s += CHANNEL_SWITCH_SECONDS
            channel = ev["channel"]
            out.measures += 1
            result = ev.get("measure_result")
            out.measures_direction += result == "direction"
            out.measures_near += result == "near"
            out.measures_silent += result == "no_signal"
        else:
            out.clears += 1
            if ev.get("clear_result") == "success":
                out.clear_success_s += CLEAR_SUCCESS_SECONDS
                out.clears_success += 1
            else:
                out.clear_fail_s += CLEAR_FAIL_SECONDS

    stats = arena.stats()
    out.cleared = stats["cleared"]
    out.total_s = stats["total_virtual_time_s"]
    out.mean_clear_time_s = stats["mean_clear_time_s"]
    out.move_distance_m = stats["move_distance_m"]
    out.nn_tour_m = nearest_neighbour_tour_m(case)
    out.detour_ratio = out.move_distance_m / out.nn_tour_m if out.nn_tour_m else 0.0
    return out


def verify(breakdown: TimeBreakdown, tol_s: float = 1e-3) -> None:
    """四项之和必须等于总虚拟时间（计时账目自洽性检查）。"""
    parts = (breakdown.move_s + breakdown.measure_s + breakdown.switch_s
             + breakdown.clear_success_s + breakdown.clear_fail_s)
    if abs(parts - breakdown.total_s) > tol_s:
        raise AssertionError(
            f"seed={breakdown.seed} 时间分解不自洽：四项和 {parts:.6f} != 总计 {breakdown.total_s:.6f}")


def profile_episode(seed: int, *, directional: int = 0, count: int | None = None,
                    config: StrategyConfig | None = None,
                    real_limit_s: float = 1200.0) -> TimeBreakdown:
    """跑一局并返回时间分解。"""
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session("timeline", arena, real_limit_s=real_limit_s)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client = RobotClient("timeline", base_url=f"http://127.0.0.1:{port}")
        summary = DogStrategy(client, config).run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    breakdown = decompose(arena, case, seed, summary.get("events"))
    verify(breakdown)
    return breakdown


def summarize(rows: list[TimeBreakdown]) -> dict:
    """汇总一批分解结果，给出四项均值与占比。"""
    if not rows:
        return {}
    def mean(attr: str) -> float:
        return statistics.fmean(getattr(r, attr) for r in rows)

    total = mean("total_s")
    means = {k: mean(k) for k in ("move_s", "measure_s", "switch_s",
                                  "clear_success_s", "clear_fail_s")}
    finished = [r.mean_clear_time_s for r in rows if r.mean_clear_time_s is not None]
    return {
        "episodes": len(rows),
        "jammers_total": sum(r.jammers for r in rows),
        "cleared_total": sum(r.cleared for r in rows),
        "clear_rate": round(sum(r.cleared for r in rows) / max(1, sum(r.jammers for r in rows)), 4),
        "mean_clear_time_s": round(statistics.fmean(finished), 2) if finished else None,
        "mean_clear_time_s_max": round(max(finished), 2) if finished else None,
        "total_virtual_time_s": round(total, 1),
        "components_s": {k: round(v, 1) for k, v in means.items()},
        "components_share": {k: round(v / total, 4) for k, v in means.items()} if total else {},
        "move_distance_m": round(mean("move_distance_m"), 1),
        "nn_tour_m": round(mean("nn_tour_m"), 1),
        "detour_ratio": round(mean("detour_ratio"), 3),
        "measures": round(mean("measures"), 1),
        "measure_yield": round(statistics.fmean(r.measure_yield for r in rows), 4),
        "clears": round(mean("clears"), 1),
    }


def _print_table(rows: list[TimeBreakdown]) -> None:
    head = (f"{'seed':>6} {'n':>3} {'clr':>4} {'总计':>7} {'T̄':>6} {'移动':>6} "
            f"{'检测':>6} {'切频':>5} {'清除✓':>6} {'清除✗':>6} {'距离m':>7} "
            f"{'NN巡回':>7} {'绕路':>5} {'#测':>4} {'有效率':>6} {'#清':>4}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r.seed:>6} {r.jammers:>3} {r.cleared:>4} {r.total_s:>7.0f} "
              f"{(r.mean_clear_time_s or 0):>6.0f} {r.move_s:>6.0f} {r.measure_s:>6.0f} "
              f"{r.switch_s:>5.0f} {r.clear_success_s:>6.0f} {r.clear_fail_s:>6.0f} "
              f"{r.move_distance_m:>7.0f} {r.nn_tour_m:>7.0f} {r.detour_ratio:>5.2f} "
              f"{r.measures:>4} {r.measure_yield:>6.2f} {r.clears:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B 题虚拟时间构成分解")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--directional", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--json", default=None, help="把逐局分解与汇总写入该 JSON 文件")
    args = parser.parse_args()

    seed_start = args.seed_start if args.seed_start is not None else (
        7001 if args.directional else 5000)
    config = StrategyConfig(directional_probe=args.directional > 0)
    rows = [profile_episode(seed_start + i, directional=args.directional,
                            count=args.count, config=config)
            for i in range(args.episodes)]
    _print_table(rows)
    summary = summarize(rows)
    print("\n== 汇总 ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    move_lb = summary["nn_tour_m"] / SPEED_MPS
    print(f"\n移动时间参考基准（最近邻巡回 / 5 m·s⁻¹）= {move_lb:.0f} s；"
          f"实际移动 {summary['components_s']['move_s']:.0f} s；"
          f"绕路倍数 {summary['detour_ratio']:.2f}x")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "episodes": [asdict(r) for r in rows]},
                      handle, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()


__all__ = ["TimeBreakdown", "decompose", "nearest_neighbour_tour_m",
           "profile_episode", "summarize", "verify"]
