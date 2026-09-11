"""两个局部优化的隔离配对实验：(a) 清除成功后立即停止 (b) 573 m 单示向度直清。

设计要点
--------
(a) **已在 HEAD 内**（commit f7ec34a 给 _spiral / _home_channel / _clear_at 三处
    加了 `state.cleared` 判断）。因此本文件把 (a) 做成**消融**（ablation）：
    OFF_A 变体复刻 f7ec34a 之前的行为，用它当基线，HEAD 行为当处理组。
    这样 (a) 的收益 = (ON_A - OFF_A) / OFF_A，方向与"新增优化"一致。

(b) 573 m 阈值。机器狗不知道到源的真实距离，必须用**可观测量**推出距离上界
    r_max。本文件用两条严格成立（非启发式、不作弊）的界：

    C0 接收半径上界：拿到示向度 => r <= R_recv <= RECEPTION_MAX_M = 1500 m。
       这是题面常量，不是真值泄露。

    C1 沉默点几何界（仅全向情形严格成立）：
       设在 Q 拿到示向度、方向单位向量 u，源在 Q + r*u；又设曾在 P 检测沉默。
       沉默 => dist(P, src) > R_recv；有信号 => r = dist(Q, src) <= R_recv。
       故 dist(P, src) > r。令 D = |PQ|、alpha = <(P-Q), u>：
           |Q + r u - P|^2 > r^2
           => |v|^2 + 2 r (v.u) > 0,  v = Q - P
           => r < D / (2 cos alpha)          （仅当 cos alpha > 0，即 P 在射线前方）
       即"射线前方 D 处沉默 => 源在 D/(2 cos alpha) 以内"。
       示向度有 ±1° 误差，为保守取 alpha_eff = alpha + 1°（cos 更小 => 界更松）。
       定向源的沉默可能来自"落在辐射扇区外"而非"超出接收半径"，此界不成立，
       故 Q4（directional_probe=True）默认关闭 C1。

    r_max = min(C0, 所有 C1) 。若 r_max <= 573 m，则单条示向度的横向不确定度
    573*tan(1°) = 10.0 m < 20 m 清除半径，无需第二基线；改为沿射线在
    (0, r_max] 上按 32 m 步长铺清除点（只打中心线）。

    注：题面给的 573 m 来自 2*r*tan(1°) <= 20（用楔形全宽）。若瞄准中心线，
    真源横向偏差只有 r*tan(1°)，锐阈值是 20/tan(1°) = 1145.9 m。本文件用
    b_threshold_m 参数暴露，默认 573（题面口径），并额外测 1146 口径。

用法
----
  python -m src.b_micro_opt --episodes 40 --seed-start 5000 --directional 0
  python -m src.b_micro_opt --episodes 40 --seed-start 6000 --directional 5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from .b_sim.arena import (
    ARENA_RADIUS,
    CHANNEL_MAX,
    CHANNEL_MIN,
    CLEAR_RADIUS_M,
    RECEPTION_MAX_M,
    Arena,
    generate_case,
)
from .b_sim.client import ClearResult, ProtocolError, RobotClient, TestEnded
from .b_sim.server import Session, make_server
from .b_sim.strategy import ChannelState, DogStrategy, StrategyConfig

ROBOT_ID = "dog-micro"

# (b) 阈值：2*r*tan(1°) <= 20 => r <= 573.0 m（题面口径，保守 2 倍）
THRESHOLD_573 = CLEAR_RADIUS_M / (2.0 * math.tan(math.radians(1.0)))
# 锐口径：瞄中心线时横向偏差 r*tan(1°) <= 20 => r <= 1145.9 m
THRESHOLD_1146 = CLEAR_RADIUS_M / math.tan(math.radians(1.0))


class MicroStrategy(DogStrategy):
    """可开关 (a)/(b) 的实验变体。

    opt_a=True  -> HEAD 行为（清除成功立刻停）
    opt_a=False -> 复刻 f7ec34a 之前（成功后仍把螺旋点打完）
    opt_b=True  -> 单示向度且可判定 r_max <= 阈值时跳过 _dogleg，直接沿射线清除
    """

    def __init__(self, client, config=None, *, opt_a: bool = True, opt_b: bool = False,
                 b_threshold_m: float = THRESHOLD_573,
                 b_use_silence_bound: bool = True,
                 b_force: bool = False) -> None:
        super().__init__(client, config)
        self.opt_a = opt_a
        self.opt_b = opt_b
        self.b_threshold_m = b_threshold_m
        self.b_use_silence_bound = b_use_silence_bound
        # b_force：无视距离上界，单示向度时直接赌"源在阈值内"，沿射线铺点。
        # 这不是作弊（只用示向度），但也不是可证明的判据——它是一个赌注。
        # 用来把"判据不可行"与"动作本身无价值"两件事分开量化。
        self.b_force = b_force
        # 仪表
        self.stat_wasted_clears = 0        # 已 cleared 后仍发出的 /clear 调用数
        self.stat_b_fired = 0              # (b) 判据触发次数
        self.stat_b_success = 0            # (b) 触发后清除成功次数
        self.stat_b_clear_calls = 0        # (b) 路径消耗的 /clear 调用数
        self.stat_b_rmax = []              # 触发时的 r_max
        self.stat_dogleg_calls = 0
        self.stat_rmax_probe = []          # 每次 estimate is None 时算出的 r_max（诊断）

    # ---- (a) 的开关面 ----

    def _clear_at(self, point, channel):
        state = self.channels[channel]
        if state.cleared:
            if self.opt_a:
                return True                     # HEAD：不发调用
            self.stat_wasted_clears += 1        # 消融：真的发出去，白烧 3 s
        state.clear_attempts += 1
        try:
            result: ClearResult = self.client.clear(point[0], point[1], channel)
        except (TestEnded, ProtocolError) as exc:
            self.events.append(f"clear stopped: {exc}")
            self._stopped = True
            return False
        self.pos = point
        if result.ok:
            state.cleared = True
        return result.ok

    def _spiral(self, state: ChannelState) -> bool:
        if state.estimate is None:
            return False
        cx, cy = state.estimate
        cfg = self.config
        for k in range(cfg.spiral_points):
            if self.opt_a and state.cleared:
                return state.cleared
            if self._stop() or state.clear_attempts >= cfg.max_clear_attempts:
                return state.cleared
            theta = 2.0 * math.pi * k / cfg.spiral_points
            point = (cx + cfg.spiral_radius_m * math.cos(theta),
                     cy + cfg.spiral_radius_m * math.sin(theta))
            if self._clear_at(point, state.channel):
                return True
        return False

    # ---- (b) 的判据：可观测的距离上界 ----

    def _distance_upper_bound(self, state: ChannelState) -> float:
        """只用可观测量给出源到最后一个示向度观测点的距离上界。

        不读真值：仅用 (1) 题面常量 RECEPTION_MAX_M，(2) 本频道自己的沉默点集合，
        (3) 最后一条示向度读数。
        """
        last = state.bearings[-1]
        r_max = RECEPTION_MAX_M
        if not self.b_use_silence_bound or self.config.directional_probe:
            return r_max
        rad = math.radians(last.svd_deg)
        ux, uy = math.cos(rad), math.sin(rad)
        for px, py in state.silent_points:
            dx, dy = px - last.x, py - last.y
            D = math.hypot(dx, dy)
            if D < 1e-6:
                continue
            cos_a = (dx * ux + dy * uy) / D
            cos_a = max(-1.0, min(1.0, cos_a))
            alpha = math.acos(cos_a) + math.radians(1.0)   # ±1° 保守放宽
            if alpha >= math.pi / 2 - 1e-9:
                continue                                    # P 不在射线前方，无界
            bound = D / (2.0 * math.cos(alpha))
            if bound < r_max:
                r_max = bound
        return r_max

    def _direct_ray_clear(self, state: ChannelState, r_max: float) -> bool:
        """已判定 r <= r_max <= 阈值：沿单条示向度铺清除点，不做交会。

        步长 32 m < 2*20 m 保证射线被覆盖；r <= 573 m 时横向偏差
        573*tan(1°) = 10 m < 20 m，只打中心线即可。
        """
        last = state.bearings[-1]
        rad = math.radians(last.svd_deg)
        ux, uy = math.cos(rad), math.sin(rad)
        step = self.config.ray_clear_step_m
        budget_start = state.clear_attempts
        r = step
        limit = r_max + CLEAR_RADIUS_M
        while r <= limit:
            if self._stop() or state.cleared:
                break
            if state.clear_attempts - budget_start >= self.config.ray_clear_max_attempts:
                break
            px, py = last.x + ux * r, last.y + uy * r
            if math.hypot(px, py) <= ARENA_RADIUS + CLEAR_RADIUS_M:
                self.stat_b_clear_calls += 1
                if self._clear_at((px, py), state.channel):
                    self.stat_b_success += 1
                    self.events.append(
                        f"ch{state.channel}: opt-b direct ray clear at r={r:.0f}m "
                        f"(r_max={r_max:.0f}m)")
                    return True
            r += step
        return state.cleared

    def _dogleg(self, state: ChannelState) -> bool:
        self.stat_dogleg_calls += 1
        return super()._dogleg(state)

    # ---- 归航主循环：复制 HEAD 版本，把 (a)/(b) 做成条件分支 ----

    def _home_channel(self, state: ChannelState) -> bool:
        cfg = self.config
        while not self._stop():
            if self.opt_a and state.cleared:
                return True
            budget = cfg.max_homing_measures + (6 if state.revived else 0)
            if len(state.silent_points) >= 8:
                budget += cfg.narrow_sector_extra_measures
            if state.homing_measures >= budget:
                self.events.append(f"ch{state.channel}: measure budget exhausted")
                return self._final_attempt(state)
            if state.clear_attempts >= cfg.max_clear_attempts:
                self.events.append(f"ch{state.channel}: clear budget exhausted")
                return self._ray_blind_clear(state)
            silent_tolerance = cfg.silent_tolerance
            if state.bearings and len(state.silent_points) >= 6:
                silent_tolerance = cfg.narrow_sector_silent_tolerance
            if state.consecutive_silent >= silent_tolerance:
                self.events.append(f"ch{state.channel}: lost signal, defer to ring probe")
                if state.bearings:
                    return self._ray_blind_clear(state)
                return False

            if state.estimate is None:
                # ---- 优化 (b) 插入点：单示向度且已可判定进入阈值内 ----
                if self.opt_b and state.bearings:
                    r_max = self._distance_upper_bound(state)
                    self.stat_rmax_probe.append(round(r_max, 1))
                    if self.b_force:
                        r_max = min(r_max, self.b_threshold_m)
                    if r_max <= self.b_threshold_m:
                        self.stat_b_fired += 1
                        self.stat_b_rmax.append(round(r_max, 1))
                        if self._direct_ray_clear(state, r_max):
                            return True
                        # 直清失败：回落到原路径（继续交会）
                if not self._dogleg(state):
                    return False
                continue

            distance = math.hypot(state.estimate[0] - self.pos[0],
                                  state.estimate[1] - self.pos[1])
            if distance <= cfg.clear_distance_m:
                if self._clear_at(state.estimate, state.channel):
                    return True
                if not self._spiral(state):
                    return False
                return state.cleared
            if not self._step_closer(state, distance):
                return False
        return False


VARIANTS = {
    "base":     dict(opt_a=False, opt_b=False),   # f7ec34a 之前（两个优化都关）
    "a_only":   dict(opt_a=True,  opt_b=False),   # = 当前 HEAD
    "b_only":   dict(opt_a=False, opt_b=True),
    "ab":       dict(opt_a=True,  opt_b=True),
    # 锐阈值口径的 (b)（诊断用，不进主表）
    "ab_1146":  dict(opt_a=True,  opt_b=True, b_threshold_m=THRESHOLD_1146),
    # (b) 只用接收半径上界（不用沉默点几何界），检验判据强度
    "ab_c0":    dict(opt_a=True,  opt_b=True, b_use_silence_bound=False),
    # 强制版：不等判据成立，单示向度就赌。用来测"动作本身"的价值上限/风险。
    "ab_force573":  dict(opt_a=True, opt_b=True, b_force=True, b_threshold_m=THRESHOLD_573),
    "ab_force1146": dict(opt_a=True, opt_b=True, b_force=True, b_threshold_m=THRESHOLD_1146),
}


def run_one(args) -> dict:
    seed, variant, directional, count = args
    kw = dict(VARIANTS[variant])
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=1200.0)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        strat = MicroStrategy(client, StrategyConfig(directional_probe=directional > 0), **kw)
        strat.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    s = arena.stats()
    return {
        "seed": seed,
        "variant": variant,
        "total": s["total_jammers"],
        "cleared": s["cleared"],
        "mean_clear_time_s": s["mean_clear_time_s"],
        "total_virtual_time_s": round(s["total_virtual_time_s"], 3),
        "measure_calls": s["measure_calls"],
        "clear_calls": s["clear_calls"],
        "move_distance_m": round(s["move_distance_m"], 1),
        "wasted_clears": strat.stat_wasted_clears,
        "b_fired": strat.stat_b_fired,
        "b_success": strat.stat_b_success,
        "b_clear_calls": strat.stat_b_clear_calls,
        "b_rmax": strat.stat_b_rmax,
        "dogleg_calls": strat.stat_dogleg_calls,
        "rmax_probe": strat.stat_rmax_probe,
    }


# ---- 配对统计 ----

def _t_crit(df: int) -> float:
    """t 分布 97.5% 分位（双侧 95%），小自由度查表 + 大样本正态近似。"""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
             25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990,
             100: 1.984, 120: 1.980}
    if df <= 0:
        return float("nan")
    keys = sorted(table)
    for k in keys:
        if df <= k:
            return table[k]
    return 1.96


def paired(treat: dict, ctrl: dict, key: str = "mean_clear_time_s") -> dict:
    """同 seed 配对差值统计。差值定义 = treat - ctrl（负数 = 变快 = 更好）。"""
    seeds = sorted(set(treat) & set(ctrl))
    diffs, rels, base_vals = [], [], []
    better = worse = tie = 0
    net_cleared = 0
    for s in seeds:
        a, b = treat[s], ctrl[s]
        net_cleared += a["cleared"] - b["cleared"]
        va, vb = a[key], b[key]
        if va is None or vb is None:
            continue
        d = va - vb
        diffs.append(d)
        base_vals.append(vb)
        rels.append(d / vb if vb else 0.0)
        if d < -1e-9:
            better += 1
        elif d > 1e-9:
            worse += 1
        else:
            tie += 1
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean_d = st.fmean(diffs)
    med_d = st.median(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    tc = _t_crit(n - 1)
    lo, hi = mean_d - tc * se, mean_d + tc * se
    mean_base = st.fmean(base_vals)
    # 相对收益按"总量比"定义（与均值指标一致），并给出配对相对差的 CI
    rel_mean = st.fmean(rels)
    rel_sd = st.stdev(rels) if n > 1 else 0.0
    rel_se = rel_sd / math.sqrt(n) if n > 1 else 0.0
    rel_lo, rel_hi = rel_mean - tc * rel_se, rel_mean + tc * rel_se
    return {
        "n": n,
        "ctrl_mean": round(mean_base, 2),
        "treat_mean": round(st.fmean([treat[s][key] for s in seeds if treat[s][key] is not None]), 2),
        "diff_mean_s": round(mean_d, 3),
        "diff_median_s": round(med_d, 3),
        "diff_sd_s": round(sd, 3),
        "ci95_s": [round(lo, 3), round(hi, 3)],
        "gain_pct": round(100.0 * mean_d / mean_base, 4),
        "rel_gain_pct_mean": round(100.0 * rel_mean, 4),
        "rel_ci95_pct": [round(100.0 * rel_lo, 4), round(100.0 * rel_hi, 4)],
        "significant": not (lo <= 0.0 <= hi),
        "better": better, "worse": worse, "tie": tie,
        "net_cleared_delta": net_cleared,
    }


def counter_stats(rows: dict, keys=("clear_calls", "measure_calls", "move_distance_m",
                                    "total_virtual_time_s", "cleared")) -> dict:
    return {k: round(st.fmean([r[k] for r in rows.values()]), 2) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--directional", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", default="base,a_only,b_only,ab,ab_1146,ab_c0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [args.seed_start + i for i in range(args.episodes)]
    jobs = [(s, v, args.directional, args.count) for v in variants for s in seeds]

    results: dict[str, dict[int, dict]] = {v: {} for v in variants}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run_one, jobs, chunksize=2):
            results[r["variant"]][r["seed"]] = r
            print(f"{r['variant']:9s} seed={r['seed']} cleared={r['cleared']}/{r['total']} "
                  f"meanT={None if r['mean_clear_time_s'] is None else round(r['mean_clear_time_s'],1)} "
                  f"clears={r['clear_calls']} waste={r['wasted_clears']} bfire={r['b_fired']}",
                  flush=True)

    comparisons = {}
    pairs = [("a_only", "base", "opt_a"),
             ("b_only", "base", "opt_b"),
             ("ab", "base", "combined"),
             ("ab", "a_only", "b_given_a"),
             ("ab_1146", "a_only", "b1146_given_a"),
             ("ab_c0", "a_only", "b_c0only_given_a"),
             ("ab_force573", "a_only", "bforce573_given_a"),
             ("ab_force1146", "a_only", "bforce1146_given_a")]
    for treat, ctrl, name in pairs:
        if treat in results and ctrl in results and results[treat] and results[ctrl]:
            comparisons[name] = paired(results[treat], results[ctrl])

    per_variant = {v: counter_stats(results[v]) for v in variants if results[v]}
    # (b) 判据诊断
    diag = {}
    for v in variants:
        if not results[v]:
            continue
        fired = sum(r["b_fired"] for r in results[v].values())
        succ = sum(r["b_success"] for r in results[v].values())
        probes = [x for r in results[v].values() for x in r["rmax_probe"]]
        rmaxes = [x for r in results[v].values() for x in r["b_rmax"]]
        diag[v] = {
            "b_fired_total": fired,
            "b_success_total": succ,
            "b_clear_calls_total": sum(r["b_clear_calls"] for r in results[v].values()),
            "dogleg_calls_total": sum(r["dogleg_calls"] for r in results[v].values()),
            "single_bearing_decision_points": len(probes),
            "rmax_probe_min": round(min(probes), 1) if probes else None,
            "rmax_probe_median": round(st.median(probes), 1) if probes else None,
            "rmax_probe_frac_le_573": round(sum(1 for x in probes if x <= THRESHOLD_573) / len(probes), 4) if probes else None,
            "rmax_probe_frac_le_1146": round(sum(1 for x in probes if x <= THRESHOLD_1146) / len(probes), 4) if probes else None,
            "fired_rmax_mean": round(st.fmean(rmaxes), 1) if rmaxes else None,
            "wasted_clears_total": sum(r["wasted_clears"] for r in results[v].values()),
        }

    payload = {
        "config": {"episodes": args.episodes, "seed_start": args.seed_start,
                   "directional": args.directional, "count": args.count,
                   "variants": variants,
                   "threshold_573_m": round(THRESHOLD_573, 2),
                   "threshold_1146_m": round(THRESHOLD_1146, 2)},
        "per_variant_means": per_variant,
        "paired": comparisons,
        "b_criterion_diag": diag,
        "episodes": {v: list(results[v].values()) for v in variants},
    }
    label = f"q{4 if args.directional else 3}"
    out = Path(args.out or f"results/b_micro_{label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== per-variant means ==")
    print(json.dumps(per_variant, ensure_ascii=False, indent=2))
    print("\n== paired (diff = treat - ctrl, negative = faster = better) ==")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print("\n== (b) criterion diagnostics ==")
    print(json.dumps(diag, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
