"""针对 Q4 残余漏失的三个候选修复，全部通过子类化实现（不碰 src/b_sim/*）。

归因结论（`src/b_miss_attrib.py deep`，17 个已知漏失源，真值判定）
-----------------------------------------------------------------
* **17/17 的 U_hard 为真** —— 可见区非空，`/clear` 只看距离，所以理论上全部可清除。
  漏失 100% 是策略缺陷，零信息缺陷。
* 14/17 拿到了**恰好 1 条**示向度就再也拿不到第二条 => `ChannelState.estimate`
  永远是 None（`_refresh` 要求 >= 2 条）=> `_home_channel` 走
  `if state.estimate is None: if not self._dogleg(state): return False`
  这条分支直接放弃，绕过了 `_ray_blind_clear`。
* 17/17 的 `belief` 里**真状态格点已被排除**（`pos_cell_alive` 全为 False），
  其中 13/17 候选集直接空掉 => `_blind_clear_sweep` 因 `not belief.alive` 跳过该频道。
* 共同物理画像：源贴着区域边界（离边界 7-81 m）且辐射轴几乎正朝区域外
  （`aim_outward_gap_deg` 0.5°-30°，中位数约 6°），可见区仅占区域 0.05%-2.2%。

三个修复
--------
R1 `RayGapRescue`     dogleg 失败但已有示向度时，改走 `_ray_blind_clear` 兜底。
R2 `RobustBeliefRescue` 用「量化容差安全」的 belief 重建候选集，让盲清除有目标。
                      根因：`DirectionalBelief` 位置网格 120 m、朝向 30°，
                      而这些源的示向度是在距源仅 20-300 m 处测到的近场观测，
                      真值格点偏移 40-108 m 在近场对应几十度方位差，
                      远超 `apply_bearing` 的 1+6=7° 容差 => 真状态被误杀。
                      `apply_silence` 同理会因 15° 朝向量化误杀边界擦过的候选。
R3 `EdgeRescue`       收尾对仍未清除的频道，沿区域边界内侧铺一圈 /clear。
                      依据：漏失源全部贴边（离边界 <= 81 m）。
`ComboRescue` = R1 + R2。

用法
----
  python -m src.b_miss_rescue --plan known                     # 17 个已知漏失种子
  python -m src.b_miss_rescue --plan paired --episodes 200 --directional 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .b_sim.arena import ARENA_RADIUS, CLEAR_RADIUS_M, RECEPTION_MIN_M
from .b_sim.directional import (
    GRID_STEP_M,
    HEADING_STEP_DEG,
    DirectionalBelief,
    in_beam,
)
from .b_sim.strategy import DogStrategy
from .b_miss_attrib import KNOWN_MISSES, MissProbe, aggregate, run_one

RESULTS = Path(__file__).resolve().parent.parent / "results"

# 位置网格格点到真实源位置的最大偏移（120 m 方格的半对角）
GRID_HALF_DIAG_M = GRID_STEP_M * math.sqrt(2.0) / 2.0
HEADING_HALF_STEP_DEG = HEADING_STEP_DEG / 2.0


class RobustBelief(DirectionalBelief):
    """量化容差安全的候选集：只做「即使算上离散化误差也必然矛盾」的排除。

    与父类的两处差别：
    1. `apply_silence` 加双重保守裕度 —— 只有当格点连同 120 m 方格整体都落在
       1000 m 内（`dist + 半对角 <= 1000`），且观测点稳稳落在扇区内部
       （角差 <= 90° - 朝向量化半步 - 近场张角）时才排除。
    2. `apply_bearing` 的角容差随「观测点到格点的距离」放大：格点可能偏离真源
       最多 85 m，在距离 d 处对应 atan(85/d) 的方位不确定度。d = 100 m 时是 40°，
       父类固定的 7° 容差在近场必然误杀真状态。
    """

    @classmethod
    def fresh(cls) -> RobustBelief:
        base = DirectionalBelief.fresh()
        return cls(candidates=base.candidates, positions=base.positions,
                   headings=base.headings)

    @staticmethod
    def _near_field_slack_deg(dist_m: float) -> float:
        d = max(dist_m, 1.0)
        return math.degrees(math.atan2(GRID_HALF_DIAG_M, d))

    def apply_silence(self, px: float, py: float) -> None:
        survivors = set()
        for pi, hi, tag in self.candidates:
            sx, sy = self.positions[pi]
            dist = math.hypot(px - sx, py - sy)
            if dist + GRID_HALF_DIAG_M > RECEPTION_MIN_M:
                survivors.add((pi, hi, tag))
                continue
            bearing = math.degrees(math.atan2(py - sy, px - sx))
            diff = abs((bearing - self.headings[hi] + 180.0) % 360.0 - 180.0)
            margin = HEADING_HALF_STEP_DEG + self._near_field_slack_deg(dist)
            if diff <= 90.0 - margin:
                continue                      # 稳稳在扇区内且必然够近 -> 真矛盾
            survivors.add((pi, hi, tag))
        self.candidates = survivors

    def apply_bearing(self, px: float, py: float, svd_deg: float,
                      tolerance_deg: float = 1.0) -> None:
        survivors = set()
        for pi, hi, tag in self.candidates:
            sx, sy = self.positions[pi]
            dist = math.hypot(sx - px, sy - py)
            bearing = math.degrees(math.atan2(sy - py, sx - px))
            tol = tolerance_deg + 6.0 + self._near_field_slack_deg(dist)
            if abs((bearing - svd_deg + 180.0) % 360.0 - 180.0) > tol:
                continue
            # in_beam 也放宽朝向量化半步
            b2 = math.degrees(math.atan2(py - sy, px - sx))
            d2 = abs((b2 - self.headings[hi] + 180.0) % 360.0 - 180.0)
            if d2 > 90.0 + HEADING_HALF_STEP_DEG + self._near_field_slack_deg(dist):
                continue
            survivors.add((pi, hi, tag))
        self.candidates = survivors


# ---------------------------------------------------------------- R1


class RayGapRescue(MissProbe):
    """dogleg 失败且已有示向度时，就地触发射线盲清除而不是直接放弃。"""

    def _dogleg(self, state):
        r = super()._dogleg(state)
        if not r and state.bearings and not state.ray_swept and not state.cleared:
            self._ray_blind_clear(state)
        return r


# ---------------------------------------------------------------- R2


class RobustBeliefRescue(MissProbe):
    """收尾前用 RobustBelief 重建候选集，让盲清除重新拿到目标。

    重建方式：从零开始，按该频道记录下来的 silent_points / bearings 顺序重放
    保守约束。只在 `_belief_search` / `_blind_clear_sweep` 入口做一次，
    不影响此前的归航行为。
    """

    def _rebuild(self, state) -> RobustBelief:
        belief = RobustBelief.fresh()
        for px, py in state.silent_points:
            belief.apply_silence(px, py)
        for b in state.bearings:
            belief.apply_bearing(b.x, b.y, b.svd_deg)
        return belief

    def _upgrade_beliefs(self) -> None:
        for state in self.channels.values():
            if state.cleared or state.belief is None:
                continue
            if isinstance(state.belief, RobustBelief):
                continue
            state.belief = self._rebuild(state)

    def _belief_search(self, *a, **kw):
        if self.config.directional_probe:
            self._upgrade_beliefs()
        return super()._belief_search(*a, **kw)

    def _blind_clear_sweep(self, *a, **kw):
        if self.config.directional_probe:
            self._upgrade_beliefs()
        return super()._blind_clear_sweep(*a, **kw)


# ---------------------------------------------------------------- R3


class EdgeRescue(MissProbe):
    """最后手段：对仍未清除的频道沿区域边界内侧铺一圈 /clear。

    依据：17/17 已知漏失源离边界 <= 81 m 且几乎正朝外辐射。
    这是纯几何兜底，不需要任何信号。成本很高，故放在最后且限制点数。
    """

    edge_inset_m: float = 18.0
    edge_step_m: float = CLEAR_RADIUS_M * 1.6
    edge_max_clears: int = 400

    def _edge_points(self) -> list[tuple[float, float]]:
        r = ARENA_RADIUS - self.edge_inset_m
        n = max(8, int(2.0 * math.pi * r / self.edge_step_m))
        return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n))
                for k in range(n)]

    def _blind_clear_sweep(self, *a, **kw):
        super()._blind_clear_sweep(*a, **kw)
        if not self.config.directional_probe:
            return
        todo = [s for s in self.channels.values() if not s.cleared and not s.verdict_empty]
        if not todo:
            return
        pts = self._edge_points()
        # 从当前位置最近的边界点开始顺着环走，避免跨场弦跳
        start = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - self.pos[0],
                                                              pts[i][1] - self.pos[1]))
        ordered = pts[start:] + pts[:start]
        budget = self.edge_max_clears
        for px, py in ordered:
            if self._stop() or budget <= 0:
                return
            for state in list(todo):
                if state.cleared:
                    todo.remove(state)
                    continue
                if self._stop() or budget <= 0:
                    return
                self._clear_at((px, py), state.channel)
                budget -= 1
            if not todo:
                return


class ComboRescue(RayGapRescue, RobustBeliefRescue):
    """R1 + R2。MRO: RayGapRescue -> RobustBeliefRescue -> MissProbe -> DogStrategy。"""


class RayEdgeRescue(RayGapRescue, EdgeRescue):
    """R1 + R3：先用射线兜底救回「听到过」的源，再用边界扫兜底救「从未听到」的源。"""


class RayEdgeLeanRescue(RayEdgeRescue):
    """R1 + R3，但边界扫只在「该频道从未收到任何信号」时启用且预算更紧。

    R3 全量版会对所有未清频道铺边界点，成本按频道数线性放大。
    实际需要边界扫的只有 `bearings == 0` 的频道（从未听到 => 射线兜底无从下手）。
    """

    edge_max_clears: int = 240

    def _blind_clear_sweep(self, *a, **kw):
        # 先跑 R1 链上的原生收尾（含父类 EdgeRescue 之前的 DogStrategy 行为）
        DogStrategy._blind_clear_sweep(self, *a, **kw)
        if not self.config.directional_probe:
            return
        todo = [s for s in self.channels.values()
                if not s.cleared and not s.verdict_empty and not s.bearings]
        if not todo:
            return
        pts = self._edge_points()
        start = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - self.pos[0],
                                                             pts[i][1] - self.pos[1]))
        budget = self.edge_max_clears
        for px, py in pts[start:] + pts[:start]:
            if self._stop() or budget <= 0:
                return
            for state in list(todo):
                if state.cleared:
                    todo.remove(state)
                    continue
                if self._stop() or budget <= 0:
                    return
                self._clear_at((px, py), state.channel)
                budget -= 1
            if not todo:
                return


def _q3_fast():
    from .b_sim.q3_fast import Q3FastStrategy
    return Q3FastStrategy


def _q4_fast():
    from .b_sim.q4_fast import Q4FastStrategy
    return Q4FastStrategy


VARIANTS = {
    "baseline": MissProbe,
    "r1_raygap": RayGapRescue,
    "r2_robust_belief": RobustBeliefRescue,
    "r3_edge": EdgeRescue,
    "combo": ComboRescue,
    "r1r3_edge": RayEdgeRescue,
    "r1r3_lean": RayEdgeLeanRescue,
    # 生产策略。上面那些变体都基于 DogStrategy，而交付入口默认跑的是下面两个 ——
    # 想验证一个改动对成绩的真实影响，基线必须选 prod_q3 / prod_q4。
    "prod_q3": _q3_fast,
    "prod_q4": _q4_fast,
}


# ---------------------------------------------------------------- harness


def _job(args):
    seed, directional, count, variant, overrides = args
    try:
        # variant 名可带 "@" 后缀区分同一策略类的不同参数组（如 r1_raygap@ring1150x7），
        # 便于在一次配对实验里同时测「修复」与「布局」的叠加效果。
        cls_key = variant.split("@", 1)[0]
        # prod_q3 / prod_q4 存的是延迟导入的工厂函数（避免模块级循环导入），
        # 其余是类本身。类也是 callable，故用 isinstance 而非 callable() 区分。
        entry = VARIANTS[cls_key]
        cls = entry if isinstance(entry, type) else entry()
        r = run_one(seed, directional, count, cls, overrides or None)
        r["variant"] = variant
        return r
    except Exception as exc:  # noqa: BLE001
        return {"seed": seed, "directional": directional, "variant": variant,
                "error": f"{type(exc).__name__}: {exc}"}


def paired_report(by_variant: dict[str, list[dict]], ref: str = "baseline") -> dict:
    out = {}
    if ref not in by_variant:
        raise SystemExit(
            f"参照臂 '{ref}' 不在 --variants 里。配对实验至少要两个臂：一个基线加一个候选。\n"
            f"当前只有: {sorted(by_variant)}\n"
            f"参照臂取 --variants 的第一个，所以把基线写在最前面：\n"
            f"  --variants prod_q4,你的变体名 --directional 5   # 问题四\n"
            f"  --variants prod_q3,你的变体名                   # 问题三")
    base = {(r["directional"], r["seed"]): r for r in by_variant[ref] if "error" not in r}
    if not base:
        errs = [r.get("error") for r in by_variant[ref] if "error" in r][:3]
        raise SystemExit(
            f"参照臂 '{ref}' 没有任何成功局，无法配对。前几条错误：{errs}")
    for name, rows in by_variant.items():
        cur = {(r["directional"], r["seed"]): r for r in rows if "error" not in r}
        keys = [k for k in base if k in cur]
        d_cleared = sum(cur[k]["cleared"] - base[k]["cleared"] for k in keys)
        dt = [cur[k]["mean_clear_time_s"] - base[k]["mean_clear_time_s"] for k in keys
              if cur[k]["mean_clear_time_s"] and base[k]["mean_clear_time_s"]]
        base_metric = st.fmean(base[k]["mean_clear_time_s"] for k in keys
                               if base[k]["mean_clear_time_s"])
        cleared = sum(cur[k]["cleared"] for k in keys)
        total = sum(cur[k]["total"] for k in keys)
        out[name] = {
            "paired_episodes": len(keys),
            "clear_rate_raw": f"{cleared}/{total}",
            # 门禁一：源总数必须与参考臂一致。不一致意味着两臂跑的不是同一批案例,
            # 速度对比无意义。最常见的成因不是变体真的改了案例生成, 而是某些种子上
            # 抛异常被当成 error 静默丢局（实测 seed 7052 因 verdict_empty 缺失丢局,
            # 使源数变成 142 vs 157, 曾被误读为案例生成差异）。命中此门禁时先查
            # rows 里的 error 条目, 再怀疑变体本身。
            "total_sources_matches_ref": total == sum(base[k]["total"] for k in keys),
            # 门禁二：清除数不得低于参考臂。评分指标是 总时间/清除数, 分子分母同向
            # 变化时容易把"快了但掉源"误读成改进, 故与速度指标分开判定。
            "clear_regression": d_cleared < 0,
            "clear_rate": round(cleared / total, 5) if total else None,
            "full_clear_episodes": sum(1 for k in keys if cur[k]["cleared"] == cur[k]["total"]),
            "d_cleared_sources": d_cleared,
            "metric_mean_s": round(st.fmean(cur[k]["mean_clear_time_s"] for k in keys
                                            if cur[k]["mean_clear_time_s"]), 1),
            "d_metric_s": round(st.fmean(dt), 1) if dt else None,
            "d_metric_pct": round(st.fmean(dt) / base_metric * 100, 2) if dt else None,
            "better_episodes": sum(1 for v in dt if v < -1),
            "worse_episodes": sum(1 for v in dt if v > 1),
            "measure_calls_mean": round(st.fmean(cur[k]["measure_calls"] for k in keys), 1),
            "clear_calls_mean": round(st.fmean(cur[k]["clear_calls"] for k in keys), 1),
            "still_missed": sorted(
                (k[0], k[1], ch) for k in keys for ch in cur[k]["missed_channels"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["known", "paired"], default="known")
    ap.add_argument("--variants", default="baseline,r1_raygap,r2_robust_belief,combo")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=7000)
    ap.add_argument("--directional", type=int, default=10)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overrides", default=None,
                    help='JSON: {"变体名": {"probe_ring_radius": 1150, '
                         '"probe_ring_stations": 7}}，给单个变体注入 StrategyConfig 覆盖')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.monotonic()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        # "策略类@参数组标签" 形式只校验 @ 前的策略类名
        if v.split("@", 1)[0] not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {sorted(VARIANTS)}")

    if args.plan == "known":
        cases = [(s, d, c) for s, d, c in KNOWN_MISSES]
    else:
        cases = [(args.seed_start + i, args.directional, args.count)
                 for i in range(args.episodes)]

    # --overrides 是 {variant: {cfg 字段: 值}} 的 JSON，用于给单个变体注入
    # StrategyConfig 覆盖（如环半径/站位数），从而测「漏失修复 x 站位布局」的叠加。
    ov_map = json.loads(args.overrides) if args.overrides else {}
    unknown = set(ov_map) - set(variants)
    if unknown:
        raise SystemExit(f"--overrides 里有未参与本次实验的变体: {sorted(unknown)}")
    tasks = [(s, d, c, v, ov_map.get(v)) for v in variants for s, d, c in cases]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, t) for t in tasks]
        done = 0
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(tasks)}", flush=True)

    by_variant: dict[str, list[dict]] = {v: [] for v in variants}
    for r in rows:
        by_variant[r["variant"]].append(r)
    for v in variants:
        by_variant[v].sort(key=lambda r: (r.get("directional", 0), r["seed"]))

    paired = paired_report(by_variant, ref=variants[0])
    aggs = {v: aggregate(by_variant[v]) for v in variants}

    print(f"\nplan={args.plan} cases={len(cases)}")
    for v in variants:
        p = paired[v]
        print(f"{v:<18} cleared={p['clear_rate_raw']} ({p['clear_rate']}) "
              f"full={p['full_clear_episodes']}/{p['paired_episodes']} "
              f"d_src={p['d_cleared_sources']:+d} metric={p['metric_mean_s']}s "
              f"d={p['d_metric_pct']:+}% meas={p['measure_calls_mean']} "
              f"clr={p['clear_calls_mean']}")
        print(f"{'':<18} still_missed={p['still_missed']}")
        if not p["total_sources_matches_ref"]:
            print(f"{'':<18} !! 源总数与参考臂不一致, 速度对比不可采信")
        if p["clear_regression"]:
            print(f"{'':<18} !! 清除数退化 {p['d_cleared_sources']:+d}, 速度收益不足以采纳")
    for v in variants:
        print(f"{v:<18} causes={aggs[v]['cause_distribution']} "
              f"u_hard_bound={aggs[v]['u_hard_bound_raw']}")

    out = {"config": vars(args), "paired": paired,
           "aggregate": {v: {k: x for k, x in aggs[v].items()
                             if k not in ("visible_frac_of_missed",)} for v in variants},
           "rows": rows, "elapsed_s": round(time.monotonic() - t0, 1)}
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = RESULTS.parent / p
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
