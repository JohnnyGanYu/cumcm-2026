"""用官方演练日志检验本地 mock 的保真度。

动机：全部结论都建立在本地 mock（src/b_sim/arena.py）之上，而它是我们自己写的。
若 mock 与官方行为有系统性偏差，则"7/10 漏失属几何不可达"这类结论可能是
mock 的产物而非题目性质。官方演练日志的 protocol_log 记录了真实模拟器对每次
measure/clear 的响应，是唯一可用的实证对照。

检验项（只用日志能直接观测到的量，不做推断）：
  1. 清除点半径分布——官方案例里被成功清除的源，其位置距原点多远？
     若官方源也能贴近 1800 m 边界，则 mock 的均匀撒点假设合理；
     若官方源都明显内缩，则 mock 制造了不存在的"贴边界"困难案例。
  2. 从 direction 响应到成功 clear 的距离尺度——间接反映有效接收半径量级。
  3. 响应类型集合与字段名——协议层是否一致（no_signal/direction/near）。
  4. near 判定：出现 near 时的清除点与后续 clear 点距离是否 <= 5 m 量级。

用法: python -m src.b_mock_fidelity
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st

from .b_sim.arena import (
    ARENA_RADIUS,
    NEAR_THRESHOLD_M,
    RECEPTION_MAX_M,
    generate_case,
)


def load_official() -> list[dict]:
    out = []
    for path in sorted(glob.glob("official_runs/*.sanitized.json")):
        with open(path, encoding="utf-8") as fh:
            out.append({"path": path, "data": json.load(fh)})
    return out


def analyse_official(runs: list[dict]) -> dict:
    clear_radii: list[float] = []      # 成功清除点距原点的半径
    resp_kinds: dict[str, int] = {}
    near_events = 0
    measure_pts = 0
    max_r_seen = 0.0
    for run in runs:
        log = run["data"].get("protocol_log") or []
        for entry in log:
            path = entry.get("path") or entry.get("endpoint") or ""
            req = entry.get("request") or {}
            resp = entry.get("response") or {}
            pos = req.get("position") or {}
            x, y = pos.get("x"), pos.get("y")
            if x is not None and y is not None:
                r = math.hypot(x, y)
                max_r_seen = max(max_r_seen, r)
            kind = resp.get("measure_result") or resp.get("clear_result")
            if kind:
                resp_kinds[kind] = resp_kinds.get(kind, 0) + 1
            if "measure" in path:
                measure_pts += 1
                if resp.get("measure_result") == "near":
                    near_events += 1
            if "clear" in path and resp.get("clear_result") == "success":
                if x is not None and y is not None:
                    clear_radii.append(math.hypot(x, y))
    return {"n_runs": len(runs), "measure_calls": measure_pts,
            "response_kinds": resp_kinds, "near_events": near_events,
            "successful_clears": len(clear_radii),
            "clear_radius_max": round(max(clear_radii), 1) if clear_radii else None,
            "clear_radius_mean": round(st.fmean(clear_radii), 1) if clear_radii else None,
            "clear_radius_p90": round(sorted(clear_radii)[int(len(clear_radii) * 0.9)], 1)
            if len(clear_radii) >= 10 else None,
            "robot_max_radius_seen": round(max_r_seen, 1)}


def analyse_mock(n_cases: int = 200, directional: int = 10) -> dict:
    """mock 生成器下，源半径与'贴边界'比例。清除点必在源 20 m 内，故用源半径。"""
    radii: list[float] = []
    for seed in range(9000, 9000 + n_cases):
        case = generate_case(seed, count=None, directional_count=directional)
        radii.extend(math.hypot(j.x, j.y) for j in case.jammers)
    radii.sort()
    near_edge = sum(1 for r in radii if ARENA_RADIUS - r <= 30.0)
    outside_recep = sum(1 for r in radii if r > RECEPTION_MAX_M)
    return {"n_sources": len(radii),
            "radius_max": round(radii[-1], 1),
            "radius_mean": round(st.fmean(radii), 1),
            "radius_p90": round(radii[int(len(radii) * 0.9)], 1),
            "within_30m_of_boundary": near_edge,
            "within_30m_pct": round(near_edge / len(radii) * 100, 2),
            "beyond_1500m": outside_recep,
            "beyond_1500m_pct": round(outside_recep / len(radii) * 100, 2)}


def main() -> None:
    runs = load_official()
    if not runs:
        print("没有找到 official_runs/*.sanitized.json")
        return
    off = analyse_official(runs)
    mock = analyse_mock()
    print("=== 官方演练日志（实测） ===")
    for k, v in off.items():
        print(f"  {k:26s} = {v}")
    print("\n=== 本地 mock 生成器（同 200 局 d10） ===")
    for k, v in mock.items():
        print(f"  {k:26s} = {v}")

    print("\n=== 对照判断 ===")
    print(f"  题目区域半径 = {ARENA_RADIUS} m，near 阈值 = {NEAR_THRESHOLD_M} m")
    if off["clear_radius_max"] is not None:
        print(f"  官方成功清除点最远半径 {off['clear_radius_max']} m "
              f"(p90={off['clear_radius_p90']}, mean={off['clear_radius_mean']})")
        print(f"  mock 源半径最远 {mock['radius_max']} m "
              f"(p90={mock['radius_p90']}, mean={mock['radius_mean']})")
        gap = ARENA_RADIUS - off["clear_radius_max"]
        print(f"  官方最远清除点距边界 {gap:.1f} m")
        if gap > 100:
            print("  => 官方样本中未观测到贴边界(<=100m)的源；mock 允许贴到边界，"
                  "可能制造了官方不会出现的困难案例。样本量小，不能定论。")
        else:
            print("  => 官方样本中也出现了接近边界的源，mock 的撒点假设与之相容。")
    with open("results/b_mock_fidelity.json", "w", encoding="utf-8") as fh:
        json.dump({"official": off, "mock": mock}, fh, ensure_ascii=False, indent=1)
    print("\nwrote results/b_mock_fidelity.json")


if __name__ == "__main__":
    main()
