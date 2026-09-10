"""官方日志移动耗时拆解 + 路径规划（TSP）收益上限判定。

动机：`src/b_time_budget.py` 只回答"移动占多少"，回答不了"移动里有多少是可省的"。
本脚本把移动距离按用途拆开，再用精确/近似 TSP 给出多条量级不同的下界，
判断"上全局路径规划"到底能赚回多少、以及收益来自哪一块。

时间常数（由官方日志残差逐调用反解，非抄文档）：
    measure         5 s，切频道再 +1 s
    clear success   5 s
    clear fail      3 s
    move            dist / 5 m/s
校验：Σ 各项 == summary.virtual_time_s，逐局误差 < 1e-5 s（见输出第一节）。

轨迹结构（从日志实测，不是假设）：
  机器狗轨迹 = 若干"扫描站"（同一坐标连测 >=8 个频道，做覆盖探测）
             + 站间"excursion"（锁定某频道后沿示向度归航 + 清除微调）
  实测站位集合 = 原点 + 12@r1250 + 18@r1750 (+ 12@r1800 + 少量内圈)，共 31~47 站。

下界体系（越往下越松，越接近可实现）：
  B_cov   仅覆盖：TSP(原点 ∪ 实测站位集合)。把该策略选定的站位走一遍的最小路程。
          与"实际站位空驶"之差 = 纯粹的访问顺序损失，TSP 立刻可拿。
  B_ring  只走 12@1250 环（config 声称的主站位）的 TSP，精确 Held-Karp。
          与 B_cov 之差 = 加密站位（1750/1800 环）的覆盖代价。
  L4      只改顺序：站位集合、每段 excursion 的内部几何（归航+清除米数）全部保留，
          只重排站位访问顺序与 excursion 挂载。== TSP 层的现实收益上限。
  L3      完美覆盖+摸源：TSP(原点 ∪ 站位 ∪ 源坐标)，零归航浪费。不可达。
  L1      全知：TSP(原点 ∪ 已清除源坐标)，精确 Held-Karp。物理地板。
  L0      全知且零检测：L1/5 + 5s×源数。信息论极限。

用法：
    python -m src.b_tsp_bound
    python -m src.b_tsp_bound 'official_runs/q4_practice_run*.sanitized.json'
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

MEASURE_S = 5.0
SWITCH_S = 1.0
CLEAR_OK_S = 5.0
CLEAR_FAIL_S = 3.0
SPEED = 5.0

STATION_MIN_CHANNELS = 8   # 判定"扫描站"：同坐标测到 >=8 个不同频道
FINE_STEP_M = 40.0         # < 该步长视为清除微调，否则视为归航步


# ---------------------------------------------------------------- 日志解析

def load_calls(path: Path) -> tuple[list[dict], dict, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    calls = []
    for e in d.get("protocol_log", []):
        if e.get("path") not in ("/measure", "/clear"):
            continue
        r = e.get("request") or {}
        resp = e.get("response") or {}
        pos = r.get("position") or {}
        calls.append({
            "kind": e["path"],
            "x": float(pos.get("x", 0.0)), "y": float(pos.get("y", 0.0)),
            "ch": r.get("channel"),
            "measure_result": resp.get("measure_result"),
            "clear_result": resp.get("clear_result"),
        })
    return calls, d.get("summary", {}) or {}, d.get("strategy_config", {}) or {}


def time_decomposition(calls: list[dict], summary: dict) -> dict:
    n_meas = n_switch = n_ok = n_fail = 0
    dist = 0.0
    px, py, pch = 0.0, 0.0, 1
    for c in calls:
        dist += math.hypot(c["x"] - px, c["y"] - py)
        if c["kind"] == "/measure":
            n_meas += 1
            if c["ch"] != pch:
                n_switch += 1
        elif c["clear_result"] == "success":
            n_ok += 1
        else:
            n_fail += 1
        px, py, pch = c["x"], c["y"], c["ch"]

    t_meas = n_meas * MEASURE_S + n_switch * SWITCH_S
    t_clear = n_ok * CLEAR_OK_S + n_fail * CLEAR_FAIL_S
    t_move = dist / SPEED
    virt = float(summary.get("virtual_time_s", 0.0))
    return {
        "measure_calls": n_meas, "channel_switches": n_switch,
        "clear_success": n_ok, "clear_fail": n_fail,
        "move_distance_m": dist,
        "t_measure_s": t_meas, "t_clear_s": t_clear, "t_move_s": t_move,
        "virtual_time_s": virt,
        "reconcile_err_s": abs(t_meas + t_clear + t_move - virt),
        "pct_measure": 100 * t_meas / virt, "pct_clear": 100 * t_clear / virt,
        "pct_move": 100 * t_move / virt,
    }


# ------------------------------------------------- 轨迹分段

def segment(calls: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for c in calls:
        key = (round(c["x"], 3), round(c["y"], 3))   # 仅用于同点归并
        if not groups or groups[-1]["key"] != key:
            # pos 保留原始未取整坐标，保证距离重构可对账到 1e-9
            groups.append({"key": key, "pos": (c["x"], c["y"]), "calls": []})
        groups[-1]["calls"].append(c)
    for g in groups:
        chs = {c["ch"] for c in g["calls"] if c["kind"] == "/measure"}
        g["is_station"] = len(chs) >= STATION_MIN_CHANNELS
        g["clear_ok"] = sum(1 for c in g["calls"]
                            if c["kind"] == "/clear" and c["clear_result"] == "success")
    return groups


def distance_breakdown(groups: list[dict]) -> dict:
    d_station = d_home = d_fine = 0.0
    prev = (0.0, 0.0)
    for g in groups:
        step = math.hypot(g["pos"][0] - prev[0], g["pos"][1] - prev[1])
        if g["is_station"]:
            d_station += step
        elif step < FINE_STEP_M:
            d_fine += step
        else:
            d_home += step
        prev = g["pos"]
    return {"d_station_transit_m": d_station, "d_homing_m": d_home,
            "d_clear_fine_m": d_fine}


def itemize(groups: list[dict]) -> list[dict]:
    """把轨迹切成有序 item 序列：站位点 + excursion 定长段。

    item 形式统一为 (entry, exit, inner_m)：
      station   : entry == exit == 站坐标, inner_m = 0
      excursion : entry/exit 为该段首末坐标, inner_m = 段内真实路程
    这样"实际总移动 == Σ 相邻 item 的 exit->entry 间距 + Σ inner_m"，
    即实际轨迹本身就是该 item 序列问题的一个可行解，故对顺序取最优必然 <= 实际。
    """
    items: list[dict] = []
    cur: list[dict] = []
    for g in groups:
        if g["is_station"]:
            if cur:
                pts = [x["pos"] for x in cur]
                items.append({
                    "kind": "excursion", "entry": pts[0], "exit": pts[-1],
                    "inner_m": sum(math.hypot(q[0] - p[0], q[1] - p[1])
                                   for p, q in zip(pts, pts[1:])),
                    "n_success": sum(x["clear_ok"] for x in cur),
                })
                cur = []
            items.append({"kind": "station", "entry": g["pos"], "exit": g["pos"],
                          "inner_m": 0.0, "n_success": g["clear_ok"]})
        else:
            cur.append(g)
    if cur:
        pts = [x["pos"] for x in cur]
        items.append({
            "kind": "excursion", "entry": pts[0], "exit": pts[-1],
            "inner_m": sum(math.hypot(q[0] - p[0], q[1] - p[1])
                           for p, q in zip(pts, pts[1:])),
            "n_success": sum(x["clear_ok"] for x in cur),
        })
    return items


# ---------------------------------------------------------------- TSP

def dmat(pts) -> np.ndarray:
    a = np.asarray(pts, dtype=float)
    return np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1])


def held_karp_open(D: np.ndarray) -> float:
    """精确开放路径 TSP：起点固定 0，访问全部节点，不返回起点。"""
    n = D.shape[0]
    if n <= 1:
        return 0.0
    m = n - 1
    dp = np.full((1 << m, m), np.inf)
    for j in range(m):
        dp[1 << j, j] = D[0, j + 1]
    for mask in range(1 << m):
        row = dp[mask]
        for j in range(m):
            if not (mask >> j) & 1 or not np.isfinite(row[j]):
                continue
            cur = row[j]
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                cand = cur + D[j + 1, k + 1]
                if cand < dp[nm, k]:
                    dp[nm, k] = cand
    return float(dp[(1 << m) - 1].min())


def path_len(D: np.ndarray, order) -> float:
    o = np.asarray(order)
    return float(D[o[:-1], o[1:]].sum())


def two_opt(D: np.ndarray, order: list[int]) -> list[int]:
    order = order[:]
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a, b = order[i - 1], order[i]
            for j in range(i + 1, n):
                c = order[j]
                d = order[j + 1] if j + 1 < n else None
                old = D[a, b] + (D[c, d] if d is not None else 0.0)
                new = D[a, c] + (D[b, d] if d is not None else 0.0)
                if new < old - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    b = order[i]
                    improved = True
    return order


def or_opt(D: np.ndarray, order: list[int]) -> list[int]:
    order = order[:]
    best = path_len(D, order)
    changed = True
    while changed:
        changed = False
        for seglen in (1, 2, 3):
            for i in range(1, len(order) - seglen + 1):
                seg = order[i:i + seglen]
                rest = order[:i] + order[i + seglen:]
                for k in range(1, len(rest) + 1):
                    cand = rest[:k] + seg + rest[k:]
                    L = path_len(D, cand)
                    if L < best - 1e-9:
                        order, best, changed = cand, L, True
                        break
                if changed:
                    break
            if changed:
                break
    return order


def tsp_open(pts, exact_limit: int = 15) -> tuple[float, str]:
    """开放路径 TSP，起点 pts[0]。小规模精确，大规模多起点 NN+2opt+OrOpt。"""
    n = len(pts)
    if n <= 1:
        return 0.0, "trivial"
    D = dmat(pts)
    if n <= exact_limit:
        return held_karp_open(D), f"exact Held-Karp n={n}"
    best = np.inf
    for hint in range(1, n):
        unv = set(range(1, n))
        order, cur = [0], 0
        order.append(hint); unv.discard(hint); cur = hint
        while unv:
            nxt = min(unv, key=lambda k: D[cur, k])
            order.append(nxt); unv.discard(nxt); cur = nxt
        order = or_opt(D, two_opt(D, order))
        best = min(best, path_len(D, order))
    return float(best), f"heuristic NNx{n-1}+2opt+OrOpt n={n}"


def mst_lb(pts) -> float:
    """开放路径 TSP 的合法下界：路径是生成树，故 MST <= OPT。"""
    D = dmat(pts)
    n = len(pts)
    inn = np.zeros(n, bool); inn[0] = True
    total = 0.0
    for _ in range(n - 1):
        sub = D[np.ix_(inn, ~inn)]
        total += float(sub.min())
        idx = np.where(~inn)[0][int(np.unravel_index(sub.argmin(), sub.shape)[1])]
        inn[idx] = True
    return total


def ring_order_loss(stations: list[tuple[float, float]]) -> list[dict]:
    """按半径分环，对比"实际访问顺序"与"按方位角单调扫"的路程差。

    定位具体缺陷：若某环被乱序访问，会产生跨越场地的弦跳（最长 2R=3600 m），
    而按角度顺扫每步只需 2R·sin(π/n)。这是纯实现缺陷，不涉及任何策略取舍。
    """
    rings: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    for i, p in enumerate(stations):
        r = int(round(math.hypot(*p) / 25.0) * 25)
        if r >= 100:
            rings.setdefault(r, []).append((i, p))
    out = []
    for r, members in sorted(rings.items()):
        if len(members) < 4:
            continue
        idx = [i for i, _ in members]
        walk = stations[idx[0]:idx[-1] + 1]
        actual = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(walk, walk[1:]))
        srt = [p for _, p in sorted(
            ((math.degrees(math.atan2(p[1], p[0])) % 360, p) for _, p in members),
            key=lambda t: t[0])]
        best = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(srt, srt[1:]))
        out.append({"radius_m": r, "n": len(members), "actual_m": actual,
                    "angle_sorted_m": best, "loss_m": actual - best,
                    "angles": [round(math.degrees(math.atan2(p[1], p[0])) % 360)
                               for _, p in members]})
    return out


# ------------------------------------------------- L4：只改顺序

def reorder_bound(items: list[dict]) -> dict:
    """只改顺序：item 集合与每个 item 的内部几何全部保留，仅重排访问次序。

    这是一个非对称的开放路径 TSP：节点 = item，边权 c(i,j) = |exit_i - entry_j|，
    每个 item 还要付固定的 inner_m（与顺序无关，直接加常数）。起点 (0,0)。
    每段 excursion 可正向或反向遍历（entry/exit 互换），两种朝向都作候选。

    可行性：实际轨迹的 item 顺序 + 各自实际朝向就是一个可行解，
    因此最优值 <= 实际总移动，L4 是货真价实的下界（对"只改顺序"这一族策略）。
    """
    fixed = sum(it["inner_m"] for it in items)
    # 节点 0 = 起点；其余每个 item 两种朝向
    nodes = [{"entry": (0.0, 0.0), "exit": (0.0, 0.0), "item": -1}]
    for k, it in enumerate(items):
        nodes.append({"entry": it["entry"], "exit": it["exit"], "item": k})
        if it["entry"] != it["exit"]:
            nodes.append({"entry": it["exit"], "exit": it["entry"], "item": k})
    N = len(nodes)
    ex = np.array([n["exit"] for n in nodes], float)
    en = np.array([n["entry"] for n in nodes], float)
    C = np.hypot(ex[:, None, 0] - en[None, :, 0], ex[:, None, 1] - en[None, :, 1])
    owner = np.array([n["item"] for n in nodes])
    n_items = len(items)

    def cost(seq: list[int]) -> float:
        s = np.asarray(seq)
        return float(C[s[:-1], s[1:]].sum())

    def greedy(hint: int | None) -> list[int]:
        done = np.zeros(n_items, bool)
        seq, cur = [0], 0
        if hint is not None:
            seq.append(hint); done[owner[hint]] = True; cur = hint
        while not done.all():
            cand = np.where(~done[owner])[0]
            nxt = int(cand[C[cur, cand].argmin()])
            seq.append(nxt); done[owner[nxt]] = True; cur = nxt
        return seq

    def improve(seq: list[int]) -> tuple[float, list[int]]:
        best = cost(seq)
        changed = True
        while changed:
            changed = False
            # 2-opt（非对称：段反转要重算，直接整体评估）
            L = len(seq)
            for i in range(1, L - 1):
                for j in range(i + 1, L):
                    cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    c = cost(cand)
                    if c < best - 1e-9:
                        seq, best, changed = cand, c, True
            # Or-opt：单 item 迁移
            for i in range(1, len(seq)):
                node = seq[i]
                rest = seq[:i] + seq[i + 1:]
                for k in range(1, len(rest) + 1):
                    cand = rest[:k] + [node] + rest[k:]
                    c = cost(cand)
                    if c < best - 1e-9:
                        seq, best, changed = cand, c, True
                        break
            # 朝向翻转
            for i in range(1, len(seq)):
                alt = [m for m in range(1, N)
                       if owner[m] == owner[seq[i]] and m != seq[i]]
                for a in alt:
                    cand = seq[:i] + [a] + seq[i + 1:]
                    c = cost(cand)
                    if c < best - 1e-9:
                        seq, best, changed = cand, c, True
                        break
        return best, seq

    best_val, best_seq = math.inf, None
    starts = [None] + [m for m in range(1, N)][:24]
    for h in starts:
        v, s = improve(greedy(h))
        if v < best_val:
            best_val, best_seq = v, s
    return {"total_m": best_val + fixed, "transit_m": best_val,
            "fixed_inner_m": fixed,
            "order": [int(owner[i]) for i in (best_seq or [])[1:]]}


# ---------------------------------------------------------------- 主流程

def analyze(path: Path) -> dict:
    calls, summary, cfg = load_calls(path)
    td = time_decomposition(calls, summary)
    groups = segment(calls)
    db = distance_breakdown(groups)
    items = itemize(groups)

    # 自检：item 分解必须严格重构实际总移动，否则 L4 的可行性论证不成立
    recon = sum(it["inner_m"] for it in items)
    prev = (0.0, 0.0)
    for it in items:
        recon += math.hypot(it["entry"][0] - prev[0], it["entry"][1] - prev[1])
        prev = it["exit"]
    assert abs(recon - td["move_distance_m"]) < 1e-6, \
        f"item 分解未能重构实际移动: {recon} vs {td['move_distance_m']}"

    stations = [g["pos"] for g in groups if g["is_station"] and g["key"] != (0.0, 0.0)]
    sources = [(c["x"], c["y"]) for c in calls
               if c["kind"] == "/clear" and c["clear_result"] == "success"]

    r_ring = float(cfg.get("probe_ring_radius") or 1250.0)
    n_ring = int(cfg.get("probe_ring_stations") or 12)
    ring12 = [(r_ring * math.cos(2 * math.pi * k / n_ring),
               r_ring * math.sin(2 * math.pi * k / n_ring)) for k in range(n_ring)]

    b_cov, m_cov = tsp_open([(0.0, 0.0)] + stations)
    b_ring, m_ring = tsp_open([(0.0, 0.0)] + ring12)
    l1, m_l1 = tsp_open([(0.0, 0.0)] + sources)
    l3, m_l3 = tsp_open([(0.0, 0.0)] + stations + sources)
    l3_lb = mst_lb([(0.0, 0.0)] + stations + sources)
    r4 = reorder_bound(items)
    l4, l4_tour, l4_svc = r4["total_m"], r4["transit_m"], r4["fixed_inner_m"]
    rings = ring_order_loss(stations)

    cleared = int(summary.get("cleared_count") or len(sources) or 1)
    fixed = td["t_measure_s"] + td["t_clear_s"]

    def metric(d_m: float) -> float:
        return (d_m / SPEED + fixed) / cleared

    D = td["move_distance_m"]
    radii = sorted({round(math.hypot(*p) / 50) * 50 for p in stations})
    return {
        "file": path.name, "cleared": cleared, "time": td,
        "dist_breakdown": db,
        "n_stations": len(stations) + 1, "station_radii": radii,
        "n_excursions": sum(1 for it in items if it["kind"] == "excursion"),
        "ring_order_loss": rings,
        "ring_loss_total_m": sum(r["loss_m"] for r in rings),
        "actual_move_m": D,
        "bounds_m": {"B_cov": b_cov, "B_ring12": b_ring, "L4": l4,
                     "L4_tour": l4_tour, "L4_service": l4_svc,
                     "L3": l3, "L3_mst_lb": l3_lb, "L1": l1},
        "methods": {"B_cov": m_cov, "B_ring12": m_ring, "L1": m_l1, "L3": m_l3},
        "coverage_order_loss_m": db["d_station_transit_m"] - b_cov,
        "ratio_actual_over_L1": D / l1,
        "metric_actual": td["virtual_time_s"] / cleared,
        "metric_L4": metric(l4), "metric_L3": metric(l3), "metric_L1": metric(l1),
        "metric_L0": (l1 / SPEED + cleared * CLEAR_OK_S) / cleared,
    }


def main(argv: list[str]) -> int:
    pats = argv[1:] or ["official_runs/q4_practice_run*.sanitized.json"]
    paths: list[Path] = []
    for p in pats:
        paths.extend(Path(q) for q in sorted(glob.glob(p)))
    if not paths:
        print("no input files")
        return 1
    rows = [analyze(p) for p in paths]
    n = len(rows)
    avg = lambda k: sum(k(r) for r in rows) / n  # noqa: E731

    print("=" * 92)
    print("一、时间拆解对账  时间常数由日志残差反解: measure 5s(+1s切频) / clear 5s成功 3s失败 / 5 m/s")
    print("=" * 92)
    h = (f"{'file':32s} {'源':>3s} {'虚拟时间s':>10s} {'移动%':>7s} {'检测%':>7s} "
         f"{'清除%':>6s} {'切频次':>7s} {'对账误差s':>10s}")
    print(h); print("-" * len(h))
    for r in rows:
        t = r["time"]
        print(f"{r['file']:32s} {r['cleared']:3d} {t['virtual_time_s']:10.1f} "
              f"{t['pct_move']:7.1f} {t['pct_measure']:7.1f} {t['pct_clear']:6.2f} "
              f"{t['channel_switches']:7d} {t['reconcile_err_s']:10.1e}")
    print(f"\n均值: 移动 {avg(lambda r: r['time']['pct_move']):.1f}%  "
          f"检测 {avg(lambda r: r['time']['pct_measure']):.1f}%  "
          f"清除 {avg(lambda r: r['time']['pct_clear']):.2f}%   "
          f"最大对账误差 {max(r['time']['reconcile_err_s'] for r in rows):.1e} s")

    print("\n" + "=" * 92)
    print("二、移动距离按用途拆解 + 实测站位结构")
    print("=" * 92)
    h = (f"{'file':32s} {'总移动m':>9s} {'站位空驶':>9s} {'归航':>8s} {'清除微调':>9s} "
         f"{'站数':>5s} {'段数':>5s} {'站位半径':>22s}")
    print(h); print("-" * len(h))
    for r in rows:
        b = r["dist_breakdown"]
        print(f"{r['file']:32s} {r['actual_move_m']:9.0f} {b['d_station_transit_m']:9.0f} "
              f"{b['d_homing_m']:8.0f} {b['d_clear_fine_m']:9.0f} {r['n_stations']:5d} "
              f"{r['n_excursions']:5d} {str(r['station_radii']):>22s}")
    tot = sum(r["actual_move_m"] for r in rows)
    print(f"\n占总移动: 站位空驶 {100*sum(r['dist_breakdown']['d_station_transit_m'] for r in rows)/tot:.1f}%  "
          f"归航 {100*sum(r['dist_breakdown']['d_homing_m'] for r in rows)/tot:.1f}%  "
          f"清除微调 {100*sum(r['dist_breakdown']['d_clear_fine_m'] for r in rows)/tot:.1f}%")

    print("\n" + "=" * 92)
    print("三、覆盖段：实际站位空驶 vs 走同一站位集合的最优 TSP（这是纯顺序损失）")
    print("=" * 92)
    h = (f"{'file':32s} {'实际空驶m':>10s} {'B_cov最优':>10s} {'顺序损失m':>10s} "
         f"{'损失占比':>9s} {'B_ring12':>9s}")
    print(h); print("-" * len(h))
    for r in rows:
        b = r["bounds_m"]; d = r["dist_breakdown"]["d_station_transit_m"]
        print(f"{r['file']:32s} {d:10.0f} {b['B_cov']:10.0f} "
              f"{r['coverage_order_loss_m']:10.0f} {100*r['coverage_order_loss_m']/d:8.1f}% "
              f"{b['B_ring12']:9.0f}")
    print(f"\n站位集合 TSP 解法: {rows[0]['methods']['B_cov']};  "
          f"12 站环: {rows[0]['methods']['B_ring12']}")

    print("\n" + "=" * 92)
    print("三之二、定位到具体缺陷：环内站位访问顺序（乱序 -> 跨场弦跳）")
    print("=" * 92)
    h = (f"{'file':32s} {'环半径':>7s} {'站数':>5s} {'实际走m':>9s} {'角度顺扫m':>10s} "
         f"{'损失m':>8s} {'损失s':>8s}")
    print(h); print("-" * len(h))
    for r in rows:
        for rg in r["ring_order_loss"]:
            print(f"{r['file']:32s} {rg['radius_m']:7d} {rg['n']:5d} "
                  f"{rg['actual_m']:9.0f} {rg['angle_sorted_m']:10.0f} "
                  f"{rg['loss_m']:8.0f} {rg['loss_m']/SPEED:8.0f}")
    print(f"\n环序损失均值 {avg(lambda r: r['ring_loss_total_m']):.0f} m "
          f"= {avg(lambda r: r['ring_loss_total_m'])/SPEED:.0f} s 虚拟时间")
    for r in rows:
        bad = [rg for rg in r["ring_order_loss"] if rg["loss_m"] > 3000]
        for rg in bad:
            print(f"  {r['file']}: r={rg['radius_m']} 访问角序 {rg['angles']}  <- 非单调")

    print("\n" + "=" * 92)
    print("四、理论下界 vs 实际行走（移动距离，米）")
    print("=" * 92)
    h = (f"{'file':32s} {'实际':>8s} {'L4只改序':>9s} {'L3完美':>8s} {'L1全知':>8s} "
         f"{'实/L4':>6s} {'实/L3':>6s} {'实/L1':>6s}")
    print(h); print("-" * len(h))
    for r in rows:
        b = r["bounds_m"]; D = r["actual_move_m"]
        print(f"{r['file']:32s} {D:8.0f} {b['L4']:9.0f} {b['L3']:8.0f} {b['L1']:8.0f} "
              f"{D/b['L4']:6.2f} {D/b['L3']:6.2f} {D/b['L1']:6.2f}")
    print(f"\nL3 启发式/MST下界 = "
          f"{sum(r['bounds_m']['L3'] for r in rows)/sum(r['bounds_m']['L3_mst_lb'] for r in rows):.3f} "
          f"(接近 1 说明 L3 已贴近真最优; MST 为路径TSP合法下界)")
    print(f"L4 构成均值: 可优化的段间空驶 {avg(lambda r: r['bounds_m']['L4_tour']):.0f} m + "
          f"不可压的段内几何(归航/清除) {avg(lambda r: r['bounds_m']['L4_service']):.0f} m")
    print(f"  实测段间空驶均值 {avg(lambda r: r['actual_move_m'] - r['bounds_m']['L4_service']):.0f} m "
          f"-> 最优 {avg(lambda r: r['bounds_m']['L4_tour']):.0f} m，"
          f"即段间空驶可压缩 "
          f"{100*(1-avg(lambda r: r['bounds_m']['L4_tour'])/avg(lambda r: r['actual_move_m']-r['bounds_m']['L4_service'])):.0f}%")

    print("\n" + "=" * 92)
    print("五、折算评分指标（虚拟时间/清除个数，秒）—— 只替换移动，检测与清除按实测保留")
    print("=" * 92)
    h = f"{'file':32s} {'实际':>9s} {'L4':>9s} {'L3':>9s} {'L1':>9s} {'L0全知零检测':>12s}"
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['file']:32s} {r['metric_actual']:9.1f} {r['metric_L4']:9.1f} "
              f"{r['metric_L3']:9.1f} {r['metric_L1']:9.1f} {r['metric_L0']:12.1f}")
    ma = avg(lambda r: r["metric_actual"])
    m4, m3, m1, m0 = (avg(lambda r: r[k]) for k in
                      ("metric_L4", "metric_L3", "metric_L1", "metric_L0"))
    print(f"\n均值 实际 {ma:.1f} s")
    print(f"  L4 只改访问顺序      -> {m4:7.1f} s  ({100*(m4-ma)/ma:+6.1f}%)  <== TSP 层现实收益上限")
    print(f"  L3 完美覆盖+摸源     -> {m3:7.1f} s  ({100*(m3-ma)/ma:+6.1f}%)  零归航浪费，需已知源位置")
    print(f"  L1 全知只摸源        -> {m1:7.1f} s  ({100*(m1-ma)/ma:+6.1f}%)  物理地板")
    print(f"  L0 全知且零检测      -> {m0:7.1f} s  ({100*(m0-ma)/ma:+6.1f}%)  信息论极限")
    print(f"\n判定：TSP 可拿 {-100*(m4-ma)/ma:.0f}%；余下 "
          f"{100*(m4-m1)/ma:.0f}% 的差距属于信息获取代价（不知道源在哪），"
          f"只能靠减少归航步数/提高示向度利用率，路径规划拿不到。")

    out = Path("results/b_tsp_bound.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"runs": rows, "mean": {
        "metric_actual": ma, "metric_L4": m4, "metric_L3": m3,
        "metric_L1": m1, "metric_L0": m0,
        "gain_L4_pct": 100 * (m4 - ma) / ma, "gain_L3_pct": 100 * (m3 - ma) / ma,
        "gain_L1_pct": 100 * (m1 - ma) / ma,
        "coverage_order_loss_m": avg(lambda r: r["coverage_order_loss_m"]),
        "ring_loss_total_m": avg(lambda r: r["ring_loss_total_m"]),
        "ring_loss_s": avg(lambda r: r["ring_loss_total_m"]) / SPEED,
    }}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
