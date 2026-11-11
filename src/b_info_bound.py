"""B 题问题 3/4 的信息论下界分析。

回答三个问题：
1. 把所有干扰源定位到 20 m 精度（清除半径）至少需要多少 bit；
2. 一次 /measure 最多能提供多少 bit（信道容量式上界 + 真实几何下的互信息）；
3. 官方实测的 610 次检测提供了多少 bit，信息效率比是多少。

依据：
- 目标区域半径 / 清除半径 / 频道数 / 干扰源数量区间 / 示向度误差：src/b_sim/arena.py
  （复刻自 problem/B题/B题.pdf + 附件1/2）
- 610 次检测：official_runs/q4_practice_run1.sanitized.json（官方客户端实测）

所有数值由本脚本实算，不使用预设常量以外的手写结果。
运行：cd <workdir> && python -m src.b_info_bound
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import Counter

import numpy as np

from src.b_sim.arena import (
    ARENA_RADIUS,
    CHANNEL_MAX,
    CHANNEL_MIN,
    CLEAR_RADIUS_M,
    DIRECTIONAL_HALF_ANGLE_DEG,
    ERROR_ABS_MAX_DEG,
    NEAR_THRESHOLD_M,
    RECEPTION_MAX_M,
    RECEPTION_MIN_M,
    SOURCE_COUNT_MAX,
    SOURCE_COUNT_MIN,
)

SVD_QUANT_DEG = 0.01          # 官方 svd_deg 保留 2 位小数
N_CHANNELS = CHANNEL_MAX - CHANNEL_MIN + 1
LOG2 = math.log2
RESULTS = "results/b_info_bound.json"
RNG_SEED = 20260910


# --------------------------------------------------------------------------
# A. 需求侧：定位到 20 m 精度需要多少 bit
# --------------------------------------------------------------------------

def position_bits(target_radius_m: float) -> float:
    """单个源位置从"半径 1800 m 圆域均匀"压缩到"半径 target 的圆盘"所需 bit。

    H = log2(面积比) = 2*log2(R/target)
    """
    return 2.0 * LOG2(ARENA_RADIUS / target_radius_m)


def position_bits_square_cell(cell_side_m: float) -> float:
    """备选判据：把源锁进边长 cell_side 的方格。"""
    return LOG2(math.pi * ARENA_RADIUS**2 / cell_side_m**2)


def requirement_bits(n_sources: int, n_directional: int = 0,
                     target_radius_m: float = CLEAR_RADIUS_M) -> dict:
    """一局案例的完整参数辨识信息量分解。"""
    pos = n_sources * position_bits(target_radius_m)
    # 哪 n 个频道被占用（1..20 无重复采样）
    chan = LOG2(math.comb(N_CHANNELS, n_sources))
    # 干扰源总数本身未知，10..16 共 7 种
    count = LOG2(SOURCE_COUNT_MAX - SOURCE_COUNT_MIN + 1)

    core = pos + chan + count

    # 问题 4 额外的"讨厌参数"：哪些源是定向的 + 定向方向
    # 方向本身不需要精确到 20 m 精度，但要判定"这个源在哪个扇区可见"必须解析它。
    flag = LOG2(math.comb(n_sources, n_directional)) if n_directional else 0.0
    # 方向解析到 ±1 度（示向度误差量级）：360/2 = 180 个可分辨值
    bearing_resolution = 2.0 * ERROR_ABS_MAX_DEG
    direction = n_directional * LOG2(360.0 / bearing_resolution)

    return {
        "n_sources": n_sources,
        "n_directional": n_directional,
        "target_radius_m": target_radius_m,
        "bits_position": pos,
        "bits_position_per_source": position_bits(target_radius_m),
        "bits_channel_subset": chan,
        "bits_source_count": count,
        "bits_core_localization": core,
        "bits_directional_flags": flag,
        "bits_directional_bearing": direction,
        "bits_total_with_nuisance": core + flag + direction,
    }


# --------------------------------------------------------------------------
# B. 供给侧：一次 /measure 的信息上界
# --------------------------------------------------------------------------

def measure_alphabet_size() -> int:
    """输出符号表大小：no_signal / near / 36000 个量化示向度。"""
    return 2 + int(round(360.0 / SVD_QUANT_DEG))


def noise_limited_capacity_bits() -> dict:
    """噪声受限的单次检测容量。

    svd = 真实方位 + eps, eps ~ U[-1,1] 度，输出量化到 0.01 度。
    给定 theta，输出在 200 个 0.01 度格子上近似均匀 => H(Y|theta) = log2(200)。
    H(Y) <= log2(36002)。
    """
    n_sym = measure_alphabet_size()
    n_noise_bins = int(round(2.0 * ERROR_ABS_MAX_DEG / SVD_QUANT_DEG))
    h_y_max = LOG2(n_sym)
    h_y_given_theta = LOG2(n_noise_bins)
    return {
        "alphabet_size": n_sym,
        "H_Y_max_bits": h_y_max,
        "H_Y_given_theta_bits": h_y_given_theta,
        "capacity_bits": h_y_max - h_y_given_theta,
        # 等价写法：方位角只能分辨到 ±1 度 => 180 个可分辨扇区
        "capacity_bits_check_180_sectors": LOG2(360.0 / (2.0 * ERROR_ABS_MAX_DEG)),
        "noise_bins": n_noise_bins,
    }


def mc_measure_mutual_information(station_xy, *, directional: bool,
                                  n_samples: int = 400_000,
                                  rng: np.random.Generator | None = None) -> dict:
    """蒙特卡洛算单个站位、单个"已占用频道"的一次检测互信息 I(theta; Y)。

    theta = (源位置 uniform 圆域, 接收半径 U[1000,1500], 定向方向 U[0,360))
    Y in {no_signal, near} U {量化示向度}

    I = H(Y) - H(Y|theta)
      H(Y)        = H(type) + p_dir * H(svd_q | dir)
      H(svd_q|dir)= H_1deg + log2(1/0.01)      (1 度直方图估微分熵，密度在 1 度尺度光滑)
      H(Y|theta)  = p_dir * log2(200)
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    px, py = station_xy

    rad = ARENA_RADIUS * np.sqrt(rng.random(n_samples))
    ang = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    sx, sy = rad * np.cos(ang), rad * np.sin(ang)
    recep = rng.uniform(RECEPTION_MIN_M, RECEPTION_MAX_M, n_samples)

    d = np.hypot(px - sx, py - sy)
    if directional:
        phi = rng.uniform(0.0, 360.0, n_samples)
        bearing_s2p = np.degrees(np.arctan2(py - sy, px - sx)) % 360.0
        gap = np.abs(bearing_s2p - phi) % 360.0
        gap = np.minimum(gap, 360.0 - gap)
        covered = gap <= DIRECTIONAL_HALF_ANGLE_DEG
    else:
        covered = np.ones(n_samples, dtype=bool)

    is_near = covered & (d <= recep) & (d <= NEAR_THRESHOLD_M)
    is_dir = covered & (d <= recep) & (d > NEAR_THRESHOLD_M)
    is_ns = ~(is_near | is_dir)

    p = np.array([is_ns.mean(), is_near.mean(), is_dir.mean()])
    h_type = -sum(pi * LOG2(pi) for pi in p if pi > 0)

    p_dir = float(p[2])
    if p_dir > 0:
        b0 = np.degrees(np.arctan2(sy[is_dir] - py, sx[is_dir] - px)) % 360.0
        eps = rng.uniform(-ERROR_ABS_MAX_DEG, ERROR_ABS_MAX_DEG, b0.size)
        svd = (b0 + eps) % 360.0
        hist, _ = np.histogram(svd, bins=360, range=(0.0, 360.0))
        q = hist / hist.sum()
        h_1deg = -sum(qi * LOG2(qi) for qi in q if qi > 0)
        # 量化到 0.01 度后的离散熵
        h_svd_q = h_1deg + LOG2(1.0 / SVD_QUANT_DEG)
    else:
        h_1deg = 0.0
        h_svd_q = 0.0

    h_noise = LOG2(2.0 * ERROR_ABS_MAX_DEG / SVD_QUANT_DEG)
    h_y = h_type + p_dir * h_svd_q
    mi = h_y - p_dir * h_noise

    return {
        "station": [float(px), float(py)],
        "station_radius_m": float(math.hypot(px, py)),
        "directional": directional,
        "p_no_signal": float(p[0]),
        "p_near": float(p[1]),
        "p_direction": p_dir,
        "H_type_bits": h_type,
        "H_svd_1deg_bits": h_1deg,
        "H_Y_bits": h_y,
        "H_Y_given_theta_bits": p_dir * h_noise,
        "mutual_information_bits": mi,
        "n_samples": n_samples,
    }


# --------------------------------------------------------------------------
# C. 实测侧：从官方日志算 610 次检测的输出分布
# --------------------------------------------------------------------------

def load_official_run(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def outcome_breakdown(run: dict) -> dict:
    types = Counter()
    svd_values = []
    stations = []
    seen_requests = set()
    for entry in run["protocol_log"]:
        if entry["path"] != "/measure":
            continue
        rid = entry["request"].get("request_id")
        if rid in seen_requests:       # 日志里有重复条目，按 request_id 去重
            continue
        seen_requests.add(rid)
        resp = entry["response"]
        res = resp.get("measure_result", "missing")
        types[res] += 1
        if res == "direction":
            svd_values.append(resp["svd_deg"])
        pos = entry["request"]["position"]
        stations.append((pos["x"], pos["y"]))
    return {
        "measure_calls_unique": sum(types.values()),
        "measure_calls_reported": run["summary"]["measure_calls"],
        "types": dict(types),
        "svd_values": svd_values,
        "stations": stations,
        "cleared_count": run["summary"]["cleared_count"],
        "clear_calls": run["summary"]["clear_calls"],
        "virtual_time_s": run["summary"]["virtual_time_s"],
    }


def realized_information_upper_bound(breakdown: dict) -> dict:
    """把实测输出序列的信息量上界算出来。

    上界 = sum over measurements of I_i，且 I_i <= H(Y_i) - H(Y_i|theta)。
    用整轮的经验边缘分布估 H(Y)：
      H(Y) = H(type_emp) + p_dir_emp * H(svd_q | dir)
    其中 H(svd_q|dir) 用 1 度直方图 + 量化修正估（样本少，直方图偏低=偏保守，
    所以另给一个"角度均匀"的宽松上界 log2(36000)）。
    这是上界而非精确值：真实检测彼此高度相关（大量重复 no_signal 几乎不带新信息）。
    """
    n = breakdown["measure_calls_unique"]
    types = breakdown["types"]
    p = {k: v / n for k, v in types.items()}
    h_type = -sum(pi * LOG2(pi) for pi in p.values() if pi > 0)
    p_dir = p.get("direction", 0.0)

    svd = np.asarray(breakdown["svd_values"], dtype=float)
    if svd.size:
        hist, _ = np.histogram(svd % 360.0, bins=36, range=(0.0, 360.0))
        q = hist / hist.sum()
        h_10deg = -sum(qi * LOG2(qi) for qi in q if qi > 0)
        h_svd_emp = h_10deg + LOG2(10.0 / SVD_QUANT_DEG)
    else:
        h_10deg = 0.0
        h_svd_emp = 0.0
    h_svd_uniform = LOG2(360.0 / SVD_QUANT_DEG)
    h_noise = LOG2(2.0 * ERROR_ABS_MAX_DEG / SVD_QUANT_DEG)

    per_meas_emp = h_type + p_dir * (h_svd_emp - h_noise)
    per_meas_loose = h_type + p_dir * (h_svd_uniform - h_noise)
    cap = noise_limited_capacity_bits()["capacity_bits"]

    return {
        "n_measurements": n,
        "type_probs": p,
        "H_type_bits": h_type,
        "p_direction": p_dir,
        "H_svd_10deg_bits": h_10deg,
        "H_svd_quantized_empirical_bits": h_svd_emp,
        "H_svd_quantized_uniform_bits": h_svd_uniform,
        "H_Y_given_theta_bits": p_dir * h_noise,
        "bits_per_measurement_empirical": per_meas_emp,
        "bits_per_measurement_loose": per_meas_loose,
        "bits_per_measurement_capacity": cap,
        "total_bits_empirical": n * per_meas_emp,
        "total_bits_loose": n * per_meas_loose,
        "total_bits_capacity": n * cap,
    }


# --------------------------------------------------------------------------
# D. 汇总
# --------------------------------------------------------------------------

def main() -> None:
    out: dict = {"constants": {
        "arena_radius_m": ARENA_RADIUS,
        "clear_radius_m": CLEAR_RADIUS_M,
        "near_threshold_m": NEAR_THRESHOLD_M,
        "reception_radius_m": [RECEPTION_MIN_M, RECEPTION_MAX_M],
        "n_channels": N_CHANNELS,
        "source_count_range": [SOURCE_COUNT_MIN, SOURCE_COUNT_MAX],
        "bearing_error_abs_max_deg": ERROR_ABS_MAX_DEG,
        "svd_quantization_deg": SVD_QUANT_DEG,
        "directional_half_angle_deg": DIRECTIONAL_HALF_ANGLE_DEG,
    }}

    print("=" * 78)
    print("A. 需求侧：定位所有干扰源到 20 m 精度需要多少 bit")
    print("=" * 78)
    print(f"单源位置信息量 = 2*log2(1800/20) = {position_bits(CLEAR_RADIUS_M):.4f} bit")
    print(f"  （备选判据：锁进 20 m 边长方格 = {position_bits_square_cell(20.0):.4f} bit）")
    print()
    print(f"{'N':>3} {'定向':>4} | {'位置':>9} {'频道子集':>9} {'总数':>6} | "
          f"{'核心需求':>9} | {'+定向讨厌参数':>13}")
    print("-" * 78)
    reqs = {}
    for n in range(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX + 1):
        r3 = requirement_bits(n, 0)
        reqs[f"q3_N{n}"] = r3
        print(f"{n:>3} {0:>4} | {r3['bits_position']:>9.2f} {r3['bits_channel_subset']:>9.2f} "
              f"{r3['bits_source_count']:>6.2f} | {r3['bits_core_localization']:>9.2f} | "
              f"{'—':>13}")
    print("-" * 78)
    for n in range(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX + 1):
        k = max(1, round(n * 5 / 12))          # 官方演练量级：约 1/3~1/2 为定向
        r4 = requirement_bits(n, k)
        reqs[f"q4_N{n}_dir{k}"] = r4
        print(f"{n:>3} {k:>4} | {r4['bits_position']:>9.2f} {r4['bits_channel_subset']:>9.2f} "
              f"{r4['bits_source_count']:>6.2f} | {r4['bits_core_localization']:>9.2f} | "
              f"{r4['bits_total_with_nuisance']:>13.2f}")
    out["requirement"] = reqs

    print()
    print("=" * 78)
    print("B. 供给侧：一次 /measure 的信息上界")
    print("=" * 78)
    cap = noise_limited_capacity_bits()
    out["capacity"] = cap
    print(f"输出符号表大小            = {cap['alphabet_size']}  (no_signal, near, 36000 个 0.01° 示向度)")
    print(f"H(Y) 上界                 = {cap['H_Y_max_bits']:.4f} bit")
    print(f"H(Y|theta)（±1° 噪声）     = {cap['H_Y_given_theta_bits']:.4f} bit  ({cap['noise_bins']} 个噪声格)")
    print(f"单次检测容量 C            = {cap['capacity_bits']:.4f} bit")
    print(f"核对（180 个可分辨扇区）   = {cap['capacity_bits_check_180_sectors']:.4f} bit")

    print()
    print("真实几何下的互信息（蒙特卡洛，单站位、频道已占用、位置先验均匀）：")
    print(f"{'站位半径':>8} {'类型':>6} | {'P(ns)':>7} {'P(near)':>8} {'P(dir)':>7} | "
          f"{'H(Y)':>8} {'I':>7}")
    print("-" * 78)
    rng = np.random.default_rng(RNG_SEED)
    mc = []
    for radius in (0.0, 600.0, 1250.0, 1750.0):
        for directional in (False, True):
            m = mc_measure_mutual_information((radius, 0.0), directional=directional,
                                              n_samples=400_000, rng=rng)
            mc.append(m)
            print(f"{radius:>8.0f} {'定向' if directional else '全向':>6} | "
                  f"{m['p_no_signal']:>7.4f} {m['p_near']:>8.5f} {m['p_direction']:>7.4f} | "
                  f"{m['H_Y_bits']:>8.4f} {m['mutual_information_bits']:>7.4f}")
    out["monte_carlo_station_mi"] = mc
    best_mi = max(m["mutual_information_bits"] for m in mc)
    print(f"\n最好站位的单次检测互信息  = {best_mi:.4f} bit（<= 容量 {cap['capacity_bits']:.4f}）")

    print()
    print("=" * 78)
    print("C. 实测侧：官方演练日志的检测输出分布")
    print("=" * 78)
    paths = sorted(glob.glob("official_runs/q4_practice_run*.sanitized.json"))
    runs = {}
    for path in paths:
        run = load_official_run(path)
        bd = outcome_breakdown(run)
        info = realized_information_upper_bound(bd)
        label = os.path.basename(path).split(".")[0]
        runs[label] = {"breakdown": {k: v for k, v in bd.items()
                                    if k not in ("svd_values", "stations")},
                       "information": info}
        print(f"\n{label}: 检测 {info['n_measurements']} 次（日志汇总报 "
              f"{bd['measure_calls_reported']}），清除 {bd['cleared_count']} 个")
        print(f"  输出类型分布: " + ", ".join(
            f"{k}={v}({v/info['n_measurements']*100:.1f}%)" for k, v in
            sorted(bd["types"].items(), key=lambda kv: -kv[1])))
        print(f"  H(type)={info['H_type_bits']:.4f}  p_dir={info['p_direction']:.4f}")
        print(f"  单次检测信息上界: 经验 {info['bits_per_measurement_empirical']:.4f} bit / "
              f"宽松 {info['bits_per_measurement_loose']:.4f} bit / "
              f"容量 {info['bits_per_measurement_capacity']:.4f} bit")
        print(f"  整轮信息上界:     经验 {info['total_bits_empirical']:.1f} bit / "
              f"宽松 {info['total_bits_loose']:.1f} bit / "
              f"容量 {info['total_bits_capacity']:.1f} bit")
    out["official_runs"] = runs

    print()
    print("=" * 78)
    print("D. 信息效率比")
    print("=" * 78)
    main_label = "q4_practice_run1"
    info = runs[main_label]["information"]
    bd = runs[main_label]["breakdown"]
    n_cleared = bd["cleared_count"]
    n_meas = info["n_measurements"]

    # 该局至少 12 个源（清掉 12 个），定向数未知；按官方演练 5/12 定向估
    k_dir = round(n_cleared * 5 / 12)
    req = requirement_bits(n_cleared, k_dir)
    req_core = req["bits_core_localization"]
    req_full = req["bits_total_with_nuisance"]

    rows = []
    for name, supplied in (
        ("容量上界 C=7.4919 bit/次", info["total_bits_capacity"]),
        ("宽松经验上界", info["total_bits_loose"]),
        ("经验上界（含实测角分布）", info["total_bits_empirical"]),
        ("最优站位互信息 × 次数", best_mi * n_meas),
    ):
        rows.append({
            "supply_model": name,
            "supplied_bits": supplied,
            "required_core_bits": req_core,
            "required_full_bits": req_full,
            "efficiency_core": req_core / supplied if supplied else None,
            "efficiency_full": req_full / supplied if supplied else None,
            "min_detections_core": math.ceil(req_core / (supplied / n_meas)) if supplied else None,
        })

    print(f"本局（{main_label}）：清除 {n_cleared} 个源，检测 {n_meas} 次，"
          f"虚拟时间 {bd['virtual_time_s']:.0f} s")
    print(f"需求：核心定位 {req_core:.2f} bit"
          f"（位置 {req['bits_position']:.2f} + 频道子集 {req['bits_channel_subset']:.2f}"
          f" + 源数 {req['bits_source_count']:.2f}）")
    print(f"      含定向讨厌参数 {req_full:.2f} bit（定向标记 {req['bits_directional_flags']:.2f}"
          f" + 方向 {req['bits_directional_bearing']:.2f}，按 {k_dir} 个定向源估）")
    print()
    print(f"{'供给模型':<26} {'供给(bit)':>11} {'效率(核心)':>11} {'效率(含讨厌)':>13} {'理论最少检测':>13}")
    print("-" * 78)
    for r in rows:
        print(f"{r['supply_model']:<26} {r['supplied_bits']:>11.1f} "
              f"{r['efficiency_core']*100:>10.2f}% {r['efficiency_full']*100:>12.2f}% "
              f"{r['min_detections_core']:>13d}")
    out["efficiency"] = {"run": main_label, "requirement": req, "rows": rows,
                         "best_station_mi_bits": best_mi}

    # 问题 3 对照（官方 q3 演练：209 次检测、清除 10 个）
    print()
    q3_meas, q3_cleared = 209, 10
    req3 = requirement_bits(q3_cleared, 0)["bits_core_localization"]
    sup3 = q3_meas * cap["capacity_bits"]
    print(f"问题 3 对照（官方演练 {q3_meas} 次检测、清除 {q3_cleared} 个）：")
    print(f"  需求 {req3:.2f} bit，容量供给 {sup3:.1f} bit，效率 {req3/sup3*100:.2f}%，"
          f"理论最少检测 {math.ceil(req3/cap['capacity_bits'])} 次")
    q4_eff = req_core / info["total_bits_capacity"]
    print(f"  问题 4 同口径效率 {q4_eff*100:.2f}% => 问题 3 的信息效率是问题 4 的 "
          f"{(req3/sup3)/q4_eff:.2f} 倍")
    out["q3_reference"] = {
        "measure_calls": q3_meas, "cleared": q3_cleared,
        "required_core_bits": req3, "supplied_capacity_bits": sup3,
        "efficiency_core": req3 / sup3,
        "min_detections_core": math.ceil(req3 / cap["capacity_bits"]),
        "q4_efficiency_same_basis": q4_eff,
        "q3_over_q4_efficiency_ratio": (req3 / sup3) / q4_eff,
    }

    print()
    print("=" * 78)
    print("E. 构造性下界（比信息论下界更可达，用于判断优化空间）")
    print("=" * 78)
    # 频道占用必须逐一试：20 个频道，每个至少 1 次检测才能排除
    ch_probe = N_CHANNELS
    # 每个源至少 2 条示向度才能交会出位置（1 条只给方向，距离未知）
    tri = 2 * n_cleared
    constructive = ch_probe + tri
    # 若允许"频道扫描与示向度采集共用检测"（同一站位换频道），下界降到 max(20, 2N)
    shared = max(ch_probe, tri)
    print(f"频道占用探测下界        = {ch_probe} 次（20 个频道各至少 1 次）")
    print(f"三角交会下界            = {tri} 次（{n_cleared} 个源 × 2 条示向度）")
    print(f"不共用时构造性下界      = {constructive} 次")
    print(f"共用检测时构造性下界    = {shared} 次")
    info_lb = math.ceil(req_core / cap["capacity_bits"])
    print(f"信息论下界（容量口径）   = {info_lb} 次  <= 构造性下界 {shared} 次 "
          f"（{'一致' if info_lb <= shared else '矛盾，需复查'}）")
    print(f"实测 {n_meas} 次 / 构造性下界 {shared} 次 = "
          f"{n_meas / shared:.1f} 倍冗余")
    print(f"实测 {n_meas} 次 / 信息论下界 {info_lb} 次 = "
          f"{n_meas / info_lb:.1f} 倍冗余")
    out["constructive_bound"] = {
        "channel_probe": ch_probe, "triangulation": tri,
        "constructive_disjoint": constructive, "constructive_shared": shared,
        "information_bound": info_lb,
        "redundancy_vs_constructive": n_meas / shared,
        "redundancy_vs_information": n_meas / info_lb,
    }

    # 为什么效率低：no_signal 占 88%，每次只值 H(type) 的一小部分
    p_ns = runs[main_label]["information"]["type_probs"].get("no_signal", 0.0)
    print(f"\n低效来源：{p_ns*100:.1f}% 的检测返回 no_signal，单次仅值 "
          f"{-LOG2(p_ns):.3f} bit（信息量 -log2 p），而一条示向度值 "
          f"{cap['capacity_bits']:.2f} bit。")
    print("=> 提升效率的杠杆是提高每次检测命中示向度的概率（站位/频道调度），"
          "而非增加检测次数。")
    out["inefficiency_source"] = {
        "p_no_signal": p_ns,
        "bits_per_no_signal": -LOG2(p_ns) if p_ns > 0 else None,
        "bits_per_direction_at_capacity": cap["capacity_bits"],
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {RESULTS}")


if __name__ == "__main__":
    main()
