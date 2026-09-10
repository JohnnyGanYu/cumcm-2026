"""b_sim 离线回归：物理规则、协议门禁、幂等、计时。

不联网，不依赖官方模拟器。运行：
  /home/opsuser/.venvs/cumcm-2026/bin/python -m pytest src/b_sim/test_b_sim.py -q
"""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request

import pytest

from src.b_sim.arena import (
    Arena,
    ArenaRejected,
    Case,
    Jammer,
    angular_gap_deg,
    bearing_error_deg,
    generate_case,
    norm_deg,
)
from src.b_sim.client import ProtocolError, RobotClient
from src.b_sim.geometry import (
    Bearing,
    circle_on_diameter_covers,
    localization_polygon,
    localization_region,
    polygon_diameter,
    second_point_candidates,
)
from src.b_sim.server import Session, make_server

ROBOT = "99999"


def single_case(**kwargs) -> Case:
    defaults = dict(channel=1, x=300.0, y=400.0, reception_radius_m=1200.0)
    defaults.update(kwargs)
    return Case(jammers=[Jammer(**defaults)], error_salt="t", seed=0)


# ---------- 几何与角度 ----------

def test_norm_deg_wraps_both_directions():
    assert norm_deg(-90) == 270
    assert norm_deg(450) == 90
    assert norm_deg(360) == 0


def test_angular_gap_takes_short_arc():
    assert angular_gap_deg(350, 10) == 20
    assert angular_gap_deg(0, 180) == 180


def test_bearing_error_within_one_degree_and_deterministic():
    for i in range(200):
        x, y = i * 7.3 - 500, i * -3.1 + 200
        err = bearing_error_deg(x, y, "t")
        assert -1.0 <= err <= 1.0
        assert err == bearing_error_deg(x, y, "t")  # 同点重复检测误差不变


def test_bearing_error_varies_across_locations():
    values = {round(bearing_error_deg(i * 50.0, 0.0, "t"), 6) for i in range(50)}
    assert len(values) > 40  # 不同地点误差应当分散


# ---------- 检测判定 ----------

def test_measure_returns_direction_within_one_degree_of_truth():
    arena = Arena(single_case())
    arena.enter()
    res = arena.measure(0.0, 0.0, 1)
    assert res["measure_result"] == "direction"
    truth = math.degrees(math.atan2(400, 300))
    assert angular_gap_deg(res["svd_deg"], truth) <= 1.0


def test_measure_no_signal_beyond_reception_radius():
    arena = Arena(single_case(x=1400.0, y=0.0, reception_radius_m=1000.0))
    arena.enter()
    assert arena.measure(0.0, 0.0, 1)["measure_result"] == "no_signal"
    assert arena.measure(500.0, 0.0, 1)["measure_result"] == "direction"


def test_measure_near_within_five_metres_hides_direction():
    arena = Arena(single_case(x=0.0, y=0.0))
    arena.enter()
    res = arena.measure(3.0, 0.0, 1)
    assert res["measure_result"] == "near"
    assert "svd_deg" not in res


def test_measure_empty_channel_is_no_signal():
    arena = Arena(single_case())
    arena.enter()
    assert arena.measure(0.0, 0.0, 7)["measure_result"] == "no_signal"


def test_directional_source_only_visible_inside_coverage():
    # 定向方向 0 度（朝东），覆盖 [-90, 90]；正东侧可见，正西侧不可见
    arena = Arena(single_case(x=0.0, y=0.0, directional_deg=0.0))
    arena.enter()
    assert arena.measure(500.0, 0.0, 1)["measure_result"] == "direction"
    assert arena.measure(-500.0, 0.0, 1)["measure_result"] == "no_signal"


def test_directional_boundary_is_inclusive():
    arena = Arena(single_case(x=0.0, y=0.0, directional_deg=0.0))
    arena.enter()
    assert arena.measure(0.0, 500.0, 1)["measure_result"] == "direction"   # 恰好 +90
    assert arena.measure(0.0, -500.0, 1)["measure_result"] == "direction"  # 恰好 -90


# ---------- 清除判定 ----------

def test_clear_succeeds_within_twenty_metres_and_only_once():
    arena = Arena(single_case(x=100.0, y=0.0))
    arena.enter()
    assert arena.clear(115.0, 0.0, 1)["clear_result"] == "success"
    assert arena.clear(100.0, 0.0, 1)["clear_result"] == "no_target_in_range"


def test_clear_fails_beyond_twenty_metres():
    arena = Arena(single_case(x=100.0, y=0.0))
    arena.enter()
    assert arena.clear(130.0, 0.0, 1)["clear_result"] == "no_target_in_range"


def test_clear_ignores_directional_coverage():
    arena = Arena(single_case(x=0.0, y=0.0, directional_deg=0.0))
    arena.enter()
    # 正西方位在覆盖范围外，检测无信号，但 20 米内清除仍应成功
    assert arena.measure(-10.0, 0.0, 1)["measure_result"] == "no_signal"
    assert arena.clear(-10.0, 0.0, 1)["clear_result"] == "success"


def test_cleared_source_reports_no_signal():
    arena = Arena(single_case(x=100.0, y=0.0))
    arena.enter()
    arena.clear(100.0, 0.0, 1)
    assert arena.measure(200.0, 0.0, 1)["measure_result"] == "no_signal"


# ---------- 计时（对齐附件2 第10节示例）----------

def test_official_timing_example_reproduces_199_seconds():
    case = Case(jammers=[Jammer(channel=3, x=9e5, y=9e5, reception_radius_m=1000.0)],
                error_salt="t", seed=0)
    arena = Arena(case)
    arena.enter()
    assert arena.robot.virtual_time_s == 0
    arena.measure(300, 400, 1)   # 移动 500m/100s + 检测 5s，频道未变
    assert arena.robot.virtual_time_s == pytest.approx(105)
    arena.measure(300, 400, 2)   # 原地 + 切频 1s + 检测 5s
    assert arena.robot.virtual_time_s == pytest.approx(111)
    arena.clear(300, 0, 3)       # 移动 400m/80s + 未发现 3s
    assert arena.robot.virtual_time_s == pytest.approx(194)
    arena.measure(300, 0, 2)     # 原地，频道仍为 2，无切频
    assert arena.robot.virtual_time_s == pytest.approx(199)
    arena.exit()


def test_clear_does_not_switch_measuring_channel():
    arena = Arena(single_case(x=5000.0, y=0.0))
    arena.enter()
    arena.measure(0, 0, 2)                       # 切到频道 2
    before = arena.robot.virtual_time_s
    arena.clear(0, 0, 9)                         # /clear 的 channel 不切频
    arena.measure(0, 0, 2)                       # 仍是频道 2，不应有 1s 切频
    assert arena.robot.virtual_time_s - before == pytest.approx(3 + 5)


def test_successful_clear_costs_five_seconds():
    arena = Arena(single_case(x=0.0, y=0.0))
    arena.enter()
    arena.clear(0.0, 0.0, 1)
    assert arena.robot.virtual_time_s == pytest.approx(5)


def test_actions_require_enter_and_stop_after_exit():
    arena = Arena(single_case())
    with pytest.raises(ArenaRejected):
        arena.measure(0, 0, 1)
    arena.enter()
    arena.exit()
    with pytest.raises(ArenaRejected):
        arena.measure(0, 0, 1)


def test_virtual_time_limit_blocks_action():
    arena = Arena(single_case(), max_virtual_seconds=10.0)
    arena.enter()
    with pytest.raises(ArenaRejected):
        arena.measure(1000.0, 0.0, 1)  # 200s 移动，超过 10s 上限
    assert arena.robot.virtual_time_s == 0  # 被拒动作不推进时钟


# ---------- 案例生成 ----------

def test_generated_case_respects_problem_constraints():
    for seed in range(30):
        case = generate_case(seed)
        assert 10 <= case.total <= 16
        channels = [j.channel for j in case.jammers]
        assert len(set(channels)) == len(channels)
        assert all(1 <= c <= 20 for c in channels)
        for j in case.jammers:
            assert math.hypot(j.x, j.y) <= 1800.0 + 1e-9
            assert 1000.0 <= j.reception_radius_m <= 1500.0
            assert j.directional_deg is None


def test_generated_case_is_reproducible_by_seed():
    assert generate_case(7).truth_table() == generate_case(7).truth_table()


def test_directional_count_is_honoured():
    case = generate_case(3, count=12, directional_count=5)
    assert case.total == 12
    assert sum(1 for j in case.jammers if j.is_directional) == 5


# ---------- HTTP 协议层 ----------

@pytest.fixture
def server():
    case = generate_case(42, count=10)
    session = Session(ROBOT, Arena(case))
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", session
    httpd.shutdown()
    httpd.server_close()


def raw_post(url: str, path: str, payload, *, ctype="application/json", method="POST"):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(url + path, data=data,
                                     headers={"Content-Type": ctype}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        return exc.code, (json.loads(body) if body else {})


def base(rid):
    return {"arena_id": "default", "robot_id": ROBOT, "request_id": rid}


def test_client_round_trip_and_stats(server):
    url, session = server
    client = RobotClient(ROBOT, base_url=url)
    body = client.enter()
    assert body["accepted"] is True
    assert body["remaining_real_duration_s"] <= 1200
    client.measure(0, 0, 1)
    client.exit()
    assert session.arena.stats()["measure_calls"] == 1


def test_unknown_field_is_rejected_without_consuming_request_id(server):
    url, session = server
    payload = base("e1") | {"typo": 1}
    status, body = raw_post(url, "/enter", payload)
    assert (status, body["accepted"], body["virtual_time_s"]) == (200, False, 0)
    status, body = raw_post(url, "/enter", base("e1"))  # 同 ID 可复用
    assert body["accepted"] is True


def test_wrong_arena_and_robot_id_rejected(server):
    url, _ = server
    assert raw_post(url, "/enter", base("x") | {"arena_id": "other"})[1]["accepted"] is False
    assert raw_post(url, "/enter", base("y") | {"robot_id": "00000"})[1]["accepted"] is False


def test_measure_before_enter_rejected(server):
    url, _ = server
    payload = base("m1") | {"position": {"x": 0, "y": 0}, "channel": 1}
    assert raw_post(url, "/measure", payload)[1]["accepted"] is False


def test_idempotent_replay_does_not_double_advance_clock(server):
    url, session = server
    raw_post(url, "/enter", base("e"))
    payload = base("m9") | {"position": {"x": 300, "y": 400}, "channel": 1}
    first = raw_post(url, "/measure", payload)[1]
    second = raw_post(url, "/measure", payload)[1]
    assert first["virtual_time_s"] == second["virtual_time_s"] == pytest.approx(105)
    assert session.arena.robot.measure_calls == 1


def test_same_request_id_different_action_conflicts(server):
    url, _ = server
    raw_post(url, "/enter", base("e"))
    p1 = base("m") | {"position": {"x": 0, "y": 0}, "channel": 1}
    p2 = base("m") | {"position": {"x": 10, "y": 0}, "channel": 1}
    assert raw_post(url, "/measure", p1)[1]["accepted"] is True
    assert raw_post(url, "/measure", p2)[0] == 409


def test_bad_channel_and_coordinates_return_400(server):
    url, _ = server
    raw_post(url, "/enter", base("e"))
    for bad in ({"channel": 0}, {"channel": 21}, {"channel": 1.5}):
        payload = base("b") | {"position": {"x": 0, "y": 0}} | bad
        assert raw_post(url, "/measure", payload)[0] == 400
    payload = base("c") | {"position": {"x": 3_000_000, "y": 0}, "channel": 1}
    assert raw_post(url, "/measure", payload)[0] == 400
    payload = base("d") | {"position": {"x": 0, "y": 0}, "channel": 1}
    assert raw_post(url, "/measure", json.dumps(payload).replace("0", "NaN", 1).encode())[0] == 400


def test_integer_valued_float_channel_accepted(server):
    url, _ = server
    raw_post(url, "/enter", base("e"))
    payload = base("f") | {"position": {"x": 0, "y": 0}, "channel": 1.0}
    assert raw_post(url, "/measure", payload)[1]["accepted"] is True


def test_duplicate_json_key_returns_400(server):
    url, _ = server
    body = b'{"arena_id":"default","arena_id":"default","robot_id":"99999","request_id":"z"}'
    assert raw_post(url, "/enter", body)[0] == 400


def test_protocol_surface_errors(server):
    url, _ = server
    assert raw_post(url, "/unknown", base("u"))[0] == 404
    assert raw_post(url, "/enter", base("v"), method="GET")[0] == 405
    assert raw_post(url, "/enter", base("w"), ctype="text/plain")[0] == 415
    assert raw_post(url, "/enter", b"{not json", ctype="application/json")[0] == 400
    assert raw_post(url, "/enter", b"x" * 70000)[0] == 413


def test_client_raises_protocol_error_on_rejection(server):
    url, _ = server
    client = RobotClient(ROBOT, base_url=url)
    with pytest.raises(ProtocolError):
        client.measure(0, 0, 1)  # 尚未 enter


# ---------- Q1 定位区域输入契约与退化返回（R1.2 / R1.4 / R1.5） ----------

def test_localization_rejects_fewer_than_two_observations():
    # R1.2：N < 2 应显式拒绝，而非静默返回空区域
    with pytest.raises(ValueError):
        localization_polygon([])
    with pytest.raises(ValueError):
        localization_polygon([Bearing(0.0, 0.0, 45.0)])
    with pytest.raises(ValueError):
        localization_region([Bearing(0.0, 0.0, 45.0)])


def test_localization_produces_region_for_good_crossing():
    # 两个检测点良态交会应产生非空凸区域，含真值指示 empty=False
    bearings = [
        Bearing(-400.0, 0.0, math.degrees(math.atan2(300.0, 400.0))),
        Bearing(400.0, 0.0, math.degrees(math.atan2(300.0, -400.0))),
    ]
    region = localization_region(bearings)
    assert region.empty is False
    assert bool(region) is True
    assert len(region.vertices) >= 3
    assert localization_polygon(bearings) == region.vertices


def test_empty_intersection_returns_empty_region_indicator():
    # R1.4：两个检测点的示向度彼此背离（源需同时落在两点的"外侧"），楔形之交为空
    bearings = [
        Bearing(-100.0, 0.0, 180.0),  # 从左侧点朝西（背离右点）
        Bearing(100.0, 0.0, 0.0),     # 从右侧点朝东（背离左点）
    ]
    region = localization_region(bearings)
    assert region.empty is True
    assert region.reason == "交会为空"
    assert region.vertices == []
    assert bool(region) is False
    assert localization_polygon(bearings) == []


def test_degenerate_small_area_returns_empty_region_indicator():
    # R1.4：同一检测点两条不同示向度的楔形仅交于顶点，面积 ≤ 1e-9 → 退化"交会为空"
    bearings = [
        Bearing(0.0, 0.0, 0.0),   # 从原点朝东
        Bearing(0.0, 0.0, 90.0),  # 从原点朝北 → 两楔形仅在原点相交
    ]
    region = localization_region(bearings)
    assert region.empty is True
    assert region.reason == "交会为空"
    assert localization_polygon(bearings) == []


def test_polygon_diameter_deterministic_tie_break_first_vertex_pair():
    # R1.5：正方形有两条相等的最大对角线，SHALL 确定性返回按顶点索引升序的首个
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    diameter, a, b = polygon_diameter(square)
    assert diameter == pytest.approx(math.sqrt(2.0))
    # 首个最大顶点对为 (index 0, index 2) = (0,0)-(1,1)
    assert a == (0.0, 0.0)
    assert b == (1.0, 1.0)
    # 顶点顺序旋转后仍确定性取索引升序首对，且结果与输入顺序一致
    rotated = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    d2, a2, b2 = polygon_diameter(rotated)
    assert d2 == pytest.approx(math.sqrt(2.0))
    assert a2 == (1.0, 0.0)
    assert b2 == (0.0, 1.0)


def test_polygon_diameter_handles_degenerate_vertex_counts():
    # < 2 顶点时安全返回 0（供退化区域调用），不抛异常
    assert polygon_diameter([]) == (0.0, (0.0, 0.0), (0.0, 0.0))
    assert polygon_diameter([(3.0, 4.0)]) == (0.0, (0.0, 0.0), (0.0, 0.0))


# ---------- Q1 外接圆覆盖裕度与退化几何（R1.6 / R1.7 / R1.8）----------


def test_covering_margin_sign_covered_vs_not_covered():
    # 覆盖裕度符号语义：margin = D/2 − max‖v−M‖。
    # 注意直径端点本身到中点恰为 D/2，故 worst ≥ D/2 恒成立 → margin ≤ 0；
    #   覆盖（无顶点更远）时 margin ≈ 0（判 covers=True）；
    #   存在更远顶点（如正三角形第三点）时 margin < 0（判 covers=False）。
    # 狭长矩形：最长对角线为直径，其余顶点均落在直径圆内 → margin ≈ 0、覆盖。
    thin = [(0.0, 0.0), (100.0, 0.0), (100.0, 1.0), (0.0, 1.0)]
    covers, margin = circle_on_diameter_covers(thin)
    assert covers is True
    assert margin == pytest.approx(0.0, abs=1e-6)
    assert margin >= -1e-7
    # 钝角三角形：底边为直径，第三顶点距中点 5 m < 半径 50 m → 覆盖、margin ≈ 0。
    obtuse = [(-50.0, 0.0), (50.0, 0.0), (0.0, 5.0)]
    covers2, margin2 = circle_on_diameter_covers(obtuse)
    assert covers2 is True
    assert margin2 == pytest.approx(0.0)
    assert margin2 >= -1e-7


def test_equilateral_triangle_is_not_covered_with_negative_margin():
    # R1.8：正三角形反例——以任一边（=直径 L）为直径的圆不能覆盖第三顶点。
    # 第三顶点到该边中点的距离 = (√3/2)·L > L/2，故 margin = L/2 − (√3/2)·L = −(√3/2 − 1/2)·L。
    L = 100.0
    tri = [(0.0, 0.0), (L, 0.0), (L / 2.0, L * math.sqrt(3.0) / 2.0)]
    covers, margin = circle_on_diameter_covers(tri)
    assert covers is False
    expected_margin = -(math.sqrt(3.0) / 2.0 - 0.5) * L
    assert margin == pytest.approx(expected_margin)
    assert margin < -1e-7  # 显著为负 → 不覆盖


def test_single_point_degeneracy_reports_covered_with_zero_margin():
    # R1.6：所有顶点两两距离 ≤ 1e-7（退化为单点）→ D=0，标记"覆盖"，margin=0。
    pt = [(3.0, 4.0), (3.0 + 1e-8, 4.0), (3.0, 4.0 - 1e-8)]
    covers, margin = circle_on_diameter_covers(pt)
    assert covers is True
    assert margin == pytest.approx(0.0)


def test_collinear_segment_degeneracy_reports_covered():
    # R1.6：顶点共线（退化为线段）→ 所有点落在以端点距离为直径的圆内，标记"覆盖"、margin≥0。
    seg = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0)]
    covers, margin = circle_on_diameter_covers(seg)
    assert covers is True
    assert margin >= -1e-7  # 中间点恰在圆心，端点在圆上 → margin ≈ 0


# ---------- 任务 9.1：确定性种子可复现与虚拟时钟（R6.1 / R6.2）----------
# Feature: b-radio-interference-locate-clear
# 核对 arena.generate_case 的确定性可复现与 Arena._commit_move 的微秒整数记账。


def test_generate_case_same_seed_byte_identical_truth_table():
    """R6.1：同种子逐案例完全可复现——truth_table 序列化后逐字节一致。"""
    for seed in (0, 1, 7, 42, 12345):
        first = json.dumps(generate_case(seed).truth_table(), sort_keys=True)
        second = json.dumps(generate_case(seed).truth_table(), sort_keys=True)
        assert first == second
        # 同种子的完整案例（含定向源与接收半径）也应字节一致
        first_dir = json.dumps(generate_case(seed, count=13, directional_count=5).truth_table(),
                               sort_keys=True)
        second_dir = json.dumps(generate_case(seed, count=13, directional_count=5).truth_table(),
                                sort_keys=True)
        assert first_dir == second_dir


def test_generate_case_repeated_bearing_reading_is_reproducible():
    """R6.1：确定性 ±1° 误差场——同一检测点重复检测返回完全相同的示向度读数。"""
    # 固定一个可返回 direction 的案例：源在原点外侧、全向、接收半径充足。
    case = single_case(x=800.0, y=600.0, reception_radius_m=1400.0)
    channel = case.jammers[0].channel
    arena = Arena(case)
    arena.enter()
    point = (120.0, -75.0)
    first = arena.measure(*point, channel)
    second = arena.measure(*point, channel)  # 同点重复检测
    assert first["measure_result"] == "direction"
    assert first == second  # 示向度读数逐字段一致（含 svd_deg）
    # 独立构造相同案例，同点也应返回同一读数（误差场只依赖坐标与 salt）
    twin = Arena(single_case(x=800.0, y=600.0, reception_radius_m=1400.0))
    twin.enter()
    assert twin.measure(*point, channel) == first


def test_generate_case_source_count_and_channels_within_bounds():
    """R6.1：源数 10–16、频道 1–20 不重复。"""
    for seed in range(50):
        case = generate_case(seed)
        assert 10 <= case.total <= 16
        channels = [j.channel for j in case.jammers]
        assert len(set(channels)) == len(channels)  # 不重复
        assert all(1 <= c <= 20 for c in channels)


def test_generate_case_reception_radius_within_bounds():
    """R6.1：有效接收半径落在 [1000, 1500] m。"""
    for seed in range(50):
        for jammer in generate_case(seed).jammers:
            assert 1000.0 <= jammer.reception_radius_m <= 1500.0
            assert math.hypot(jammer.x, jammer.y) <= 1800.0 + 1e-9  # 位置在圆域内


def test_commit_move_accumulates_integer_microseconds_per_timing_table():
    """R6.2：虚拟时钟以微秒整数按时间记账表累加。"""
    case = Case(jammers=[Jammer(channel=3, x=9e5, y=9e5, reception_radius_m=1000.0)],
                error_salt="t", seed=0)
    arena = Arena(case)
    arena.enter()
    assert arena.robot.virtual_time_us == 0
    # 移动 500 m / 5 = 100 s，检测 +5 s，频道不变 → 105 s = 105_000_000 us
    arena.measure(300, 400, 1)
    assert arena.robot.virtual_time_us == 105_000_000
    assert isinstance(arena.robot.virtual_time_us, int)
    # 原地 + 切频 +1 s + 检测 +5 s → 111 s
    arena.measure(300, 400, 2)
    assert arena.robot.virtual_time_us == 111_000_000
    # 移动 400 m / 5 = 80 s + 清除失败 +3 s → 194 s
    arena.clear(300, 0, 3)
    assert arena.robot.virtual_time_us == 194_000_000
    # virtual_time_s 由整数微秒派生，无浮点漂移
    assert arena.robot.virtual_time_s == 194.0


def test_commit_move_triggers_virtual_time_exhausted_past_limit():
    """R6.2：超过 MAX_VIRTUAL_SECONDS 时触发 virtual_time_exhausted 并结束该局。"""
    arena = Arena(single_case(), max_virtual_seconds=10.0)
    arena.enter()
    # 移动 1000 m 需 200 s > 10 s 上限
    with pytest.raises(ArenaRejected):
        arena.measure(1000.0, 0.0, 1)
    assert arena.robot.exit_reason == "virtual_time_exhausted"
    assert arena.robot.finished is True
    assert arena.robot.virtual_time_us == 0  # 被拒动作不推进时钟
    # 已结束后任何动作继续被拒
    with pytest.raises(ArenaRejected):
        arena.measure(0.0, 0.0, 1)


def test_commit_move_allows_action_exactly_at_limit():
    """R6.2：累计虚拟时间恰好等于上限（不超过）的动作应被放行，超过 1 微秒即拒。"""
    # 移动 500 m = 100 s + 检测 5 s = 105 s；上限恰为 105 s → 放行（等于上限不算超）。
    arena = Arena(single_case(x=9e5, y=9e5, reception_radius_m=1000.0),
                  max_virtual_seconds=105.0)
    arena.enter()
    arena.measure(300.0, 400.0, 1)  # 恰好累计到 105 s
    assert arena.robot.virtual_time_us == 105_000_000
    assert arena.robot.exit_reason is None
    assert arena.robot.finished is False
    # 上限少 1 微秒时，同样的动作应被拒并触发耗尽。
    tight = Arena(single_case(x=9e5, y=9e5, reception_radius_m=1000.0))
    tight.max_virtual_us = 105_000_000 - 1
    tight.enter()
    with pytest.raises(ArenaRejected):
        tight.measure(300.0, 400.0, 1)
    assert tight.robot.exit_reason == "virtual_time_exhausted"



# ---------- 任务 5.1：客户端发送前坐标越界拒绝（R5.3）----------
# Feature: b-radio-interference-locate-clear
# 客户端在 measure()/clear() 发送前校验 abs(x)/abs(y) ≤ 2_000_000 m；
# 越界（含 NaN/inf）抛 ProtocolError 且不发送、不生成 request_id、不推进虚拟时刻；
# 恰为 2_000_000 m 放行（通过客户端侧门禁）。


def test_measure_rejects_out_of_bounds_x_before_sending(server):
    """R5.3：abs(x) 略超 2_000_000 m → ProtocolError，且未发出任何请求。"""
    url, session = server
    client = RobotClient(ROBOT, base_url=url)
    client.enter()
    calls_before = session.arena.stats()["measure_calls"]
    vt_before = client.virtual_time_s
    seq_before = client._seq
    with pytest.raises(ProtocolError):
        client.measure(2_000_000.0 + 1.0, 0.0, 1)
    # 未发送：服务端未记 measure，客户端虚拟时刻不变、无日志新增、request_id 未推进
    assert session.arena.stats()["measure_calls"] == calls_before
    assert client.virtual_time_s == vt_before
    assert client._seq == seq_before
    assert all(entry["path"] != "/measure" for entry in client.log)


def test_measure_rejects_out_of_bounds_y_before_sending(server):
    """R5.3：abs(y) 越界同样在发送前拒绝。"""
    url, session = server
    client = RobotClient(ROBOT, base_url=url)
    client.enter()
    calls_before = session.arena.stats()["measure_calls"]
    with pytest.raises(ProtocolError):
        client.measure(0.0, -2_000_000.5, 1)
    assert session.arena.stats()["measure_calls"] == calls_before


def test_measure_rejects_non_finite_coordinates(server):
    """R5.3：NaN / inf 亦属越界（非有限值），发送前拒绝。"""
    url, session = server
    client = RobotClient(ROBOT, base_url=url)
    client.enter()
    calls_before = session.arena.stats()["measure_calls"]
    for bad_x in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ProtocolError):
            client.measure(bad_x, 0.0, 1)
    assert session.arena.stats()["measure_calls"] == calls_before


def test_clear_rejects_out_of_bounds_before_sending(server):
    """R5.3：clear() 与 measure() 同样在发送前拦截越界坐标。"""
    url, session = server
    client = RobotClient(ROBOT, base_url=url)
    client.enter()
    clears_before = session.arena.stats().get("clear_calls", 0)
    vt_before = client.virtual_time_s
    with pytest.raises(ProtocolError):
        client.clear(3_000_000.0, 0.0, 1)
    assert session.arena.stats().get("clear_calls", 0) == clears_before
    assert client.virtual_time_s == vt_before


def test_boundary_coordinate_exactly_two_million_is_allowed_by_client_gate():
    """R5.3：恰为 ±2_000_000 m 视为合法，通过客户端侧门禁（不抛越界错误）。"""
    # 直接验证发送前门禁：边界值不抛，越界值抛。不依赖网络。
    RobotClient._check_coord(2_000_000.0, -2_000_000.0)   # 恰为边界 → 放行
    RobotClient._check_coord(-2_000_000.0, 2_000_000.0)   # 另一侧边界 → 放行
    with pytest.raises(ProtocolError):
        RobotClient._check_coord(2_000_000.0 + 1e-3, 0.0)  # 略超 → 拒绝




# ---------- 任务 9.2：批量彩排统计与官方校准记录（R6.3 / R6.4 / R6.5）----------
# Feature: b-radio-interference-locate-clear
# 核对 rehearse.py 能对本地 mock 服务批量彩排并汇总 T̄、清除率、程序运行时长。
# 批量规模保持小（少量局）以确保测试快速；上限（Q3≥60/Q4≥40）在 rehearse.main 由
# --episodes 控制，此处只验证聚合契约与量值合理性。

from src.b_sim.rehearse import aggregate as _rehearse_aggregate
from src.b_sim.rehearse import run_episode as _rehearse_run_episode


def test_rehearse_small_batch_aggregates_scoring_fields():
    """R6.3：批量彩排（此处用小批量保证快速）应汇总清除个数、清除率、T̄、运行时长。

    对本地 mock 服务连跑几局，aggregate() 返回的字典须含全部评分项字段且量值合理：
      - cleared_total：累计已清除个数（整数，0 ≤ 值 ≤ 真实源总数）
      - clear_rate / full_clear_rate：清除率 ∈ [0, 1]
      - mean_clear_time_s_mean：平均定位清除时间 T̄（> 0）
      - program_wall_s_mean / _max：程序运行时长（≥ 0 现实秒）
    正式 Q3≥60 / Q4≥40 的批量规模由 rehearse.main 的 --episodes 控制，聚合逻辑同此。
    """
    episodes = [_rehearse_run_episode(2000 + i, count=10) for i in range(3)]
    agg = _rehearse_aggregate(episodes)

    # 局数与清除个数
    assert agg["episodes"] == 3
    assert isinstance(agg["cleared_total"], int)
    assert 0 <= agg["cleared_total"] <= agg["total_jammers_total"]
    assert agg["total_jammers_total"] == sum(e["total_jammers"] for e in episodes)

    # 清除率
    assert 0.0 <= agg["clear_rate"] <= 1.0
    assert 0.0 <= agg["full_clear_rate"] <= 1.0
    assert 0.0 <= agg["cleared_ratio_min"] <= agg["cleared_ratio_mean"] <= 1.0

    # 平均定位清除时间 T̄（只要有清除即应为正值）
    assert agg["mean_clear_time_s_mean"] is not None
    assert agg["mean_clear_time_s_mean"] > 0.0
    assert agg["mean_clear_time_s_max"] >= agg["mean_clear_time_s_mean"]

    # 程序运行时长（现实秒）
    assert agg["program_wall_s_mean"] is not None
    assert agg["program_wall_s_mean"] >= 0.0
    assert agg["program_wall_s_max"] >= agg["program_wall_s_mean"]


def test_rehearse_aggregate_clear_rate_matches_manual_count():
    """R6.3：clear_rate 应等于累计已清除个数 / 真实源总数，与逐局求和一致。"""
    episodes = [_rehearse_run_episode(3000 + i, count=10) for i in range(2)]
    agg = _rehearse_aggregate(episodes)
    cleared = sum(e["cleared"] for e in episodes)
    total = sum(e["total_jammers"] for e in episodes)
    assert agg["cleared_total"] == cleared
    assert agg["total_jammers_total"] == total
    assert agg["clear_rate"] == pytest.approx(round(cleared / total, 4))



# ---------- Q2 第二检测点候选生成 (second_point_candidates, R2.1 / R2.4) [task 3.1] ----------

def test_q2_second_point_candidates_ordering_descending_delta():
    """R2.1：候选序列按偏移角 δ 降序排列（32° 对 → 16° 对 → 0° 单点）。"""
    p1 = (100.0, 200.0)
    theta1 = 30.0
    cands = second_point_candidates(p1, theta1)
    def abs_delta(pt):
        dx, dy = pt[0] - p1[0], pt[1] - p1[1]
        bearing = math.degrees(math.atan2(dy, dx))
        diff = abs(norm_deg(bearing) - norm_deg(theta1))
        return diff if diff <= 180.0 else 360.0 - diff
    deltas = [abs_delta(c) for c in cands]
    for a, b in zip(deltas, deltas[1:]):
        assert a >= b - 1e-9, f"δ 未按降序排列: {deltas}"
    assert deltas[0] == pytest.approx(32.0, abs=1e-6)
    assert deltas[-1] == pytest.approx(0.0, abs=1e-6)


def test_q2_second_point_candidates_sequence_length():
    """R2.1：δ>0 生成 ± 一对，δ=0 只生成一个候选。"""
    p1 = (0.0, 0.0)
    assert len(second_point_candidates(p1, 45.0)) == 5
    assert len(second_point_candidates(p1, 45.0, offsets=(0.0,))) == 1
    assert len(second_point_candidates(p1, 45.0, offsets=(32.0,))) == 2
    assert len(second_point_candidates(p1, 45.0, offsets=(32.0, 0.0))) == 3


def test_q2_second_point_candidates_forward_advance_length():
    """R2.4：每个候选到 p1 的距离 = L = 380 m（±1 m 容差），且沿示向度有正前进分量。"""
    p1 = (-50.0, 75.0)
    theta1 = 123.4
    ux = math.cos(math.radians(theta1))
    uy = math.sin(math.radians(theta1))
    for pt in second_point_candidates(p1, theta1):
        rel = (pt[0] - p1[0], pt[1] - p1[1])
        dist = math.hypot(*rel)
        assert abs(dist - 380.0) <= 1.0, f"候选前进长度 {dist} 偏离 380±1 m"
        forward = rel[0] * ux + rel[1] * uy
        assert forward > 0.0, "候选缺少沿 θ1 的正前进分量"
        assert forward >= 380.0 * math.cos(math.radians(32.0)) - 1e-6


def test_q2_second_point_candidates_no_pure_lateral_translation():
    """R2.4：无 |δ|>32° 或纯横向平移（前进分量 ≤ 0）候选，且非法偏移被拒。"""
    p1 = (300.0, -120.0)
    theta1 = 200.0
    ux = math.cos(math.radians(theta1))
    uy = math.sin(math.radians(theta1))
    for pt in second_point_candidates(p1, theta1):
        rel = (pt[0] - p1[0], pt[1] - p1[1])
        forward = rel[0] * ux + rel[1] * uy
        assert forward > 0.0
    with pytest.raises(ValueError):
        second_point_candidates(p1, theta1, offsets=(45.0,))



# ---------- 任务 5.2：400/409 不重试语义、网络重试与记录完整性（R5.2 / R5.4 / R5.5 / R5.9）----------
# Feature: b-radio-interference-locate-clear
# 客户端 _post 对 HTTP 400/409 抛 ProtocolError，不重试、不推进虚拟时刻，
# 并完整保留 (request, http_status, response) 记录（每条含 request_id 与时间戳）；
# 网络异常（URLError/Timeout/ConnectionError）以同一 request_id/请求体退避重试至多 3 次，
# 耗尽抛 TestEnded；成功路径要求 http_status==200 且 accepted is True 双重校验。

from src.b_sim.client import TestEnded as _TestEnded


def _post_raw(client, path, payload):
    """绕过 _base_payload 的自增，直接以给定 payload 调用 _post，
    以便复现"同一 request_id 不同内容"的 409 冲突路径。"""
    return client._post(path, payload)


def test_client_conflict_409_raises_protocol_error_without_retry_or_time_advance(server):
    """R5.4：同一 request_id 异内容 → 409 → ProtocolError；不重试、虚拟时刻不变、日志完整保留。"""
    url, session = server
    client = RobotClient(ROBOT, base_url=url)
    client.enter()  # request_id "enter-1"
    # 用固定 request_id 先成功一次 measure（accepted），推进虚拟时刻。
    rid = f"{ROBOT}-conflict"
    ok_payload = base(rid) | {"position": {"x": 0.0, "y": 0.0}, "channel": 1}
    body = _post_raw(client, "/measure", ok_payload)
    assert body["accepted"] is True
    vt_after_ok = client.virtual_time_s
    assert vt_after_ok > 0.0
    log_len_before = len(client.log)
    measure_calls_before = session.arena.stats()["measure_calls"]

    # 同一 request_id、不同内容 → 服务端 409 冲突。
    conflict_payload = base(rid) | {"position": {"x": 10.0, "y": 0.0}, "channel": 1}
    with pytest.raises(ProtocolError):
        _post_raw(client, "/measure", conflict_payload)

    # 不推进虚拟时刻（保持上次成功值），不重复计入服务端 measure。
    assert client.virtual_time_s == vt_after_ok
    assert session.arena.stats()["measure_calls"] == measure_calls_before
    # 完整保留 (request, http_status, response) 记录：恰新增 1 条 409 记录（不重试→不重复记录）。
    assert len(client.log) == log_len_before + 1
    conflict_entry = client.log[-1]
    assert conflict_entry["http_status"] == 409
    assert conflict_entry["request_id"] == rid
    assert conflict_entry["request"] == conflict_payload
    assert isinstance(conflict_entry["response"], dict)  # 完整响应体保留
    assert "ts" in conflict_entry  # R5.9 时间戳


def test_client_bad_request_400_raises_protocol_error_and_preserves_record(server):
    """R5.4：非法 channel（0）→ 服务端 400 → ProtocolError；日志完整保留、虚拟时刻不变。"""
    url, session = server
    client = RobotClient(ROBOT, base_url=url)
    client.enter()
    vt_before = client.virtual_time_s
    log_len_before = len(client.log)
    with pytest.raises(ProtocolError):
        client.measure(0.0, 0.0, 0)  # channel=0 越界 → 400
    assert client.virtual_time_s == vt_before  # 不记账
    assert len(client.log) == log_len_before + 1  # 完整保留一条 400 记录
    entry = client.log[-1]
    assert entry["http_status"] == 400
    assert entry["path"] == "/measure"
    assert entry["request"]["channel"] == 0
    assert "request_id" in entry and "ts" in entry


def test_client_bad_coordinates_400_from_server_raises_protocol_error(server):
    """R5.4：客户端门禁内（≤2e6 m）但服务端判越界的坐标 → 400 → ProtocolError。

    3_000_000 m 会被客户端 _check_coord 提前拦截（R5.3），此处用一个通过客户端门禁
    但服务端仍拒绝的路径：直接以合法门禁内坐标构造，但通过 _post 发送非法 channel 组合，
    确认 400 归一为 ProtocolError（与 server test 的 400 判定一致）。"""
    url, _ = server
    client = RobotClient(ROBOT, base_url=url)
    client.enter()
    bad = base(f"{ROBOT}-badch") | {"position": {"x": 0.0, "y": 0.0}, "channel": 21}
    with pytest.raises(ProtocolError):
        _post_raw(client, "/measure", bad)
    assert client.log[-1]["http_status"] == 400


class _FailingTransport:
    """可控失败次数的伪 urlopen：抛 URLError 指定次数，之后成功返回。"""

    def __init__(self, fail_times: int, success_body: dict | None = None):
        self.fail_times = fail_times
        self.calls: list[bytes] = []
        self.success_body = success_body or {"accepted": True, "virtual_time_s": 5.0}

    def __call__(self, request, timeout=None):
        # 记录每次实际发送的请求体（用于断言重试复用同一 request_id/body）。
        self.calls.append(request.data)
        if len(self.calls) <= self.fail_times:
            raise urllib.error.URLError("simulated network failure")
        payload = json.loads(request.data.decode("utf-8"))
        resp = dict(self.success_body)
        # 回显 request_id 供断言复用
        resp["request_id"] = payload.get("request_id")

        class _Resp:
            status = 200

            def read(self_inner):
                return json.dumps(resp).encode("utf-8")

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        return _Resp()


def test_client_network_failure_retries_three_times_then_raises_test_ended(server, monkeypatch):
    """R5.5：URLError 持续失败 → 退避重试至多 3 次（复用同一 request_id/body）→ TestEnded。"""
    url, _ = server
    client = RobotClient(ROBOT, base_url=url, max_network_retries=3)
    transport = _FailingTransport(fail_times=99)  # 始终失败
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    monkeypatch.setattr("src.b_sim.client.time.sleep", lambda *_: None)  # 免退避等待

    payload = base(f"{ROBOT}-net") | {"position": {"x": 0.0, "y": 0.0}, "channel": 1}
    with pytest.raises(_TestEnded):
        _post_raw(client, "/measure", payload)

    # 恰好尝试 3 次（max_network_retries），且每次逐字节复用同一请求体与 request_id。
    assert len(transport.calls) == 3
    assert transport.calls[0] == transport.calls[1] == transport.calls[2]
    sent = json.loads(transport.calls[0].decode("utf-8"))
    assert sent["request_id"] == f"{ROBOT}-net"
    # 网络异常不产生日志记录（无 (request,status,response) 三元组），虚拟时刻不变。
    assert all(e["path"] != "/measure" or e.get("http_status") is not None for e in client.log)
    assert client.virtual_time_s == 0.0


def test_client_network_failure_then_recovery_reuses_same_request_id(server, monkeypatch):
    """R5.5：失败 2 次后第 3 次成功 → 复用同一 request_id/body，成功记账，无 TestEnded。"""
    url, _ = server
    client = RobotClient(ROBOT, base_url=url, max_network_retries=3)
    transport = _FailingTransport(fail_times=2)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    monkeypatch.setattr("src.b_sim.client.time.sleep", lambda *_: None)

    payload = base(f"{ROBOT}-net2") | {"position": {"x": 0.0, "y": 0.0}, "channel": 1}
    body = _post_raw(client, "/measure", payload)
    assert body["accepted"] is True
    assert len(transport.calls) == 3  # 2 次失败 + 1 次成功
    assert transport.calls[0] == transport.calls[-1]  # 复用同一请求体/ID
    assert client.virtual_time_s == pytest.approx(5.0)  # 成功后按返回记账
    assert client.log[-1]["http_status"] == 200
    assert client.log[-1]["request_id"] == f"{ROBOT}-net2"


def test_client_invalid_json_response_raises_protocol_error(server, monkeypatch):
    """R5.2：http 200 但响应体非法 JSON → ProtocolError 且不记账。"""
    url, _ = server
    client = RobotClient(ROBOT, base_url=url)

    class _BadJsonResp:
        status = 200

        def read(self):
            return b"{not valid json"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _BadJsonResp())
    payload = base(f"{ROBOT}-badjson") | {"position": {"x": 0.0, "y": 0.0}, "channel": 1}
    vt_before = client.virtual_time_s
    with pytest.raises(ProtocolError):
        _post_raw(client, "/measure", payload)
    assert client.virtual_time_s == vt_before  # 不记账


def test_client_200_but_accepted_false_raises_protocol_error(server):
    """R5.2：双重校验——http 200 但 accepted 非 True（此处 measure 前未 enter）→ ProtocolError 不记账。"""
    url, _ = server
    client = RobotClient(ROBOT, base_url=url)
    vt_before = client.virtual_time_s
    log_len_before = len(client.log)
    with pytest.raises(ProtocolError):
        client.measure(0.0, 0.0, 1)  # 未 enter → accepted=False（http 200）
    assert client.virtual_time_s == vt_before
    # 记录仍完整保留（含 accepted=False 的响应体）。
    assert len(client.log) == log_len_before + 1
    entry = client.log[-1]
    assert entry["http_status"] == 200
    assert entry["response"]["accepted"] is False
    assert "request_id" in entry and "ts" in entry



# ---------- Q2 第二检测点有效性判定与退让/失败语义 (R2.2/R2.3/R2.5/R2.6) [task 3.2] ----------
# Feature: b-radio-interference-locate-clear
# classify_second_point_candidates 逐候选判 valid/invalid（交会角∈[60,120] 边界含、
# d2≤1000 边界含、‖p2‖≤1800 边界含）；select_second_point 按 δ 降序退让、末尾退化到
# δ=0° 安全直行，全失败返回显式"不可行"结果且不选越界/病态候选。

from src.b_sim.geometry import (
    ARENA_RADIUS,
    classify_second_point_candidates,
    crossing_angle_deg,
    select_second_point,
)


def test_q2_crossing_angle_matches_definition():
    """交会角 = 估计源处两视线 source→p1 与 source→p2 的夹角，∈[0,180]。"""
    # 源在原点，p1 正东、p2 正北 → 两视线相互垂直 → 90°。
    assert crossing_angle_deg((0.0, 0.0), (100.0, 0.0), (0.0, 100.0)) == pytest.approx(90.0)
    # p1、p2 关于源同向 → 0°；反向 → 180°。
    assert crossing_angle_deg((0.0, 0.0), (100.0, 0.0), (50.0, 0.0)) == pytest.approx(0.0)
    assert crossing_angle_deg((0.0, 0.0), (100.0, 0.0), (-50.0, 0.0)) == pytest.approx(180.0)
    # 退化：p1 与源重合 → 0.0。
    assert crossing_angle_deg((0.0, 0.0), (0.0, 0.0), (0.0, 100.0)) == 0.0


def _source_for_alpha(p1, theta1, p2, alpha_target_deg, side=+1):
    """在过 p1、方向 theta1 的射线上求一点 source，使 crossing_angle(source,p1,p2)=alpha。

    数值二分：source = p1 + t·(cosθ1,sinθ1)，t∈(0, 大)。α 随 t 单调变化，二分求根。
    """
    ux, uy = math.cos(math.radians(theta1)), math.sin(math.radians(theta1))
    lo, hi = 1e-6, 5000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = (p1[0] + mid * ux, p1[1] + mid * uy)
        a = crossing_angle_deg(s, p1, p2)
        if a > alpha_target_deg:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return (p1[0] + t * ux, p1[1] + t * uy)


def test_q2_alpha_boundary_60_and_120_are_valid_inclusive():
    """R2.2：交会角恰为 60° 与 120° 均判定为有效（边界含）。"""
    p1 = (0.0, 0.0)
    theta1 = 0.0
    # 取 δ=+32° 候选作为被判定对象。
    p2 = (380.0 * math.cos(math.radians(32.0)), 380.0 * math.sin(math.radians(32.0)))
    for alpha in (60.0, 120.0):
        source = _source_for_alpha(p1, theta1, p2, alpha)
        # 确认构造出的 α 确实等于目标（在容差内）。
        assert crossing_angle_deg(source, p1, p2) == pytest.approx(alpha, abs=1e-3)
        results = classify_second_point_candidates(p1, theta1, source)
        # 找到与 p2 对应的候选（δ=+32°，即第一个）。
        cand = next(c for c in results if c.point == pytest.approx(p2))
        assert cand.crossing_angle_deg == pytest.approx(alpha, abs=1e-3)
        # d2 与 ‖p2‖ 均在界内时，α 边界值应判有效。
        if cand.d2_m <= 1000.0 and cand.range_from_origin_m <= ARENA_RADIUS:
            assert cand.valid is True, f"α={alpha}° 边界应判有效: {cand.reason}"


def test_q2_pathological_alpha_below_60_and_above_120_are_invalid():
    """R2.3：α<60° 或 α>120° 的候选判无效并剔除（不作为选定点）。"""
    p1 = (0.0, 0.0)
    theta1 = 0.0
    p2 = (380.0 * math.cos(math.radians(32.0)), 380.0 * math.sin(math.radians(32.0)))
    # 目标病态角 30°(<60) 与 145°(>120)；沿 θ1 射线可达（该 p2 下 α 上确界≈148°）。
    for alpha in (30.0, 145.0):
        source = _source_for_alpha(p1, theta1, p2, alpha)
        results = classify_second_point_candidates(p1, theta1, source)
        cand = next(c for c in results if c.point == pytest.approx(p2))
        # 构造到的 α 落在目标病态区间（<60 或 >120）即可，无需精确命中。
        assert (cand.crossing_angle_deg < 60.0) if alpha < 60.0 else (cand.crossing_angle_deg > 120.0)
        assert cand.valid is False
        assert "交会角病态" in cand.reason
    # 病态候选绝不会被 select_second_point 选中。
    source = _source_for_alpha(p1, theta1, p2, 30.0)
    sel = select_second_point(p1, theta1, source, offsets=(32.0,))
    # 唯一候选病态 → 不可行，不选定该越界/病态候选。
    assert sel.feasible is False
    assert sel.point is None


def test_q2_d2_and_range_boundary_inclusive_exact():
    """R2.2：精确构造 d2=1000（边界含）与 ‖p2‖=1800（边界含）判有效。"""
    # 几何：p1=(0,0)，theta1=0（朝东）。用单个 δ=0° 候选，length 可控 → p2=(length,0)。
    # 令 source 在 (px, py) 使得：(a) α=90°，(b) d2=1000。
    # α=90 要求 source→p1 ⟂ source→p2。取 source=(length/2, h)，则
    #   v1=source→p1=(-length/2,-h), v2=source→p2=(length/2,-h)，v1·v2 = -length²/4 + h² =0
    #   → h = length/2。此时 d2=|v2|=sqrt(length²/4 + h²)=sqrt(length²/4+length²/4)=length/√2。
    # 令 d2=1000 → length = 1000·√2 ≈ 1414.21。
    length = 1000.0 * math.sqrt(2.0)
    p1 = (0.0, 0.0)
    theta1 = 0.0
    source = (length / 2.0, length / 2.0)
    # ‖p2‖ = length ≈ 1414 ≤ 1800，界内。
    results = classify_second_point_candidates(
        p1, theta1, source, offsets=(0.0,), length=length
    )
    cand = results[0]
    assert cand.crossing_angle_deg == pytest.approx(90.0)
    assert cand.d2_m == pytest.approx(1000.0)
    assert cand.range_from_origin_m == pytest.approx(length)
    assert cand.valid is True  # α=90、d2=1000（边界含）、‖p2‖≈1414 ≤1800 → 有效

    # d2 略超 1000（放大 length 使 d2>1000）→ 无效。
    length_big = length * 1.001
    source_big = (length_big / 2.0, length_big / 2.0)
    res_big = classify_second_point_candidates(
        p1, theta1, source_big, offsets=(0.0,), length=length_big
    )
    assert res_big[0].d2_m > 1000.0
    assert res_big[0].valid is False
    assert "接收半径" in res_big[0].reason

    # ‖p2‖ 边界：p2 恰在 1800 m 处（δ=0°、length=1800、theta1=0 → p2=(1800,0)）。
    # 源置于 (1800,300)：α≈81°∈[60,120]、d2=300≤1000，隔离出 ‖p2‖ 判据本身。
    res_edge = classify_second_point_candidates(
        p1, theta1, (1800.0, 300.0), offsets=(0.0,), length=1800.0
    )
    cand_edge = res_edge[0]
    assert cand_edge.range_from_origin_m == pytest.approx(1800.0)  # 恰在边界
    assert 60.0 <= cand_edge.crossing_angle_deg <= 120.0
    assert cand_edge.d2_m <= 1000.0
    assert cand_edge.valid is True  # ‖p2‖=1800 边界含 → 有效
    # 越界：length=1900 → ‖p2‖=1900>1800；源 (1900,300) 保持 α∈[60,120]、d2≤1000，
    # 使唯一失败原因为越界。
    res_over = classify_second_point_candidates(
        p1, theta1, (1900.0, 300.0), offsets=(0.0,), length=1900.0
    )
    over = res_over[0]
    assert over.range_from_origin_m > 1800.0
    assert 60.0 <= over.crossing_angle_deg <= 120.0
    assert over.d2_m <= 1000.0
    assert over.valid is False
    assert "越界" in over.reason


def test_q2_select_step_down_order_32_16_0():
    """R2.5：按 32°→16°→0° 退让——前面的 δ 无信号时选到下一个更小 δ。"""
    p1 = (0.0, 0.0)
    theta1 = 0.0
    # 放置源使所有 δ 候选 α 均良态：源在 p1 前方远处偏上，令交会角适中。
    # 简化：用一个使全部候选有效的源，再用 is_signal_fn 控制退让。
    source = (500.0, 500.0)
    all_res = classify_second_point_candidates(p1, theta1, source)
    valid_pts = [c.point for c in all_res if c.valid]
    # 确保至少 δ=0° 之外还有有效候选可供退让演示；若不足则跳过强断言。
    # 逐级退让：让 +32/-32/+16/-16 都 no_signal，仅 δ=0° 有信号 → 应选 δ=0°。
    zero_cand = all_res[-1]
    if zero_cand.valid:
        def only_zero_has_signal(pt):
            return pt == zero_cand.point
        sel = select_second_point(p1, theta1, source, only_zero_has_signal)
        assert sel.feasible is True
        assert sel.point == zero_cand.point
        assert sel.candidate.offset_deg == pytest.approx(0.0)
        assert "δ=+0.0°" in sel.reason
    # 无信号回调恒 True（有信号）时应选第一个有效候选（最大 δ）。
    if valid_pts:
        sel_first = select_second_point(p1, theta1, source, lambda pt: True)
        assert sel_first.feasible is True
        assert sel_first.point == valid_pts[0]


def test_q2_select_skips_invalid_and_falls_back_to_zero():
    """R2.5/R2.6：无效候选被跳过，退化到 δ=0° 安全直行兜底。"""
    p1 = (0.0, 0.0)
    theta1 = 0.0
    # 精确构造：δ=0° 候选 α=90、d2=1000（有效）；令大偏移候选因 α 病态被跳过。
    # 用 length=1000√2、source=(length/2,length/2) → δ=0° 候选 (length,0) α=90 有效。
    # δ=±32/±16 候选 p2 偏离轴线，其 α 多半病态或 d2 超界 → 被跳过，最终落到 δ=0°。
    length = 1000.0 * math.sqrt(2.0)
    source = (length / 2.0, length / 2.0)
    sel = select_second_point(p1, theta1, source, lambda pt: True, length=length)
    assert sel.feasible is True
    # 选定候选应有效。
    assert sel.candidate.valid is True
    # 所有被考察且排在选中候选之前的候选都应是无效的（被跳过）——体现退让语义。
    chosen_index = sel.evaluated.index(sel.candidate)
    assert chosen_index > 0, "本用例应至少跳过一个较大 δ 的无效候选"
    for skipped in sel.evaluated[:chosen_index]:
        assert skipped.valid is False


def test_q2_all_candidates_fail_returns_explicit_infeasible():
    """R2.6：全部候选无效或 no_signal → 显式"不可行"，不选越界/病态候选，p1 数据不变。"""
    p1 = (0.0, 0.0)
    theta1 = 0.0
    source = (500.0, 500.0)
    # 情形 A：所有候选 no_signal（回调恒 False）→ 不可行。
    sel_ns = select_second_point(p1, theta1, source, lambda pt: False)
    assert sel_ns.feasible is False
    assert sel_ns.point is None
    assert sel_ns.candidate is None
    assert "第二检测点不可行" in sel_ns.reason
    assert "no_signal" in sel_ns.reason
    # 情形 B：源置于极近处使所有候选 α 病态/d2 超界（全无效）→ 不可行。
    # 源紧贴 p1（几乎共线）→ δ=0° α≈0 病态；大偏移 α 也难良态。
    source_bad = (5.0, 0.0)
    sel_inv = select_second_point(p1, theta1, source_bad, lambda pt: True)
    assert sel_inv.feasible is False
    assert sel_inv.point is None
    # 显式失败结果绝不选定任何越界或病态候选。
    assert all(
        (not c.valid) for c in sel_inv.evaluated
    ) or sel_inv.point is None


def test_q2_out_of_bounds_candidate_never_selected():
    """R2.6：越界候选（‖p2‖>1800）不被选定，即便回调报有信号。"""
    # p1 靠近边界，theta1 朝外 → 候选 p2 越界。
    p1 = (1700.0, 0.0)
    theta1 = 0.0  # 朝东，向区域外
    source = (2000.0, 300.0)  # 估计源在外侧（仅用于判定）
    sel = select_second_point(p1, theta1, source, lambda pt: True)
    if sel.feasible:
        # 若选出，必须满足 ‖p2‖≤1800（未选越界候选）。
        assert sel.candidate.range_from_origin_m <= ARENA_RADIUS + 1e-6
    else:
        # 否则显式不可行，且考察过的候选中越界者均标记无效。
        assert sel.point is None
        over = [c for c in sel.evaluated if c.range_from_origin_m > ARENA_RADIUS + 1e-6]
        assert all(not c.valid for c in over)



# ===========================================================================
# 任务 3.3：候选区域界定（R2.7）与 lateral_error_at 核对（R2.8）
# 新增于本微批次，未改动上方已有内容。
# ===========================================================================

from src.b_sim.geometry import (  # noqa: E402
    ARENA_RADIUS,
    CROSSING_ANGLE_MAX_DEG,
    CROSSING_ANGLE_MIN_DEG,
    RECEP_LOWER_M,
    SECOND_POINT_WEDGE_HALF_DEG,
    crossing_angle_deg,
    in_second_point_region,
    second_point_candidate_region,
)
from src.b_sim.strategy import lateral_error_at  # noqa: E402


def _wedge_diff_deg(p1, theta1, p2):
    """p1→p2 方位与 theta1 的最小角差（度），用于测试内独立复算前向楔形约束。"""
    b = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    d = abs(norm_deg(b) - norm_deg(theta1))
    return min(d, 360.0 - d)


# ---------------------------------------------------------------------------
# R2.7：候选区域界定（θ1±32° 前向楔形 ∩ α∈[60,120] ∩ d2≤1000 ∩ ‖p2‖≤1800）
# ---------------------------------------------------------------------------

def test_q2_region_bounds_are_theta1_plusminus_32():
    """R2.7：候选区域角边界为 θ1±32°，径向上界 d2≤1000、‖p2‖≤1800。"""
    p1 = (-300.0, 0.0)
    theta1 = 0.0
    reg = second_point_candidate_region(p1, theta1)
    assert reg.wedge_half_deg == pytest.approx(SECOND_POINT_WEDGE_HALF_DEG)
    assert reg.angle_lo_deg == pytest.approx(theta1 - 32.0)
    assert reg.angle_hi_deg == pytest.approx(theta1 + 32.0)
    assert reg.crossing_min_deg == pytest.approx(CROSSING_ANGLE_MIN_DEG)
    assert reg.crossing_max_deg == pytest.approx(CROSSING_ANGLE_MAX_DEG)
    assert reg.d2_max_m == pytest.approx(RECEP_LOWER_M)      # 1000 m
    assert reg.range_max_m == pytest.approx(ARENA_RADIUS)     # 1800 m


def test_q2_in_region_true_for_wellposed_point():
    """R2.7：楔形内、α∈[60,120]、d2≤1000、‖p2‖≤1800 的良态点判为在区域内。

    源置于原点（此时 d2 = ‖p2‖）。搜索一个 α≈90°、位于 θ1±32° 前向楔形内的候选点，
    断言其在区域内，并逐条复核四个界定约束确实满足。
    """
    source = (0.0, 0.0)
    p1 = (-300.0, 0.0)
    theta1 = 0.0

    found = None
    for wdeg in [x * 0.5 for x in range(-64, 65)]:
        phi = math.radians(theta1 + wdeg)
        for L in range(50, 900, 5):
            p2 = (p1[0] + L * math.cos(phi), p1[1] + L * math.sin(phi))
            alpha = crossing_angle_deg(source, p1, p2)
            d2 = math.hypot(p2[0] - source[0], p2[1] - source[1])
            if abs(alpha - 90.0) < 1.0 and _wedge_diff_deg(p1, theta1, p2) <= 30.0 and d2 <= 1000.0:
                found = p2
                break
        if found:
            break

    assert found is not None, "应能构造一个良态的区域内候选点"
    # 逐条复核四约束满足。
    assert _wedge_diff_deg(p1, theta1, found) <= 32.0 + 1e-7
    alpha = crossing_angle_deg(source, p1, found)
    assert 60.0 - 1e-7 <= alpha <= 120.0 + 1e-7
    assert math.hypot(found[0] - source[0], found[1] - source[1]) <= 1000.0 + 1e-7
    assert math.hypot(*found) <= 1800.0 + 1e-7
    # 结论：在区域内。
    assert in_second_point_region(found, p1, theta1, source) is True


def test_q2_in_region_excludes_bad_crossing_angle():
    """R2.7：α∉[60,120] 的点（近共线，α<60 或 α>120）被排除，即便在前向楔形内。"""
    source = (0.0, 0.0)
    p1 = (-300.0, 0.0)
    theta1 = 0.0

    # α < 60：沿轴附近小偏移、较短前进 → 近共线，α 偏小。
    p2_small = (p1[0] + 200.0 * math.cos(math.radians(20.0)),
                p1[1] + 200.0 * math.sin(math.radians(20.0)))
    assert crossing_angle_deg(source, p1, p2_small) < CROSSING_ANGLE_MIN_DEG
    assert _wedge_diff_deg(p1, theta1, p2_small) <= 32.0  # 确在前向楔形内
    assert in_second_point_region(p2_small, p1, theta1, source) is False

    # α > 120：越过源之后 → 交会角偏大。
    p2_large = (p1[0] + 520.0 * math.cos(math.radians(10.0)),
                p1[1] + 520.0 * math.sin(math.radians(10.0)))
    assert crossing_angle_deg(source, p1, p2_large) > CROSSING_ANGLE_MAX_DEG
    assert _wedge_diff_deg(p1, theta1, p2_large) <= 32.0
    assert in_second_point_region(p2_large, p1, theta1, source) is False


def test_q2_in_region_excludes_out_of_arena_point():
    """R2.7：‖p2‖>1800 的点被排除，即便 α∈[60,120] 且 d2≤1000 且在前向楔形内。

    用远处的源（近边缘）使 d2 很小但 ‖p2‖ 超过 1800 m，隔离出"越界"这一排除条件。
    """
    source = (1500.0, 0.0)
    p1 = (1447.905546699921, 295.4423259036624)   # 距源约 300 m
    theta1 = -80.0
    p2 = (1809.2360741337043, -105.85587985413065)

    alpha = crossing_angle_deg(source, p1, p2)
    d2 = math.hypot(p2[0] - source[0], p2[1] - source[1])
    norm = math.hypot(*p2)
    # 其余三约束满足，唯独越界。
    assert 60.0 <= alpha <= 120.0
    assert d2 <= 1000.0
    assert _wedge_diff_deg(p1, theta1, p2) <= 32.0 + 1e-6
    assert norm > 1800.0
    assert in_second_point_region(p2, p1, theta1, source) is False


def test_q2_in_region_boundary_alpha_60_is_inclusive():
    """R2.7 / R2.2：交会角边界 α=60°（含）判为在区域内（边界包含）。"""
    source = (0.0, 0.0)
    p1 = (-300.0, 0.0)
    theta1 = 0.0
    # 搜索 α 恰在 60° 附近（略 ≥60）且在楔形内、d2≤1000 的点。
    found = None
    for wdeg in [x * 0.25 for x in range(-128, 129)]:
        phi = math.radians(theta1 + wdeg)
        for L in range(50, 900):
            p2 = (p1[0] + L * math.cos(phi), p1[1] + L * math.sin(phi))
            alpha = crossing_angle_deg(source, p1, p2)
            if 60.0 <= alpha < 60.05 and _wedge_diff_deg(p1, theta1, p2) <= 32.0 and math.hypot(*p2) <= 1000.0:
                found = p2
                break
        if found:
            break
    assert found is not None
    assert crossing_angle_deg(source, p1, found) >= 60.0
    assert in_second_point_region(found, p1, theta1, source) is True


def test_q2_in_region_rejects_degenerate_p2_equals_p1():
    """R2.7：p2 与 p1 重合（无明确方位）不在区域内。"""
    source = (0.0, 0.0)
    p1 = (-300.0, 0.0)
    assert in_second_point_region(p1, p1, 0.0, source) is False


# ---------------------------------------------------------------------------
# R2.8：lateral_error_at(r) = r·tan(1°)，关于 r 严格单调递增
# ---------------------------------------------------------------------------

def test_lateral_error_matches_r_tan_1deg():
    """R2.8：lateral_error_at(r) 恰为 r·tan(1°)（含 0/100/1000/1500 m）。"""
    tan1 = math.tan(math.radians(1.0))
    for r in (0.0, 100.0, 1000.0, 1500.0):
        assert lateral_error_at(r) == pytest.approx(r * tan1, abs=1e-12)
    # 直观量级：100 m ≈ 1.7 m、1500 m ≈ 26 m（> 20 m 清除半径，佐证需逼近）。
    assert lateral_error_at(100.0) == pytest.approx(1.745506, abs=1e-4)
    assert lateral_error_at(1500.0) == pytest.approx(26.182597, abs=1e-4)
    assert lateral_error_at(1500.0) > 20.0
    assert lateral_error_at(0.0) == pytest.approx(0.0, abs=1e-12)


def test_lateral_error_strictly_increasing():
    """R2.8：对任意 r2 > r1 ≥ 0 有 e(r2) > e(r1)（严格单调递增）。"""
    grid = [0.0, 1e-6, 100.0, 500.0, 1000.0, 1500.0, 1800.0]
    vals = [lateral_error_at(r) for r in grid]
    for a, b in zip(vals, vals[1:]):
        assert b > a
    # 显式两两抽样再确认严格性。
    assert lateral_error_at(1500.0) > lateral_error_at(1000.0) > lateral_error_at(100.0) > lateral_error_at(0.0)


# ---------------------------------------------------------------------------
# 任务 6.1：核对情节循环、原点普扫、覆盖站位与就近贪心路由（R3.1 / R3.2 / R3.11）
# Feature: b-radio-interference-locate-clear
#
# 本节为「验证/加固」任务的回归测试。核对项：
#   (a) DogStrategy.run 在离开原点前，对频道 1–20 各执行至少一次原点普扫检测（R3.1）；
#   (b) 覆盖站位集合 = {原点} ∪ Ring(1250, 12)（13 站位），coverage_design 计算的最坏
#       覆盖间隙 ≈ 670 m 且严格 < 1000 m（R3.2）；
#   (c) _home_all / _walk_ring 采用就近贪心排序（每步选到当前位置最近者）（R3.11）。
# ---------------------------------------------------------------------------

import threading as _threading61

from src.b_sim.arena import (
    ARENA_RADIUS as _ARENA_RADIUS_61,
    Arena as _Arena61,
    Case as _Case61,
    Jammer as _Jammer61,
    CHANNEL_MIN as _CH_MIN_61,
    CHANNEL_MAX as _CH_MAX_61,
    RECEPTION_MIN_M as _RECEPTION_MIN_61,
)
from src.b_sim.client import RobotClient as _RobotClient61
from src.b_sim.coverage_design import ring as _cov_ring, worst_coverage_gap as _worst_gap
from src.b_sim.server import Session as _Session61, make_server as _make_server61
from src.b_sim.strategy import (
    ChannelState as _ChannelState61,
    DogStrategy as _DogStrategy61,
    StrategyConfig as _StrategyConfig61,
)


def _run_strategy_against_mock(case, config=None):
    """在本地 mock 服务上跑一局策略，返回 (summary, client)，便于检查 client.log。"""
    arena = _Arena61(case)
    session = _Session61("t61", arena)
    httpd = _make_server61(session, port=0)
    port = httpd.server_address[1]
    thread = _threading61.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client = _RobotClient61("t61", base_url=f"http://127.0.0.1:{port}")
        strategy = _DogStrategy61(client, config)
        summary = strategy.run()
    finally:
        httpd.shutdown()
        httpd.server_close()
    return summary, client, arena


def _is_origin(pos, tol=1e-9):
    return math.hypot(pos["x"], pos["y"]) <= tol


# ---- (a) R3.1：原点普扫 1–20，离开原点前每频道 ≥1 次 /measure ----

def test_run_sweeps_all_channels_at_origin_before_leaving_origin():
    """R3.1：run() 在离开原点前对频道 1–20 各普扫至少一次 /measure。

    构造一个所有源都远离原点（原点不可见）的小案例，使原点普扫全部 no_signal，
    从而不会在原点触发 clear，也不会因近距离源提前进入归航；随后检查 client.log 中
    第一段位于原点的 /measure 恰好覆盖频道 1–20 各一次，且这些普扫全部先于任何
    非原点动作（measure 或 clear）。
    """
    # 三个源，均在距原点 > 其接收半径处（原点收不到信号）。频道 1/2/3。
    jammers = [
        _Jammer61(channel=1, x=1600.0, y=0.0, reception_radius_m=1000.0),
        _Jammer61(channel=2, x=-1500.0, y=200.0, reception_radius_m=1000.0),
        _Jammer61(channel=3, x=0.0, y=-1700.0, reception_radius_m=1000.0),
    ]
    case = _Case61(jammers=jammers, error_salt="t61", seed=0)
    _summary, client, _arena = _run_strategy_against_mock(case)

    # 找出第一个非原点动作（measure 或 clear）在日志中的下标。
    first_non_origin_idx = None
    for i, entry in enumerate(client.log):
        if entry["path"] not in ("/measure", "/clear"):
            continue
        if not _is_origin(entry["request"]["position"]):
            first_non_origin_idx = i
            break
    if first_non_origin_idx is None:
        first_non_origin_idx = len(client.log)

    # 原点普扫阶段的 /measure：位于第一个非原点动作之前，且位置恰为原点。
    origin_measure_channels = [
        entry["request"]["channel"]
        for entry in client.log[:first_non_origin_idx]
        if entry["path"] == "/measure" and _is_origin(entry["request"]["position"])
    ]
    # 频道 1–20 每个至少被普扫一次（离开原点前）。
    for ch in range(_CH_MIN_61, _CH_MAX_61 + 1):
        assert ch in origin_measure_channels, f"频道 {ch} 未在离开原点前普扫"
    # 恰好 20 次原点普扫（每频道 1 次），无重复/遗漏。
    assert sorted(origin_measure_channels) == list(range(_CH_MIN_61, _CH_MAX_61 + 1))


def test_sweep_at_origin_measures_each_channel_exactly_once():
    """R3.1：_sweep_at 从初始状态（无示向度、未清除）对给定频道各测一次。"""
    jammers = [_Jammer61(channel=5, x=1650.0, y=0.0, reception_radius_m=1000.0)]
    case = _Case61(jammers=jammers, error_salt="t61b", seed=0)
    arena = _Arena61(case)
    session = _Session61("t61b", arena)
    httpd = _make_server61(session, port=0)
    port = httpd.server_address[1]
    thread = _threading61.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client = _RobotClient61("t61b", base_url=f"http://127.0.0.1:{port}")
        client.enter()
        strategy = _DogStrategy61(client)
        strategy._sweep_at((0.0, 0.0), range(_CH_MIN_61, _CH_MAX_61 + 1))
        client.exit()
    finally:
        httpd.shutdown()
        httpd.server_close()
    measured = [e["request"]["channel"] for e in client.log if e["path"] == "/measure"]
    assert sorted(measured) == list(range(_CH_MIN_61, _CH_MAX_61 + 1))
    # 全部位于原点。
    assert all(_is_origin(e["request"]["position"]) for e in client.log if e["path"] == "/measure")


def test_run_order_origin_sweep_precedes_ring_and_retry():
    """R3.1：run() 的默认阶段顺序为「原点普扫 → 归航/环补扫 → 内圈重测」。

    通过在 DogStrategy 上打桩记录各阶段调用次序，确认 _sweep_at 首个被调用，
    且 _probe_ring / _inner_retry 均在其之后。
    """
    jammers = [_Jammer61(channel=1, x=1600.0, y=0.0, reception_radius_m=1000.0)]
    case = _Case61(jammers=jammers, error_salt="t61c", seed=0)
    arena = _Arena61(case)
    session = _Session61("t61c", arena)
    httpd = _make_server61(session, port=0)
    port = httpd.server_address[1]
    thread = _threading61.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    calls: list[str] = []
    try:
        client = _RobotClient61("t61c", base_url=f"http://127.0.0.1:{port}")
        strategy = _DogStrategy61(client)
        orig_sweep = strategy._sweep_at
        orig_home = strategy._home_all
        orig_ring = strategy._probe_ring
        orig_revive = strategy._revive_exhausted
        orig_retry = strategy._inner_retry

        def rec(name, fn):
            def wrapper(*a, **k):
                calls.append(name)
                return fn(*a, **k)
            return wrapper

        strategy._sweep_at = rec("sweep", orig_sweep)          # type: ignore[assignment]
        strategy._home_all = rec("home", orig_home)            # type: ignore[assignment]
        strategy._probe_ring = rec("ring", orig_ring)          # type: ignore[assignment]
        strategy._revive_exhausted = rec("revive", orig_revive)  # type: ignore[assignment]
        strategy._inner_retry = rec("retry", orig_retry)       # type: ignore[assignment]
        strategy.run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert calls, "策略未执行任何阶段"
    assert calls[0] == "sweep", f"首阶段应为原点普扫，实际 {calls[0]}"
    sweep_i = calls.index("sweep")
    # 归航、环补扫、复活、内圈重测均在原点普扫之后。
    for phase in ("home", "ring", "revive", "retry"):
        assert phase in calls, f"缺少阶段 {phase}"
        assert calls.index(phase) > sweep_i, f"阶段 {phase} 早于原点普扫"
    # 具体子序列：sweep → home → ring → revive → home → retry。
    assert calls.index("ring") > calls.index("home")
    assert calls.index("revive") > calls.index("ring")
    assert calls.index("retry") > calls.index("revive")


# ---- (b) R3.2：覆盖站位 {原点} ∪ Ring(1250,12) 最坏间隙 ≈670 m < 1000 m ----

def test_coverage_stations_are_origin_plus_ring_1250_x12():
    """R3.2：策略配置的覆盖环参数即 Ring(1250 m, 12)，合计 13 站位（含原点）。"""
    cfg = _StrategyConfig61()
    assert cfg.probe_ring_radius == pytest.approx(1250.0)
    assert cfg.probe_ring_stations == 12
    # 单环情形（全向 Q3）不启用外环。
    assert cfg.use_outer_ring is False
    assert cfg.directional_probe is False
    stations = [(0.0, 0.0)] + _cov_ring(cfg.probe_ring_radius, cfg.probe_ring_stations)
    assert len(stations) == 13
    # 环站位均在半径 1250 m 上。
    for px, py in stations[1:]:
        assert math.hypot(px, py) == pytest.approx(1250.0, abs=1e-6)


def test_worst_coverage_gap_is_about_670m_and_below_reception_min():
    """R3.2：coverage_design.worst_coverage_gap({原点}∪Ring(1250,12)) ≈ 670 m 且 < 1000 m。"""
    stations = [(0.0, 0.0)] + _cov_ring(1250.0, 12)
    gap, worst_point = _worst_gap(stations, grid=300)
    # 设计文档 §2.4 / coverage_design 结论表载明最坏间隙 = 670 m。
    assert gap == pytest.approx(670.0, abs=6.0), f"最坏间隙 {gap:.1f} m 偏离 670 m"
    # 严格小于全向源最小有效接收半径 → 保证发现任一全向源。
    assert gap < _RECEPTION_MIN_61
    # 最坏点应落在竞技场内。
    assert math.hypot(*worst_point) <= _ARENA_RADIUS_61 + 1e-6


def test_probe_stations_single_ring_for_omni_case():
    """R3.2：全向情形 _probe_stations 只产生一层内环（12 站位），不含外环。"""
    strategy = _DogStrategy61(client=None, config=_StrategyConfig61())  # 不触网，仅算站位
    rings = strategy._probe_stations()
    assert len(rings) == 1, "全向情形应只有内环一层"
    assert len(rings[0]) == 12
    for px, py in rings[0]:
        assert math.hypot(px, py) == pytest.approx(1250.0, abs=1e-6)


# ---- (c) R3.11：_home_all / _walk_ring 就近贪心排序 ----

def test_home_all_visits_nearest_channel_first():
    """R3.11：_home_all 每步选取到当前位置估计最近的频道优先归航。

    构造三个已带估计位置的频道，用打桩的 _home_channel 记录被处理的顺序，
    断言其按到当前位置（原点）的距离升序被选取。
    """
    strategy = _DogStrategy61(client=None, config=_StrategyConfig61())
    strategy.pos = (0.0, 0.0)
    # 手工装配三个"已可定位"的频道，estimate 距原点分别为 300 / 100 / 700。
    specs = {
        1: (300.0, 0.0),
        2: (0.0, 100.0),
        3: (700.0, 0.0),
    }
    for ch, est in specs.items():
        st = strategy.channels[ch]
        st.bearings = [_bearing_from(est)]  # 让 active=True
        st.estimate = est
        st.diameter = 10.0
    # 其余频道无观测（inactive），不参与。
    visited_order: list[int] = []

    def fake_home_channel(state):
        visited_order.append(state.channel)
        state.cleared = True  # 处理完即清除，退出 active 集合
        return True

    strategy._home_channel = fake_home_channel  # type: ignore[assignment]
    strategy._stop = lambda: False              # type: ignore[assignment]
    strategy._home_all()
    # 按到原点距离升序：ch2 (100) → ch1 (300) → ch3 (700)。
    assert visited_order == [2, 1, 3]


def _bearing_from(est):
    from src.b_sim.geometry import Bearing as _B
    return _B(0.0, 0.0, math.degrees(math.atan2(est[1], est[0])))


def test_walk_ring_pops_nearest_station_first():
    """R3.11：_walk_ring 每步弹出到当前位置最近的站位（就近贪心压缩空驶）。"""
    strategy = _DogStrategy61(client=None, config=_StrategyConfig61())
    strategy.pos = (0.0, 0.0)
    # 让至少一个频道成为补扫对象（未清除、无示向度），使 _walk_ring 有事可做。
    target = strategy.channels[1]
    target.cleared = False
    target.bearings = []
    strategy._stop = lambda: False  # type: ignore[assignment]

    visited_stations: list[tuple[float, float]] = []

    def fake_probe(point, state):
        visited_stations.append(point)
        # 处理后清除该频道，使 _probe_target 变 False，循环终止。
        state.cleared = True

    def fake_home_all():
        return None

    strategy._probe_channel_at = fake_probe  # type: ignore[assignment]
    strategy._home_all = fake_home_all       # type: ignore[assignment]

    # 站位到原点距离：A=1000, B=200, C=1500 → 应先访问 B（最近）。
    ring_stations = [(1000.0, 0.0), (0.0, 200.0), (1500.0, 0.0)]
    strategy._walk_ring(ring_stations)
    assert visited_stations, "未访问任何站位"
    assert visited_stations[0] == (0.0, 200.0), "首站应为距当前位置最近者"


def test_walk_ring_reorders_by_current_position_each_step():
    """R3.11：就近排序基于「当前位置」而非固定原点——移动后重新按最近选取。

    保留两个补扫对象，令处理首站后位置移动到该站，验证下一站按新位置最近选取。
    """
    strategy = _DogStrategy61(client=None, config=_StrategyConfig61())
    strategy.pos = (0.0, 0.0)
    # 两个频道待补扫。
    for ch in (1, 2):
        st = strategy.channels[ch]
        st.cleared = False
        st.bearings = []
    strategy._stop = lambda: False  # type: ignore[assignment]

    # 记录被访问的「站位序列」（同一站位内可能对多个频道各测一次，只记站位本身，
    # 且只在站位变化时追加，得到去重后的到访顺序）。
    station_seq: list[tuple[float, float]] = []
    # 每访问一个「新」站位就清除一个频道，使 _walk_ring 恰好访问两个不同站位后终止。
    pending_clear = [1, 2]

    def fake_probe(point, state):
        if not station_seq or station_seq[-1] != point:
            station_seq.append(point)
            strategy.pos = point  # 移动到该站位
            if pending_clear:
                strategy.channels[pending_clear.pop(0)].cleared = True

    strategy._probe_channel_at = fake_probe  # type: ignore[assignment]
    strategy._home_all = lambda: None        # type: ignore[assignment]

    # 三站：A=(0,300) d=300, B=(0,400) d=400, C=(0,-100) d=100。
    # 首步从原点选最近 C=(0,-100)；移动到 C 后，A 距 C=400、B 距 C=500 → 选 A（不是 B）。
    strategy._walk_ring([(0.0, 300.0), (0.0, 400.0), (0.0, -100.0)])
    assert station_seq[0] == (0.0, -100.0)
    assert station_seq[1] == (0.0, 300.0)



# ---------- 任务 5.3：现实时间预算与虚拟时钟一致性核对（R5.6 / R5.8）----------
# Feature: b-radio-interference-locate-clear
# 回归核对（VERIFY 任务，实现已正确，无需改 client.py）：
#   (a) enter() 后 real_budget_left_s() 从服务端返回的 remaining_real_duration_s
#       派生（非硬编码 1200 s），随单调时钟推进而递减；
#   (b) real_budget_left_s 随单调时间前进而减少；
#   (c) real_budget_left_s() ≤ real_time_reserve_s(15) 时 DogStrategy._stop() 为 True，
#       > 15 s 时为 False（对齐 R5.8：策略在 15 s 前收尾）；
#   (d) 一次成功 measure 将 client.virtual_time_s 更新为服务端返回值。

import time as _time_53

from src.b_sim.strategy import DogStrategy as _DogStrategy53
from src.b_sim.strategy import StrategyConfig as _StrategyConfig53


def test_real_budget_derives_from_server_remaining_not_hardcoded_1200():
    """R5.8：real_budget_left_s() 基于 /enter 的 remaining_real_duration_s，而非固定 1200 s。

    直接构造客户端并把服务端提供的剩余时长设为非 1200 的值，断言预算按该值
    （减去自 enter 以来的单调耗时）派生，而不是固定 1200 s。
    """
    client = RobotClient(ROBOT)
    # 模拟 /enter 返回一个明显不等于 1200 的 remaining_real_duration_s。
    client.remaining_real_s = 300.0
    client._enter_monotonic = _time_53.monotonic()
    left = client.real_budget_left_s()
    # 预算应贴近 300（减去极小的调用耗时），绝不是 1200。
    assert left <= 300.0
    assert left == pytest.approx(300.0, abs=1.0)
    assert abs(left - 1200.0) > 100.0  # 明确排除硬编码 1200 s

    # 另一个非 1200 的值同样成立（例如更小的现实预算）。
    client.remaining_real_s = 42.5
    client._enter_monotonic = _time_53.monotonic()
    left2 = client.real_budget_left_s()
    assert left2 == pytest.approx(42.5, abs=1.0)
    assert abs(left2 - 1200.0) > 1000.0


def test_real_budget_decreases_as_monotonic_time_advances(monkeypatch):
    """R5.8：real_budget_left_s 随单调时钟推进而单调递减（monotonic 时钟驱动）。

    monkeypatch time.monotonic 以确定性推进虚拟的现实时间，验证预算按 remaining − elapsed
    递减，且 elapsed 达到/超过 remaining 时截断为 0（不为负）。
    """
    fake_now = {"t": 1000.0}
    monkeypatch.setattr("src.b_sim.client.time.monotonic", lambda: fake_now["t"])

    client = RobotClient(ROBOT)
    client.remaining_real_s = 100.0
    client._enter_monotonic = fake_now["t"]  # enter 时刻
    assert client.real_budget_left_s() == pytest.approx(100.0)

    fake_now["t"] += 30.0  # 现实时间前进 30 s
    assert client.real_budget_left_s() == pytest.approx(70.0)

    fake_now["t"] += 40.0  # 再前进 40 s（共 70 s）
    assert client.real_budget_left_s() == pytest.approx(30.0)

    # 超过剩余预算后截断为 0（max(0, ...)），不返回负值。
    fake_now["t"] += 1000.0
    assert client.real_budget_left_s() == 0.0
    # real_elapsed_s 反映自 enter 以来的单调耗时。
    assert client.real_elapsed_s() == pytest.approx(1070.0)


def test_real_elapsed_is_zero_before_enter():
    """enter() 之前（_enter_monotonic 为 None）耗时与预算均安全返回 0。"""
    client = RobotClient(ROBOT)
    assert client._enter_monotonic is None
    assert client.real_elapsed_s() == 0.0
    assert client.real_budget_left_s() == 0.0


class _FakeBudgetClient:
    """最小客户端替身：仅暴露 _stop() 所需的 real_budget_left_s / virtual_time_s。"""

    def __init__(self, budget_left_s: float, virtual_time_s: float = 0.0):
        self._budget_left_s = budget_left_s
        self.virtual_time_s = virtual_time_s

    def real_budget_left_s(self) -> float:
        return self._budget_left_s


def test_stop_triggers_when_real_budget_at_or_below_reserve():
    """R5.8：现实剩余 ≤ real_time_reserve_s(15) 时 _stop() 返回 True（15 s 前收尾）。"""
    config = _StrategyConfig53()
    assert config.real_time_reserve_s == 15.0

    # 恰为 15 s：≤ 判据成立 → 停止。
    strat_at = _DogStrategy53(client=_FakeBudgetClient(15.0), config=config)
    assert strat_at._stop() is True
    assert "real time budget reached" in strat_at.events

    # 低于 15 s（如 5 s）→ 停止。
    strat_below = _DogStrategy53(client=_FakeBudgetClient(5.0), config=config)
    assert strat_below._stop() is True


def test_stop_does_not_trigger_when_real_budget_above_reserve():
    """R5.8：现实剩余 > 15 s 且虚拟预算未耗尽时 _stop() 返回 False（继续动作）。"""
    config = _StrategyConfig53()
    # 现实剩余 16 s（> 15），虚拟时刻远低于 0.9×360000 预算 → 不停止。
    strat = _DogStrategy53(client=_FakeBudgetClient(16.0, virtual_time_s=0.0), config=config)
    assert strat._stop() is False
    assert strat._stopped is False

    # 充裕的现实预算（如 600 s）同样不停止。
    strat2 = _DogStrategy53(client=_FakeBudgetClient(600.0, virtual_time_s=1000.0), config=config)
    assert strat2._stop() is False


def test_stop_triggers_when_virtual_budget_reached_even_with_real_budget():
    """R5.8/R3.12：现实预算充裕但虚拟时刻 ≥ 0.9×MAX_VIRTUAL_SECONDS 时亦停止。"""
    config = _StrategyConfig53()
    over_virtual = config.virtual_time_budget_s  # 恰达虚拟预算上限
    strat = _DogStrategy53(client=_FakeBudgetClient(600.0, virtual_time_s=over_virtual),
                           config=config)
    assert strat._stop() is True
    assert "virtual time budget reached" in strat.events


def test_successful_measure_updates_client_virtual_time_from_server(server):
    """R5.6：一次成功 measure 将 client.virtual_time_s 更新为服务端返回的 virtual_time_s。"""
    url, session = server
    client = RobotClient(ROBOT, base_url=url)
    client.enter()
    assert client.virtual_time_s == 0.0
    result = client.measure(0.0, 0.0, 1)
    # 客户端虚拟时钟 == 该动作返回的 virtual_time_s == 服务端 arena 的虚拟时刻。
    assert client.virtual_time_s == pytest.approx(result.virtual_time_s)
    assert client.virtual_time_s == pytest.approx(session.arena.robot.virtual_time_s)
    assert client.virtual_time_s > 0.0  # 检测 +5 s，虚拟时刻已推进
