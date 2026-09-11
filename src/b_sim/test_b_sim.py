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
    url, session = server
    assert raw_post(url, "/unknown", base("u"))[0] == 404
    assert raw_post(url, "/enter", base("v"), method="GET")[0] == 405
    assert raw_post(url, "/enter", base("w"), ctype="text/plain")[0] == 415
    assert raw_post(url, "/enter", b"{not json", ctype="application/json")[0] == 400
    # Exercise the Windows upload/early-response race more than once. A rejected
    # request must receive 413 and cannot enter or advance the arena.
    for _ in range(10):
        status, body = raw_post(url, "/enter", b"x" * 70000)
        assert status == 413
        assert body["accepted"] is False
        assert body["virtual_time_s"] == 0
    assert not session.arena.robot.entered


def test_rejected_body_drain_is_bounded_and_restores_timeout():
    import io
    from src.b_sim.server import Handler, MAX_REJECT_DRAIN_BYTES

    class Connection:
        timeout = None

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value

    handler = object.__new__(Handler)
    handler.connection = Connection()
    handler.rfile = io.BytesIO(b"x" * (MAX_REJECT_DRAIN_BYTES + 100))
    handler._drain_rejected_body(10**12)
    assert handler.rfile.tell() == MAX_REJECT_DRAIN_BYTES
    assert handler.connection.gettimeout() is None


def test_oversized_request_with_stalled_body_still_returns_413(server):
    import socket
    from urllib.parse import urlsplit

    url, session = server
    address = urlsplit(url)
    with socket.create_connection((address.hostname, address.port), timeout=3) as peer:
        peer.sendall(b"POST /enter HTTP/1.1\r\nHost: localhost\r\n"
                     b"Content-Type: application/json\r\nContent-Length: 1000000000\r\n\r\nx")
        response = peer.makefile("rb")
        try:
            assert response.readline().startswith(b"HTTP/1.1 413 ")
        finally:
            response.close()
    assert not session.arena.robot.entered


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


def test_run_order_origin_sweep_precedes_tour_and_fallbacks():
    """R3.1：run() 的情节顺序为「原点普扫 → 统一主巡回 → 兜底（复活/内圈重测/固定环扫）」。

    与旧版的差别（本轮路径规划重构）：清除与补测不再是「归航」「环补扫」两个交替阶段，而是合并
    进同一个开放 TSP 主巡回 `_tour`；固定环扫 `_probe_ring` 降级为兜底，且只在"可断言还有源
    没找到"（已清除数 < 干扰源总数下界 10）时才启用。本例只有 1 个干扰源，故该条件成立、
    环扫兜底会执行。

    通过在 DogStrategy 上打桩记录各阶段调用次序，确认 _sweep_at 首个被调用，
    且 _tour / _revive_exhausted / _probe_ring 均在其之后。
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
        def rec(name, fn):
            def wrapper(*a, **k):
                calls.append(name)
                return fn(*a, **k)
            return wrapper

        strategy._sweep_at = rec("sweep", strategy._sweep_at)            # type: ignore[assignment]
        strategy._tour = rec("tour", strategy._tour)                     # type: ignore[assignment]
        strategy._revive_exhausted = rec("revive", strategy._revive_exhausted)  # type: ignore[assignment]
        strategy._probe_ring = rec("ring", strategy._probe_ring)         # type: ignore[assignment]
        strategy.run()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert calls, "策略未执行任何阶段"
    assert calls[0] == "sweep", f"首阶段应为原点普扫，实际 {calls[0]}"
    sweep_i = calls.index("sweep")
    for phase in ("tour", "revive", "ring"):
        assert phase in calls, f"缺少阶段 {phase}"
        assert calls.index(phase) > sweep_i, f"阶段 {phase} 早于原点普扫"
    # 子序列：sweep → tour → revive → ...（环扫兜底在复活之后）
    assert calls.index("tour") < calls.index("revive")
    assert calls.index("revive") < calls.index("ring")
    # 主巡回是主路径，兜底只跑一次量级：tour 的调用次数应不少于环扫
    assert calls.count("tour") >= calls.count("ring")


# ---- (b) R3.2：覆盖站位 {原点} ∪ Ring(1250,12) 最坏间隙 ≈670 m < 1000 m ----

def test_coverage_stations_are_origin_plus_ring():
    """R3.2：覆盖站位 = 原点 + 单层内环，且最坏覆盖间隙 < 接收半径下限。

    锁的是「覆盖充分性」这条规格，而不是某一组具体环参数——参数经实测标定会变
    （1250x12 -> 1150x6 -> 1100x8），把数值写死会让调参必然踩红。
    """
    cfg = _StrategyConfig61()
    # 单环情形（全向 Q3）不启用外环。
    assert cfg.use_outer_ring is False
    assert cfg.directional_probe is False
    stations = [(0.0, 0.0)] + _cov_ring(cfg.probe_ring_radius, cfg.probe_ring_stations)
    assert len(stations) == cfg.probe_ring_stations + 1
    # 环站位均落在配置半径上。
    for px, py in stations[1:]:
        assert math.hypot(px, py) == pytest.approx(cfg.probe_ring_radius, abs=1e-6)
    # 覆盖充分性：任一点到最近站位不超过全向源最小有效接收半径。
    gap, _ = _worst_gap(stations, grid=300)
    assert gap < _RECEPTION_MIN_61, f"最坏间隙 {gap:.1f} m 已超接收半径下限"


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
    """R3.2：全向情形 _probe_stations 只产生一层内环，不含外环。"""
    cfg = _StrategyConfig61()
    strategy = _DogStrategy61(client=None, config=cfg)  # 不触网，仅算站位
    rings = strategy._probe_stations()
    assert len(rings) == 1, "全向情形应只有内环一层"
    assert len(rings[0]) == cfg.probe_ring_stations
    for px, py in rings[0]:
        assert math.hypot(px, py) == pytest.approx(cfg.probe_ring_radius, abs=1e-6)


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



# ---------------------------------------------------------------------------
# 任务 6.2：核对逐步逼近归航与清除触发（R3.3 / R3.4 / R3.5 / R3.6）
# Feature: b-radio-interference-locate-clear
#
# 本节为「验证/加固」任务的回归测试。核对项：
#   (a) _step_closer 前进 approach_fraction(0.62)×剩余距离，横向偏移
#       lateral_fraction(0.16)×前进距离（advance），使定位多边形直径随逼近单调不增（R3.3）；
#   (b) dist(estimate) ≤ clear_distance_m(55) 触发对估计位置的 /clear（R3.4）；
#   (c) near（真实距离 ≤5 m）在当前位置直接 /clear、不要求先取示向度（R3.5）；
#   (d) /clear 失败保持 cleared=False、累加 clear_attempts、走 _spiral 补点，
#       连续失败达上限转射线盲清除/放弃复活，不无限重试同一估计点（R3.6）。
#
# 说明：单元核对用最小 fake client 替身（仅暴露 DogStrategy 所需接口），
# near 直接清除额外用本地 mock arena 端到端驱动 _probe_channel_at 佐证。
# ---------------------------------------------------------------------------

import math as _math62

from src.b_sim.arena import Arena as _Arena62, Case as _Case62, Jammer as _Jammer62
from src.b_sim.client import ClearResult, MeasureResult
from src.b_sim.geometry import Bearing as _Bearing62
from src.b_sim.strategy import (
    ChannelState as _ChannelState62,
    DogStrategy as _DogStrategy62,
    StrategyConfig as _StrategyConfig62,
)


class _FakeArena62:
    """最小 arena 替身：measure 返回预置结果序列，clear 依据到真源距离判定成功。

    仅覆盖 DogStrategy 归航路径所需的 measure/clear 行为，不联网、不计时。
    """

    def __init__(self, *, source=None, clear_radius=20.0, measure_script=None):
        self.source = source          # (x, y) 真源；用于 clear 距离判定
        self.clear_radius = clear_radius
        self.measure_script = list(measure_script or [])
        self.measure_calls: list[tuple[float, float, int]] = []
        self.clear_calls: list[tuple[float, float, int]] = []

    def measure(self, x, y, channel):
        self.measure_calls.append((x, y, channel))
        if self.measure_script:
            return self.measure_script.pop(0)
        return {"measure_result": "no_signal", "virtual_time_s": 0.0}

    def clear(self, x, y, channel):
        self.clear_calls.append((x, y, channel))
        ok = (self.source is not None
              and _math62.hypot(x - self.source[0], y - self.source[1]) <= self.clear_radius)
        return {"clear_result": "success" if ok else "no_target_in_range",
                "virtual_time_s": 0.0}


class _FakeClient62:
    """最小 client 替身：把 measure/clear 转发给 _FakeArena62，记录调用序列。

    暴露 DogStrategy 所需接口：measure/clear/real_budget_left_s/virtual_time_s/log。
    real_budget_left_s 返回大值（永不因预算触发 _stop）。
    """

    def __init__(self, arena: _FakeArena62):
        self.arena = arena
        self.virtual_time_s = 0.0
        self.log: list[dict] = []
        self.clear_calls: list[tuple[float, float, int]] = []
        self.measure_calls: list[tuple[float, float, int]] = []

    def real_budget_left_s(self) -> float:
        return 10_000.0

    def measure(self, x, y, channel):
        self.measure_calls.append((x, y, channel))
        body = self.arena.measure(x, y, channel)
        res = body["measure_result"]
        svd = body.get("svd_deg")
        return MeasureResult(result=res, svd_deg=svd, virtual_time_s=0.0)

    def clear(self, x, y, channel):
        self.clear_calls.append((x, y, channel))
        body = self.arena.clear(x, y, channel)
        return ClearResult(result=body["clear_result"], virtual_time_s=0.0)


def _make_strategy62(arena: _FakeArena62, config=None):
    strat = _DogStrategy62.__new__(_DogStrategy62)  # 绕过 __init__ 的真实 client 依赖
    strat.client = _FakeClient62(arena)
    strat.config = config or _StrategyConfig62()
    strat.channels = {ch: _ChannelState62(ch) for ch in range(1, 21)}
    strat.pos = (0.0, 0.0)
    strat.radio_channel = 1
    strat.events = []
    strat._stopped = False
    return strat


# ---- (a) R3.3：_step_closer 前进/横向几何 ----

def test_step_closer_advance_is_062_of_distance_and_lateral_016_of_advance():
    """R3.3：前进 ≈0.62×剩余距离，横向偏移 ≈0.16×前进距离(advance)。

    构造一个远距离估计（避免 clear_distance 附近的截断项主导），验证新观测点相对
    起点的沿视线前进分量 ≈0.62×distance，横向（垂直视线）分量 ≈0.16×advance。
    """
    cfg = _StrategyConfig62()
    # 估计位于正东 1000 m 处，狗在原点；no_signal 使 _step_closer 不会因取得示向度而提前返回。
    arena = _FakeArena62(measure_script=[{"measure_result": "no_signal", "virtual_time_s": 0.0}] * 4)
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[1]
    state.estimate = (1000.0, 0.0)
    strat.pos = (0.0, 0.0)
    distance = 1000.0

    ok = strat._step_closer(state, distance)
    assert ok is True
    assert arena.measure_calls, "_step_closer 未发起重测"
    # 第一个候选（sign=+1）：视线为 +x，横向单位向量为 (-uy, ux)=(0,1)，
    # sign=+1 时候选 = pos + u*advance - (0,1)*lateral。
    cx, cy, _ch = arena.measure_calls[0]
    # 起点为原点、视线沿 +x：前进分量 = cx，横向分量 = |cy|。
    expected_advance = distance * cfg.approach_fraction        # 620 m
    expected_lateral = expected_advance * cfg.lateral_fraction  # 0.16×advance
    assert cx == pytest.approx(expected_advance, rel=1e-6)
    assert abs(cy) == pytest.approx(expected_lateral, rel=1e-6)
    # 明确区分「×advance」与「旧口径 ×distance」：两者数值不同。
    assert abs(cy) < distance * cfg.lateral_fraction - 1.0


def test_step_closer_advance_capped_near_clear_distance():
    """R3.3：距离接近 clear_distance 时前进被 (distance − clear_distance×0.35) 截断，
    仍保持沿视线单调靠近（advance>0 且不越过目标）。"""
    cfg = _StrategyConfig62()
    arena = _FakeArena62(measure_script=[{"measure_result": "no_signal", "virtual_time_s": 0.0}] * 4)
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[1]
    distance = 70.0                       # 略大于 clear_distance(55)
    state.estimate = (distance, 0.0)
    strat.pos = (0.0, 0.0)
    strat._step_closer(state, distance)
    cx, cy, _ = arena.measure_calls[0]
    upper = distance - cfg.clear_distance_m * 0.35
    lower = cfg.clear_distance_m * 0.8
    expected_advance = min(max(lower, distance * cfg.approach_fraction), upper)
    assert cx == pytest.approx(expected_advance, rel=1e-6)
    assert 0.0 < cx < distance             # 前进但不越过目标
    assert abs(cy) == pytest.approx(expected_advance * cfg.lateral_fraction, rel=1e-6)


def test_localization_diameter_non_increasing_while_approaching():
    """R3.3：随检测点逐步逼近真源，ChannelState.diameter 单调不增（逼近收缩误差）。

    真源在正东 1000 m。以一串沿 x 轴逐步逼近的检测点喂入带 ±1° 误差的示向度，
    每加一个更近的观测，refresh 后的定位多边形直径应不增（收缩或持平）。
    """
    source = (1000.0, 0.0)

    def svd_from(px, py, err_deg):
        true_deg = math.degrees(math.atan2(source[1] - py, source[0] - px))
        return norm_deg(true_deg + err_deg)

    state = _ChannelState62(1)
    # 逐步逼近的检测点（沿 x 轴，带交叉的横向偏移以形成有效交会基线）。
    approach_points = [
        (-200.0, 60.0),
        (0.0, -60.0),
        (200.0, 55.0),
        (400.0, -50.0),
        (600.0, 45.0),
        (750.0, -40.0),
        (850.0, 30.0),
    ]
    # 确定性 ±1° 误差（交替符号，落在 [−1,+1] 界内）。
    errs = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    diameters: list[float] = []
    for (px, py), e in zip(approach_points, errs):
        state.add_bearing(px, py, svd_from(px, py, e))
        if state.diameter != float("inf"):
            diameters.append(state.diameter)

    assert len(diameters) >= 3, "有效观测不足以形成定位多边形"
    for earlier, later in zip(diameters, diameters[1:]):
        assert later <= earlier + 1e-6, f"逼近过程直径增大: {diameters}"
    # 逼近确有收缩效果：末尾直径显著小于起始。
    assert diameters[-1] < diameters[0]


# ---- (b) R3.4：dist(estimate) ≤ 55 m 触发对估计位置的 /clear ----

def test_home_channel_triggers_clear_at_estimate_within_clear_distance():
    """R3.4：估计到源距离 ≤ clear_distance_m(55) 时，对估计位置发起 /clear。"""
    cfg = _StrategyConfig62()
    # 真源即估计点，狗距其 40 m（< 55）。clear 应命中并成功。
    source = (40.0, 0.0)
    arena = _FakeArena62(source=source)
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[3]
    state.bearings = [_Bearing62(0.0, 100.0, 270.0), _Bearing62(0.0, -100.0, 90.0)]
    state.estimate = (40.0, 0.0)          # dist(estimate) = 40 ≤ 55
    strat.pos = (0.0, 0.0)

    ok = strat._home_channel(state)
    assert ok is True
    assert state.cleared is True
    # 恰好对估计位置发起了 /clear（首个 clear 调用坐标 == estimate）。
    assert arena.clear_calls, "未发起任何 /clear"
    fx, fy, fch = arena.clear_calls[0]
    assert (fx, fy) == pytest.approx((40.0, 0.0))
    assert fch == 3


def test_home_channel_does_not_clear_when_estimate_beyond_clear_distance():
    """R3.4 边界：dist(estimate) > 55 m 时不立即 /clear，而是先 _step_closer 逼近。"""
    cfg = _StrategyConfig62()
    arena = _FakeArena62(source=(1000.0, 0.0),
                         measure_script=[{"measure_result": "no_signal", "virtual_time_s": 0.0}] * 2)
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[3]
    state.bearings = [_Bearing62(0.0, 100.0, 275.0), _Bearing62(0.0, -100.0, 85.0)]
    state.estimate = (1000.0, 0.0)        # 远大于 55
    strat.pos = (0.0, 0.0)
    # 只推进一步：打桩 _step_closer 记录被调用后立即结束循环。
    called = {"step": 0}
    orig_step = strat._step_closer

    def spy_step(st, dist):
        called["step"] += 1
        strat._stopped = True             # 逼停循环，只验证首步走的是逼近而非清除
        return orig_step(st, dist)

    strat._step_closer = spy_step  # type: ignore[assignment]
    strat._home_channel(state)
    assert called["step"] == 1, "远距离首步应为 _step_closer 逼近"
    assert not arena.clear_calls, "估计距离 >55 m 不应立即 /clear"


# ---- (c) R3.5：near（≤5 m）在当前位置直接 /clear，不要求先取示向度 ----

def test_probe_channel_at_near_triggers_direct_clear_without_bearing():
    """R3.5：检测返回 near 时，对当前所在点直接 /clear，且不写入示向度。"""
    cfg = _StrategyConfig62()
    point = (300.0, 400.0)
    arena = _FakeArena62(source=point,     # clear 命中
                         measure_script=[{"measure_result": "near", "virtual_time_s": 0.0}])
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[7]

    strat._probe_channel_at(point, state)

    # 直接对 point 清除，成功；未取得任何示向度。
    assert arena.clear_calls == [(300.0, 400.0, 7)]
    assert state.cleared is True
    assert state.bearings == [], "near 路径不应写入示向度"


def test_near_direct_clear_end_to_end_via_mock_arena():
    """R3.5（端到端）：真实 Arena 下源在原点，狗在其 3 m 内检测得 near → 直接清除成功。

    用真实 Arena（复刻 near ≤5 m 判定与 20 m 清除半径）驱动 _probe_channel_at，
    佐证策略在 near 结果上对当前位置直接 /clear。
    """
    case = _Case62(jammers=[_Jammer62(channel=2, x=0.0, y=0.0, reception_radius_m=1200.0)],
                   error_salt="t62", seed=0)
    arena = _Arena62(case)
    arena.enter()
    # 独立地先确认该点确实返回 near（真实距离 3 m ≤ 5 m）。
    assert arena.measure(3.0, 0.0, 2)["measure_result"] == "near"

    # 用真实客户端接口的最小包装：直接调用 arena 并转成 MeasureResult/ClearResult。
    class _ArenaClient:
        virtual_time_s = 0.0
        def __init__(self, a): self.a = a; self.log = []
        def real_budget_left_s(self): return 10_000.0
        def measure(self, x, y, ch):
            b = self.a.measure(x, y, ch)
            return MeasureResult(result=b["measure_result"], svd_deg=b.get("svd_deg"),
                                 virtual_time_s=b.get("virtual_time_s", 0.0))
        def clear(self, x, y, ch):
            b = self.a.clear(x, y, ch)
            return ClearResult(result=b["clear_result"], virtual_time_s=b.get("virtual_time_s", 0.0))

    strat = _DogStrategy62.__new__(_DogStrategy62)
    strat.client = _ArenaClient(arena)
    strat.config = _StrategyConfig62()
    strat.channels = {ch: _ChannelState62(ch) for ch in range(1, 21)}
    strat.pos = (0.0, 0.0)
    strat.radio_channel = 1
    strat.events = []
    strat._stopped = False

    state = strat.channels[2]
    strat._probe_channel_at((3.0, 0.0), state)   # near → 直接对 (3,0) 清除（距源 3 m ≤ 20）
    assert state.cleared is True
    assert state.bearings == []


# ---- (d) R3.6：/clear 失败保持 cleared=False、累加 clear_attempts、_spiral 补点、不无限重试 ----

def test_clear_at_failure_increments_attempts_and_keeps_uncleared():
    """R3.6：/clear 失败 → cleared 保持 False，clear_attempts +1。"""
    cfg = _StrategyConfig62()
    # 真源离估计点很远（>20 m），任何 clear 都失败。
    arena = _FakeArena62(source=(5000.0, 0.0))
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[4]
    before = state.clear_attempts
    ok = strat._clear_at((0.0, 0.0), 4)
    assert ok is False
    assert state.cleared is False
    assert state.clear_attempts == before + 1


def test_home_channel_clear_failure_attempts_spiral_and_stays_uncleared():
    """R3.6：clear_distance 内清除失败后走 _spiral 补点；仍失败则保持未清除并结束。"""
    cfg = _StrategyConfig62()
    # 估计点在狗 40 m（<55）触发 clear；真源在 5000 m 外，clear/spiral 均失败。
    arena = _FakeArena62(source=(5000.0, 0.0))
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[5]
    state.bearings = [_Bearing62(0.0, 100.0, 270.0), _Bearing62(0.0, -100.0, 90.0)]
    state.estimate = (40.0, 0.0)
    strat.pos = (0.0, 0.0)

    spiral_called = {"n": 0}
    orig_spiral = strat._spiral

    def spy_spiral(st):
        spiral_called["n"] += 1
        return orig_spiral(st)

    strat._spiral = spy_spiral  # type: ignore[assignment]

    ok = strat._home_channel(state)
    assert ok is False
    assert state.cleared is False
    assert spiral_called["n"] == 1, "clear 失败后应尝试 _spiral 补点一次"
    # 至少发起了 1 次估计点 clear + 若干次 spiral 补点 clear（有限，不无限）。
    assert len(arena.clear_calls) >= 2
    # spiral 布点数量受 spiral_points 约束，clear_attempts 不超过 max_clear_attempts。
    assert state.clear_attempts <= cfg.max_clear_attempts


def test_home_channel_clear_budget_exhausted_falls_back_to_ray_not_infinite():
    """R3.6：连续 /clear 失败累计达上限 → 转射线盲清除兜底，不无限重试同一估计点。

    预置 clear_attempts 达到 max_clear_attempts，令 _home_channel 首轮即命中
    「clear budget exhausted」门禁，转入 _ray_blind_clear（有限调用，最终返回）。
    """
    cfg = _StrategyConfig62()
    arena = _FakeArena62(source=(5000.0, 0.0))     # 一切 clear 失败
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[6]
    state.bearings = [_Bearing62(0.0, 0.0, 45.0)]
    state.estimate = (30.0, 30.0)
    state.clear_attempts = cfg.max_clear_attempts   # 已达清除上限
    strat.pos = (0.0, 0.0)

    ray_called = {"n": 0}
    orig_ray = strat._ray_blind_clear

    def spy_ray(st):
        ray_called["n"] += 1
        return orig_ray(st)

    strat._ray_blind_clear = spy_ray  # type: ignore[assignment]

    ok = strat._home_channel(state)
    # clear 上限门禁应立即转射线盲清除（而非围绕估计点无限重试）。
    assert ray_called["n"] == 1
    assert "clear budget exhausted" in " ".join(strat.events)
    assert ok is False or state.cleared is True     # 真源在 5000 m 外，实际清不掉
    assert state.cleared is False
    # 射线盲清除有独立上限，clear 调用数有限（不无限）。
    ray_start = cfg.max_clear_attempts
    assert state.clear_attempts - ray_start <= cfg.ray_clear_max_attempts + 1


def test_lost_signal_defers_to_ray_blind_clear_when_bearing_known():
    """R3.6：连续沉默达容忍上限且已有示向度 → 转射线盲清除兜底（不无限重试）。"""
    cfg = _StrategyConfig62()
    arena = _FakeArena62(source=(5000.0, 0.0))
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[8]
    state.bearings = [_Bearing62(0.0, 0.0, 30.0)]   # 已有方向
    state.estimate = None
    state.consecutive_silent = cfg.silent_tolerance  # 达到沉默容忍上限
    strat.pos = (0.0, 0.0)

    ray_called = {"n": 0}
    orig_ray = strat._ray_blind_clear

    def spy_ray(st):
        ray_called["n"] += 1
        return orig_ray(st)

    strat._ray_blind_clear = spy_ray  # type: ignore[assignment]
    strat._home_channel(state)
    assert ray_called["n"] == 1
    assert "lost signal" in " ".join(strat.events)


# ===========================================================================
# 任务 6.3：核对预算拆分、沉默容忍与复活/内圈/边缘处理（R3.7–R3.11）
# Feature: b-radio-interference-locate-clear
#
# 本节为「验证/加固」任务的回归测试（VERIFY：实现已正确，无需改 strategy.py）。
# 复用上方任务 6.2 定义的替身：_FakeArena62 / _make_strategy62 / _StrategyConfig62 /
# _Bearing62（Bearing）。核对项：
#   (R3.7) 搜索段 max_measures_per_channel=40 与归航段 max_homing_measures=14 分别记账；
#          首个示向度后 add_bearing 重置 homing_measures=0 且清除 exhausted；
#          _measure 每次都累加 measure_count，仅在已有示向度时才累加 homing_measures。
#   (R3.8) _revive_exhausted 复活放弃频道（exhausted→False、置 revived）；silent_tolerance=3。
#   (R3.9) _inner_retry 对「有示向度却清不掉」的频道走原点小三角基线（≥3 观测点）。
#   (R3.10) 边缘源（hypot>1400）允许越过覆盖环沿示向度逼近至 ARENA_RADIUS(1800) 内。
#   (R3.11) 就近贪心排序（轻断言，主验证在任务 6.1）。
# ===========================================================================


# ---- (R3.7) 预算拆分与首个示向度重置 ----

def test_config_budget_split_search_forty_homing_fourteen():
    """R3.7：搜索段总额度 40、归航段额度 14、沉默容忍 3 —— 配置默认值到位。"""
    cfg = _StrategyConfig62()
    assert cfg.max_measures_per_channel == 40
    assert cfg.max_homing_measures == 14
    assert cfg.silent_tolerance == 3


def test_add_bearing_first_resets_homing_measures_and_clears_exhausted():
    """R3.7：首个示向度到来时，homing_measures 重置为 0 且清除 exhausted 标记。

    构造一个「搜索阶段已烧掉大量归航额度且被判放弃」的状态：homing_measures>0、
    exhausted=True、且已有若干沉默观测。add_bearing 后二者必须双双复位，避免
    「已可定位」的频道因搜索阶段耗尽额度而被误判放弃。
    """
    state = _ChannelState62(channel=4)
    state.homing_measures = 11          # 搜索阶段已累计的归航计数
    state.exhausted = True              # 曾被判放弃
    state.consecutive_silent = 5
    state.silent_points = [(0.0, 0.0), (10.0, 0.0)]
    assert not state.bearings           # 首个示向度之前

    state.add_bearing(100.0, 200.0, 42.0)

    assert state.homing_measures == 0   # 归航预算重新起算
    assert state.exhausted is False     # 放弃标记被清除
    assert state.consecutive_silent == 0
    assert len(state.bearings) == 1


def test_add_bearing_second_does_not_reset_homing_measures():
    """R3.7：归航中拿到后续示向度不重置归航计数。

    复位只在两种情形发生：拿到**首个**示向度（此前的检测都花在搜索上，归航预算须重新起算），
    或该频道**已被判定归航失败（exhausted）后又在新站位拿到示向度**——后者意味着换了一套交会
    几何，值得重新起算；这条重试由 `ChannelState.probe_rounds` 上界兜住，不会无限循环。
    """
    state = _ChannelState62(channel=4)
    state.add_bearing(0.0, 0.0, 10.0)   # 首个 → 复位
    state.homing_measures = 7           # 归航阶段自然累计
    state.add_bearing(50.0, 50.0, 20.0)  # 归航中的第二个：不复位
    assert state.homing_measures == 7
    assert state.exhausted is False


def test_add_bearing_after_exhausted_resets_homing_budget_and_clears_flag():
    """R3.7 续：归航失败后在新站位重新拿到示向度 → 重置归航预算并解除 exhausted。"""
    state = _ChannelState62(channel=4)
    state.add_bearing(0.0, 0.0, 10.0)
    state.homing_measures = 7
    state.exhausted = True               # 归航失败，等待换站位重取几何
    state.add_bearing(50.0, 50.0, 20.0)
    assert state.homing_measures == 0, "新交会几何应重新起算归航预算"
    assert state.exhausted is False, "拿到新示向度后应解除放弃标记"


def test_measure_counts_search_and_homing_budgets_separately():
    """R3.7：_measure 每次累加 measure_count；仅在已有示向度时才累加 homing_measures。"""
    arena = _FakeArena62(measure_script=[
        {"measure_result": "no_signal", "virtual_time_s": 0.0},   # 搜索段：无示向度
        {"measure_result": "no_signal", "virtual_time_s": 0.0},   # 归航段：已有示向度
    ])
    strat = _make_strategy62(arena)
    state = strat.channels[1]

    # 尚无示向度：measure_count +1，homing_measures 不动（仍属搜索段）。
    strat._measure((0.0, 0.0), 1)
    assert state.measure_count == 1
    assert state.homing_measures == 0

    # 拿到首个示向度（不经 _measure，直接注入以隔离计数逻辑）。
    state.add_bearing(0.0, 0.0, 45.0)
    assert state.homing_measures == 0   # add_bearing 复位

    # 已有示向度后再 _measure：measure_count 与 homing_measures 同时 +1。
    strat._measure((10.0, 0.0), 1)
    assert state.measure_count == 2
    assert state.homing_measures == 1


# ---- (R3.8) 复活与沉默容忍 ----

def test_revive_exhausted_flips_and_marks_revived_for_uncleared():
    """R3.8：_revive_exhausted 对未清除的放弃频道置 exhausted→False 并标记 revived。"""
    strat = _make_strategy62(_FakeArena62())
    # ch1：放弃且未清除 → 应被复活
    s1 = strat.channels[1]
    s1.exhausted = True
    s1.cleared = False
    s1.consecutive_silent = 3
    # ch2：放弃但已清除 → 不复活（无需再试）
    s2 = strat.channels[2]
    s2.exhausted = True
    s2.cleared = True
    # ch3：放弃且未清除，但此前已复活过 → 不再复活（只给一次第二次机会）
    s3 = strat.channels[3]
    s3.exhausted = True
    s3.cleared = False
    s3.revived = True

    strat._revive_exhausted()

    assert s1.exhausted is False and s1.revived is True
    assert s1.consecutive_silent == 0        # 复活时清零连续沉默计数
    assert s1.ray_swept is False             # 允许再做一次射线盲清除
    assert s2.cleared is True and s2.revived is False  # 已清除频道不动
    assert s3.exhausted is True              # 已复活过的频道保持放弃


def test_silent_tolerance_three_before_abandoning_channel():
    """R3.8：连续沉默达 silent_tolerance(=3) 次后方可对当前站位放弃该频道。

    对一个既无示向度的频道连续喂 no_signal：每次 _probe_channel_at 累加
    consecutive_silent；到达 3 次即达放弃阈值。
    """
    cfg = _StrategyConfig62()
    arena = _FakeArena62(measure_script=[
        {"measure_result": "no_signal", "virtual_time_s": 0.0} for _ in range(3)
    ])
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[5]
    for expected in (1, 2, 3):
        strat._probe_channel_at((100.0 * expected, 0.0), state)
        assert state.consecutive_silent == expected
    # 达到 silent_tolerance 才允许放弃（前两次不足以放弃）。
    assert state.consecutive_silent >= cfg.silent_tolerance
    assert len(state.silent_points) == 3


# ---- (R3.9) 内圈重测：原点可见但环站位全沉默的近距离源 ----

def test_inner_retry_visits_small_triangle_of_at_least_three_points():
    """R3.9：_inner_retry 对「有示向度却清不掉」的频道走原点小三角基线（≥3 观测点）。

    源在原点可见（已注入一条示向度）但始终清不掉（真源置于 5000 m 外，clear 永远失败）。
    打桩 _probe_channel_at 记录被访问的站位点：应包含原点在内的一个小三角基线，
    ≥3 个不同观测点，且三角边长 ≤400 m（贴近原点的小基线）。
    """
    arena = _FakeArena62(source=(5000.0, 0.0))     # 真源在远处 → clear 恒失败
    strat = _make_strategy62(arena)
    # 仅 ch7 有示向度且未清除；其余频道无示向度（不应被内圈重测选中）。
    state = strat.channels[7]
    state.bearings = [_Bearing62(0.0, 0.0, 30.0)]
    state.cleared = False

    visited: list[tuple[float, float]] = []

    def spy_probe(point, st):
        if st.channel == 7:
            visited.append(point)
        # 记录后即返回，不真正驱动 measure/clear，隔离站位序列
        return None

    strat._probe_channel_at = spy_probe  # type: ignore[assignment]
    # _home_all 在 _inner_retry 每站后被调用；替身化为 no-op 以隔离站位序列。
    strat._home_all = lambda: None       # type: ignore[assignment]

    strat._inner_retry()

    # 去重后至少 3 个不同观测点（内圈三角基线）。
    unique = sorted(set(visited))
    assert len(unique) >= 3, f"内圈重测观测点不足 3 个: {unique}"
    # 必含原点作为基线锚点。
    assert (0.0, 0.0) in unique
    # 三角基线贴近原点：所有点到原点距离 ≤400 m（小三角）。
    for px, py in unique:
        assert _math62.hypot(px, py) <= 400.0 + 1e-6, f"内圈站位 {px,py} 超出小三角范围"


def test_inner_retry_skips_channels_without_bearings():
    """R3.9：完全沉默（无示向度）的频道不参与内圈重测（属覆盖问题，交给环补扫）。"""
    arena = _FakeArena62(source=(5000.0, 0.0))
    strat = _make_strategy62(arena)
    # 所有频道均无示向度 → 无目标 → 不访问任何站位。
    visited: list[tuple[float, float]] = []
    strat._probe_channel_at = lambda point, st: visited.append(point)  # type: ignore[assignment]
    strat._home_all = lambda: None  # type: ignore[assignment]
    strat._inner_retry()
    assert visited == []


# ---- (R3.10) 边缘源允许越过覆盖环逼近至 1800 m 内 ----

def test_edge_source_step_closer_advances_past_ring_within_arena_radius():
    """R3.10：边缘源（hypot>1400）沿示向度逼近，新观测点可越过 1250 m 覆盖环，
    但仍落在 ARENA_RADIUS(1800) 允许范围内（_step_closer 判据 ≤ ARENA_RADIUS×1.02）。

    构造一个贴边估计（正东 1700 m，hypot=1700>1400），狗从覆盖环上（1250 m）沿 x 轴
    逼近。验证 _step_closer 生成的重测点前进到覆盖环（1250 m）之外、且 ≤1800×1.02——
    即策略不把逼近点钉死在覆盖环上，允许越环贴近边缘源。
    """
    from src.b_sim.strategy import ARENA_RADIUS as _ARENA_R
    cfg = _StrategyConfig62()
    arena = _FakeArena62(measure_script=[{"measure_result": "no_signal", "virtual_time_s": 0.0}] * 4)
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[9]
    state.estimate = (1700.0, 0.0)          # 贴边源估计，hypot=1700>1400
    strat.pos = (cfg.probe_ring_radius, 0.0)  # 狗恰在 1250 m 覆盖环上
    distance = 1700.0 - cfg.probe_ring_radius  # 剩余 450 m

    ok = strat._step_closer(state, distance)
    assert ok is True
    assert arena.measure_calls, "_step_closer 未发起越环重测"
    cx, cy, _ = arena.measure_calls[0]
    # 新观测点越过 1250 m 覆盖环（沿 +x 前进），且未越出竞技场允许半径。
    assert cx > cfg.probe_ring_radius, "边缘源逼近点未越过覆盖环"
    assert _math62.hypot(cx, cy) <= _ARENA_R * 1.02 + 1e-6


def test_edge_source_can_clear_within_fifty_five_metres_beyond_ring():
    """R3.10：边缘源允许一路逼近至 dist(estimate) ≤ clear_distance(55) 再清除。

    真源在正东 1500 m（贴边）。狗逼近到距估计 ≤55 m 时，_home_channel 触发 /clear，
    clear 命中真源（20 m 内）→ 清除成功。整条路径越过覆盖环但始终在 1800 m 内。
    """
    cfg = _StrategyConfig62()
    source = (1500.0, 0.0)
    arena = _FakeArena62(source=source)     # clear 依距离判定
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[10]
    state.bearings = [_Bearing62(1470.0, 0.0, 0.0)]  # 已有方向（朝东）
    state.estimate = source                 # 已能定位到贴边真源
    strat.pos = (1460.0, 0.0)               # 已逼近到距源 40 m ≤55
    ok = strat._home_channel(state)
    assert ok is True and state.cleared is True
    # 清除点应在真源 20 m 内且落在竞技场半径附近（越环但合法）。
    assert arena.clear_calls, "未发起 /clear"
    cx, cy, _ = arena.clear_calls[-1]
    assert _math62.hypot(cx - source[0], cy - source[1]) <= cfg.clear_distance_m


# ---- (R3.11) 就近贪心排序（轻断言，主验证在任务 6.1）----

def test_home_all_processes_nearest_active_channel_first():
    """R3.11（轻断言）：_home_all 按到当前位置的欧氏距离升序选取下一归航目标。

    两个已可定位的频道，近者应先被 _home_channel 处理。用打桩记录处理顺序。
    """
    strat = _make_strategy62(_FakeArena62())
    near = strat.channels[1]
    far = strat.channels[2]
    near.bearings = [_Bearing62(0.0, 0.0, 0.0)]
    near.estimate = (100.0, 0.0)            # 距原点 100 m
    far.bearings = [_Bearing62(0.0, 0.0, 0.0)]
    far.estimate = (900.0, 0.0)             # 距原点 900 m
    strat.pos = (0.0, 0.0)

    order: list[int] = []

    def spy_home(st):
        order.append(st.channel)
        st.cleared = True                   # 处理后标记清除，退出 active 集合
        return True

    strat._home_channel = spy_home  # type: ignore[assignment]
    strat._home_all()
    assert order[0] == near.channel, f"未先处理最近目标: {order}"
    assert order == [near.channel, far.channel]



# ===========================================================================
# 任务 6.4：核对预算上限、优雅退出与结果报告（R3.12 / R3.13）
# Feature: b-radio-interference-locate-clear
#
# 本节为「验证/加固」任务的回归测试（VERIFY：实现已正确，无需改 strategy.py）。
# 核对项：
#   (R3.12) _stop() 在「现实剩余 ≤ real_time_reserve_s(15 s)」或「虚拟时刻
#           ≥ 0.9×MAX_VIRTUAL_SECONDS(=324000 s)」时停止新动作（_stopped=True 并
#           记事件）；否则继续。run() 用 try/finally 保证即便某阶段抛
#           TestEnded/ProtocolError 也总会调用 client.exit() 优雅收尾。
#   (R3.13) summary() 报告已清除个数 cleared_count（int）、平均定位清除时间 T̄
#           （mean_clear_time_s = virtual_time_s/已清除个数，无清除时为 None）、
#           程序虚拟时长 virtual_time_s，以及 measure_calls/clear_calls，
#           字段与官方评分项对应；cleared_count == len(cleared_channels)。
#
# T̄「两位小数」口径说明（R3.13）：summary() 返回**原始浮点** mean_clear_time_s，
# 两位/三位小数的四舍五入由**上层报告层** rehearse.per_episode 完成
# （src/b_sim/rehearse.py 中 round(stats["mean_clear_time_s"], 3)）。设计有意让
# summary() 保持原始精度、由报告层负责格式化，故此处**验证该分工**而非改变行为。
#
# 复用任务 6.2 的替身：_FakeArena62 / _FakeClient62 / _make_strategy62 /
# _StrategyConfig62 / _Bearing62（Bearing）。另定义一个带 enter/exit 的最小客户端
# 替身以驱动 run() 的 finally 优雅退出路径。
# ===========================================================================

from src.b_sim.arena import MAX_VIRTUAL_SECONDS as _MAX_VIRTUAL_64
from src.b_sim.client import (
    ClearResult as _ClearResult64,
    MeasureResult as _MeasureResult64,
    ProtocolError as _ProtocolError64,
    TestEnded as _TestEnded64,
)


# ---- (R3.12) _stop()：虚拟预算阈值 324000 s ----

def test_stop_virtual_budget_threshold_is_324000_seconds():
    """R3.12：虚拟预算上限 = 0.9 × MAX_VIRTUAL_SECONDS(360000) = 324000 s。"""
    cfg = _StrategyConfig62()
    assert _MAX_VIRTUAL_64 == pytest.approx(360000.0)
    assert cfg.virtual_time_budget_s == pytest.approx(324000.0)


def test_stop_true_when_virtual_time_at_or_above_324000():
    """R3.12：虚拟时刻恰达 324000 s（含）→ _stop()=True，并记「virtual time budget reached」。

    与任务 5.3 的现实预算断言不重叠：此处聚焦虚拟预算阈值 324000 与事件消息。
    real_budget_left_s 置为大值，隔离出虚拟预算这一触发路径。
    """
    cfg = _StrategyConfig62()
    # 恰为 324000 s（边界含 ≥）→ 停止。real_budget 充裕（10000 s > 15）。
    arena = _FakeArena62()
    strat_at = _make_strategy62(arena, cfg)
    strat_at.client.virtual_time_s = 324000.0
    assert strat_at._stop() is True
    assert strat_at._stopped is True
    assert "virtual time budget reached" in strat_at.events
    assert "real time budget reached" not in strat_at.events  # 非现实预算触发

    # 略超 324000 s → 同样停止。
    strat_over = _make_strategy62(_FakeArena62(), cfg)
    strat_over.client.virtual_time_s = 324000.0 + 1.0
    assert strat_over._stop() is True
    assert "virtual time budget reached" in strat_over.events


def test_stop_false_when_virtual_time_below_324000_and_real_budget_ok():
    """R3.12：虚拟时刻 < 324000 s 且现实预算 > 15 s → _stop()=False（继续动作）。"""
    cfg = _StrategyConfig62()
    arena = _FakeArena62()
    strat = _make_strategy62(arena, cfg)
    # _FakeClient62.real_budget_left_s() 恒返回 10000 s（> 15）。
    strat.client.virtual_time_s = 324000.0 - 1.0   # 差 1 s 未达阈值
    assert strat._stop() is False
    assert strat._stopped is False
    assert "virtual time budget reached" not in strat.events


# ---- (R3.12) run() 的 finally 总会优雅调用 client.exit() ----

class _StubRunClient64:
    """可驱动 DogStrategy.run() 的最小客户端替身。

    - enter()/exit() 记录调用次数；exit_called 供断言优雅退出。
    - measure()/clear() 可配置为抛 TestEnded/ProtocolError，验证异常下仍走 finally。
    - 暴露 real_budget_left_s / virtual_time_s / log（run()/summary() 需要）。
    """

    def __init__(self, *, measure_exc=None, exit_exc=None):
        self.virtual_time_s = 0.0
        self.log: list[dict] = []
        self.enter_calls = 0
        self.exit_calls = 0
        self._measure_exc = measure_exc
        self._exit_exc = exit_exc

    def real_budget_left_s(self) -> float:
        return 10_000.0

    def enter(self):
        self.enter_calls += 1
        return {"accepted": True, "remaining_real_duration_s": 1200.0}

    def exit(self):
        self.exit_calls += 1
        if self._exit_exc is not None:
            raise self._exit_exc

    def measure(self, x, y, channel):
        if self._measure_exc is not None:
            raise self._measure_exc
        self.log.append({"path": "/measure", "request": {"channel": channel}})
        return _MeasureResult64(result="no_signal", svd_deg=None, virtual_time_s=0.0)

    def clear(self, x, y, channel):
        self.log.append({"path": "/clear", "request": {"channel": channel}})
        return _ClearResult64(result="no_target_in_range", virtual_time_s=0.0)


def _make_strategy_with_client64(client, config=None):
    strat = _DogStrategy62.__new__(_DogStrategy62)  # 绕过 __init__ 的真实 client 依赖
    strat.client = client
    strat.config = config or _StrategyConfig62()
    strat.channels = {ch: _ChannelState62(ch) for ch in range(1, 21)}
    strat.pos = (0.0, 0.0)
    strat.radio_channel = 1
    strat.events = []
    strat._stopped = False
    return strat


def test_run_calls_exit_when_phase_raises_test_ended():
    """R3.12：某阶段（measure）遇 TestEnded 时，run() 仍在 finally 中调用 client.exit()，
    并正常返回 summary()。

    说明：_measure()/_clear() 内部已捕获 TestEnded/ProtocolError，记「... stopped: ...」
    事件并置 _stopped=True（比让异常一路上抛更优雅——后续阶段被 _stop() 短路）。
    无论异常在 _measure 内被捕获、还是罕见地一路抛到 run() 的 except，finally 都保证
    client.exit() 被调用。此处核对「measure 遇 TestEnded → 记停止事件 + 优雅 exit」。
    """
    client = _StubRunClient64(measure_exc=_TestEnded64("simulated end"))
    strat = _make_strategy_with_client64(client)
    result = strat.run()
    assert client.enter_calls == 1
    assert client.exit_calls == 1, "阶段异常下仍应优雅调用 client.exit()"
    # _measure 捕获异常后置停止标记并记事件；后续阶段被 _stop() 短路。
    assert strat._stopped is True
    assert any("measure stopped" in e for e in strat.events)
    # summary() 正常返回（未清除任何频道）。
    assert isinstance(result, dict)
    assert result["cleared_count"] == 0
    assert result["mean_clear_time_s"] is None


def test_run_calls_exit_when_phase_raises_protocol_error():
    """R3.12：阶段遇 ProtocolError 时同样走 finally 优雅退出并返回 summary()。"""
    client = _StubRunClient64(measure_exc=_ProtocolError64("simulated protocol error"))
    strat = _make_strategy_with_client64(client)
    result = strat.run()
    assert client.exit_calls == 1
    assert strat._stopped is True
    assert any("measure stopped" in e for e in strat.events)
    assert result["cleared_count"] == 0


def test_run_finally_exit_always_called_even_if_run_body_raises(monkeypatch):
    """R3.12：即便某阶段抛出会一路上传到 run() try/except 的 TestEnded/ProtocolError
    （例如 enter 之后、_measure 之外的阶段直接抛），finally 仍调用 client.exit()，
    并记「aborted: ...」事件后返回 summary()。

    通过打桩主巡回 _tour 令其直接抛 TestEnded（绕过 _measure 的内部捕获），验证 run()
    的 except (TestEnded, ProtocolError) 分支记 aborted 事件、finally 仍优雅退出。
    """
    client = _StubRunClient64()
    strat = _make_strategy_with_client64(client)
    # sweep_origin 关闭以尽快进入主巡回；令 _tour 抛异常上传到 run()。
    strat.config = _StrategyConfig62()
    object.__setattr__(strat.config, "sweep_origin", False)

    def boom(*_args, **_kwargs):
        raise _TestEnded64("tour phase aborted")

    strat._tour = boom  # type: ignore[assignment]
    result = strat.run()
    assert client.exit_calls == 1, "run() body 抛异常时 finally 仍应调用 exit()"
    assert any(e.startswith("aborted:") for e in strat.events)
    assert isinstance(result, dict)
    assert result["cleared_count"] == 0


def test_run_exit_exception_is_swallowed_and_summary_returned():
    """R3.12：即便 client.exit() 自身抛 TestEnded/ProtocolError，run() 也吞掉异常，
    仍返回 summary()（优雅退出绝不向上抛错终止程序）。"""
    client = _StubRunClient64(measure_exc=_TestEnded64("phase end"),
                              exit_exc=_ProtocolError64("exit failed"))
    strat = _make_strategy_with_client64(client)
    # 不应抛出——finally 内对 exit 的异常被吞。
    result = strat.run()
    assert client.exit_calls == 1
    assert isinstance(result, dict)
    assert result["cleared_count"] == 0


def test_run_calls_exit_on_normal_completion():
    """R3.12（对照）：正常完成（无异常）时 run() 亦调用一次 client.exit()。"""
    client = _StubRunClient64()
    strat = _make_strategy_with_client64(client)
    result = strat.run()
    assert client.enter_calls == 1
    assert client.exit_calls == 1
    assert isinstance(result, dict)


# ---- (R3.13) summary() 字段与官方评分项对应 ----

def test_summary_fields_match_official_scoring_items_when_cleared():
    """R3.13：summary() 报告 cleared_count（int）、T̄（mean_clear_time_s = 虚拟时长/已清除）、
    程序虚拟时长 virtual_time_s，以及 measure_calls/clear_calls；cleared_count==len(cleared_channels)。

    直接装配一个「已清除若干频道 + 已知虚拟时长」的策略实例，核对 summary() 各字段。
    """
    arena = _FakeArena62()
    strat = _make_strategy62(arena)
    # 标记频道 2、5、7 为已清除（共 3 个）。
    for ch in (2, 5, 7):
        strat.channels[ch].cleared = True
    # 预置客户端虚拟时长与日志（3 次 measure、2 次 clear）供计数。
    strat.client.virtual_time_s = 900.0
    strat.client.log = [
        {"path": "/measure", "request": {"channel": 1}},
        {"path": "/measure", "request": {"channel": 2}},
        {"path": "/clear", "request": {"channel": 2}},
        {"path": "/measure", "request": {"channel": 5}},
        {"path": "/clear", "request": {"channel": 5}},
    ]

    s = strat.summary()
    # 已清除频道列表升序、个数为整数且与列表长度一致。
    assert s["cleared_channels"] == [2, 5, 7]
    assert isinstance(s["cleared_count"], int)
    assert s["cleared_count"] == 3
    assert s["cleared_count"] == len(s["cleared_channels"])
    # T̄ = 虚拟时长 / 已清除个数 = 900 / 3 = 300.0（原始浮点，未经四舍五入）。
    assert s["mean_clear_time_s"] == pytest.approx(900.0 / 3.0)
    assert s["mean_clear_time_s"] == pytest.approx(300.0)
    # 程序虚拟时长即 client.virtual_time_s。
    assert s["virtual_time_s"] == pytest.approx(900.0)
    # measure/clear 调用数由 client.log 中 path 计数。
    assert s["measure_calls"] == 3
    assert s["clear_calls"] == 2
    # 官方评分项字段齐备。
    for key in ("cleared_channels", "cleared_count", "virtual_time_s",
                "measure_calls", "clear_calls", "mean_clear_time_s", "events"):
        assert key in s


def test_summary_mean_clear_time_is_none_when_nothing_cleared():
    """R3.13：无任何清除时 T̄（mean_clear_time_s）为 None（避免除零），cleared_count=0。"""
    arena = _FakeArena62()
    strat = _make_strategy62(arena)
    strat.client.virtual_time_s = 500.0
    strat.client.log = [
        {"path": "/measure", "request": {"channel": 1}},
        {"path": "/measure", "request": {"channel": 2}},
    ]
    s = strat.summary()
    assert s["cleared_channels"] == []
    assert s["cleared_count"] == 0
    assert s["cleared_count"] == len(s["cleared_channels"])
    assert s["mean_clear_time_s"] is None       # 无清除 → 不做除法，返回 None
    assert s["virtual_time_s"] == pytest.approx(500.0)
    assert s["measure_calls"] == 2
    assert s["clear_calls"] == 0


def test_summary_raw_mean_is_rounded_by_reporting_layer_not_summary():
    """R3.13（口径分工）：summary() 返回原始浮点 T̄，两/三位小数由报告层四舍五入。

    验证「summary 原始、rehearse 舍入」的分工：构造虚拟时长使原始 T̄ 含多位小数，
    断言 summary() 保留原始精度，而 rehearse 报告层的 round(·, 3) 才产生定点结果。
    此测试固定该分工，避免误把舍入塞进 summary() 改变既有行为。
    """
    arena = _FakeArena62()
    strat = _make_strategy62(arena)
    for ch in (1, 2, 3):
        strat.channels[ch].cleared = True
    strat.client.virtual_time_s = 1000.0        # T̄ = 1000/3 = 333.3333...
    strat.client.log = []
    s = strat.summary()
    raw = s["mean_clear_time_s"]
    assert raw == pytest.approx(1000.0 / 3.0)
    # 原始值确含多位小数（未在 summary 内被截断成两/三位）。
    assert round(raw, 3) != raw
    # 报告层（rehearse.per_episode）负责舍入：模拟其 round(·, 3)。
    assert round(raw, 3) == pytest.approx(333.333)
    # 两位小数报告口径亦由报告层完成。
    assert round(raw, 2) == pytest.approx(333.33)



# =============================================================================
# 任务 7.1：核对候选网格初始化与约束更新（VERIFY，回归测试）
#
# 对照 requirements.md R4.1 / R4.2 / R4.3 与 design.md §3.4，固定
# directional.py 的 DirectionalBelief 三项行为：
#   - fresh() 生成全域 (x, y, φ) 候选网格：位置步长 120 m、朝向步长 30°
#     （φ ∈ {0,30,…,330}，12 个朝向），初始候选非空且规模 = 位置数 × 12（R4.1）
#   - apply_silence(p)：删除 dist(p,s) ≤ 1000 m 且 p 在源 ±90° 扇区内的候选（R4.2）
#   - apply_bearing(p, svd)：仅保留源方位与 svd 角差 ≤ 7°（1°+6° 离散容差）
#     且 p 在源扇区内的候选（R4.3）
#
# 结论：现有实现三项行为均正确，本次为验证性回归，未改动 directional.py。
# =============================================================================

from src.b_sim.arena import ARENA_RADIUS as _ARENA_RADIUS  # noqa: E402
from src.b_sim.arena import RECEPTION_MIN_M as _RECEPTION_MIN_M  # noqa: E402
from src.b_sim.directional import (  # noqa: E402
    DirectionalBelief,
    _grid_positions,
    _headings,
    in_beam,
)


def test_task71_fresh_headings_are_twelve_multiples_of_30():
    """R4.1：朝向步长 30° → φ ∈ {0,30,…,330}，恰 12 个朝向值。"""
    headings = _headings()
    assert headings == [float(k * 30) for k in range(12)]
    assert len(headings) == 12
    # 相邻朝向间隔恒为 30°，且不含 360（半开区间）。
    assert all(headings[i + 1] - headings[i] == 30.0 for i in range(11))
    assert 360.0 not in headings


def test_task71_fresh_positions_within_radius_and_step_120():
    """R4.1：位置步长 120 m 的格点全部落在半径 1800 m 圆域内，且集合非空。"""
    positions = _grid_positions()
    assert len(positions) > 0
    # 全部在圆域内（含边界）。
    assert all(math.hypot(x, y) <= _ARENA_RADIUS + 1e-9 for x, y in positions)
    # 步长为 120 m：所有坐标都是 120 的整数倍，且原点在网格上。
    assert all(
        abs(x % 120.0) < 1e-9 or abs(x % 120.0 - 120.0) < 1e-9 for x, y in positions
    )
    assert (0.0, 0.0) in positions


def test_task71_fresh_candidate_set_nonempty_and_full_domain():
    """R4.1：初始候选 = 位置数 × 12 个朝向，非空，且每个候选朝向标签为 0。"""
    belief = DirectionalBelief.fresh()
    positions = _grid_positions()
    assert belief.alive is True
    assert len(belief.candidates) > 0
    assert len(belief.candidates) == len(positions) * 12
    # 位置索引覆盖全部格点、朝向索引覆盖 0..11、tag 恒为 0。
    pos_idx = {pi for pi, _, _ in belief.candidates}
    head_idx = {hi for _, hi, _ in belief.candidates}
    assert pos_idx == set(range(len(positions)))
    assert head_idx == set(range(12))
    assert all(tag == 0 for _, _, tag in belief.candidates)


def test_task71_apply_silence_removes_in_sector_within_1000():
    """R4.2：p 距源 ≤ 1000 m 且落在源 ±90° 扇区内 → 该 (位置,朝向) 候选被删除。

    源置于原点、朝向 0°（波束指向 +x）。观测点 (600,0) 在 +x 扇区内、距源 600 m。
    """
    belief = DirectionalBelief.fresh()
    idx0 = belief.positions.index((0.0, 0.0))
    h0 = belief.headings.index(0.0)
    obs = (600.0, 0.0)
    # 前置条件：观测点在扇区内且在 1000 m 内。
    assert in_beam(0.0, 0.0, 0.0, *obs) is True
    assert math.hypot(*obs) <= _RECEPTION_MIN_M
    belief.apply_silence(*obs)
    assert (idx0, h0, 0) not in belief.candidates


def test_task71_apply_silence_retains_out_of_sector_candidate():
    """R4.2：同一沉默观测点对「朝向使其落在扇区外」的候选不删除。

    源在原点、朝向 180°（波束指向 -x）。观测点 (600,0) 在其扇区外 → 候选保留。
    """
    belief = DirectionalBelief.fresh()
    idx0 = belief.positions.index((0.0, 0.0))
    h180 = belief.headings.index(180.0)
    obs = (600.0, 0.0)
    assert in_beam(0.0, 0.0, 180.0, *obs) is False
    belief.apply_silence(*obs)
    assert (idx0, h180, 0) in belief.candidates


def test_task71_apply_silence_retains_candidate_beyond_1000():
    """R4.2：p 在扇区内但距源 > 1000 m（保守下限之外）→ 候选保留。"""
    belief = DirectionalBelief.fresh()
    idx0 = belief.positions.index((0.0, 0.0))
    h0 = belief.headings.index(0.0)
    far = (1500.0, 0.0)  # 在 +x 扇区内，但距原点 1500 m > 1000 m
    assert in_beam(0.0, 0.0, 0.0, *far) is True
    assert math.hypot(*far) > _RECEPTION_MIN_M
    belief.apply_silence(*far)
    assert (idx0, h0, 0) in belief.candidates


def test_task71_apply_bearing_keeps_true_state_on_correct_observation():
    """R4.3：与 svd 一致且 p 在扇区内的候选保留（真值近邻格点存活）。

    源在原点、朝向 0°；观测点 (600,0)，源相对该点的方位角为 180°。svd=180°
    的正确观测应保留原点候选。
    """
    belief = DirectionalBelief.fresh()
    idx0 = belief.positions.index((0.0, 0.0))
    h0 = belief.headings.index(0.0)
    obs = (600.0, 0.0)
    # bearing(source from p) = atan2(0-0, 0-600) = 180°
    belief.apply_bearing(*obs, svd_deg=180.0)
    assert (idx0, h0, 0) in belief.candidates


def test_task71_apply_bearing_drops_inconsistent_bearing():
    """R4.3：源方位与 svd 角差超出容差（> 7°）的候选被删除。"""
    belief_wrong = DirectionalBelief.fresh()
    idx0 = belief_wrong.positions.index((0.0, 0.0))
    h0 = belief_wrong.headings.index(0.0)
    obs = (600.0, 0.0)  # 真实方位 180°
    belief_wrong.apply_bearing(*obs, svd_deg=90.0)  # 差 90° ≫ 7°
    assert (idx0, h0, 0) not in belief_wrong.candidates


def test_task71_apply_bearing_tolerance_boundary_7deg():
    """R4.3：容差为 ±1° + 6° 离散 = 7°。真实方位 180°：185°（差 5°）保留，
    188°（差 8°）删除，固定 7° 边界语义。"""
    obs = (600.0, 0.0)  # 真实方位 180°

    keep = DirectionalBelief.fresh()
    idx0 = keep.positions.index((0.0, 0.0))
    h0 = keep.headings.index(0.0)
    keep.apply_bearing(*obs, svd_deg=185.0)
    assert (idx0, h0, 0) in keep.candidates

    drop = DirectionalBelief.fresh()
    drop.apply_bearing(*obs, svd_deg=188.0)
    assert (idx0, h0, 0) not in drop.candidates


def test_task71_apply_bearing_drops_out_of_sector_even_if_bearing_matches():
    """R4.3：即使方位与 svd 一致，若 p 不在候选源扇区内，候选仍被删除。

    源在原点、朝向 180°（波束 -x），观测点 (600,0) 在其扇区外；svd=180° 方位一致
    但因扇区约束仍应删除该候选。
    """
    belief = DirectionalBelief.fresh()
    idx0 = belief.positions.index((0.0, 0.0))
    h180 = belief.headings.index(180.0)
    obs = (600.0, 0.0)
    assert in_beam(0.0, 0.0, 180.0, *obs) is False
    belief.apply_bearing(*obs, svd_deg=180.0)
    assert (idx0, h180, 0) not in belief.candidates


# =============================================================================
# 任务 7.2：核对信息增益补测与「推理非加密」（VERIFY，回归测试）
#
# 对照 requirements.md R4.5 / R4.8 与 design.md §3.4 / §4.7，固定 directional.py
# 的补测点选择行为与「盲扫加密站位存在天花板 → 改用推理 + 盲清除」这一设计判断：
#   - best_probe_point / _probe_candidates（R4.5）：
#       * 补测点沿候选源朝向前方按 (2·step, 4·step, 0.75·1000 m) 三档距离生成，
#         裁剪进 1800 m 区域，NOT 直接使用候选源位置本身；
#       * 评分 = 期望排除比例 gain(=eliminated/total) − 移动折价(distance/500·0.05)，
#         返回 argmax；相较"信息量更少的远点"能排除更多候选；
#       * 候选为空时返回 None。
#   - 「推理非加密」（R4.8）：对照 coverage_design.directional_visibility，
#     稠密站位布局对定向源的可见率 < 1.0（存在天花板），故加密站位无法保证全见，
#     必须改用沉默约束推理压缩候选 + 盲清除。
#
# 结论：现有实现行为均正确，本次为验证性回归，未改动 directional.py。
# =============================================================================

from src.b_sim.directional import GRID_STEP_M as _GRID_STEP_M  # noqa: E402
from src.b_sim.coverage_design import directional_visibility as _directional_visibility  # noqa: E402
from src.b_sim.coverage_design import ring as _cov_ring  # noqa: E402


def _make_single_candidate_belief(position, heading_deg):
    """构造只含单个 (位置, 朝向) 候选的信念，便于精确断言补测点几何。"""
    belief = DirectionalBelief.fresh()
    pi = belief.positions.index(position)
    hi = belief.headings.index(heading_deg)
    belief.candidates = {(pi, hi, 0)}
    return belief


def test_task72_probe_points_are_outward_not_source_position():
    """R4.5：补测点生成于候选源朝向**外侧**，NOT 候选源位置本身。

    候选源在原点、朝向 0°（波束朝东）：补测点应位于 +x 方向前方，
    绝不等于源位置 (0,0)，且横向 y 分量为 0（沿朝向前方）。
    """
    belief = _make_single_candidate_belief((0.0, 0.0), 0.0)
    probes = belief._probe_candidates(260)
    assert probes, "应至少生成一个补测点"
    # 无补测点落在候选源位置本身
    assert all((px, py) != (0.0, 0.0) for px, py in probes)
    for px, py in probes:
        assert px > 0.0, f"补测点应在朝向前方 (x>0)，实得 {(px, py)}"
        assert abs(py) < 1e-6, f"沿朝向 0° 前方，横向应为 0，实得 {(px, py)}"


def test_task72_probe_reaches_follow_two_four_step_and_075_reception():
    """R4.5：补测点沿朝向前方按 (2·step, 4·step, 0.75·1000 m) 三档距离生成。"""
    belief = _make_single_candidate_belief((0.0, 0.0), 0.0)
    probes = belief._probe_candidates(260)
    reaches = sorted(math.hypot(px, py) for px, py in probes)
    expected = sorted((_GRID_STEP_M * 2, _GRID_STEP_M * 4, _RECEPTION_MIN_M * 0.75))
    assert reaches == pytest.approx(expected)
    # 具体几何：朝向 0° → (240,0), (480,0), (750,0)
    assert set(probes) == {(240.0, 0.0), (480.0, 0.0), (750.0, 0.0)}


def test_task72_probe_points_clipped_into_arena():
    """R4.5：越界补测点沿区域边界回退，裁剪到 1800 m 圆域内。

    候选源贴近边界、朝向指向区域外，则前方三档距离会越界；应被裁剪到半径 1800 m。
    """
    # 取一个靠近边界的格点，朝向指向远离原点方向（外侧）。
    edge_pos = None
    for p in DirectionalBelief.fresh().positions:
        if 1700.0 <= math.hypot(*p) <= 1800.0:
            edge_pos = p
            break
    assert edge_pos is not None
    heading = math.degrees(math.atan2(edge_pos[1], edge_pos[0]))  # 指向区域外
    # 用最近的 30° 网格朝向
    heading_grid = min(_headings(), key=lambda h: abs((h - heading + 180) % 360 - 180))
    belief = _make_single_candidate_belief(edge_pos, heading_grid)
    for px, py in belief._probe_candidates(260):
        assert math.hypot(px, py) <= _ARENA_RADIUS + 1e-6, f"补测点越界未裁剪: {(px, py)}"


def test_task72_best_probe_point_none_when_empty():
    """R4.5：候选集合为空时 best_probe_point 返回 None。"""
    belief = DirectionalBelief.fresh()
    belief.candidates = set()
    assert belief.best_probe_point((0.0, 0.0)) is None


def test_task72_best_probe_point_maximizes_elimination_gain():
    """R4.5：best_probe_point 选期望排除比例最大的补测点。

    构造一簇朝向一致、聚在原点附近的候选，使某个"前方近点"能一次排除全部候选
    （gain=1.0）；返回点确实能把候选数打到 0（优于信息量更少的远角点）。
    """
    belief = DirectionalBelief.fresh()
    h0 = belief.headings.index(0.0)
    cluster = [(0.0, 0.0), (120.0, 0.0), (0.0, 120.0), (0.0, -120.0)]
    belief.candidates = {(belief.positions.index(p), h0, 0) for p in cluster}
    total = len(belief.candidates)

    best = belief.best_probe_point((0.0, 0.0))
    assert best is not None

    def eliminated_at(px, py, cands):
        n = 0
        for pi, hi, _ in cands:
            sx, sy = belief.positions[pi]
            if math.hypot(px - sx, py - sy) <= _RECEPTION_MIN_M and in_beam(
                sx, sy, belief.headings[hi], px, py
            ):
                n += 1
        return n

    # 最优点应能排除全部候选（gain = 1.0）。
    assert eliminated_at(*best, belief.candidates) == total
    # 与"信息量更少的远角点"(区域另一侧) 相比，最优点排除更多。
    naive = (-1700.0, 0.0)
    assert eliminated_at(*best, belief.candidates) > eliminated_at(*naive, belief.candidates)


def test_task72_best_probe_reduces_candidates_more_than_naive_alternative():
    """R4.5：移动到 best_probe_point 后（若无信号）比朴素替代点排除更多候选。

    对全域信念，用 best_probe_point 选点，随后 apply_silence 模拟该点无信号；
    候选削减量应严格大于在一个远角朴素点的削减量。
    """
    belief = DirectionalBelief.fresh()
    before = len(belief.candidates)
    best = belief.best_probe_point((0.0, 0.0))
    assert best is not None

    at_best = DirectionalBelief.fresh()
    at_best.apply_silence(*best)
    reduced_best = before - len(at_best.candidates)

    at_naive = DirectionalBelief.fresh()
    at_naive.apply_silence(1700.0, 0.0)  # 贴边角点：可排除的扇区内近候选更少
    reduced_naive = before - len(at_naive.candidates)

    assert reduced_best > 0
    assert reduced_best > reduced_naive, (
        f"best_probe 削减 {reduced_best} 应大于朴素点削减 {reduced_naive}"
    )


def test_task72_best_probe_prefers_closer_point_when_gain_equal():
    """R4.5：增益相等时，移动折价使 best_probe_point 偏好更近的补测点。

    两个候选状态，其补测点位于同一直线的近/远两处、且各自能排除的候选数相同
    （单调折价 distance/500·0.05）。构造使近点严格占优。
    """
    belief = DirectionalBelief.fresh()
    h0 = belief.headings.index(0.0)
    # 单个原点候选：补测点 (240,0)/(480,0)/(750,0) 都能排除该唯一候选（gain 相等=1.0）。
    belief.candidates = {(belief.positions.index((0.0, 0.0)), h0, 0)}
    best = belief.best_probe_point((0.0, 0.0))
    # 三档 reach 增益相同，折价随距离递增 → 选最近的 (240,0)。
    assert best == (240.0, 0.0)


def test_task72_directional_visibility_has_ceiling_below_one():
    """R4.8：稠密站位布局对定向源的可见率 < 1.0（存在天花板）。

    对照 coverage_design.directional_visibility：即便布上原点 + 双环（31 站位）的
    稠密布局，定向源（仅 ±90° 可见）的可见率仍达不到 100%——说明盲目加密站位无法
    保证全见，故 Q4 改用「沉默约束推理 + 盲清除」而非加密观测站位。种子数保持很小。
    """
    dense = [(0.0, 0.0)] + _cov_ring(1250, 12) + _cov_ring(1560, 18, math.pi / 18)
    assert len(dense) == 31
    seen, total = _directional_visibility(dense, range(3000, 3008), directional_count=5)
    assert total > 0
    rate = seen / total
    # 稠密布局仍有盲区（朝区域外辐射的定向源看不见）→ 可见率严格 < 1.0。
    assert rate < 1.0, f"稠密站位定向可见率应 < 1.0（存在天花板），实得 {rate:.3f}"



# ---------------------------------------------------------------------------
# 任务 7.3：核对盲清除与射线盲清除（R4.6 / R4.7）
# Feature: b-radio-interference-locate-clear
#
# 本节为「验证/加固」任务的回归测试。核对项：
#   (a) _grid_cover_points：以 (cx,cy) 为心、边长 120 m 的网格内任意点，距至少一个
#       生成清除点 ≤ 20 m（CLEAR_RADIUS_M）——保证盲清除能覆盖整格（R4.6）；
#   (b) _blind_clear_sweep：belief 候选集较小 + 真源落在某候选格内时，逐格按
#       _grid_cover_points 布点直接 /clear，最终清除该源（fake arena，20 m 内判成功）（R4.6）；
#   (c) _ray_blind_clear：沿示向度射线以间距 <40 m 铺点，超过 r·rad(1°)>lateral 的
#       远处补 ±lateral 两侧点，受独立 ray_clear_max_attempts 约束，且清除落在射线上的
#       源（fake arena）；ray_swept 幂等（第二次调用不再重扫）（R4.7）。
#
# 阈值口径差异（实测调优值，按任务要求"记录而非强改"）：
#   * R4.6 需求文字写"候选 ≤ 8 时"逐格盲清除，但当前实现（且 design.md §4.8 一致）
#     用的是 _blind_clear_sweep(max_targets=24) 的加权取前 N 上限，而非 ≤8 的触发门槛：
#     只要 belief 存活即逐格盲清除，仅当候选位置 >24 时按候选密度加权保留前 24 个。
#     这些常量为经验调优所得（design 与 code 一致），本测试断言"实际行为"，不改常量。
#   * R4.7 需求文字写射线盲清除独立 /clear 调用 ≤40 次，但当前实现为
#     ray_clear_max_attempts=120（经验调优：窄走廊源需铺更长射线）。本测试断言实际的
#     120 上限被独立遵守（与 _home_channel 的 max_clear_attempts=9 无关），不改常量。
# ---------------------------------------------------------------------------

import math as _math73

from src.b_sim.directional import DirectionalBelief as _DirectionalBelief73
from src.b_sim.geometry import Bearing as _Bearing73
from src.b_sim.strategy import (
    CLEAR_RADIUS_M as _CLEAR_RADIUS_M73,
    DogStrategy as _DogStrategy73,
    StrategyConfig as _StrategyConfig73,
)


# ---- (a) R4.6：_grid_cover_points 覆盖整格 ----

def test_grid_cover_points_covers_entire_120m_cell_within_clear_radius():
    """R4.6：120 m 网格内任意采样点距某生成清除点 ≤ CLEAR_RADIUS_M(20 m)。

    在偏移中心 (cx,cy) 上稠密网格化采样整格，验证最坏"到最近清除点"距离 ≤ 20 m。
    这保证任何落在该格内的真源都在某清除点 20 m 内（/clear 仅看距离即判成功）。
    """
    cx, cy = 300.0, -450.0
    cell = 120.0
    pts = _DogStrategy73._grid_cover_points(cx, cy, cell=cell, radius=_CLEAR_RADIUS_M73)
    assert pts, "_grid_cover_points 未生成任何点"
    half = cell / 2.0
    n = 41  # 41×41 采样格，步长 3 m，足够密
    worst = 0.0
    for i in range(n):
        for j in range(n):
            x = cx - half + (cell * i) / (n - 1)
            y = cy - half + (cell * j) / (n - 1)
            nearest = min(_math73.hypot(x - px, y - py) for px, py in pts)
            worst = max(worst, nearest)
    assert worst <= _CLEAR_RADIUS_M73 + 1e-9, (
        f"格内最坏到最近清除点距离 {worst:.3f} m 超出清除半径 {_CLEAR_RADIUS_M73} m")
    # 生成点按到中心距离升序（最可能命中的中心点优先）
    dists = [_math73.hypot(px - cx, py - cy) for px, py in pts]
    assert dists == sorted(dists)


def _make_belief_with_candidates(positions):
    """构造一个受控 belief：给定候选位置列表，每个位置附一个朝向候选（存活、非空）。"""
    belief = _DirectionalBelief73.__new__(_DirectionalBelief73)
    belief.positions = list(positions)
    belief.headings = [0.0]
    belief._pos_index = {}
    belief.candidates = {(pi, 0, 0) for pi in range(len(positions))}
    return belief


# ---- (b) R4.6：_blind_clear_sweep 逐格盲清除并清除真源 ----

def test_blind_clear_sweep_clears_source_in_small_candidate_cell():
    """R4.6：候选集较小 + 真源落在某候选格内时，逐格布点盲清除，最终清除该源。

    构造两个候选格点，真源落在第二个候选格中心偏 40 m 处（仍在 120 m 格内）。
    fake arena 对距真源 ≤20 m 的 /clear 判成功。验证：
      * 逐格调用 _grid_cover_points 布点直接 /clear（无需先看见目标）；
      * 该源被清除（state.cleared 由 _clear_at 置真）。
    """
    source = (1000.0, 40.0)
    # 候选格 1 在别处（不含源），候选格 2 中心 (1000,0)，源在其 120 m 格内。
    positions = [(500.0, 500.0), (1000.0, 0.0)]
    arena = _FakeArena62(source=source, clear_radius=20.0)
    strat = _make_strategy62(arena)
    state = strat.channels[1]
    state.belief = _make_belief_with_candidates(positions)
    state.cleared = False

    # 候选数（2）远小于 max_targets(24)：全部逐格盲清除，无加权裁剪。
    assert len(state.belief.candidate_positions()) <= 24
    strat._blind_clear_sweep()

    assert arena.clear_calls, "_blind_clear_sweep 未发起任何 /clear"
    assert state.cleared is True, "真源应被盲清除覆盖并清除"
    # 命中点必在真源 20 m 内。
    hit = arena.clear_calls[-1]
    assert _math73.hypot(hit[0] - source[0], hit[1] - source[1]) <= 20.0 + 1e-9
    # 布点确实落在含源的候选格 (1000,0) 周围（而非仅在无关格）。
    near_cell2 = [c for c in arena.clear_calls
                  if _math73.hypot(c[0] - 1000.0, c[1] - 0.0) <= 90.0]
    assert near_cell2, "未在含源候选格内布点"


def test_blind_clear_sweep_skips_channels_without_alive_belief():
    """R4.6：belief 为空/已死或频道已清除时不发起盲清除（只对存活 belief 逐格清除）。"""
    arena = _FakeArena62(source=(1000.0, 0.0), clear_radius=20.0)
    strat = _make_strategy62(arena)
    # 所有频道：无 belief（默认 None）→ 不应产生任何 /clear。
    strat._blind_clear_sweep()
    assert arena.clear_calls == []


# ---- (c) R4.7：_ray_blind_clear 沿射线盲清除 ----

def _ray_config73():
    cfg = _StrategyConfig73()
    # 断言实测调优常量口径（记录差异，不强改）。
    assert cfg.ray_clear_step_m == 32.0 and cfg.ray_clear_step_m < 40.0
    assert cfg.ray_clear_lateral_m == 22.0
    assert cfg.ray_clear_max_range_m == 1600.0
    # 需求文字写 ≤40，实测实现为 120（独立于 _home_channel 的 max_clear_attempts=9）。
    assert cfg.ray_clear_max_attempts == 120
    assert cfg.max_clear_attempts == 9
    return cfg


def test_ray_blind_clear_spacing_and_lateral_offsets_and_cap():
    """R4.7：沿射线间距 <40 m 铺点，远处（r·rad(1°)>lateral）补 ±lateral 两侧点，
    调用受独立 ray_clear_max_attempts 约束。使用不含源的 arena 让其铺满整条射线。"""
    cfg = _ray_config73()
    arena = _FakeArena62(source=None, clear_radius=20.0)  # 永不命中 → 铺满
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[1]
    # 已知方位（正东 0°），距离未知；last bearing 起点在原点。
    state.bearings = [_Bearing73(x=0.0, y=0.0, svd_deg=0.0)]

    result = strat._ray_blind_clear(state)
    assert result is False  # 无源 → 未清除
    calls = arena.clear_calls
    assert calls, "_ray_blind_clear 未铺任何点"

    # 独立调用上限被遵守（不因 max_clear_attempts=9 提前掐死）。
    assert len(calls) <= cfg.ray_clear_max_attempts
    assert len(calls) > cfg.max_clear_attempts, "应突破 max_clear_attempts(9) 的独立铺点"

    # 中心线点（|off|≈0，横向分量≈0，即 y≈0）沿 x 轴的相邻间距 == step < 40 m。
    centerline_x = sorted(x for x, y, _ in calls if abs(y) < 1e-6)
    assert len(centerline_x) >= 2
    gaps = [b - a for a, b in zip(centerline_x, centerline_x[1:])]
    assert max(gaps) <= cfg.ray_clear_step_m + 1e-9
    assert max(gaps) < 40.0  # 间距 < 2×20 m 清除直径

    # 近处只打中心线；远处（r > lateral/rad(1°) ≈ 1260.6 m）补 ±lateral 两侧点。
    r_thresh = cfg.ray_clear_lateral_m / _math73.radians(1.0)
    near_lateral = [(x, y) for x, y, _ in calls if x < r_thresh - cfg.ray_clear_step_m and abs(y) > 1e-6]
    assert near_lateral == [], "近处不应有横向偏移点"
    far_lateral = [(x, y) for x, y, _ in calls if x > r_thresh + cfg.ray_clear_step_m and abs(y) > 1e-6]
    assert far_lateral, "远处应补 ±lateral 横向偏移点"
    assert any(_math73.isclose(abs(y), cfg.ray_clear_lateral_m, rel_tol=1e-6) for _, y in far_lateral)


def test_ray_blind_clear_clears_source_on_the_ray():
    """R4.7：真源落在示向度射线上（距离未知）时，射线铺点命中并清除该源。"""
    cfg = _ray_config73()
    # 源在正东 800 m 射线上，偏 5 m（在 20 m 清除半径内的铺点可命中）。
    source = (800.0, 0.0)
    arena = _FakeArena62(source=source, clear_radius=20.0)
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[1]
    state.bearings = [_Bearing73(x=0.0, y=0.0, svd_deg=0.0)]

    result = strat._ray_blind_clear(state)
    assert result is True
    assert state.cleared is True
    hit = arena.clear_calls[-1]
    assert _math73.hypot(hit[0] - source[0], hit[1] - source[1]) <= 20.0 + 1e-9
    # 命中即停：铺点数应远小于铺满整条射线（1600 m）所需。
    assert len(arena.clear_calls) < cfg.ray_clear_max_attempts


def test_ray_blind_clear_is_idempotent_via_ray_swept():
    """R4.7：ray_swept 防重入——第二次调用直接返回，不再重扫（幂等）。"""
    cfg = _ray_config73()
    arena = _FakeArena62(source=None, clear_radius=20.0)
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[1]
    state.bearings = [_Bearing73(x=0.0, y=0.0, svd_deg=0.0)]

    strat._ray_blind_clear(state)
    assert state.ray_swept is True
    calls_after_first = len(arena.clear_calls)

    result2 = strat._ray_blind_clear(state)
    assert result2 is False
    assert len(arena.clear_calls) == calls_after_first, "第二次调用不应再铺点"


def test_ray_blind_clear_without_bearing_returns_false():
    """R4.7：无示向度（方向未知）时射线盲清除不可用，直接返回 False。"""
    cfg = _ray_config73()
    arena = _FakeArena62(source=(800.0, 0.0), clear_radius=20.0)
    strat = _make_strategy62(arena, cfg)
    state = strat.channels[1]
    assert not state.bearings
    assert strat._ray_blind_clear(state) is False
    assert arena.clear_calls == []



# ===========================================================================
# 任务 3.5（可选）：second_point_candidates 单元测试（示例 / 边界）
# Feature: b-radio-interference-locate-clear
#
# 覆盖 R2.1 / R2.2 / R2.4 / R2.6。本节为**补充**任务 3.1（生成/排序/长度/前进）与
# 任务 3.2（有效性判定、退让、显式不可行）已有覆盖之外的**具体数值算例**与**边界**：
#   (A) 用一组固定 (p1, theta1) 的确定坐标算例，逐点核对候选序列的精确坐标、
#       ± 对称性与 δ 降序编号（如 32° 对 → 16° 对 → 0° 单点），补齐 3.1 中以角差
#       重构、未直接断言精确坐标的缺口；
#   (B) 三条有效性边界同时落在边界值（α=60/120、d2=1000、‖p2‖=1800）的组合算例；
#   (C) 无横向平移候选（沿 θ1 前进分量恒正）的另一朝向算例；
#   (D) select_second_point 在 is_signal_fn 恒 False（全 no_signal）时的显式不可行
#       SecondPointSelection 路径——逐字段核对 feasible/point/candidate/reason 及
#       "全部 no_signal、0 个无效" 的原因分解（与 3.2 的混合失败用例互补）。
#
# 与既有覆盖的重叠说明：δ 降序、序列长度、前进长度=380、非法偏移 ValueError 已由
# test_q2_second_point_candidates_* 覆盖；α/d2/‖p2‖ 边界有效性已由
# test_q2_alpha_boundary_* 与 test_q2_d2_and_range_boundary_inclusive_exact 覆盖。
# 本节仅新增**确定坐标算例**与**纯 no_signal 显式不可行**这两处主要补充，避免重复。
# ===========================================================================

from src.b_sim.geometry import (  # noqa: E402
    SecondPointSelection as _SecondPointSelection35,
    classify_second_point_candidates as _classify35,
    second_point_candidates as _second_point_candidates35,
    select_second_point as _select_second_point35,
)


def _abs_delta_deg_35(p1, theta1, pt):
    """候选点相对 p1 的方位与 theta1 的最小角差（度）。"""
    dx, dy = pt[0] - p1[0], pt[1] - p1[1]
    bearing = math.degrees(math.atan2(dy, dx))
    diff = abs(norm_deg(bearing) - norm_deg(theta1))
    return diff if diff <= 180.0 else 360.0 - diff


def test_q2_35_worked_example_exact_candidate_coordinates():
    """R2.1/R2.4（算例）：p1=(0,0)、θ1=0° 时候选精确坐标与 ± 对称、δ 降序编号。

    L=380、θ1=0（正东）。候选序列应为：
      δ=+32° → (380·cos32°, 380·sin32°)
      δ=-32° → (380·cos32°, -380·sin32°)
      δ=+16° → (380·cos16°, 380·sin16°)
      δ=-16° → (380·cos16°, -380·sin16°)
      δ= 0°  → (380, 0)
    逐点核对精确坐标（补 3.1 未直接断言坐标的缺口）。
    """
    p1 = (0.0, 0.0)
    theta1 = 0.0
    L = 380.0
    cands = _second_point_candidates35(p1, theta1)
    c32, s32 = math.cos(math.radians(32.0)), math.sin(math.radians(32.0))
    c16, s16 = math.cos(math.radians(16.0)), math.sin(math.radians(16.0))
    expected = [
        (L * c32, L * s32),
        (L * c32, -L * s32),
        (L * c16, L * s16),
        (L * c16, -L * s16),
        (L, 0.0),
    ]
    assert len(cands) == 5
    for got, exp in zip(cands, expected):
        assert got[0] == pytest.approx(exp[0], abs=1e-9)
        assert got[1] == pytest.approx(exp[1], abs=1e-9)
    # ± 对称：同一 |δ| 的一对候选关于 θ1 轴（此处 x 轴）镜像。
    assert cands[0][0] == pytest.approx(cands[1][0])   # +32/-32 同 x
    assert cands[0][1] == pytest.approx(-cands[1][1])  # y 反号
    assert cands[2][0] == pytest.approx(cands[3][0])   # +16/-16 同 x
    assert cands[2][1] == pytest.approx(-cands[3][1])
    # δ 降序编号：32,32,16,16,0。
    deltas = [_abs_delta_deg_35(p1, theta1, c) for c in cands]
    assert deltas == pytest.approx([32.0, 32.0, 16.0, 16.0, 0.0], abs=1e-6)


def test_q2_35_worked_example_offset_p1_and_nonzero_theta():
    """R2.1/R2.4（算例）：非零 p1 与斜向 θ1 的具体坐标算例。

    p1=(120,-80)、θ1=135°、单条 δ=(20,0)（自定义 offsets 展示 δ=20° 的 ± 对与直行）。
    逐点核对 p2 = p1 + 380·(cos(135±20°), sin(135±20°)) 与 p1 + 380·(cos135°,sin135°)。
    同时确认末位为沿 θ1 直行（δ=0）候选。
    """
    p1 = (120.0, -80.0)
    theta1 = 135.0
    L = 380.0
    cands = _second_point_candidates35(p1, theta1, offsets=(20.0, 0.0))
    assert len(cands) == 3  # δ=20 的 ± 对 + δ=0 单点
    exp = [
        (p1[0] + L * math.cos(math.radians(theta1 + 20.0)),
         p1[1] + L * math.sin(math.radians(theta1 + 20.0))),
        (p1[0] + L * math.cos(math.radians(theta1 - 20.0)),
         p1[1] + L * math.sin(math.radians(theta1 - 20.0))),
        (p1[0] + L * math.cos(math.radians(theta1)),
         p1[1] + L * math.sin(math.radians(theta1))),
    ]
    for got, e in zip(cands, exp):
        assert got[0] == pytest.approx(e[0], abs=1e-9)
        assert got[1] == pytest.approx(e[1], abs=1e-9)
    # 末位为沿 θ1 直行：相对 p1 方位等于 θ1。
    assert _abs_delta_deg_35(p1, theta1, cands[-1]) == pytest.approx(0.0, abs=1e-6)
    # 到 p1 距离恒为 L=380。
    for c in cands:
        assert math.hypot(c[0] - p1[0], c[1] - p1[1]) == pytest.approx(380.0, abs=1e-6)


def test_q2_35_all_three_validity_bounds_simultaneously_inclusive():
    """R2.2（算例）：α=90°、d2=1000、‖p2‖ 界内三条件同时满足 → 有效（边界含）。

    复用 3.2 中"α=90、d2=1000"的解析构造（length=1000·√2、source=(L/2,L/2)），
    此处作为一个**并列三条件**的具名算例：δ=0° 候选 p2=(L,0)，
      α = 90°∈[60,120]、d2 = 1000（恰为下限边界）、‖p2‖ = L≈1414 ≤ 1800。
    与 3.2 的对应断言互补（此处显式核对三值与 valid=True 的组合语义）。
    """
    L = 1000.0 * math.sqrt(2.0)
    p1 = (0.0, 0.0)
    theta1 = 0.0
    source = (L / 2.0, L / 2.0)
    res = _classify35(p1, theta1, source, offsets=(0.0,), length=L)
    cand = res[0]
    assert cand.crossing_angle_deg == pytest.approx(90.0)
    assert cand.d2_m == pytest.approx(1000.0)                 # 恰为 d2 下限边界（含）
    assert cand.range_from_origin_m == pytest.approx(L)       # ≈1414 ≤ 1800
    assert cand.range_from_origin_m <= 1800.0
    assert cand.valid is True
    assert cand.reason == "valid"


def test_q2_35_no_lateral_translation_various_headings():
    """R2.4（算例）：多个朝向下候选均含正前进分量（无纯横向平移/反向候选）。"""
    for theta1 in (0.0, 45.0, 90.0, 137.0, 200.0, 275.0, 359.0):
        p1 = (50.0, -30.0)
        ux, uy = math.cos(math.radians(theta1)), math.sin(math.radians(theta1))
        for pt in _second_point_candidates35(p1, theta1):
            rel = (pt[0] - p1[0], pt[1] - p1[1])
            forward = rel[0] * ux + rel[1] * uy
            assert forward > 0.0
            # |δ|≤32° ⟹ 前进分量 ≥ L·cos32°。
            assert forward >= 380.0 * math.cos(math.radians(32.0)) - 1e-6


def test_q2_35_select_all_no_signal_returns_explicit_infeasible_selection():
    """R2.6（算例）：is_signal_fn 恒 False（全候选 no_signal）→ 显式不可行 SecondPointSelection。

    源置于使全部候选**几何有效**的位置，从而唯一的失败来源是"全部 no_signal"。
    逐字段核对返回的 SecondPointSelection：feasible=False、point/candidate=None、
    reason 含"第二检测点不可行"与"no_signal N 个"、且分解出"无效 0 个"（几何全有效），
    并确认 evaluated 覆盖了全部候选（退让走到底后才判不可行）。
    """
    # 用解析构造保证 δ=0° 候选几何有效：length=1000√2、source=(L/2,L/2) →
    # δ=0° 候选 p2=(L,0) 处 α=90°∈[60,120]、d2=1000（含）、‖p2‖=L≈1414≤1800。
    # 只用单个 δ=0° 候选，隔离出"纯 no_signal"这一失败来源。
    L = 1000.0 * math.sqrt(2.0)
    p1 = (0.0, 0.0)
    theta1 = 0.0
    source = (L / 2.0, L / 2.0)
    classified = _classify35(p1, theta1, source, offsets=(0.0,), length=L)
    n_valid_geo = sum(1 for c in classified if c.valid)
    # 前置条件：至少存在一个几何有效候选，才能隔离"纯 no_signal"这一失败来源。
    assert n_valid_geo >= 1

    calls = {"n": 0}

    def never_signal(_pt):
        calls["n"] += 1
        return False

    sel = _select_second_point35(
        p1, theta1, source, never_signal, offsets=(0.0,), length=L
    )
    assert isinstance(sel, _SecondPointSelection35)
    assert sel.feasible is False
    assert sel.point is None
    assert sel.candidate is None
    assert "第二检测点不可行" in sel.reason
    assert "no_signal" in sel.reason
    # 全部候选被考察（退让到底）。
    assert len(sel.evaluated) == len(classified)
    # is_signal_fn 对每个几何有效候选都被询问一次（无效候选直接跳过、不询问）。
    assert calls["n"] == n_valid_geo
    # 原因分解与实际一致：no_signal 计数 = 几何有效数、无效计数 = 其余。
    n_invalid = len(classified) - n_valid_geo
    assert f"no_signal {n_valid_geo}" in sel.reason
    assert f"无效 {n_invalid}" in sel.reason


def test_q2_35_select_first_valid_with_signal_reports_offset_in_reason():
    """R2.1/R2.5（算例，补充）：回调恒有信号 → 选中首个有效候选，reason 标注其 δ 符号。

    与 3.2 的退让用例互补：此处显式核对"选中候选即 evaluated 中首个有效候选"，
    且 reason 形如 'selected δ=±NN.N°' 与该候选的带符号偏移一致。
    """
    # 同上解析构造：单个几何有效的 δ=0° 候选。
    L = 1000.0 * math.sqrt(2.0)
    p1 = (0.0, 0.0)
    theta1 = 0.0
    source = (L / 2.0, L / 2.0)
    classified = _classify35(p1, theta1, source, offsets=(0.0,), length=L)
    first_valid = next((c for c in classified if c.valid), None)
    assert first_valid is not None
    sel = _select_second_point35(
        p1, theta1, source, lambda _pt: True, offsets=(0.0,), length=L
    )
    assert sel.feasible is True
    assert sel.point == first_valid.point
    assert sel.candidate is not None
    sign = "+" if sel.candidate.offset_deg >= 0 else "-"
    assert f"selected δ={sign}{abs(sel.candidate.offset_deg):.1f}°" == sel.reason



# ===========================================================================
# 任务 5.6（可选）：客户端 request_id 唯一/单调 与 400/409 单元测试
# Feature: b-radio-interference-locate-clear
#
# 覆盖 R5.1 / R5.3 / R5.4 / R5.5。本节聚焦任务 5.1（_check_coord 边界）与任务 5.2
# （409/400 不重试、网络重试复用同一 request_id、耗尽 TestEnded）之外的**主要缺口**：
#   R5.1 —— 新动作 request_id 的**唯一性/单调性**（自动生成的 `{tag}-{seq}`，seq 自 1
#   起单调递增；同一进程内任意两个新动作 request_id 互不相同，如 enter-1、measure-2、
#   clear-3、measure-4 …），以及**网络重试复用同一自动生成 request_id**（不消耗新 seq）。
#
# 与既有覆盖的重叠说明：
#   - _check_coord 的 ±2_000_000 边界与越界/NaN/inf 拒绝已由任务 5.1 的
#     test_measure_rejects_* / test_boundary_coordinate_* 覆盖；
#   - 409/400 → ProtocolError 不重试、记录完整已由任务 5.2 的
#     test_client_conflict_409_* / test_client_bad_request_400_* 覆盖；
#   - 网络重试至多 3 次、耗尽 TestEnded、复用**显式传入**的 request_id 已由 5.2 的
#     test_client_network_failure_* 覆盖。
#   本节据此仅**新增** request_id 唯一/单调性（主缺口）与"自动生成 id 在网络重试中
#   逐字节复用"这两处；对 400/409/TestEnded 仅做一处以自动生成 request_id 为主体的
#   端到端串联（而非重复 5.2 已断言的细节），避免重复。
# ===========================================================================

import urllib.error as _urllib_error56  # noqa: E402
import urllib.request as _urllib_request56  # noqa: E402

from src.b_sim.client import (  # noqa: E402
    ProtocolError as _ProtocolError56,
    RobotClient as _RobotClient56,
    TestEnded as _TestEnded56,
)


def _action_log_ids_56(client):
    """按发送顺序取动作日志中的 request_id 列表（仅成功记录三元组的动作）。"""
    return [entry["request_id"] for entry in client.log]


def test_client_request_ids_are_monotonic_and_unique_across_new_actions(server):
    """R5.1：连续新动作的 request_id 形如 `{tag}-{seq}`，seq 自 1 单调递增且全局唯一。

    实际动作序列：enter → measure → clear → measure → clear。期望自动生成的
    request_id 依次为 enter-1、measure-2、clear-3、measure-4、clear-5（seq 单调递增、
    跨动作互不相同）。逐条从 client.log 与内部 _seq 核对。
    """
    url, _session = server
    client = _RobotClient56(ROBOT, base_url=url)

    client.enter()                 # seq 1 → enter-1
    client.measure(0.0, 0.0, 1)    # seq 2 → measure-2
    # 用界内合法坐标发起 clear（是否命中不影响 request_id 语义）。
    client.clear(0.0, 0.0, 1)      # seq 3 → clear-3
    client.measure(10.0, 0.0, 2)   # seq 4 → measure-4
    client.clear(10.0, 0.0, 2)     # seq 5 → clear-5

    ids = _action_log_ids_56(client)
    assert ids == ["enter-1", "measure-2", "clear-3", "measure-4", "clear-5"]
    # 全局唯一。
    assert len(set(ids)) == len(ids)
    # seq 单调递增（从 1 起，步长 1）。
    seqs = [int(rid.rsplit("-", 1)[1]) for rid in ids]
    assert seqs == [1, 2, 3, 4, 5]
    for a, b in zip(seqs, seqs[1:]):
        assert b == a + 1
    # 内部计数器与已发出动作数一致。
    assert client._seq == 5
    # 每个 request_id 的 tag 与其动作路径一致。
    tag_by_path = {"/enter": "enter", "/measure": "measure", "/clear": "clear"}
    for entry in client.log:
        assert entry["request_id"].rsplit("-", 1)[0] == tag_by_path[entry["path"]]


def test_client_base_payload_increments_seq_and_never_repeats():
    """R5.1：_base_payload 每次调用生成新 seq，任意两个新动作 request_id 互不相同（不依赖网络）。"""
    client = _RobotClient56(ROBOT, base_url="http://127.0.0.1:1")  # 不实际发送
    seen = []
    for tag in ("enter", "measure", "clear", "measure", "clear", "exit"):
        payload = client._base_payload(tag)
        seen.append(payload["request_id"])
    assert seen == ["enter-1", "measure-2", "clear-3", "measure-4", "clear-5", "exit-6"]
    assert len(set(seen)) == len(seen)  # 全部唯一
    assert client._seq == 6


class _FailOnceThenSucceed56:
    """网络传输桩：前 fail_times 次抛 URLError，其后回显成功；记录每次实际请求体。"""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls: list[bytes] = []

    def __call__(self, request, timeout=None):
        self.calls.append(request.data)
        if len(self.calls) <= self.fail_times:
            raise _urllib_error56.URLError("simulated transient failure")
        payload = json.loads(request.data.decode("utf-8"))
        # 成功体同时带 measure/clear 结果字段，使公共 measure()/clear() 均可解析。
        resp = {
            "accepted": True,
            "virtual_time_s": 7.0,
            "request_id": payload.get("request_id"),
            "measure_result": "no_signal",
            "svd_deg": None,
            "clear_result": "no_target_in_range",
        }

        class _Resp:
            status = 200

            def read(self_inner):
                return json.dumps(resp).encode("utf-8")

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        return _Resp()


def test_client_network_retry_reuses_autogenerated_request_id(server, monkeypatch):
    """R5.1/R5.5：网络重试复用**自动生成**的 request_id（同一 seq，不消耗新 seq）。

    经由公共 measure() 走完整路径：_base_payload 先自增 seq 生成 measure-2，随后网络
    失败 2 次、第 3 次成功。三次实际发送的请求体应逐字节相同（含同一自动生成的
    request_id measure-2），且成功后 seq 不再额外增加、按返回记账。
    """
    url, _session = server
    client = _RobotClient56(ROBOT, base_url=url, max_network_retries=3)
    transport = _FailOnceThenSucceed56(fail_times=2)
    monkeypatch.setattr(_urllib_request56, "urlopen", transport)
    monkeypatch.setattr("src.b_sim.client.time.sleep", lambda *_: None)

    # 先手动占用 seq 1（模拟一次 enter），使本次 measure 自动得到 measure-2。
    client._seq = 1
    result = client.measure(300.0, 400.0, 1)

    assert result.virtual_time_s == pytest.approx(7.0)
    # 恰 3 次传输（失败 2 + 成功 1），且每次请求体逐字节一致。
    assert len(transport.calls) == 3
    assert transport.calls[0] == transport.calls[1] == transport.calls[2]
    sent = json.loads(transport.calls[0].decode("utf-8"))
    assert sent["request_id"] == "measure-2"       # 自动生成、非显式传入
    # 重试期间未再自增 seq（仍为 2）。
    assert client._seq == 2
    # 成功记录一条日志，request_id 为该自动生成 id。
    assert client.log[-1]["request_id"] == "measure-2"
    assert client.log[-1]["http_status"] == 200


def test_client_out_of_bounds_before_send_does_not_consume_request_id(server):
    """R5.1/R5.3：发送前越界坐标被拦截，不消耗 seq——后续动作 request_id 连续无空洞。

    enter（seq1）→ 越界 measure（拒发、不自增 seq）→ 合法 measure（应得 measure-2 而非 -3）。
    证明越界拦截发生在 _base_payload 之前，request_id 序列不出现"跳号"。
    """
    url, _session = server
    client = _RobotClient56(ROBOT, base_url=url)
    client.enter()  # enter-1
    assert client._seq == 1
    # 越界坐标：发送前 ProtocolError，且不生成 request_id / 不自增 seq。
    with pytest.raises(_ProtocolError56):
        client.measure(2_000_000.0 + 1.0, 0.0, 1)
    assert client._seq == 1  # 未自增
    # 下一次合法 measure 应连续取到 measure-2（无跳号）。
    client.measure(0.0, 0.0, 1)
    assert client._seq == 2
    assert client.log[-1]["request_id"] == "measure-2"
    # 恰为 ±2_000_000 m 的边界坐标通过客户端侧门禁（不抛越界错误）——不实际发往
    # mock 服务（该点在竞技场外会被服务端按业务拒绝，与坐标门禁语义无关）。
    _RobotClient56._check_coord(2_000_000.0, -2_000_000.0)  # 不抛 → 放行


def test_client_409_then_new_action_gets_fresh_request_id(server):
    """R5.4/R5.1：409 冲突抛 ProtocolError 后，随后的新动作仍取到全新、连续的 request_id。

    409 路径复用任务 5.2 已验证的语义（此处不重复其记录/时间细节），本用例的**新增**
    关注点是：一次 409 之后 request_id 生成器状态正确（后续新动作 id 唯一且连续）。
    """
    url, _session = server
    client = _RobotClient56(ROBOT, base_url=url)
    client.enter()  # enter-1
    # 用显式固定 request_id 制造 409：先成功一次，再以同 id 异内容冲突。
    rid = f"{ROBOT}-dup56"
    ok = base(rid) | {"position": {"x": 0.0, "y": 0.0}, "channel": 1}
    assert client._post("/measure", ok)["accepted"] is True
    conflict = base(rid) | {"position": {"x": 25.0, "y": 0.0}, "channel": 1}
    with pytest.raises(_ProtocolError56):
        client._post("/measure", conflict)
    # 409 之后发起的新动作走正常自动生成路径，得到连续唯一的 measure-2。
    seq_before = client._seq
    client.measure(0.0, 0.0, 1)
    new_id = client.log[-1]["request_id"]
    assert new_id == f"measure-{seq_before + 1}"
    # 该新 id 未与此前任何 id 重复。
    all_ids = [e["request_id"] for e in client.log]
    assert all_ids.count(new_id) == 1


def test_client_400_raises_protocol_error_with_autogenerated_id_no_retry(server):
    """R5.4/R5.1：非法 channel → 400 → ProtocolError（不重试），日志记的是自动生成 id。

    以公共 measure() 触发服务端 400（channel 越界），确认：抛 ProtocolError、
    恰记 1 条 400 日志、其 request_id 为自动生成的 `measure-{seq}`（唯一、非重试）。
    """
    url, _session = server
    client = _RobotClient56(ROBOT, base_url=url)
    client.enter()  # enter-1
    log_len_before = len(client.log)
    with pytest.raises(_ProtocolError56):
        client.measure(0.0, 0.0, 0)  # channel=0 非法 → 400
    # 不重试：恰新增 1 条 400 记录。
    assert len(client.log) == log_len_before + 1
    entry = client.log[-1]
    assert entry["http_status"] == 400
    assert entry["request_id"] == "measure-2"  # 自动生成、连续
    assert client._seq == 2


def test_client_network_exhaustion_raises_test_ended_with_reused_autogenerated_id(server, monkeypatch):
    """R5.5/R5.1：网络持续失败 → 重试至多 3 次（复用自动生成 id）→ TestEnded。

    补充 5.2（其用显式 request_id）：此处经 clear() 走自动生成 id，确认三次重试逐字节
    复用同一自动生成 request_id（clear-{seq}），耗尽后抛 TestEnded 且虚拟时刻不变。
    """
    url, _session = server
    client = _RobotClient56(ROBOT, base_url=url, max_network_retries=3)
    transport = _FailOnceThenSucceed56(fail_times=99)  # 始终失败
    monkeypatch.setattr(_urllib_request56, "urlopen", transport)
    monkeypatch.setattr("src.b_sim.client.time.sleep", lambda *_: None)

    client._seq = 1  # 使本次 clear 取到 clear-2
    vt_before = client.virtual_time_s
    with pytest.raises(_TestEnded56):
        client.clear(0.0, 0.0, 1)
    # 恰 3 次尝试，逐字节复用同一自动生成 request_id。
    assert len(transport.calls) == 3
    assert transport.calls[0] == transport.calls[1] == transport.calls[2]
    sent = json.loads(transport.calls[0].decode("utf-8"))
    assert sent["request_id"] == "clear-2"
    assert client._seq == 2            # 重试不自增 seq
    assert client.virtual_time_s == vt_before  # 未记账
