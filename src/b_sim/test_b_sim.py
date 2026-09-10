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
