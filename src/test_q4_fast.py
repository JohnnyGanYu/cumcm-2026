"""Regression tests for the deployed Q4 optimization and protocol boundary."""
import math
import random
import threading
from types import SimpleNamespace

import pytest

from src.b_sim.arena import Arena, generate_case
from src.b_sim.client import RobotClient
from src.b_sim.geometry import Bearing
from src.b_sim.q4_fast import Q4FastStrategy, bounded_region
from src.b_sim.server import Session, make_server
from src.b_sim.strategy import DogStrategy, StrategyConfig


def strategy():
    return Q4FastStrategy(SimpleNamespace(log=[], virtual_time_s=0))


def test_bounded_region_preserves_realizable_rounded_bearings():
    rng = random.Random(450)
    for _ in range(100):
        angle, radius = rng.uniform(0, math.tau), 1800 * math.sqrt(rng.random())
        source = (radius * math.cos(angle), radius * math.sin(angle))
        bearings = []
        for _ in range(6):
            angle, radius = rng.uniform(0, math.tau), rng.uniform(20, 1500)
            observer = (source[0] + radius * math.cos(angle), source[1] + radius * math.sin(angle))
            reading = round((math.degrees(angle) + 180 + rng.uniform(-1, 1)) % 360, 2)
            bearings.append(Bearing(*observer, reading))
        polygon = bounded_region(bearings)
        assert polygon
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            cross = (b[0]-a[0])*(source[1]-a[1])-(b[1]-a[1])*(source[0]-a[0])
            assert cross >= -1e-6


def test_weak_intersection_respects_arena_and_reception_bounds():
    readings = [Bearing(0, 0, 60), Bearing(10, 0, 60)]
    polygon = bounded_region(readings)
    assert polygon
    assert all(math.hypot(*p) <= 1800 / math.cos(math.pi/48) + 1e-8 for p in polygon)
    assert all(math.dist(p,(b.x,b.y)) <= 1500 / math.cos(math.pi/24) + 1e-8
               for p in polygon for b in readings)


def test_dogleg_uses_observer_and_allows_outside_arena(monkeypatch):
    dog = strategy()
    dog.pos = (-300, 100)
    state = dog.channels[1]
    state.add_bearing(2200, 0, 180)
    calls = []
    def probe(point, state):
        calls.append(point)
        if len(calls) == 2:
            state.bearings.append(Bearing(*point, 180))
    monkeypatch.setattr(dog, '_probe_channel_at', probe)
    assert dog._dogleg(state)
    assert calls[0] == (-300, 100)
    assert calls[1] == pytest.approx((2200+390*math.cos(math.radians(192)),
                                    390*math.sin(math.radians(192))))
    assert math.hypot(*calls[1]) > 1800


def test_current_pose_measurement_can_supply_missing_baseline(monkeypatch):
    dog = strategy()
    dog.pos = (200, 200)
    state = dog.channels[1]
    state.add_bearing(0, 0, 0)
    calls = []
    def probe(point, state):
        calls.append(point)
        state.bearings.append(Bearing(*point, 330))
    monkeypatch.setattr(dog, '_probe_channel_at', probe)
    assert dog._dogleg(state)
    assert calls == [(200, 200)]


def test_failed_homing_does_not_consume_relocated_station(monkeypatch):
    dog = strategy()
    dog.relocate_inner = True
    dog.inner_radius = 900.0
    for state in dog.channels.values():
        state.feasible = SimpleNamespace(possible=True, detection_possible=lambda *args: True)
    target = dog.channels[1]
    target.bearings = [Bearing(0,0,0), Bearing(100,100,350)]
    target.estimate = (900,0)
    assert (900,0) not in dog._cover_plan()
    monkeypatch.setattr(DogStrategy, '_home_channel', lambda *_: False)
    assert not dog._home_channel(target)
    target.exhausted = True
    assert (900,0) in dog._cover_plan()
    assert (900,0) not in dog._visited_fast_stations


def test_successful_homing_replaces_station_only_with_actual_sweep(monkeypatch):
    dog = strategy()
    dog.relocate_inner = True
    dog.inner_radius = 900.0
    dog.pos = (920,0)
    target = dog.channels[1]
    monkeypatch.setattr(DogStrategy, '_home_channel', lambda *_: True)
    calls = []
    monkeypatch.setattr(dog, '_visit_cover_station', lambda point: calls.append(point))
    assert dog._home_channel(target)
    assert calls == [(920,0)]
    assert (900,0) in dog._visited_fast_stations


def test_http_results_match_in_process_protocol():
    from src.b_eval.dev_measure import InProcessClient
    cfg = StrategyConfig(directional_probe=True)
    case = dict(directional_count=5)
    expected_arena = Arena(generate_case(7007, **case))
    expected = Q4FastStrategy(InProcessClient(expected_arena), cfg).run()
    arena = Arena(generate_case(7007, **case))
    httpd = make_server(Session('q4-fast-test', arena), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client = RobotClient('q4-fast-test', base_url=f'http://127.0.0.1:{httpd.server_address[1]}')
        actual = Q4FastStrategy(client, cfg).run()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
    assert actual == expected
    assert arena.stats()['cleared_ratio'] == 1


def test_official_factory_keeps_q3_and_rollback_on_legacy():
    from src.run_official import make_strategy
    client = SimpleNamespace(log=[], virtual_time_s=0)
    # The script imports the same runtime under its delivery-package namespace.
    assert type(make_strategy(client, directional=False)).__name__ == 'DogStrategy'
    assert type(make_strategy(client, directional=True, legacy_q4=True)).__name__ == 'DogStrategy'
    assert type(make_strategy(client, directional=True)).__name__ == 'Q4FastStrategy'


def test_delivery_sync_checks_runtime_and_entry_point(tmp_path, monkeypatch):
    from src import pack_dist_windows as pack
    source = tmp_path / 'src' / 'b_sim'
    target = tmp_path / 'dist' / 'b_sim'
    source.mkdir(parents=True)
    for name in pack.RUNTIME_MODULES:
        (source / name).write_text('# ' + name, encoding='utf-8')
    (source.parent / 'run_official.py').write_text('# new Q4 entry', encoding='utf-8')
    monkeypatch.setattr(pack, 'SOURCE_DIR', source)
    monkeypatch.setattr(pack, 'TARGET_DIR', target)
    assert 'q4_fast.py' in pack.RUNTIME_MODULES
    assert pack.sync() == len(pack.RUNTIME_MODULES) + 1
    assert pack.sync(check_only=True) == 0
    (target.parent / 'run_official.py').write_text('# stale entry', encoding='utf-8')
    assert pack.sync(check_only=True) == 1


def test_official_entry_writes_complete_result_log(tmp_path, monkeypatch):
    import json
    import sys
    from src import run_official
    from src.b_eval.dev_measure import InProcessClient
    arena = Arena(generate_case(7007, directional_count=5))
    monkeypatch.setattr(run_official, 'RobotClient', lambda **kwargs: InProcessClient(arena))
    monkeypatch.setattr(sys, 'argv', ['run_official.py','--robot-id','cli-regression','--directional'])
    monkeypatch.chdir(tmp_path)
    run_official.main()
    logs = list(tmp_path.glob('official_log_q4_practice_*.json'))
    assert len(logs) == 1
    result = json.loads(logs[0].read_text(encoding='utf-8'))
    assert result['crash'] is None
    assert result['summary']['strategy_profile'] == 'q4-fast-v2'
    assert result['summary']['cleared_count'] == arena.case.total
    assert result['strategy_config']['directional_probe'] is True
    assert result['paper_table1_row']['平均定位清除时间_s'] == round(arena.stats()['mean_clear_time_s'],4)
