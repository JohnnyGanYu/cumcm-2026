"""Q3 deployment regressions: geometric safety, count logic and real protocol use."""
import json
import math
import random
import sys
import threading
from types import SimpleNamespace

import pytest

from src.b_q3_220_client import InProcessClient
from src.b_sim.arena import Arena, Case, Jammer, generate_case
from src.b_sim.client import RobotClient
from src.b_sim.geometry import Bearing
from src.b_sim.q3_fast import Q3FastStrategy, nearest_cover_point
from src.b_sim.server import Session, make_server
from src.b_sim.strategy import StrategyConfig


def bare_strategy():
    return Q3FastStrategy(SimpleNamespace(log=[], virtual_time_s=0))


def search_strategy():
    dog = bare_strategy()
    for state in dog.channels.values():
        state.feasible = SimpleNamespace(possible=True)
    return dog


def test_nearest_lens_point_uses_both_active_disks():
    # Two radius-10 disks at (+/-6, 0) have top intersection (0, 8).
    # Symmetry and the disks' top envelope make this the nearest point to (0, 20).
    point = nearest_cover_point([(-6, 0), (6, 0)], (0, 20), radius=10)
    assert point == pytest.approx((0, 8))
    assert math.dist(point, (0, 20)) == pytest.approx(12)


def test_single_vertex_region_reduces_to_one_disk():
    assert nearest_cover_point([(2, 3)], (32, 3), radius=10) == pytest.approx((12, 3))


def test_missing_or_disjoint_clearance_regions_have_no_certified_point():
    assert nearest_cover_point([], (0, 0)) is None
    assert nearest_cover_point([(-11, 0), (11, 0)], (0, 100), radius=10) is None


def test_feasible_current_position_requires_no_movement():
    origin = (1, 2)
    assert nearest_cover_point([(-6, 0), (6, 0)], origin, radius=10) == origin


def test_certified_point_really_clears_sources_inside_convex_region():
    rng = random.Random(220)
    vertices = [(195, -106), (206, -106), (206, -94), (195, -94)]
    for _ in range(30):
        source = (rng.uniform(195, 206), rng.uniform(-106, -94))
        origin = (rng.uniform(-1800, 1800), rng.uniform(-1800, 1800))
        point = nearest_cover_point(vertices, origin)
        assert point is not None
        assert math.dist(point, source) <= 19.9 + 1e-8
        # Verify against the unchanged physical clear rule, not just the solver.
        arena = Arena(Case([Jammer(1, *source, 1200)], error_salt='lens-test', seed=0))
        arena.enter()
        assert arena.clear(*point, 1)['clear_result'] == 'success'


def test_strategy_clear_uses_certified_region_and_shortens_travel():
    source = (1000, 1000)
    arena = Arena(Case([Jammer(1, *source, 1500)], error_salt='q3-clear-test', seed=0))
    client = InProcessClient(arena)
    dog = Q3FastStrategy(client)
    client.enter()
    state = dog.channels[1]
    dog._probe_channel_at((700, 700), state)
    dog._probe_channel_at((700, 1300), state)
    assert state.estimate is not None
    start, estimate = dog.pos, state.estimate
    assert dog._clear_at(estimate, 1)
    assert arena.case.cleared_count == 1
    assert math.dist(dog.pos, source) <= 19.9 + 1e-8
    assert math.dist(start, dog.pos) < math.dist(start, estimate)


def test_dogleg_belongs_to_observer_even_when_robot_is_elsewhere(monkeypatch):
    dog = bare_strategy()
    dog.pos = (-300, 100)
    state = dog.channels[1]
    observer = (2200, 0)
    state.add_bearing(*observer, 180)
    probes = []

    def probe(point, state):
        probes.append(point)
        if len(probes) == 2:
            state.bearings.append(Bearing(*point, 180))

    monkeypatch.setattr(dog, '_probe_channel_at', probe)
    assert dog._dogleg(state)
    assert probes[0] == (-300, 100)
    assert math.dist(probes[1], observer) == pytest.approx(390)
    assert probes[1][0] < observer[0]
    assert math.hypot(*probes[1]) > 1800  # Legal robot positions can be outside the source disk.
    assert math.dist(probes[1], probes[0]) > 1000


def test_current_position_can_supply_second_bearing_without_dogleg(monkeypatch):
    dog = bare_strategy()
    dog.pos = (200, 200)
    state = dog.channels[1]
    state.add_bearing(0, 0, 0)
    probes = []

    def probe(point, state):
        probes.append(point)
        state.bearings.append(Bearing(*point, 330))

    monkeypatch.setattr(dog, '_probe_channel_at', probe)
    assert dog._dogleg(state)
    assert probes == [(200, 200)]


def test_sixteen_distinct_known_channels_stop_unknown_channel_search():
    dog = search_strategy()
    for channel in range(1, 9):
        dog.channels[channel].cleared = True  # A near/clear result also proves existence.
    for channel in range(9, 17):
        dog.channels[channel].bearings = [Bearing(0, 0, 0)]
    assert not dog._search_target(dog.channels[17])


def test_fifteen_known_channels_do_not_prove_unseen_channel_empty():
    dog = search_strategy()
    for channel in range(1, 16):
        dog.channels[channel].bearings = [Bearing(0, 0, 0)]
    # Multiple readings of one channel must not be counted as distinct sources.
    dog.channels[1].bearings.extend([Bearing(100, 100, 0)] * 5)
    assert dog._search_target(dog.channels[16])


def test_known_count_gate_preserves_search_for_observed_failed_source():
    dog = search_strategy()
    for channel in range(1, 17):
        dog.channels[channel].bearings = [Bearing(0, 0, 0)]
    state = dog.channels[9]
    state.exhausted = True
    assert dog._search_target(state)
    assert not dog._search_target(dog.channels[17])


@pytest.mark.parametrize('seed', [7002, 7008])
def test_http_and_in_process_q3_cases_match_exactly_and_clear_all(seed):
    expected_arena = Arena(generate_case(seed, directional_count=0))
    expected = Q3FastStrategy(InProcessClient(expected_arena), StrategyConfig()).run()
    arena = Arena(generate_case(seed, directional_count=0))
    server = make_server(Session('q3-fast-test', arena), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RobotClient('q3-fast-test', base_url=f'http://127.0.0.1:{server.server_address[1]}')
        actual = Q3FastStrategy(client, StrategyConfig()).run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert actual == expected
    assert arena.stats() == expected_arena.stats()
    assert arena.case.cleared_count == arena.case.total


def test_official_factory_selects_q3_default_and_independent_rollbacks():
    from src.run_official import make_strategy
    client = SimpleNamespace(log=[], virtual_time_s=0)
    assert type(make_strategy(client, directional=False)).__name__ == 'Q3FastStrategy'
    assert type(make_strategy(client, directional=False, legacy_q3=True)).__name__ == 'DogStrategy'
    assert type(make_strategy(client, directional=True)).__name__ == 'Q4FastStrategy'
    assert type(make_strategy(client, directional=True, legacy_q4=True)).__name__ == 'DogStrategy'
    assert type(make_strategy(client, directional=True, legacy_q3=True)).__name__ == 'Q4FastStrategy'
    assert type(make_strategy(client, directional=False, legacy_q4=True)).__name__ == 'Q3FastStrategy'


def test_real_official_q3_entry_writes_complete_result_log(tmp_path, monkeypatch):
    from src import run_official
    arena = Arena(generate_case(7008, directional_count=0))
    monkeypatch.setattr(run_official, 'RobotClient', lambda **kwargs: InProcessClient(arena))
    monkeypatch.setattr(sys, 'argv', ['run_official.py', '--robot-id', 'q3-cli-regression'])
    monkeypatch.chdir(tmp_path)
    run_official.main()
    logs = list(tmp_path.glob('official_log_q3_practice_*.json'))
    assert len(logs) == 1
    result = json.loads(logs[0].read_text(encoding='utf-8'))
    assert result['crash'] is None
    assert result['summary']['strategy_profile'] == 'q3-fast-v2'
    assert result['summary']['cleared_count'] == arena.case.total
    assert result['strategy_config']['directional_probe'] is False
    assert result['reported_total_jammers'] is None  # Protocol success does not reveal the private total.
    assert result['paper_table1_row']['平均定位清除时间_s'] == round(arena.stats()['mean_clear_time_s'], 4)
    assert result['protocol_log'][0]['path'] == '/enter'
    assert result['protocol_log'][-1]['path'] == '/exit'
