"""Regression checks for observation-based completion and optimized search."""
import pytest
from src.b_sim.arena import Arena, Case, Jammer
from src.b_sim.client import ClearResult, MeasureResult
from src.b_sim.strategy import DogStrategy, StrategyConfig
from src.b_sim.geometry import Bearing
from src.b_sim.silence_coverage import certified_stations


class LocalClient:
    """Minimal test transport; strategy receives only public action replies."""
    def __init__(self, arena):
        self.arena = arena
        self.log = []
        self.virtual_time_s = 0.0

    def _record(self, path, result):
        self.virtual_time_s = self.arena.robot.virtual_time_s
        self.log.append({"path": path})
        return result

    def enter(self):
        return self._record("/enter", self.arena.enter())

    def exit(self):
        return self._record("/exit", self.arena.exit())

    def measure(self, x, y, channel):
        result = self._record("/measure", self.arena.measure(x, y, channel))
        return MeasureResult(result["measure_result"], result.get("svd_deg"), self.virtual_time_s)

    def clear(self, x, y, channel):
        result = self._record("/clear", self.arena.clear(x, y, channel))
        return ClearResult(result["clear_result"], self.virtual_time_s)

    def real_budget_left_s(self):
        return 1200.0


def test_sixteenth_success_stops_remaining_scan_and_exits():
    arena = Arena(Case([Jammer(ch, 0, 0, 1000) for ch in range(1, 17)], "stop", 0))
    client = LocalClient(arena)
    strategy = DogStrategy(client, StrategyConfig(directional_probe=True))
    strategy.run()
    assert arena.case.cleared_count == 16
    assert arena.robot.measure_calls == 16
    assert arena.robot.clear_calls == 16
    assert client.log[-2:]==[{"path": "/clear"}, {"path": "/exit"}]
    assert arena.robot.finished


def test_fifteen_successes_do_not_imply_completion():
    arena = Arena(Case([Jammer(ch, 0, 0, 1000) for ch in range(1, 16)], "stop", 0))
    client = LocalClient(arena)
    strategy = DogStrategy(client)
    client.enter()
    for channel in range(1, 16):
        assert strategy._clear_at((0, 0), channel)
    assert not strategy._stop()
    assert not strategy.channels[16].verdict_empty


def test_failed_clear_does_not_reach_source_count_upper_bound():
    arena = Arena(Case([Jammer(ch, 0, 0, 1000) for ch in range(1, 16)], "stop", 0))
    strategy = DogStrategy(LocalClient(arena))
    strategy.client.enter()
    for channel in range(1, 16):
        strategy._clear_at((0, 0), channel)
    assert not strategy._clear_at((0, 0), 16)
    assert not strategy._stop()


def test_planned_stations_and_another_channels_silence_cannot_certify_empty():
    strategy = DogStrategy(LocalClient(Arena(Case([], "proof", 0))),
                           StrategyConfig(certified_search=True, directional_probe=True))
    strategy.client.enter()
    strategy._probe_stations()
    assert not strategy._stop()
    first, second = strategy.channels[1], strategy.channels[2]
    first.silent_points = certified_stations()
    strategy._maybe_verdict_empty(first)
    strategy._maybe_verdict_empty(second)
    assert first.verdict_empty
    assert not second.verdict_empty
    assert not strategy._stop()
    assert strategy.summary()["completion_basis"] == "unresolved"


def test_positive_bearing_cannot_be_overridden_by_silent_coverage():
    strategy = DogStrategy(LocalClient(Arena(Case([], "proof", 0))),
                           StrategyConfig(certified_search=True, directional_probe=True))
    state = strategy.channels[7]
    state.bearings = [Bearing(0, 0, 90)]
    state.exhausted = True
    state.silent_points = certified_stations()
    strategy._maybe_verdict_empty(state)
    assert not state.verdict_empty
    assert 7 in strategy.summary()["unresolved_channels"]


def test_anchored_dogleg_uses_the_bearing_origin_after_robot_moves():
    strategy = DogStrategy(LocalClient(Arena(Case([], "anchor", 0))),
                           StrategyConfig(anchored_dogleg=True))
    strategy.client.enter()
    state = strategy.channels[2]
    state.add_bearing(100, 200, 0)
    strategy.pos = (-700, -600)
    visited = []

    def probe(point, target):
        visited.append(point)
        target.add_bearing(*point, 1)

    strategy._probe_channel_at = probe
    assert strategy._dogleg(state)
    import math
    assert math.dist(visited[0], (100, 200)) == pytest.approx(380)
    assert math.dist(visited[0], (-700, -600)) > 1000


def test_boundary_outward_source_is_found_and_completion_is_certified():
    import math
    angle = math.radians(15)
    case = Case([Jammer(7, 1800*math.cos(angle), 1800*math.sin(angle), 1000, 15)], "edge", 0)
    arena = Arena(case)
    strategy = DogStrategy(LocalClient(arena), StrategyConfig(
        certified_search=True, coverage_pruning=True, anchored_dogleg=True,
        bounded_homing=True, certified_clear=True, directional_probe=True))
    report = strategy.run()
    assert case.cleared_count == 1
    assert report["completion_basis"] == "continuous_coverage"
    assert not report["unresolved_channels"]
    assert len(report["certified_empty_channels"]) == 19
    assert any(math.hypot(e["x"], e["y"]) > 1800 for e in arena.robot.trace)
    assert strategy.client.log[-1]["path"] == "/exit"


def test_time_budget_exit_never_claims_a_coverage_certificate():
    arena = Arena(Case([Jammer(4, 1700, 0, 1000, 0)], "budget", 0))
    strategy = DogStrategy(LocalClient(arena), StrategyConfig(
        certified_search=True, directional_probe=True, virtual_time_budget_s=10))
    report = strategy.run()
    assert report["completion_basis"] == "unresolved"
    assert report["unresolved_channels"]
    assert not report["certified_empty_channels"]
    assert strategy.client.log[-1]["path"] == "/exit"
