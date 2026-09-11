"""A completed geometric certificate is distinct from giving up a search."""
from .strategy import ChannelState, DogStrategy, StrategyConfig
from .silence_coverage import certified_stations
from .geometry import Bearing


class Client:
    virtual_time_s = 0.0
    log = []


def dog(enabled=True):
    return DogStrategy(Client(), StrategyConfig(directional_probe=True, certified_silence=enabled))


def test_only_complete_actual_silence_certifies_empty():
    strategy = dog()
    state = strategy.channels[1]
    # Planned waypoints and other channels' history are not evidence for ch1.
    strategy._cover_cache = ((1,), certified_stations('rings25'))
    strategy.channels[2].silent_points = certified_stations('rings25')
    assert not strategy._certify_silence(state)
    state.silent_points = certified_stations('rings25')[:-1]
    assert not strategy._certify_silence(state)
    state.silent_points = certified_stations('rings25')
    assert strategy._certify_silence(state)
    assert state.certified_empty and not state.abandoned and not state.possible
    assert strategy._cover_cache is None
    assert strategy.summary()['certified_empty_channels'] == [1]


def test_abandonment_and_empty_candidate_grid_are_not_certificates():
    strategy = dog()
    state = strategy.channels[1]
    state.abandoned = True
    assert state.verdict_empty  # Historical read-only alias retained.
    assert not strategy._certify_silence(state)
    summary = strategy.summary()
    assert summary['certified_empty_channels'] == []
    assert summary['completion_basis'] == 'unresolved'
    assert 1 in summary['unresolved_channels']


def test_positive_observation_and_disabled_switch_prevent_certificate():
    for enabled, has_bearing, cleared in ((False, False, False), (True, True, False), (True, False, True)):
        strategy = dog(enabled)
        state = strategy.channels[1]
        state.silent_points = certified_stations('rings25')
        if has_bearing:
            state.bearings.append(Bearing(0, 0, 30))
        state.cleared = cleared
        assert not strategy._certify_silence(state)
        assert not state.certified_empty


def test_completion_requires_all_unknowns_certified_or_sixteen_successes():
    strategy = dog()
    for ch in range(1, 11):
        strategy.channels[ch].cleared = True
    assert strategy.summary()['completion_basis'] == 'unresolved'
    for ch in range(11, 21):
        state = strategy.channels[ch]
        state.silent_points = certified_stations('rings25')
        assert strategy._certify_silence(state)
    assert strategy.summary()['completion_basis'] == 'continuous_coverage'
    for state in strategy.channels.values():
        state.certified_empty = False
    for ch in range(11, 17):
        strategy.channels[ch].cleared = True
    assert strategy.summary()['completion_basis'] == 'source_count_upper_bound'
    assert strategy.summary()['unresolved_channels'] == []


def test_malformed_measurement_never_enters_silent_evidence():
    from types import SimpleNamespace
    import pytest
    from .client import ProtocolError
    for result in ('direction', 'unknown'):
        strategy = dog()
        strategy._measure = lambda *args: SimpleNamespace(result=result, svd_deg=None)
        with pytest.raises(ProtocolError):
            strategy._probe_channel_at((0, 0), strategy.channels[1])
        assert strategy.channels[1].silent_points == []
        assert not strategy.channels[1].certified_empty
