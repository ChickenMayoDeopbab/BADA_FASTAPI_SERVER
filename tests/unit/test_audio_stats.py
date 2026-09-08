
import logging

import pytest

import app.core.audio_stats as audio_stats
from app.core.audio_stats import (
    BYTES_PER_MS,
    GAP_THRESHOLD_MS,
    TurnAudioStats,
    starvation,
    starvation_gaps,
)

_100MS = 3200


def test_constants_match_audio_contract() -> None:
    assert BYTES_PER_MS == 32
    assert GAP_THRESHOLD_MS == 80.0


def test_starvation_zero_when_arrivals_keep_ahead_of_playhead() -> None:
    sends = [(0.0, _100MS), (100.0, _100MS), (200.0, _100MS)]
    assert starvation(sends, prebuffer_ms=0.0) == [0.0, 0.0, 0.0]


def test_starvation_measures_late_arrival_against_playhead() -> None:
    sends = [(0.0, _100MS), (250.0, _100MS)]
    assert starvation(sends, prebuffer_ms=0.0) == [0.0, 150.0]


def test_starvation_accumulates_inserted_silence_into_playhead() -> None:
    sends = [(0.0, _100MS), (250.0, _100MS), (300.0, _100MS), (700.0, _100MS)]
    assert starvation(sends, prebuffer_ms=0.0) == [0.0, 150.0, 0.0, 250.0]


def test_starvation_prebuffer_absorbs_jitter() -> None:
    sends = [(0.0, _100MS), (250.0, _100MS)]
    assert starvation(sends, prebuffer_ms=300.0) == [0.0, 0.0]
    assert starvation(sends, prebuffer_ms=100.0) == [0.0, 50.0]


def test_starvation_is_relative_to_first_send_not_absolute_clock() -> None:
    sends = [(5000.0, _100MS), (5250.0, _100MS)]
    assert starvation(sends, prebuffer_ms=0.0) == [0.0, 150.0]


def test_starvation_empty() -> None:
    assert starvation([], prebuffer_ms=0.0) == []


def test_gaps_list_playhead_position_and_length_above_threshold() -> None:
    sends = [(0.0, _100MS), (250.0, _100MS), (300.0, _100MS), (700.0, _100MS)]
    assert starvation_gaps(sends, prebuffer_ms=0.0) == [(250.0, 150.0), (700.0, 250.0)]


def test_gaps_below_threshold_are_not_counted_but_still_shift_playhead() -> None:
    sends = [(0.0, _100MS), (150.0, _100MS), (330.0, _100MS)]
    assert starvation(sends, prebuffer_ms=0.0) == [0.0, 50.0, 80.0]
    assert starvation_gaps(sends, prebuffer_ms=0.0) == [(330.0, 80.0)]
    assert starvation_gaps(sends, prebuffer_ms=0.0, threshold_ms=100.0) == []


def _clock(values: list[float]):
    it = iter(values)

    def _now() -> float:
        return next(it)

    return _now


def test_turn_stats_single_turn_metrics() -> None:
    stats = TurnAudioStats(clock=_clock([1000.0, 1250.0, 1300.0]))
    stats.record(b"\x00" * _100MS)
    stats.record(b"\x00" * _100MS)
    stats.record(b"\x00" * _100MS)

    assert stats.as_metrics() == {
        "pcm_chunks": 3,
        "pcm_bytes": 3 * _100MS,
        "audio_ms": 300.0,
        "odd_chunks": 0,
        "send_wall_ms": 300.0,
        "arrival_rtf": 1.0,
        "max_gap_ms": 250.0,
        "gap0_count": 1,
        "gap300_count": 0,
        "engine_chunks": 0,
    }


def test_turn_stats_counts_odd_byte_chunks_and_fractional_audio() -> None:
    stats = TurnAudioStats(clock=_clock([0.0, 10.0, 20.0]))
    stats.record(b"\x00" * 3201)
    stats.record(b"\x00" * 3200)
    stats.record(b"\x00" * 7)
    m = stats.as_metrics()
    assert m["odd_chunks"] == 2
    assert m["pcm_bytes"] == 6408
    assert m["audio_ms"] == round(6408 / 32, 1)


def test_turn_stats_arrival_rtf_above_one_means_starving_without_prebuffer() -> None:
    stats = TurnAudioStats(clock=_clock([0.0, 400.0]))
    stats.record(b"\x00" * _100MS)
    stats.record(b"\x00" * _100MS)
    m = stats.as_metrics()
    assert m["arrival_rtf"] == 2.0
    assert m["gap0_count"] == 1
    assert m["gap300_count"] == 0


def test_turn_stats_gap300_counts_only_starvation_beyond_prebuffer() -> None:
    stats = TurnAudioStats(clock=_clock([0.0, 500.0]))
    stats.record(b"\x00" * _100MS)
    stats.record(b"\x00" * _100MS)
    m = stats.as_metrics()
    assert m["gap0_count"] == 1
    assert m["gap300_count"] == 1


def test_turn_stats_empty_turn() -> None:
    assert TurnAudioStats(clock=_clock([])).as_metrics() == {
        "pcm_chunks": 0,
        "pcm_bytes": 0,
        "audio_ms": 0.0,
        "odd_chunks": 0,
        "send_wall_ms": None,
        "arrival_rtf": None,
        "max_gap_ms": None,
        "gap0_count": 0,
        "gap300_count": 0,
        "engine_chunks": 0,
    }


def test_turn_stats_single_chunk_has_no_intervals() -> None:
    stats = TurnAudioStats(clock=_clock([42.0]))
    stats.record(b"\x00" * _100MS)
    m = stats.as_metrics()
    assert m["send_wall_ms"] == 0.0
    assert m["arrival_rtf"] == 0.0
    assert m["max_gap_ms"] is None
    assert m["gap0_count"] == 0


def test_turn_stats_default_clock_is_module_now_ms(monkeypatch) -> None:
    monkeypatch.setattr(audio_stats, "now_ms", _clock([0.0, 250.0]))
    stats = TurnAudioStats()
    stats.record(b"\x00" * _100MS)
    stats.record(b"\x00" * _100MS)
    assert stats.as_metrics()["gap0_count"] == 1


def test_turn_stats_ignores_empty_chunk() -> None:
    stats = TurnAudioStats(clock=_clock([0.0]))
    stats.record(b"")
    assert stats.as_metrics()["pcm_chunks"] == 0


@pytest.mark.parametrize("prebuffer", [0.0, 300.0])
def test_gap_counts_agree_with_starvation_gaps(prebuffer: float) -> None:
    times = [0.0, 90.0, 400.0, 420.0, 900.0]
    stats = TurnAudioStats(clock=_clock(times))
    for _ in times:
        stats.record(b"\x00" * _100MS)
    key = "gap0_count" if prebuffer == 0.0 else "gap300_count"
    expected = len(starvation_gaps([(t, _100MS) for t in times], prebuffer_ms=prebuffer))
    assert stats.as_metrics()[key] == expected


def test_module_has_no_side_effect_logging(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    TurnAudioStats(clock=_clock([0.0])).record(b"\x00\x00")
    assert caplog.records == []


def test_engine_chunks_default_zero_and_counts_record_engine() -> None:
    stats = TurnAudioStats(clock=_clock([0.0]))
    assert stats.as_metrics()["engine_chunks"] == 0
    stats.record_engine(b"\x00" * 7)
    stats.record_engine(b"")
    stats.record_engine(b"\x00" * 3200)
    assert stats.as_metrics()["engine_chunks"] == 2
    assert stats.as_metrics()["pcm_chunks"] == 0


def test_arrival_rtf_survives_audio_ms_rounding_to_zero() -> None:
    stats = TurnAudioStats(clock=_clock([0.0, 100.0]))
    stats.record(b"\x00")
    stats.record(b"\x00")
    m = stats.as_metrics()
    assert m["audio_ms"] == 0.1
    assert m["arrival_rtf"] == round(100.0 / (2 / 32), 3)
