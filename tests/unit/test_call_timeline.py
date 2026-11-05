import time

from app.services.call_timeline import (
    _MAX_CLOCK_PAIRS,
    _MAX_EVENTS,
    CallTimeline,
    recording_to_epoch_ms,
)

T0 = 1_762_300_000_000


class Clock:
    def __init__(self, start: float = T0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms


def _stream(tl, clock, chunks, seconds, *, wall_step_ms=100, arrivals=None):
    for _ in range(round(seconds * 10)):
        rec = chunks / 10
        tl.audio(rec)
        if arrivals is not None:
            arrivals.append((rec, clock()))
        chunks += 1
        clock.advance(wall_step_ms)
    return chunks


def test_real_time_audio_keeps_one_anchor_and_a_gap_adds_one() -> None:
    clock = Clock()
    tl = CallTimeline(clock)

    chunks = _stream(tl, clock, 0, 6.0)
    clock.advance(4000)
    _stream(tl, clock, chunks, 1.0)

    assert tl.anchors == [[T0, 0.0], [T0 + 10000, 6.0]]
    assert [e["kind"] for e in tl.events] == ["audio_first"]
    assert tl.events[0]["at_ms"] == T0


def test_small_jitter_does_not_add_anchors() -> None:
    clock = Clock()
    tl = CallTimeline(clock)

    chunks = _stream(tl, clock, 0, 1.0)
    clock.advance(250)
    _stream(tl, clock, chunks, 1.0)

    assert tl.anchors == [[T0, 0.0]]


def test_anchor_just_below_a_round_number_still_maps_its_own_chunk() -> None:
    clock = Clock()
    tl = CallTimeline(clock)
    tl.audio(0.0)
    clock.advance(4000)
    tl.audio(2.9996)

    assert tl.recording_to_epoch_ms(2.9996) == T0 + 4000
    assert tl.recording_to_epoch_ms(2.9995) <= T0 + 3000


def test_recording_to_epoch_stays_monotonic_and_close_through_gaps_and_bursts() -> None:
    clock = Clock()
    tl = CallTimeline(clock)
    arrivals: list[tuple[float, float]] = []

    chunks = _stream(tl, clock, 0, 3.0, arrivals=arrivals)
    clock.advance(4000)
    chunks = _stream(tl, clock, chunks, 2.0, arrivals=arrivals)
    clock.advance(1000)
    chunks = _stream(tl, clock, chunks, 2.0, wall_step_ms=10, arrivals=arrivals)
    chunks = _stream(tl, clock, chunks, 2.0, arrivals=arrivals)

    grid = [tl.recording_to_epoch_ms(x / 1000) for x in range(chunks * 100)]
    assert all(b >= a for a, b in zip(grid, grid[1:], strict=False))
    assert all(abs(tl.recording_to_epoch_ms(r) - wall) <= 300 for r, wall in arrivals)
    assert tl.recording_to_epoch_ms(2.95) == T0 + 2950
    assert tl.recording_to_epoch_ms(3.0) == T0 + 7000


def test_mapping_without_anchors_is_none_and_extrapolates_before_the_first() -> None:
    assert recording_to_epoch_ms([], 1.0) is None
    assert recording_to_epoch_ms([[T0, 2.0]], 1.5) == T0 - 500
    assert recording_to_epoch_ms([[T0, 0.0], [T0 + 500, 1.0]], 0.8) == T0 + 500


def test_ai_audio_is_marked_once_per_playback_and_mute_only_on_change() -> None:
    tl = CallTimeline(Clock())

    tl.ai_audio(1)
    tl.ai_audio(1)
    tl.ai_audio(None)
    tl.ai_audio(2)
    tl.mute(True)
    tl.mute(True)
    tl.mute(False)
    tl.mute(False)

    assert [(e["kind"], e.get("playback")) for e in tl.events] == [
        ("ai_audio_start", 1),
        ("ai_audio_start", 2),
        ("mute_on", None),
        ("mute_off", None),
    ]


def test_clock_pairs_take_only_sane_integer_phone_times() -> None:
    clock = Clock()
    tl = CallTimeline(clock)

    for bad in (True, 1.5, "1762300000000", None, T0 + 2 * 86_400_000):
        tl.clock_pair(bad)
    tl.clock_pair(T0 - 40)

    assert tl.clock_pairs == [[T0 - 40, T0]]

    for i in range(_MAX_CLOCK_PAIRS + 5):
        tl.clock_pair(T0 + i)
    assert len(tl.clock_pairs) == _MAX_CLOCK_PAIRS

    near_zero = CallTimeline(Clock(0))
    near_zero.clock_pair(True)
    near_zero.clock_pair(1)
    assert near_zero.clock_pairs == [[1, 0]]


def test_intent_is_kept_from_the_first_frame_and_sanitized() -> None:
    clock = Clock()
    tl = CallTimeline(clock)

    tl.intend({"v": 1, "source": "s" * 40, "phone_ms": T0 - 10})
    clock.advance(1000)
    tl.intend({"v": 2, "source": "other", "phone_ms": T0 + 990})

    assert tl.intent == {"v": 1, "source": "s" * 32, "at_ms": T0}
    assert tl.clock_pairs == [[T0 - 10, T0], [T0 + 990, T0 + 1000]]

    odd = CallTimeline(Clock())
    odd.intend({"v": "1", "source": 3})
    assert odd.intent == {"v": None, "source": None, "at_ms": T0}
    assert odd.clock_pairs == []


def test_events_stop_at_the_cap() -> None:
    tl = CallTimeline(Clock())

    for _ in range(_MAX_EVENTS + 10):
        tl.mark("user_turn_open", turn=1)

    assert len(tl.events) == _MAX_EVENTS


def test_as_dict_carries_everything_for_later_alignment() -> None:
    clock = Clock()
    tl = CallTimeline(clock)
    tl.intend({"v": 1, "phone_ms": T0})
    tl.audio(0.0)
    tl.mark("call_close", reason="USER_END")

    assert tl.as_dict() == {
        "v": 1,
        "intent": {"v": 1, "source": None, "at_ms": T0},
        "events": [
            {"kind": "audio_first", "at_ms": T0},
            {"kind": "call_close", "at_ms": T0, "reason": "USER_END"},
        ],
        "anchors": [[T0, 0.0]],
        "clock_pairs": [[T0, T0]],
    }


def test_default_clock_is_epoch_ms_and_never_goes_back() -> None:
    tl = CallTimeline()

    first = tl.now_ms()
    later = [tl.now_ms() for _ in range(1000)]

    assert abs(first - time.time() * 1000) < 1000
    assert all(b >= a for a, b in zip([first, *later], later, strict=False))
