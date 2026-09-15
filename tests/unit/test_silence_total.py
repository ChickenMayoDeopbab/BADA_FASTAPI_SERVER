import pytest

from app.schemas.frames import EndReason
from app.services import pipeline as pipeline_mod
from app.services.pipeline import _State
from app.services.stt import STTEvent, STTEventType
from tests.unit.test_transcript_frames import _make_pipeline

BEGIN, END = STTEventType.SPEECH_BEGIN, STTEventType.SPEECH_END


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(pipeline_mod, "time", c)
    return c


def _listening(clock: _Clock):
    p = _make_pipeline()
    p._listening_since = clock.now
    return p


async def _at(clock: _Clock, p, t: float, kind: STTEventType) -> None:
    clock.now = t
    await p._handle_stt_event(STTEvent(type=kind))


@pytest.mark.asyncio
async def test_leading_silence_is_counted(clock) -> None:
    p = _listening(clock)
    await _at(clock, p, 3.0, BEGIN)
    assert p._silence_total == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_pause_between_utterances_is_counted(clock) -> None:
    p = _listening(clock)
    await _at(clock, p, 2.0, BEGIN)
    await _at(clock, p, 3.0, END)
    await _at(clock, p, 10.0, BEGIN)
    assert p._silence_total == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_speaking_time_is_not_silence(clock) -> None:
    p = _listening(clock)
    await _at(clock, p, 1.0, BEGIN)
    await _at(clock, p, 9.0, END)
    await _at(clock, p, 12.0, BEGIN)
    assert p._silence_total == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_short_gap_is_not_counted(clock) -> None:
    p = _listening(clock)
    await _at(clock, p, 1.0, BEGIN)
    await _at(clock, p, 2.0, END)
    await _at(clock, p, 3.0, BEGIN)
    assert p._silence_total == 0.0


@pytest.mark.asyncio
async def test_close_counts_trailing_silence(clock) -> None:
    p = _listening(clock)
    await _at(clock, p, 1.0, BEGIN)
    await _at(clock, p, 2.0, END)
    clock.now = 5.0
    await p._close(EndReason.END_CALL)
    assert p._silence_total == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_close_ignores_trailing_gap_under_threshold(clock) -> None:
    p = _listening(clock)
    await _at(clock, p, 1.0, BEGIN)
    await _at(clock, p, 2.0, END)
    clock.now = 3.0
    await p._close(EndReason.END_CALL)
    assert p._silence_total == 0.0


@pytest.mark.asyncio
async def test_speech_events_outside_listening_do_not_touch_the_counter(clock) -> None:
    p = _listening(clock)
    p._state = _State.THINKING
    await _at(clock, p, 5.0, BEGIN)
    await _at(clock, p, 6.0, END)
    assert p._silence_total == 0.0
    assert p._listening_since == 0.0


@pytest.mark.asyncio
async def test_close_during_speech_counts_nothing(clock) -> None:
    p = _listening(clock)
    await _at(clock, p, 1.0, BEGIN)
    clock.now = 9.5
    await p._close(EndReason.END_CALL)
    assert p._silence_total == 0.0


@pytest.mark.asyncio
async def test_repeated_begin_without_end_does_not_count_speech(clock) -> None:
    p = _listening(clock)
    await _at(clock, p, 1.0, BEGIN)
    await _at(clock, p, 8.0, BEGIN)
    assert p._silence_total == 0.0
