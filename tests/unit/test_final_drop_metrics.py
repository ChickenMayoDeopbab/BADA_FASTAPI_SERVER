import asyncio
import logging

import pytest

from app.services.pipeline import _State, _TurnTimings
from app.services.stt import STTEvent, STTEventType
from tests.unit.test_transcript_frames import _make_pipeline as _listening_pipeline
from tests.unit.test_turn_watchdog import _FakeTTSClient, _HappyLLM
from tests.unit.test_turn_watchdog import _make_pipeline as _turn_pipeline


def _metrics(caplog, name: str):
    return [r for r in caplog.records
            if r.name == "app.metrics" and getattr(r, "metric", None) == name]


async def _final(p, text: str) -> None:
    await p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text=text))


@pytest.mark.asyncio
async def test_final_dropped_while_ai_speaking(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _listening_pipeline()
    p._state = _State.SPEAKING
    p._turn_final_at = 0.0
    p._start_turn = lambda text, *, final_at: pytest.fail("버려져야 할 FINAL 로 턴이 시작됐다")

    await _final(p, "당연하지, 내가 소리를 꺼 놨으니까.")

    [rec] = _metrics(caplog, "final_dropped")
    assert rec.reason == "state"
    assert rec.state == _State.SPEAKING.value
    assert rec.chars == len("당연하지, 내가 소리를 꺼 놨으니까.")
    assert rec.since_final_ms is not None and rec.since_final_ms >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("state", "time_up", "text", "reason"), [
    (_State.LISTENING, False, "   ", "empty"),
    (_State.LISTENING, True, "늦은 발화", "time_up"),
    (_State.THINKING, False, "그거는 하죠.", "state"),
])
async def test_final_dropped_reasons(caplog, state, time_up, text, reason) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _listening_pipeline()
    p._state = state
    p._time_up = time_up
    p._start_turn = lambda text, *, final_at: None

    await _final(p, text)

    [rec] = _metrics(caplog, "final_dropped")
    assert rec.reason == reason


@pytest.mark.asyncio
async def test_final_arriving_while_closing_leaves_a_trace(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _listening_pipeline()
    p._state = _State.CLOSING

    await _final(p, "감사합니다")

    [rec] = _metrics(caplog, "final_dropped")
    assert rec.reason == "closing"
    assert rec.chars == len("감사합니다")


@pytest.mark.asyncio
async def test_accepted_final_is_not_counted_as_dropped(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _listening_pipeline()
    p._start_turn = lambda text, *, final_at: None

    await _final(p, "여보세요")
    await p._handle_stt_event(STTEvent(type=STTEventType.INTERIM, text="여보"))

    assert _metrics(caplog, "final_dropped") == []


def test_emotion_ms_is_final_to_emotion_frame() -> None:
    m = _TurnTimings(final_at=100.0, emotion_at=160.0).as_metrics()
    assert m["emotion_ms"] == 60.0
    assert _TurnTimings(final_at=100.0).as_metrics()["emotion_ms"] is None


@pytest.mark.asyncio
async def test_turn_records_emotion_and_listening_resumed(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _turn_pipeline(_HappyLLM(), _FakeTTSClient())

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    [turn] = _metrics(caplog, "voice_turn")
    assert turn.emotion_ms is not None, "앱이 마이크 게이트를 닫는 시점(감정 프레임)을 몰라서는 창을 못 잰다"
    [resumed] = _metrics(caplog, "listening_resumed")
    assert resumed.recovered is False
    assert resumed.since_turn_done_ms is not None and resumed.since_turn_done_ms >= 0


@pytest.mark.asyncio
async def test_recovery_path_marks_listening_resumed_as_recovered(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _turn_pipeline(_HappyLLM(), _FakeTTSClient())

    async def boom(ctx, suggestion):
        raise RuntimeError("hint boom")

    p._maybe_send_script_hint = boom

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    [resumed] = _metrics(caplog, "listening_resumed")
    assert resumed.recovered is True


@pytest.mark.asyncio
async def test_dropped_final_measures_time_since_turn_start(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _listening_pipeline()
    p._last_audio_at = None
    p._turn_done_at = 123.0

    async def _no_turn(user_utterance, timings):
        await asyncio.sleep(0)

    p._run_turn = _no_turn

    await _final(p, "여보세요")
    assert p._turn_done_at is None, "이전 턴의 완료 시각이 새 턴에 남으면 복귀 지연이 틀린다"
    await _final(p, "그거는 하죠.")

    [rec] = _metrics(caplog, "final_dropped")
    assert rec.reason == "state"
    assert rec.since_final_ms is not None and rec.since_final_ms >= 0
