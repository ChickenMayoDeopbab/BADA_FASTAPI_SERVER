
import asyncio
import logging
from types import SimpleNamespace

import pytest

import app.core.audio_stats as audio_stats
from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.services.pipeline import VoicePipeline, _State, _TurnTimings

_100MS = 3200
_NEW_FIELDS = (
    "pcm_chunks", "pcm_bytes", "audio_ms", "odd_chunks", "send_wall_ms",
    "arrival_rtf", "max_gap_ms", "gap0_count", "gap300_count",
)


class _FakeWS:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.pcm: list[bytes] = []

    async def send_json(self, payload: dict) -> None:
        self.frames.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.pcm.append(data)


class _ThreeChunkTTSSession:
    async def begin(self, emotion) -> None:
        pass

    async def stream(self, text_source):
        async for _ in text_source:
            pass
        yield b"\x00" * _100MS
        yield b"\x00" * _100MS
        yield b"\x00" * 3201

    async def aclose(self) -> None:
        pass


class _TTSClient:
    def __init__(self, session) -> None:
        self._session = session

    async def open(self, voice_id=None):
        return self._session


class _HappyLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네")
        yield LLMEvent(type=LLMEventType.TURN_END)


def _make_pipeline(llm, tts) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = _FakeWS()
    p._session_id = "sess-audio"
    p._session = {}
    p._llm = llm
    p._tts = tts
    p._settings = SimpleNamespace(tts_coalesce_ms=0)
    p._state = _State.THINKING
    p._history = []
    p._current_step = 1
    p._ws_alive = True
    p._time_up = False
    p._closing = asyncio.Event()
    p._turn_task = None
    p._listening_since = None
    p._silence_total = 0.0
    p._tremor_buf = bytearray()
    p._user_turn_intervals = []
    p._turn_open_at = None
    p._script_len = 0
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0
    return p


def _metric(caplog, name: str):
    recs = [r for r in caplog.records if r.name == "app.metrics" and getattr(r, "metric", None) == name]
    assert len(recs) == 1, f"{name} 지표 {len(recs)}건"
    return recs[0]


def _stepping_clock(step_ms: float):
    state = {"t": 0.0}

    def _now() -> float:
        state["t"] += step_ms
        return state["t"]

    return _now


@pytest.mark.asyncio
async def test_voice_turn_carries_send_stats(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    monkeypatch.setattr(audio_stats, "now_ms", _stepping_clock(250.0))
    p = _make_pipeline(_HappyLLM(), _TTSClient(_ThreeChunkTTSSession()))

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    rec = _metric(caplog, "voice_turn")
    for f in _NEW_FIELDS:
        assert hasattr(rec, f), f"voice_turn 에 {f} 없음"
    assert rec.pcm_chunks == 3
    assert rec.pcm_bytes == 2 * _100MS + 3201
    assert rec.odd_chunks == 1
    assert rec.send_wall_ms == 500.0
    assert rec.max_gap_ms == 250.0
    assert rec.gap0_count == 2
    assert rec.gap300_count == 0
    assert rec.arrival_rtf == round(500.0 / rec.audio_ms, 3)
    assert p._ai_pcm_bytes == 2 * _100MS + 3201


@pytest.mark.asyncio
async def test_voice_turn_stats_count_only_delivered_chunks(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    monkeypatch.setattr(audio_stats, "now_ms", _stepping_clock(10.0))

    class _DyingWS(_FakeWS):
        async def send_bytes(self, data: bytes) -> None:
            if len(self.pcm) == 1:
                raise RuntimeError("closed")
            self.pcm.append(data)

    p = _make_pipeline(_HappyLLM(), _TTSClient(_ThreeChunkTTSSession()))
    p._ws = _DyingWS()

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    rec = _metric(caplog, "voice_turn")
    assert rec.pcm_chunks == 1
    assert p._ws_alive is False


@pytest.mark.asyncio
async def test_fallback_audio_metric_from_speak_fallback(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    monkeypatch.setattr(audio_stats, "now_ms", _stepping_clock(10.0))
    p = _make_pipeline(None, _TTSClient(_ThreeChunkTTSSession()))

    await asyncio.wait_for(p._speak_fallback(), timeout=2.0)

    rec = _metric(caplog, "fallback_audio")
    assert rec.session_id == "sess-audio"
    assert rec.pcm_chunks == 3
    assert rec.odd_chunks == 1
    assert rec.gap0_count == 0
