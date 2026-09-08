
import asyncio
import logging
from types import SimpleNamespace

import pytest

from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.services import pipeline as pipeline_mod
from app.services.pipeline import VoicePipeline, _State, _TurnTimings

_SRC = [b"\x01" * 3201, b"\x02" * 7, b"\x03" * 1600] + [b"\x04" * 3200] * 5


class _FakeWS:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.pcm: list[bytes] = []

    async def send_json(self, payload: dict) -> None:
        self.frames.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.pcm.append(data)


class _OddChunkTTSSession:
    async def begin(self, emotion) -> None:
        pass

    async def stream(self, text_source):
        async for _ in text_source:
            pass
        for c in _SRC:
            yield c
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        pass


class _TTSClient:
    async def open(self, voice_id=None):
        return _OddChunkTTSSession()


class _HappyLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네")
        yield LLMEvent(type=LLMEventType.TURN_END)


def _make_pipeline(coalesce_ms: int | None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = _FakeWS()
    p._session_id = "sess-coalesce"
    p._session = {}
    p._llm = _HappyLLM()
    p._tts = _TTSClient()
    if coalesce_ms is not None:
        p._settings = SimpleNamespace(tts_coalesce_ms=coalesce_ms)
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


def _voice_turn(caplog):
    recs = [r for r in caplog.records if r.name == "app.metrics" and getattr(r, "metric", None) == "voice_turn"]
    assert len(recs) == 1
    return recs[0]


@pytest.mark.asyncio
async def test_turn_sends_even_merged_chunks_and_counts_engine_chunks(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline(coalesce_ms=320)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    sent = p._ws.pcm
    assert sent, "PCM 이 안 나갔다"
    assert all(len(c) % 2 == 0 for c in sent)
    assert sum(len(c) for c in sent) == sum(len(c) for c in _SRC)
    assert len(sent) < len(_SRC)
    assert b"".join(sent) == b"".join(_SRC)
    rec = _voice_turn(caplog)
    assert rec.engine_chunks == len(_SRC)
    assert rec.pcm_chunks == len(sent)
    assert rec.odd_chunks == 0
    assert p._ai_pcm_bytes == sum(len(c) for c in _SRC)


@pytest.mark.asyncio
async def test_coalesce_disabled_is_passthrough(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline(coalesce_ms=0)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert p._ws.pcm == _SRC
    rec = _voice_turn(caplog)
    assert rec.engine_chunks == rec.pcm_chunks == len(_SRC)
    assert rec.odd_chunks == 2


@pytest.mark.asyncio
async def test_default_when_settings_missing_is_320ms(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline(coalesce_ms=None)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert all(len(c) % 2 == 0 for c in p._ws.pcm)
    assert len(p._ws.pcm) < len(_SRC)


@pytest.mark.asyncio
async def test_fallback_speech_is_merged_too(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline(coalesce_ms=320)

    await asyncio.wait_for(p._speak_fallback(), timeout=2.0)

    assert all(len(c) % 2 == 0 for c in p._ws.pcm)
    assert b"".join(p._ws.pcm) == b"".join(_SRC)
    recs = [r for r in caplog.records if getattr(r, "metric", None) == "fallback_audio"]
    assert len(recs) == 1 and recs[0].engine_chunks == len(_SRC)


@pytest.mark.asyncio
async def test_ws_death_closes_merged_generator_immediately(monkeypatch) -> None:
    events: list[str] = []

    class _ManyChunkSession(_OddChunkTTSSession):
        async def stream(self, text_source):
            async for _ in text_source:
                pass
            try:
                for _ in range(50):
                    yield b"\x00" * 3200
                    await asyncio.sleep(0)
            finally:
                events.append("stream_closed")

    class _DyingWS(_FakeWS):
        async def send_bytes(self, data: bytes) -> None:
            if len(self.pcm) == 1:
                events.append("send_failed")
                raise RuntimeError("closed")
            self.pcm.append(data)

    class _Client:
        async def open(self, voice_id=None):
            return _ManyChunkSession()

    real = pipeline_mod.coalesce_pcm

    def _observed(*args, **kwargs):
        gen = real(*args, **kwargs)

        class _Proxy:
            def __aiter__(self):
                return self

            async def __anext__(self):
                return await gen.__anext__()

            async def aclose(self):
                events.append("aclose")
                await gen.aclose()

        return _Proxy()

    monkeypatch.setattr(pipeline_mod, "coalesce_pcm", _observed)
    p = _make_pipeline(coalesce_ms=320)
    p._tts = _Client()
    p._ws = _DyingWS()

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert events[:3] == ["send_failed", "aclose", "stream_closed"]
