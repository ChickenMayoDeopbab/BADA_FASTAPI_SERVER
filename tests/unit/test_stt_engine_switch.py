import asyncio
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.frames import EndReason
from app.services import pipeline as pipeline_module
from app.services.pipeline import VoicePipeline, _State
from app.services.stt import (
    GeminiLiveSTTClient,
    GoogleSTTClient,
    STTError,
    STTEvent,
    STTEventType,
    build_stt_client,
)


def _settings(**overrides) -> Settings:
    base = get_settings().model_dump()
    base.update(overrides)
    return Settings.model_validate(base)

def test_default_engine_is_chirp_with_gemini_defaults() -> None:
    s = get_settings()
    assert s.stt_engine == "chirp"
    assert s.gemini_stt_model == "gemini-3.5-transcribe-live"
    assert s.gemini_stt_silence_ms == 500


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(stt_engine="whisper")


def test_factory_builds_chirp_client() -> None:
    with patch("app.services.stt.SpeechAsyncClient"):
        client = build_stt_client(_settings(stt_engine="chirp"))
    assert isinstance(client, GoogleSTTClient)
    assert client.multi_utterance is False


def test_factory_builds_gemini_client_from_settings() -> None:
    s = _settings(
        stt_engine="gemini_live", gemini_stt_model="m-x", gemini_stt_silence_ms=300, google_stt_language="ko-KR"
    )
    client = build_stt_client(s)
    assert isinstance(client, GeminiLiveSTTClient)
    assert client.multi_utterance is True
    assert client._model == "m-x"
    assert client._silence_duration_ms == 300
    assert client._language == "ko-KR"


def test_pipeline_constructor_uses_factory(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(pipeline_module, "build_stt_client", lambda settings: sentinel)
    monkeypatch.setattr(pipeline_module, "LLMClient", lambda: object())
    monkeypatch.setattr(pipeline_module, "RecordingStorageService", lambda s: object())
    p = VoicePipeline(ws=object(), session_id="s-1", session={}, settings=get_settings(), spring=object())
    assert p._stt is sentinel


class _FakeSTT:
    def __init__(self, events: list[STTEvent], *, multi_utterance: bool) -> None:
        self._events = events
        self.multi_utterance = multi_utterance
        self.closed = False

    def stream(self, queue, first_chunk=None):
        fake = self

        async def _gen():
            try:
                for ev in fake._events:
                    yield ev
            finally:
                fake.closed = True

        return _gen()


def _make_pipeline(stt: _FakeSTT) -> tuple[VoicePipeline, list[STTEvent]]:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-test"
    p._closing = asyncio.Event()
    p._audio_queue = asyncio.Queue()
    p._state = _State.LISTENING
    p._stt = stt
    handled: list[STTEvent] = []

    async def record(event) -> None:
        handled.append(event)

    p._handle_stt_event = record
    return p, handled


_FINAL_A = STTEvent(type=STTEventType.FINAL, text="가")
_INTERIM_B = STTEvent(type=STTEventType.INTERIM, text="나")
_FINAL_B = STTEvent(type=STTEventType.FINAL, text="나")


async def test_single_utterance_engine_stops_after_first_final() -> None:
    stt = _FakeSTT([_FINAL_A, _INTERIM_B, _FINAL_B], multi_utterance=False)
    p, handled = _make_pipeline(stt)
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(b"pcm")
    await p._consume_one_stream(queue)
    assert handled == [_FINAL_A]
    assert stt.closed


async def test_multi_utterance_engine_keeps_stream_after_final() -> None:
    stt = _FakeSTT([_FINAL_A, _INTERIM_B, _FINAL_B], multi_utterance=True)
    p, handled = _make_pipeline(stt)
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(b"pcm")
    await p._consume_one_stream(queue)
    assert handled == [_FINAL_A, _INTERIM_B, _FINAL_B]


async def test_consumer_stt_error_closes_error() -> None:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-test"
    p._closing = asyncio.Event()
    p._audio_queue = asyncio.Queue()
    closed: list[EndReason] = []

    async def fake_close(reason: EndReason) -> None:
        closed.append(reason)
        p._closing.set()

    async def raise_stt_error(queue) -> None:
        raise STTError("engine down")

    p._close = fake_close
    p._consume_one_stream = raise_stt_error
    await p._stt_consumer()
    assert closed == [EndReason.ERROR]


async def test_multi_utterance_engine_still_recycles_stream_after_deadline(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_SECONDS", 0.0)
    stt = _FakeSTT([_FINAL_A, _INTERIM_B, _FINAL_B], multi_utterance=True)
    p, handled = _make_pipeline(stt)
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(b"pcm")
    await p._consume_one_stream(queue)
    assert handled == [_FINAL_A]
    assert stt.closed
