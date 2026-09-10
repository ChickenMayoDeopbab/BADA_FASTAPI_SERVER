import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from google.genai import errors, types
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

import app.services.stt as stt_module
from app.services.stt import (
    AUDIO_EOS,
    GeminiLiveSTTClient,
    GoogleSTTClient,
    STTError,
    STTEvent,
    STTEventType,
    STTIdleTimeoutError,
    STTStreamAbortedError,
)

_PCM_MIME = "audio/pcm;rate=16000"


def _interim(text: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            interim_input_transcription=types.Transcription(text=text)
        )
    )


def _final(text: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            input_transcription=types.Transcription(text=text, finished=True)
        )
    )


def _activity(kind: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        voice_activity=types.VoiceActivity(voice_activity_type=kind)
    )


class _FakeSession:

    def __init__(self, messages, *, raise_after=None) -> None:
        self._messages = list(messages)
        self._raise_after = raise_after
        self.sent: list[dict] = []
        self._closed = asyncio.Event()

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)

    async def close(self) -> None:
        self._closed.set()

    async def receive(self):
        for msg in self._messages:
            yield msg
        if self._raise_after is not None:
            raise self._raise_after
        await self._closed.wait()
        raise errors.APIError(1000, {"message": "OK"})


class _FakeLive:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.connect_calls: list[dict] = []

    @asynccontextmanager
    async def connect(self, *, model, config):
        self.connect_calls.append({"model": model, "config": config})
        yield self._session


class _FakeGenaiClient:
    def __init__(self, session: _FakeSession) -> None:
        self.aio = type("Aio", (), {})()
        self.aio.live = _FakeLive(session)


def _make_client(session: _FakeSession) -> GeminiLiveSTTClient:
    client = GeminiLiveSTTClient(
        api_key="k", model="gemini-3.5-transcribe-live", language="ko-KR", silence_duration_ms=500
    )
    client._client = _FakeGenaiClient(session)
    return client


async def _collect(client, queue, first_chunk=None) -> list[STTEvent]:
    return [e async for e in client.stream(queue, first_chunk=first_chunk)]


def _queue_with(*items) -> "asyncio.Queue[bytes | None]":
    q: asyncio.Queue[bytes | None] = asyncio.Queue()
    for it in items:
        q.put_nowait(it)
    return q


def test_stt_errors_share_a_base() -> None:
    assert issubclass(STTIdleTimeoutError, STTError)
    assert issubclass(STTStreamAbortedError, STTError)


def test_multi_utterance_flag_per_engine() -> None:
    assert GeminiLiveSTTClient.multi_utterance is True
    with patch("app.services.stt.SpeechAsyncClient"):
        assert GoogleSTTClient(project_id="p", location="global", model="m", language="ko-KR").multi_utterance is False


def test_config_text_modality_language_and_silence() -> None:
    client = GeminiLiveSTTClient(api_key="k", model="m", language="ko-KR", silence_duration_ms=700)
    cfg = client._build_config()
    assert cfg.response_modalities == ["TEXT"]
    assert cfg.input_audio_transcription.language_codes == ["ko-KR"]
    assert cfg.realtime_input_config.automatic_activity_detection.silence_duration_ms == 700
    assert cfg.input_audio_transcription.mode is None


async def test_interim_and_final_mapped_and_stream_continues_after_final() -> None:
    session = _FakeSession([_interim("안녕"), _final("안녕하세요"), _interim("다음")])
    client = _make_client(session)
    events = await _collect(client, _queue_with(AUDIO_EOS))
    assert [(e.type, e.text) for e in events] == [
        (STTEventType.INTERIM, "안녕"),
        (STTEventType.FINAL, "안녕하세요"),
        (STTEventType.INTERIM, "다음"),
    ]


async def test_voice_activity_mapped_to_speech_events() -> None:
    session = _FakeSession([_activity("ACTIVITY_START"), _activity("ACTIVITY_END")])
    client = _make_client(session)
    events = await _collect(client, _queue_with(AUDIO_EOS))
    assert [e.type for e in events] == [STTEventType.SPEECH_BEGIN, STTEventType.SPEECH_END]


async def test_empty_transcripts_are_skipped() -> None:
    session = _FakeSession([_interim(""), _final("")])
    client = _make_client(session)
    assert await _collect(client, _queue_with(AUDIO_EOS)) == []


async def test_audio_chunks_sent_as_pcm_blobs_in_order() -> None:
    session = _FakeSession([])
    client = _make_client(session)
    await _collect(client, _queue_with(b"\x02\x02", AUDIO_EOS), first_chunk=b"\x01\x01")
    blobs = [s["audio"] for s in session.sent if "audio" in s]
    assert [b.data for b in blobs] == [b"\x01\x01", b"\x02\x02"]
    assert {b.mime_type for b in blobs} == {_PCM_MIME}


async def test_eos_sends_audio_stream_end_and_ends_normally() -> None:
    session = _FakeSession([])
    client = _make_client(session)
    events = await _collect(client, _queue_with(AUDIO_EOS))
    assert events == []
    assert session.sent[-1] == {"audio_stream_end": True}


async def test_connect_uses_model_and_config() -> None:
    session = _FakeSession([])
    client = _make_client(session)
    await _collect(client, _queue_with(AUDIO_EOS))
    call = client._client.aio.live.connect_calls[0]
    assert call["model"] == "gemini-3.5-transcribe-live"
    assert isinstance(call["config"], types.LiveConnectConfig)


@pytest.mark.parametrize(
    "exc",
    [
        errors.APIError(1006, {"message": "Abnormal closure."}),
        errors.APIError(1000, {"message": "OK"}),
        ConnectionClosedError(None, None),
        ConnectionClosedOK(None, None),
    ],
)
async def test_connection_closed_before_eos_is_stream_aborted(exc) -> None:
    session = _FakeSession([_interim("가")], raise_after=exc)
    client = _make_client(session)
    with pytest.raises(STTStreamAbortedError):
        await _collect(client, asyncio.Queue())


@pytest.mark.parametrize(
    "exc",
    [
        errors.ServerError(500, {"message": "boom"}),
        errors.ClientError(403, {"message": "denied"}),
    ],
)
async def test_http_api_error_is_stt_error_not_aborted(exc) -> None:
    session = _FakeSession([], raise_after=exc)
    client = _make_client(session)
    with pytest.raises(STTError) as info:
        await _collect(client, asyncio.Queue())
    assert not isinstance(info.value, STTStreamAbortedError)


class _SendFailingSession(_FakeSession):
    async def send_realtime_input(self, **kwargs) -> None:
        raise RuntimeError("send side broke")


async def test_sender_failure_surfaces_as_stream_aborted_and_is_logged(caplog) -> None:
    import logging

    session = _SendFailingSession([_interim("가")])
    client = _make_client(session)
    with caplog.at_level(logging.WARNING, logger="app.services.stt"), pytest.raises(STTStreamAbortedError) as info:
        await asyncio.wait_for(_collect(client, _queue_with(b"\x01\x01")), timeout=2.0)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert any("송신" in r.getMessage() for r in caplog.records if r.name == "app.services.stt")

async def _closed_within(event: asyncio.Event, delay: float) -> bool:
    try:
        await asyncio.wait_for(event.wait(), delay)
        return True
    except TimeoutError:
        return False


class _TrailingFinalSession(_FakeSession):

    def __init__(self, messages, trailing, *, flush_delay: float = 0.05) -> None:
        super().__init__(messages)
        self._trailing = list(trailing)
        self._flush_delay = flush_delay
        self._eos = asyncio.Event()

    async def send_realtime_input(self, **kwargs) -> None:
        await super().send_realtime_input(**kwargs)
        if kwargs.get("audio_stream_end"):
            self._eos.set()

    async def receive(self):
        for msg in self._messages:
            yield msg
        await self._eos.wait()
        if not await _closed_within(self._closed, self._flush_delay):
            for msg in self._trailing:
                yield msg
        raise errors.APIError(1000, {"message": "OK"})


async def test_trailing_transcript_after_eos_is_not_cut_off() -> None:
    session = _TrailingFinalSession([_interim("안녕")], [_final("안녕하세요")])
    client = _make_client(session)
    events = await asyncio.wait_for(_collect(client, _queue_with(AUDIO_EOS)), timeout=3.0)
    assert [(e.type, e.text) for e in events] == [
        (STTEventType.INTERIM, "안녕"),
        (STTEventType.FINAL, "안녕하세요"),
    ]


async def test_eos_grace_is_bounded_when_server_never_closes(monkeypatch) -> None:
    monkeypatch.setattr(stt_module, "_EOS_GRACE_SECONDS", 0.05)
    session = _TrailingFinalSession([], [], flush_delay=10.0)
    client = _make_client(session)
    assert await asyncio.wait_for(_collect(client, _queue_with(AUDIO_EOS)), timeout=3.0) == []
