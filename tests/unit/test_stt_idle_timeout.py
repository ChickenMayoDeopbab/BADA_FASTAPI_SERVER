import asyncio
import logging
from unittest.mock import patch

import pytest
from google.api_core.exceptions import Aborted, OutOfRange, ServiceUnavailable

from app.schemas.frames import EndReason
from app.services.pipeline import VoicePipeline
from app.services.stt import (
    AUDIO_EOS,
    GoogleSTTClient,
    STTIdleTimeoutError,
    STTStreamAbortedError,
)

# Google STT v2가 오디오 미수신으로 스트림을 닫을 때 실제로 오는 메시지.
_IDLE_MSG = (
    "Audio Timeout Error: Long duration elapsed without audio. "
    "Audio should be sent close to real time."
)

# Google STT v2가 요청(config/오디오) 무수신으로 스트림을 끊을 때 오는 메시지.
_ABORT_MSG = "Stream timed out after receiving no more client requests."


def _make_stt_client() -> GoogleSTTClient:
    # SpeechAsyncClient 생성은 자격증명을 건드리므로 패치로 막는다.
    with patch("app.services.stt.SpeechAsyncClient"):
        return GoogleSTTClient(
            project_id="p",
            location="global",
            model="m",
            language="ko-KR",
        )


class _RaisingResponses:
    """첫 응답 수신 시 주어진 예외를 던지는 가짜 응답 스트림."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._exc


class _FakeSpeechClient:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def streaming_recognize(self, requests=None):
        return _RaisingResponses(self._exc)


async def _drain(client: GoogleSTTClient) -> None:
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    async for _ in client.stream(queue):
        pass


# --- stt.stream(): idle-timeout 구분 ---------------------------------------

async def test_idle_timeout_converted_to_stt_idle_timeout(caplog) -> None:
    client = _make_stt_client()
    client._client = _FakeSpeechClient(OutOfRange(_IDLE_MSG))
    with caplog.at_level(logging.INFO, logger="app.services.stt"), pytest.raises(STTIdleTimeoutError):
        await _drain(client)
    # idle 은 장애가 아니므로 exception(traceback) 로그를 남기지 않는다.
    assert all(r.exc_info is None for r in caplog.records
               if r.name == "app.services.stt")


async def test_other_out_of_range_is_reraised() -> None:
    client = _make_stt_client()
    client._client = _FakeSpeechClient(OutOfRange("some other out of range"))
    with pytest.raises(OutOfRange):
        await _drain(client)


async def test_unavailable_is_reraised() -> None:
    client = _make_stt_client()
    client._client = _FakeSpeechClient(ServiceUnavailable("backend unavailable"))
    with pytest.raises(ServiceUnavailable):
        await _drain(client)


# --- stt.stream(): 무요청 ABORTED 구분 --------------------------------------

async def test_aborted_no_requests_converted_to_stream_aborted(caplog) -> None:
    client = _make_stt_client()
    client._client = _FakeSpeechClient(Aborted(_ABORT_MSG))
    with caplog.at_level(logging.INFO, logger="app.services.stt"), pytest.raises(STTStreamAbortedError):
        await _drain(client)
    # 복구 가능한 상황이므로 exception(traceback) 로그를 남기지 않는다.
    assert all(r.exc_info is None for r in caplog.records
               if r.name == "app.services.stt")


async def test_other_aborted_is_reraised() -> None:
    client = _make_stt_client()
    client._client = _FakeSpeechClient(Aborted("some other aborted reason"))
    with pytest.raises(Aborted):
        await _drain(client)


# --- stt.stream(): first_chunk 선전송 ---------------------------------------

class _CapturingSpeechClient:
    """요청 스트림을 소비해 기록하고 빈 응답 스트림을 돌려준다."""

    def __init__(self) -> None:
        self.requests: list = []

    async def streaming_recognize(self, requests=None):
        captured = self.requests

        async def _responses():
            async for req in requests:
                captured.append(req)
            if False:  # pragma: no cover - async generator 마커
                yield

        return _responses()


async def test_first_chunk_sent_right_after_config() -> None:
    client = _make_stt_client()
    fake = _CapturingSpeechClient()
    client._client = fake

    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    queue.put_nowait(AUDIO_EOS)
    async for _ in client.stream(queue, first_chunk=b"pcm-first"):
        pass

    assert len(fake.requests) == 2
    assert fake.requests[0].recognizer  # config 요청
    assert fake.requests[1].audio == b"pcm-first"


# --- pipeline._stt_consumer(): 사유 분기 -----------------------------------

def _make_pipeline() -> tuple[VoicePipeline, list[EndReason]]:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-test"
    p._closing = asyncio.Event()
    p._audio_queue = asyncio.Queue()
    closed: list[EndReason] = []

    async def fake_close(reason: EndReason) -> None:
        closed.append(reason)
        p._closing.set()

    p._close = fake_close
    return p, closed


async def test_consumer_idle_timeout_closes_no_audio() -> None:
    p, closed = _make_pipeline()

    async def raise_idle(queue) -> None:
        raise STTIdleTimeoutError()

    p._consume_one_stream = raise_idle
    await p._stt_consumer()
    assert closed == [EndReason.NO_AUDIO]


async def test_consumer_api_error_closes_error() -> None:
    p, closed = _make_pipeline()

    async def raise_api(queue) -> None:
        raise ServiceUnavailable("backend unavailable")

    p._consume_one_stream = raise_api
    await p._stt_consumer()
    assert closed == [EndReason.ERROR]


async def test_consumer_stream_aborted_keeps_session_and_reopens() -> None:
    p, closed = _make_pipeline()
    calls = 0

    async def raise_aborted_then_close(queue) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise STTStreamAbortedError()
        p._closing.set()

    p._consume_one_stream = raise_aborted_then_close
    await p._stt_consumer()
    assert closed == []  # 세션이 죽지 않는다
    assert calls == 2  # 스트림 재오픈(재루프)


# --- pipeline._consume_one_stream(): 지연 오픈 -------------------------------

class _FakeSTT:
    """stream() 호출 여부와 first_chunk 전달을 기록하는 가짜 STT."""

    def __init__(self) -> None:
        self.calls: list[bytes | None] = []

    def stream(self, queue, first_chunk=None):
        self.calls.append(first_chunk)

        async def _events():
            if False:  # pragma: no cover - async generator 마커
                yield

        return _events()


def _make_lazy_pipeline() -> tuple[VoicePipeline, _FakeSTT]:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-test"
    p._closing = asyncio.Event()
    p._audio_queue = asyncio.Queue()
    p._stt = _FakeSTT()
    return p, p._stt


async def test_lazy_open_waits_and_times_out_without_audio(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pipeline._STT_NO_AUDIO_TIMEOUT", 0.01)
    p, fake = _make_lazy_pipeline()

    with pytest.raises(STTIdleTimeoutError):
        await p._consume_one_stream(p._audio_queue)
    assert fake.calls == []  # 오디오 없으면 스트림을 열지 않는다


async def test_lazy_open_passes_first_chunk() -> None:
    p, fake = _make_lazy_pipeline()
    p._audio_queue.put_nowait(b"pcm-first")

    await p._consume_one_stream(p._audio_queue)
    assert fake.calls == [b"pcm-first"]  # 첫 청크 도착 후에만 오픈


async def test_lazy_open_eos_returns_without_stream() -> None:
    p, fake = _make_lazy_pipeline()
    p._audio_queue.put_nowait(AUDIO_EOS)

    await p._consume_one_stream(p._audio_queue)
    assert fake.calls == []
