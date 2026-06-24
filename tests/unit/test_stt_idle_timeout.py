import asyncio
import logging
from unittest.mock import patch

import pytest
from google.api_core.exceptions import OutOfRange, ServiceUnavailable

from app.schemas.frames import EndReason
from app.services.pipeline import VoicePipeline
from app.services.stt import GoogleSTTClient, STTIdleTimeoutError

# Google STT v2가 오디오 미수신으로 스트림을 닫을 때 실제로 오는 메시지.
_IDLE_MSG = (
    "Audio Timeout Error: Long duration elapsed without audio. "
    "Audio should be sent close to real time."
)


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
