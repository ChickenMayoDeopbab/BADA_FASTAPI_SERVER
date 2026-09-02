import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.services import qwen_tts as qwen_mod
from app.services.qwen_tts import (
    QwenRealtimeTTSClient,
    QwenTTSClient,
    QwenTTSUnavailableError,
    realtime_slot_active,
    try_acquire_realtime_tts,
)


def _settings(url: str | None = "http://tts.test", realtime: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        qwen_tts_url=url,
        qwen_tts_timeout=5.0,
        qwen_tts_health_timeout=1.0,
        qwen_tts_realtime_enabled=realtime,
    )


def _ready_transport(ready: bool = True) -> httpx.MockTransport:
    return httpx.MockTransport(lambda _r: httpx.Response(200, json={"ready": ready}))


def _stream_transport(calls: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True})
        calls.append({"path": request.url.path, **json.loads(request.read())})

        async def _chunks():
            yield b"\x01"
            yield b"\x02"

        return httpx.Response(200, content=_chunks())

    return httpx.MockTransport(handler)


async def test_realtime_stream_synthesizes_per_sentence() -> None:
    calls: list[dict] = []
    client = QwenRealtimeTTSClient(_settings(), transport=_stream_transport(calls))
    session = await client.open("elevenlabs-voice-id")
    try:
        await session.begin()

        async def source():
            yield "안녕하세요. 반갑"
            yield "습니다"

        out = [chunk async for chunk in session.stream(source())]
    finally:
        await session.aclose()

    assert out == [b"\x01", b"\x02", b"\x01", b"\x02"], "문장 2개 × 청크 2개"
    assert [c["text"] for c in calls] == ["안녕하세요. ", "반갑습니다 "]
    assert {c["voice"] for c in calls} == {"ai"}, "Qwen 보이스는 ai 고정"
    assert {c["path"] for c in calls} == {"/v1/tts/stream"}


async def test_realtime_stream_requires_begin() -> None:
    client = QwenRealtimeTTSClient(_settings(), transport=_stream_transport([]))
    session = await client.open()
    try:
        async def source():
            yield "안녕하세요."

        with pytest.raises(RuntimeError):
            _ = [chunk async for chunk in session.stream(source())]
    finally:
        await session.aclose()


@pytest.mark.parametrize(
    "handler",
    [
        lambda _r: httpx.Response(503),
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("boom", request=r)),
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("hang", request=r)),
    ],
    ids=["http-503", "connect-error", "read-timeout"],
)
async def test_realtime_stream_errors_normalized(handler) -> None:
    client = QwenRealtimeTTSClient(
        _settings(), transport=httpx.MockTransport(handler)
    )
    session = await client.open()
    try:
        await session.begin()

        async def source():
            yield "안녕하세요."

        with pytest.raises(QwenTTSUnavailableError):
            _ = [chunk async for chunk in session.stream(source())]
    finally:
        await session.aclose()


async def test_acquire_disabled_when_flag_off() -> None:
    client, reason = await try_acquire_realtime_tts(_settings(realtime=False))
    assert client is None and reason == "disabled"


async def test_acquire_disabled_when_url_missing() -> None:
    client, reason = await try_acquire_realtime_tts(_settings(url=None))
    assert client is None and reason == "disabled"


async def test_acquire_busy_when_slot_taken() -> None:
    semaphore = qwen_mod._semaphore()
    await semaphore.acquire()
    try:
        client, reason = await asyncio.wait_for(
            try_acquire_realtime_tts(_settings(), transport=_ready_transport()),
            timeout=1.0,
        )
        assert client is None and reason == "busy"
    finally:
        semaphore.release()


async def test_acquire_unhealthy_releases_semaphore() -> None:
    client, reason = await try_acquire_realtime_tts(
        _settings(), transport=_ready_transport(ready=False)
    )
    assert client is None and reason == "unhealthy"
    assert qwen_mod._semaphore().locked() is False, "실패 시 슬롯을 되돌려야 한다"
    assert realtime_slot_active() is False


async def test_acquire_success_holds_slot_until_release() -> None:
    client, reason = await try_acquire_realtime_tts(
        _settings(), transport=_ready_transport()
    )
    assert client is not None and reason is None
    assert realtime_slot_active() is True
    assert qwen_mod._semaphore().locked() is True

    client.release_slot()
    assert realtime_slot_active() is False
    assert qwen_mod._semaphore().locked() is False

    client.release_slot()
    await qwen_mod._semaphore().acquire()
    try:
        assert qwen_mod._semaphore().locked() is True
    finally:
        qwen_mod._semaphore().release()


async def test_batch_falls_back_fast_while_realtime_holds_slot(monkeypatch) -> None:
    client, _ = await try_acquire_realtime_tts(_settings(), transport=_ready_transport())
    assert client is not None
    monkeypatch.setattr(qwen_mod, "_SLOT_WAIT_TIMEOUT", 0.05)
    batch = QwenTTSClient(
        _settings(),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"\x01")),
    )
    try:
        with pytest.raises(QwenTTSUnavailableError):
            await asyncio.wait_for(batch.synth("ai", "안녕하세요"), timeout=1.0)
    finally:
        client.release_slot()


async def test_realtime_busy_while_batch_connect_holds() -> None:
    batch = QwenTTSClient(
        _settings(),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"\x01")),
    )
    async with batch.connect():
        client, reason = await asyncio.wait_for(
            try_acquire_realtime_tts(_settings(), transport=_ready_transport()),
            timeout=1.0,
        )
        assert client is None and reason == "busy"
