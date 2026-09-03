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
    has_free_worker,
    try_acquire_realtime_tts,
)


def _settings(
    url: str | None = "http://tts.test",
    realtime: bool = True,
    urls: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        qwen_tts_url=url,
        qwen_tts_urls=urls,
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
        calls.append(
            {"path": request.url.path, "host": request.url.host, **json.loads(request.read())}
        )

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


async def test_has_free_worker_reflects_pool_state() -> None:
    assert has_free_worker(_settings(url=None)) is False, "워커가 없으면 False"

    settings = _settings(urls="http://w1.test,http://w2.test")
    assert has_free_worker(settings) is True

    first = qwen_mod._pool(settings).get_nowait()
    assert has_free_worker(settings) is True, "하나 남았으면 아직 True"
    second = qwen_mod._pool(settings).get_nowait()
    assert has_free_worker(settings) is False, "다 나가면 False"

    qwen_mod._pool(settings).put_nowait(first)
    assert has_free_worker(settings) is True
    qwen_mod._pool(settings).put_nowait(second)


async def test_batch_returns_worker_after_synth() -> None:
    settings = _settings()
    batch = QwenTTSClient(
        settings, transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"\x01"))
    )
    before = qwen_mod._pool(settings).qsize()
    await batch.synth("ai", "안녕하세요")
    assert qwen_mod._pool(settings).qsize() == before, "합성 후 워커가 풀로 돌아와야 한다"

    async with batch.connect() as session:
        await session.synth("ai", "안녕하세요")
        assert qwen_mod._pool(settings).qsize() == before - 1, "connect 중엔 워커를 잡고 있어야 한다"
    assert qwen_mod._pool(settings).qsize() == before, "connect 종료 후 반납돼야 한다"


async def test_batch_returns_worker_when_synth_fails() -> None:
    settings = _settings()
    batch = QwenTTSClient(
        settings, transport=httpx.MockTransport(lambda _r: httpx.Response(404))
    )
    before = qwen_mod._pool(settings).qsize()
    with pytest.raises(QwenTTSUnavailableError):
        await batch.synth("ai", "안녕하세요")
    assert qwen_mod._pool(settings).qsize() == before, "실패해도 워커는 되돌아와야 한다"


async def test_acquire_busy_when_pool_empty() -> None:
    settings = _settings()
    taken = qwen_mod._pool(settings).get_nowait()
    try:
        client, reason = await asyncio.wait_for(
            try_acquire_realtime_tts(settings, transport=_ready_transport()),
            timeout=1.0,
        )
        assert client is None and reason == "busy"
    finally:
        qwen_mod._pool(settings).put_nowait(taken)


async def test_acquire_unhealthy_returns_worker_to_pool() -> None:
    settings = _settings()
    client, reason = await try_acquire_realtime_tts(
        settings, transport=_ready_transport(ready=False)
    )
    assert client is None and reason == "unhealthy"
    assert has_free_worker(settings) is True, "실패 시 워커를 되돌려야 한다"


async def test_acquire_holds_worker_until_release() -> None:
    settings = _settings()
    client, reason = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    assert client is not None and reason is None
    assert has_free_worker(settings) is False

    client.release_slot()
    assert has_free_worker(settings) is True

    client.release_slot()
    assert qwen_mod._pool(settings).qsize() == 1


async def test_pool_size_matches_configured_workers() -> None:
    settings = _settings(urls="http://w1.test,http://w2.test, http://w3.test/ ")
    held = []
    for _ in range(3):
        client, reason = await try_acquire_realtime_tts(
            settings, transport=_ready_transport()
        )
        assert client is not None and reason is None
        held.append(client)

    client, reason = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    assert client is None and reason == "busy", "워커 수를 넘는 통화는 EL 로 가야 한다"

    assert {c._base_url for c in held} == {
        "http://w1.test",
        "http://w2.test",
        "http://w3.test",
    }, "통화마다 서로 다른 워커를 잡아야 한다"

    held[0].release_slot()
    client, reason = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    assert client is not None and reason is None, "반납한 워커는 다시 쓸 수 있어야 한다"


async def test_call_keeps_the_same_worker_across_turns() -> None:
    settings = _settings(urls="http://w1.test,http://w2.test")
    calls: list[dict] = []
    client, _ = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    assert client is not None
    client._transport = _stream_transport(calls)

    for _ in range(2):
        session = await client.open()
        try:
            await session.begin()

            async def source():
                yield "안녕하세요."

            _ = [chunk async for chunk in session.stream(source())]
        finally:
            await session.aclose()

    hosts = {c["host"] for c in calls}
    assert len(hosts) == 1, f"한 통화의 턴들이 여러 워커로 흩어졌다: {hosts}"


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


async def test_batch_uses_another_worker_while_a_call_holds_one() -> None:
    settings = _settings(urls="http://w1.test,http://w2.test")
    call, _ = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    assert call is not None
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(200, content=b"\x01")

    free_host = httpx.URL("http://w2.test").host
    batch = QwenTTSClient(settings, transport=httpx.MockTransport(handler))
    try:
        pcm = await asyncio.wait_for(batch.synth("ai", "안녕하세요"), timeout=1.0)
        assert pcm == b"\x01", "EL 로 밀리지 않고 Qwen 이 합성해야 한다"
        assert seen == [free_host], "통화가 잡지 않은 워커로 가야 한다"

        seen.clear()
        async with batch.connect() as session:
            await session.synth("ai", "안녕하세요")
            await session.synth("user", "네 알겠습니다")
        assert seen == [free_host, free_host], f"connect 도 빈 워커로 가야 한다: {seen}"
    finally:
        call.release_slot()


async def test_batch_blocks_when_single_worker_is_taken(monkeypatch) -> None:
    settings = _settings()
    call, _ = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    assert call is not None
    monkeypatch.setattr(qwen_mod, "_SLOT_WAIT_TIMEOUT", 0.05)
    batch = QwenTTSClient(
        settings, transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"\x01"))
    )
    try:
        with pytest.raises(QwenTTSUnavailableError):
            await asyncio.wait_for(batch.synth("ai", "안녕하세요"), timeout=1.0)
    finally:
        call.release_slot()


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
