from types import SimpleNamespace

import httpx
import pytest

from app.services.qwen_tts import QwenTTSClient, QwenTTSUnavailableError


def _settings(url: str | None = "http://tts.test") -> SimpleNamespace:
    return SimpleNamespace(
        qwen_tts_url=url,
        qwen_tts_timeout=5.0,
        qwen_tts_health_timeout=1.0,
    )


def _client(handler, url: str | None = "http://tts.test") -> QwenTTSClient:
    return QwenTTSClient(_settings(url), transport=httpx.MockTransport(handler))


def _raise(exc_type):
    def _handler(request: httpx.Request) -> httpx.Response:
        raise exc_type("boom", request=request)

    return _handler


# --- 비활성 (qwen_tts_url 미설정) ---


async def test_disabled_when_url_missing() -> None:
    client = QwenTTSClient(_settings(url=None))
    assert client.enabled is False
    assert await client.healthy() is False
    with pytest.raises(QwenTTSUnavailableError):
        await client.synth("ai", "안녕하세요")


async def test_disabled_when_url_blank() -> None:
    assert QwenTTSClient(_settings(url="")).enabled is False



async def test_healthy_true_when_ready() -> None:
    client = _client(lambda _r: httpx.Response(200, json={"ready": True}))
    assert await client.healthy() is True


async def test_healthy_false_when_not_ready() -> None:
    client = _client(lambda _r: httpx.Response(200, json={"ready": False}))
    assert await client.healthy() is False


async def test_healthy_false_on_connect_error() -> None:
    assert await _client(_raise(httpx.ConnectError)).healthy() is False


async def test_healthy_false_on_timeout() -> None:
    assert await _client(_raise(httpx.ReadTimeout)).healthy() is False


async def test_healthy_false_on_server_error() -> None:
    assert await _client(lambda _r: httpx.Response(503)).healthy() is False


async def test_synth_returns_pcm_and_sends_voice_and_text() -> None:
    seen: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return httpx.Response(200, content=b"\x01\x02\x03\x04")

    assert await _client(_handler).synth("ai", "안녕하세요") == b"\x01\x02\x03\x04"
    assert seen["url"] == "http://tts.test/v1/tts"
    assert b'"voice"' in seen["body"] and b'"ai"' in seen["body"]


async def test_synth_timeout_raises_unavailable() -> None:
    with pytest.raises(QwenTTSUnavailableError):
        await _client(_raise(httpx.ReadTimeout)).synth("ai", "안녕하세요")


async def test_synth_connect_error_raises_unavailable() -> None:
    with pytest.raises(QwenTTSUnavailableError):
        await _client(_raise(httpx.ConnectError)).synth("ai", "안녕하세요")


async def test_synth_http_error_raises_unavailable() -> None:
    with pytest.raises(QwenTTSUnavailableError):
        await _client(lambda _r: httpx.Response(503)).synth("ai", "안녕하세요")


async def test_synth_empty_body_raises_unavailable() -> None:
    with pytest.raises(QwenTTSUnavailableError):
        await _client(lambda _r: httpx.Response(200, content=b"")).synth("ai", "안녕하세요")



async def test_synth_serializes_concurrent_calls() -> None:
    import asyncio

    inflight = 0
    peak = 0

    async def _handler(_r: httpx.Request) -> httpx.Response:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.01)
        inflight -= 1
        return httpx.Response(200, content=b"\x01\x02")

    client = _client(_handler)
    await asyncio.gather(*(client.synth("ai", "안녕하세요") for _ in range(4)))
    assert peak == 1, "서버가 동시 1건만 처리하므로 클라이언트도 직렬화해야 한다"


async def test_synth_serializes_across_separate_instances() -> None:
    import asyncio

    inflight = 0
    peak = 0

    async def _handler(_r: httpx.Request) -> httpx.Response:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.01)
        inflight -= 1
        return httpx.Response(200, content=b"\x01\x02")

    clients = [_client(_handler) for _ in range(4)]
    await asyncio.gather(*(c.synth("ai", "안녕하세요") for c in clients))
    assert peak == 1, "서로 다른 인스턴스 사이에서도 GPU 서버 호출은 직렬화돼야 한다"


async def test_connect_reuses_one_client_across_turns() -> None:
    c = _client(lambda _r: httpx.Response(200, content=b"\x01\x02"))
    orig = c._client
    opened = 0

    def _counting(timeout):
        nonlocal opened
        opened += 1
        return orig(timeout)

    c._client = _counting
    async with c.connect() as s:
        for _ in range(5):
            await s.synth("ai", "안녕하세요")
    assert opened == 1, "connect() 안에서는 커넥션을 하나만 열어야 한다"


async def test_synth_without_connect_opens_per_call() -> None:
    c = _client(lambda _r: httpx.Response(200, content=b"\x01\x02"))
    orig = c._client
    opened = 0

    def _counting(timeout):
        nonlocal opened
        opened += 1
        return orig(timeout)

    c._client = _counting
    for _ in range(3):
        await c.synth("ai", "안녕하세요")
    assert opened == 3


async def test_synth_retries_on_5xx_then_succeeds() -> None:
    calls = 0

    def _handler(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503) if calls < 3 else httpx.Response(200, content=b"\x01\x02")

    assert await _client(_handler).synth("ai", "안녕하세요") == b"\x01\x02"
    assert calls == 3


async def test_synth_does_not_retry_on_4xx() -> None:
    calls = 0

    def _handler(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    with pytest.raises(QwenTTSUnavailableError):
        await _client(_handler).synth("ai", "안녕하세요")
    assert calls == 1, "4xx 는 재시도해도 같다"


async def test_synth_does_not_retry_on_timeout() -> None:
    calls = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(QwenTTSUnavailableError):
        await _client(_handler).synth("ai", "안녕하세요")
    assert calls == 1, "이미 타임아웃만큼 기다렸다 — 재시도하면 최악 지연이 배가 된다"


async def test_connect_failure_becomes_unavailable_error() -> None:
    c = _client(lambda _r: httpx.Response(200, content=b"\x01\x02"))

    def _boom(_timeout):
        raise OSError("socket 고갈")

    c._client = _boom
    with pytest.raises(QwenTTSUnavailableError):
        async with c.connect():
            pass


async def test_connect_holds_semaphore_for_whole_dialogue() -> None:
    import asyncio

    order: list[str] = []

    async def _handler(_r: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        return httpx.Response(200, content=b"\x01\x02")

    async def dialogue(tag: str, turns: int) -> None:
        c = _client(_handler)
        async with c.connect() as s:
            for i in range(turns):
                await s.synth("ai", "안녕하세요")
                order.append(f"{tag}{i}")

    await asyncio.gather(dialogue("A", 3), dialogue("B", 3))
    joined = "".join(order)
    assert joined in ("A0A1A2B0B1B2", "B0B1B2A0A1A2"), f"대화가 섞였다: {joined}"
