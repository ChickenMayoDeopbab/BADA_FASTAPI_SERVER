from types import SimpleNamespace

import httpx
import pytest

from app.services.qwen_tts import QwenTTSClient, QwenTTSUnavailableError


def _settings(url: str | None = "http://tts.test") -> SimpleNamespace:
    return SimpleNamespace(
        qwen_tts_url=url,
        qwen_tts_urls=None,
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


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (lambda _r: httpx.Response(200, json={"ready": True, "busy": False}), "ready"),
        (lambda _r: httpx.Response(200, json={"ready": True, "busy": True}), "busy"),
        (lambda _r: httpx.Response(200, json={"ready": True}), "ready"),
        (lambda _r: httpx.Response(200, json={"ready": False}), "warming"),
        (lambda _r: httpx.Response(200, json=[1]), "unhealthy"),
        (lambda _r: httpx.Response(503), "unhealthy"),
        (_raise(httpx.ConnectError), "down"),
        (_raise(httpx.ConnectTimeout), "unhealthy"),
        (_raise(httpx.ReadTimeout), "unhealthy"),
    ],
)
async def test_health_state_classifies(handler, expected) -> None:
    assert await _client(handler).health_state() == expected


async def test_health_state_unhealthy_when_url_missing() -> None:
    assert await QwenTTSClient(_settings(url=None)).health_state() == "unhealthy"


async def test_healthy_true_when_busy() -> None:
    """예시 굽기는 urls[0] 만 보고 다른 워커로 합성한다 — busy 를 섞으면 EL 로 굽는다"""
    client = _client(lambda _r: httpx.Response(200, json={"ready": True, "busy": True}))
    assert await client.healthy() is True


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


async def test_health_state_records_worker_voices_only_when_listed() -> None:
    listed = _client(lambda r: httpx.Response(200, json={"ready": True, "voices": ["ai", "female_old_soft"]}))
    assert await listed.health_state() == "ready" and listed.voices == frozenset({"ai", "female_old_soft"})
    odd = _client(lambda r: httpx.Response(200, json={"ready": True, "voices": "ai"}))
    assert await odd.health_state() == "ready" and odd.voices is None


async def test_unknown_voice_404_is_voice_missing_and_not_retried() -> None:
    from app.services.qwen_tts import QwenVoiceMissingError

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"detail": "unknown voice: male_old_rough"})

    with pytest.raises(QwenVoiceMissingError) as err:
        await _client(handler).synth("male_old_rough", "네.")
    assert isinstance(err.value, QwenTTSUnavailableError) and len(calls) == 1
    assert err.value.voice == "male_old_rough"


async def test_other_4xx_is_not_voice_missing() -> None:
    from app.services.qwen_tts import QwenVoiceMissingError

    with pytest.raises(QwenTTSUnavailableError) as err:
        await _client(lambda r: httpx.Response(422)).synth("ai", "네.")
    assert not isinstance(err.value, QwenVoiceMissingError)


def _multi(handler, urls: str) -> QwenTTSClient:
    s = _settings(None)
    s.qwen_tts_urls = urls
    return QwenTTSClient(s, transport=httpx.MockTransport(handler))


async def test_synth_body_is_exactly_voice_and_text() -> None:
    import json

    seen: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, content=b"\x01\x02")

    await _client(_handler).synth("ai", "안녕하세요")
    assert seen["body"] == {"voice": "ai", "text": "안녕하세요"}


async def test_synth_ns_sends_flag_and_reads_headers() -> None:
    import json

    seen: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, content=b"\x01\x02", headers={"X-Non-Streaming": "1", "X-Voice-Ref": "0a1b2c3d"})

    result = await _client(_handler).synth_ns("user", "예약할게요")
    assert seen["body"] == {"voice": "user", "text": "예약할게요", "non_streaming_mode": True}
    assert (result.pcm, result.ns, result.ref, result.worker) == (b"\x01\x02", True, "0a1b2c3d", "http://tts.test")


@pytest.mark.parametrize("headers", [{"X-Non-Streaming": "0"}, {}])
async def test_synth_ns_without_flag_header_reports_not_ns(headers: dict) -> None:
    result = await _client(lambda _r: httpx.Response(200, content=b"\x01\x02", headers=headers)).synth_ns("ai", "네.")
    assert result.ns is False and result.ref is None


@pytest.mark.parametrize("status", [500, 503])
async def test_synth_ns_does_not_retry_server_errors(status: int) -> None:
    from app.services.qwen_tts import QwenCapHitError

    calls: list[int] = []

    def _handler(_r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status)

    with pytest.raises(QwenTTSUnavailableError) as err:
        await _client(_handler).synth_ns("ai", "네.")
    assert len(calls) == 1 and not isinstance(err.value, QwenCapHitError)


async def test_synth_ns_classifies_failures() -> None:
    from app.services.qwen_tts import (
        QwenCapHitError,
        QwenStalledError,
        QwenUnreachableError,
        QwenVoiceMissingError,
    )

    with pytest.raises(QwenCapHitError):
        await _client(lambda _r: httpx.Response(502)).synth_ns("ai", "네.")
    with pytest.raises(QwenStalledError):
        await _client(_raise(httpx.ReadTimeout)).synth_ns("ai", "네.")
    with pytest.raises(QwenUnreachableError):
        await _client(_raise(httpx.ConnectError)).synth_ns("ai", "네.")
    with pytest.raises(QwenVoiceMissingError):
        await _client(lambda _r: httpx.Response(404)).synth_ns("ai", "네.")
    with pytest.raises(QwenTTSUnavailableError) as err:
        await _client(lambda _r: httpx.Response(422)).synth_ns("ai", "네.")
    assert not isinstance(err.value, (QwenCapHitError, QwenStalledError, QwenVoiceMissingError))


async def test_synth_ns_read_timeout_grows_with_text() -> None:
    from app.services.qwen_tts import ns_timeout

    seen: list[float] = []
    connects: list[float] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        connects.append(request.extensions["timeout"]["connect"])
        return httpx.Response(200, content=b"\x01\x02", headers={"X-Non-Streaming": "1"})

    client = _client(_handler)
    await client.synth_ns("ai", "네.")
    await client.synth_ns("ai", "가" * 300)
    assert seen == [ns_timeout(5.0, "네."), ns_timeout(5.0, "가" * 300)]
    assert connects == [5.0, 5.0]
    assert ns_timeout(30.0, "가" * 47) == 30.0
    assert ns_timeout(30.0, "가" * 300) == pytest.approx(110.0)


async def test_synth_ns_inside_connect_uses_the_held_worker() -> None:
    hosts: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, content=b"\x01", headers={"X-Non-Streaming": "1"})

    client = _multi(_handler, "http://w1.test,http://w2.test")
    async with client.connect() as session:
        held = session._session_url
        results = [await session.synth_ns("ai", "네.") for _ in range(3)]
    assert hosts == [httpx.URL(held).host] * 3
    assert {r.worker for r in results} == {held}


def _refs_handler(table: dict):
    def _handler(request: httpx.Request) -> httpx.Response:
        entry = table[request.url.host]
        if entry == "down":
            raise httpx.ConnectError("refused", request=request)
        body = {"ready": True, "voices": ["ai", "user"]}
        if entry is not None:
            body["refs"] = entry
        return httpx.Response(200, json=body)

    return _handler


@pytest.mark.parametrize(
    ("table", "state", "refs", "unreachable", "missing"),
    [
        ({"a": {"ai": "11111111", "user": "22222222"}, "b": {"ai": "11111111", "user": "22222222"}},
         "ok", {"ai": "11111111", "user": "22222222"}, (), ()),
        ({"a": "down", "b": {"ai": "11111111", "user": "22222222"}},
         "ok", {"ai": "11111111", "user": "22222222"}, ("http://a",), ()),
        ({"a": "down", "b": "down"}, "down", {}, ("http://a", "http://b"), ()),
        ({"a": None, "b": {"ai": "11111111", "user": "22222222"}}, "no_refs", {}, (), ()),
        ({"a": {"ai": "11111111", "user": "22222222"}, "b": {"ai": "33333333", "user": "22222222"}},
         "rolling", {"user": "22222222"}, (), ()),
        ({"a": {"ai": "11111111"}, "b": {"ai": "11111111"}}, "missing", {"ai": "11111111"}, (), ("user",)),
        ({"a": {"ai": "11111111", "user": "22222222", "user_v2": "99999999"},
          "b": {"ai": "11111111", "user": "22222222"}},
         "ok", {"ai": "11111111", "user": "22222222"}, (), ()),
        ({"a": {"ai": "stale", "user": "22222222"}, "b": {"ai": "stale", "user": "22222222"}},
         "missing", {"user": "22222222"}, (), ("ai",)),
        ({"a": None, "b": {"ai": "33333333", "user": "22222222"}, "c": {"ai": "11111111", "user": "22222222"}},
         "no_refs", {}, (), ()),
    ],
)
async def test_health_refs_states(table, state, refs, unreachable, missing) -> None:
    urls = ",".join(f"http://{host}" for host in table)
    result = await _multi(_refs_handler(table), urls).health_refs(["ai", "user"])
    assert (result.state, result.refs, result.unreachable, result.missing) == (state, refs, unreachable, missing)
