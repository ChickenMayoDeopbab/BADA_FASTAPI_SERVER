import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from app.core.config import Settings
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
    speed: float | None = None,
) -> SimpleNamespace:
    settings = SimpleNamespace(
        qwen_tts_url=url,
        qwen_tts_urls=urls,
        qwen_tts_timeout=5.0,
        qwen_tts_health_timeout=1.0,
        qwen_tts_realtime_enabled=realtime,
    )
    if speed is not None:
        settings.qwen_tts_speed = speed
    return settings


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
    session = await client.open("male_old_rough")
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
    assert {c["voice"] for c in calls} == {"male_old_rough"}, "열 때 받은 칸 보이스로 합성"
    assert {c["path"] for c in calls} == {"/v1/tts/stream"}


def _pcm_chunks(seconds: float, seed: int = 0) -> list[bytes]:
    pcm = np.random.default_rng(seed).normal(0, 3000, int(16000 * seconds)).astype("<i2").tobytes()
    return [pcm[:4480]] + [pcm[i : i + 5120] for i in range(4480, len(pcm), 5120)]


def _chunk_transport(chunks: list[bytes], error: Exception | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        async def _chunks():
            for c in chunks:
                yield c
            if error is not None:
                raise error

        return httpx.Response(200, content=_chunks())

    return httpx.MockTransport(handler)


async def _speak(settings: SimpleNamespace, transport: httpx.MockTransport) -> list[bytes]:
    session = await QwenRealtimeTTSClient(settings, transport=transport).open()
    await session.begin()
    try:
        async def source():
            yield "안녕하세요. 반갑"
            yield "습니다."

        return [chunk async for chunk in session.stream(source())]
    finally:
        await session.aclose()


async def test_realtime_speed_shortens_each_sentence() -> None:
    out = await _speak(_settings(speed=1.1), _chunk_transport(_pcm_chunks(1.0)))
    tol = 160
    first = round(2880 + (16000 - 2880) / 1.1) + tol
    rest = round(16000 / 1.1) + tol
    assert sum(map(len, out)) == 2 * (first + rest)
    assert all(len(c) % 2 == 0 for c in out)


async def test_warmup_goes_to_first_sentence_with_sound() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = b"".join(_pcm_chunks(1.0)) if len(calls) > 1 else b""
        return httpx.Response(200, content=body)

    out = await _speak(_settings(speed=1.1), httpx.MockTransport(handler))
    assert len(calls) == 2
    assert sum(map(len, out)) == 2 * (round(2880 + (16000 - 2880) / 1.1) + 160)


async def test_short_first_sentence_heard_only_through_flush_still_counts() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        seconds = 1.0 if len(calls) > 1 else 0.02
        return httpx.Response(200, content=b"".join(_pcm_chunks(seconds)))

    out = await _speak(_settings(speed=1.1), httpx.MockTransport(handler))
    assert sum(map(len, out)) == 2 * ((320 + 160) + (round(16000 / 1.1) + 160))


async def test_realtime_speed_one_yields_worker_chunks_as_is() -> None:
    chunks = _pcm_chunks(1.0)
    got = await _speak(_settings(speed=1.0), _chunk_transport(chunks))
    assert got == chunks + chunks


async def test_transport_error_mid_sentence_does_not_flush() -> None:
    chunks = _pcm_chunks(1.0)[:5]
    session = await QwenRealtimeTTSClient(
        _settings(speed=1.15), transport=_chunk_transport(chunks, httpx.ReadError("boom"))
    ).open()
    await session.begin()
    got: list[bytes] = []
    try:
        async def source():
            yield "안녕하세요."

        with pytest.raises(QwenTTSUnavailableError):
            async for c in session.stream(source()):
                got.append(c)
    finally:
        await session.aclose()
    n_in = sum(map(len, chunks)) // 2
    assert 0 < sum(map(len, got)) < 2 * (round(2880 + (n_in - 2880) / 1.15) + 160)


def test_realtime_speed_is_clamped() -> None:
    assert QwenRealtimeTTSClient(_settings(speed=3.0)).speed == 1.25
    assert QwenRealtimeTTSClient(_settings(speed=float("nan"))).speed == 1.0
    assert QwenRealtimeTTSClient(_settings()).speed == 1.0


def _real_settings() -> Settings:
    return Settings(
        _env_file=None,
        redis_url="redis://localhost:6379/0",
        jwt_secret="secret",
        internal_secret="internal",
        google_project_id="project",
        gemini_api_key="gemini",
        anthropic_api_key="anthropic",
        elevenlabs_api_key="eleven",
        elevenlabs_voice_id="voice",
        spring_boot_internal_url="http://spring",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
    )


def test_env_speed_reaches_realtime_client(monkeypatch) -> None:
    monkeypatch.delenv("QWEN_TTS_SPEED", raising=False)
    assert _real_settings().qwen_tts_speed == 1.0, "기본값은 동작 변화 없음"
    monkeypatch.setenv("QWEN_TTS_SPEED", "1.15")
    assert QwenRealtimeTTSClient(_real_settings()).speed == 1.15


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


# --- 배정 시 busy 워커 건너뛰기 (계획 0064 F101) ---

_READY = {"ready": True, "busy": False}
_BUSY = {"ready": True, "busy": True}
_WARMING = {"ready": False, "busy": False}


def _health_by_host(
    states: dict, seen: list[str] | None = None, delay: float = 0.0
) -> httpx.MockTransport:
    """워커(host)별 /health 응답. 예외 타입을 넣으면 그 예외로 실패"""

    async def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request.url.host)
        if delay:
            await asyncio.sleep(delay)
        state = states[request.url.host]
        if isinstance(state, type) and issubclass(state, Exception):
            raise state("boom", request=request)
        return httpx.Response(200, json=state)

    return httpx.MockTransport(handler)


def _pool_hosts(settings) -> list[str]:
    return [httpx.URL(u).host for u in qwen_mod._pool(settings)._queue]


def _three() -> SimpleNamespace:
    return _settings(urls="http://w1.test,http://w2.test,http://w3.test")


async def _acquire(settings, transport):
    return await asyncio.wait_for(
        try_acquire_realtime_tts(settings, transport=transport), timeout=1.0
    )


async def test_acquire_skips_busy_worker_and_takes_next(monkeypatch) -> None:
    metrics: list[tuple[str, dict]] = []
    monkeypatch.setattr(qwen_mod, "log_metric", lambda name, **kw: metrics.append((name, kw)))
    settings = _settings(urls="http://w1.test,http://w2.test")
    seen: list[str] = []

    client, reason = await _acquire(
        settings, _health_by_host({"w1.test": _BUSY, "w2.test": _READY}, seen)
    )

    assert client is not None and reason is None
    assert client._base_url == "http://w2.test", "락이 잡힌 워커를 통화에 주면 안 된다"
    assert seen == ["w1.test", "w2.test"]
    assert _pool_hosts(settings) == ["w1.test"], "건너뛴 워커는 풀로 돌아가야 한다"
    assert metrics == [
        ("realtime_tts_worker_skip", {"url": "http://w1.test", "state": "busy"})
    ]


async def test_acquire_all_busy_reports_worker_busy_and_restores_pool() -> None:
    settings = _three()
    seen: list[str] = []
    states = {"w1.test": _BUSY, "w2.test": _BUSY, "w3.test": _BUSY}

    client, reason = await _acquire(settings, _health_by_host(states, seen))

    assert client is None and reason == "worker_busy"
    assert seen == ["w1.test", "w2.test", "w3.test"], "워커마다 한 번씩만 봐야 한다"
    assert _pool_hosts(settings) == ["w1.test", "w2.test", "w3.test"]


async def test_acquire_unhealthy_stops_without_trying_next() -> None:
    settings = _settings(urls="http://w1.test,http://w2.test")
    seen: list[str] = []
    states = {"w1.test": httpx.ConnectTimeout, "w2.test": _READY}

    client, reason = await _acquire(settings, _health_by_host(states, seen))

    assert client is None and reason == "unhealthy"
    assert seen == ["w1.test"], "박스가 죽었으면 나머지에 헬스 타임아웃을 더 쓰면 안 된다"
    assert _pool_hosts(settings) == ["w2.test", "w1.test"]


async def test_acquire_busy_then_unhealthy_returns_every_worker() -> None:
    settings = _three()
    states = {"w1.test": _BUSY, "w2.test": httpx.ConnectTimeout, "w3.test": _READY}

    client, reason = await _acquire(settings, _health_by_host(states))

    assert client is None and reason == "unhealthy"
    assert sorted(_pool_hosts(settings)) == ["w1.test", "w2.test", "w3.test"]


async def test_acquire_skips_restarting_worker() -> None:
    """워밍업 중인 워커는 포트를 아직 안 열어 연결 거부가 온다"""
    settings = _settings(urls="http://w1.test,http://w2.test")

    client, reason = await _acquire(
        settings, _health_by_host({"w1.test": httpx.ConnectError, "w2.test": _READY})
    )

    assert client is not None and reason is None
    assert client._base_url == "http://w2.test", "하나만 재기동된 워커 때문에 EL 로 가면 안 된다"


async def test_acquire_only_down_or_warming_reports_unhealthy() -> None:
    settings = _settings(urls="http://w1.test,http://w2.test")

    client, reason = await _acquire(
        settings, _health_by_host({"w1.test": httpx.ConnectError, "w2.test": _WARMING})
    )

    assert client is None and reason == "unhealthy"
    assert _pool_hosts(settings) == ["w1.test", "w2.test"]


async def test_acquire_treats_missing_busy_field_as_free() -> None:
    settings = _settings(urls="http://w1.test")

    client, reason = await _acquire(settings, _health_by_host({"w1.test": {"ready": True}}))

    assert client is not None and reason is None


async def test_concurrent_acquire_does_not_recheck_busy_worker() -> None:
    settings = _settings(urls="http://w1.test,http://w2.test")
    seen: list[str] = []
    transport = _health_by_host({"w1.test": _BUSY, "w2.test": _READY}, seen, delay=0.01)

    results = await asyncio.wait_for(
        asyncio.gather(
            try_acquire_realtime_tts(settings, transport=transport),
            try_acquire_realtime_tts(settings, transport=transport),
        ),
        timeout=1.0,
    )

    got = [c._base_url for c, _ in results if c is not None]
    assert got == ["http://w2.test"]
    assert sorted(reason for c, reason in results if c is None) == ["worker_busy"]
    assert seen.count("w1.test") == 1, "busy 워커를 바로 풀에 넣으면 같은 통화가 다시 꺼내 본다"
    assert _pool_hosts(settings) == ["w1.test"]


async def test_acquire_cancelled_mid_check_returns_every_worker() -> None:
    settings = _settings(urls="http://w1.test,http://w2.test")
    transport = _health_by_host({"w1.test": _BUSY, "w2.test": _READY}, delay=0.2)

    task = asyncio.create_task(try_acquire_realtime_tts(settings, transport=transport))
    await asyncio.sleep(0.3)  # w1 은 건너뛰었고 w2 를 확인하는 중
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert sorted(_pool_hosts(settings)) == ["w1.test", "w2.test"]


async def test_realtime_open_without_voice_uses_ai() -> None:
    calls: list[dict] = []
    client = QwenRealtimeTTSClient(_settings(), transport=_stream_transport(calls))
    session = await client.open()
    try:
        await session.begin()

        async def source():
            yield "네."

        _ = [chunk async for chunk in session.stream(source())]
    finally:
        await session.aclose()
    assert {c["voice"] for c in calls} == {"ai"}


async def test_acquire_copies_worker_voices_to_client() -> None:
    settings = _settings(url="http://voices.test")
    transport = httpx.MockTransport(
        lambda _r: httpx.Response(200, json={"ready": True, "voices": ["ai", "user", "male_old_rough", 3]})
    )
    client, reason = await try_acquire_realtime_tts(settings, transport=transport)
    try:
        assert reason is None and client.voices == frozenset({"ai", "user", "male_old_rough"})
    finally:
        client.release_slot()


async def _speak_session(settings, transport, sentences=("안녕하세요. 반갑", "습니다.")):
    session = await QwenRealtimeTTSClient(settings, transport=transport).open()
    await session.begin()
    try:
        async def source():
            for s in sentences:
                yield s

        out = [chunk async for chunk in session.stream(source())]
    finally:
        await session.aclose()
    return out, session


@pytest.mark.parametrize("speed", [1.0, 1.1])
async def test_request_metric_does_not_change_yielded_bytes(speed) -> None:
    chunks = _pcm_chunks(1.0)
    on = _settings(speed=speed)
    off = _settings(speed=speed)
    off.qwen_tts_req_metric = False
    got_on, s_on = await _speak_session(on, _chunk_transport(chunks))
    got_off, s_off = await _speak_session(off, _chunk_transport(chunks))
    assert got_on == got_off
    assert len(s_on.request_log) == 2 and s_off.request_log == []
    assert s_off.turn_metrics() == {}


async def test_request_log_per_request_with_raw_ms() -> None:
    chunks = _pcm_chunks(1.0)
    _, session = await _speak_session(_settings(), _chunk_transport(chunks))
    log = session.request_log
    assert [r["outcome"] for r in log] == ["eos", "eos"]
    assert all(r["raw_ms"] == round(sum(map(len, chunks)) / 32, 1) for r in log)
    assert [r["chars"] for r in log] == [len("안녕하세요."), len("반갑습니다.")]
    assert all(r["voice"] == "ai" and r["ns"] is None for r in log)
    assert {"decay_ms", "cut", "hard", "trail_ms", "lead_ms", "inner_ends", "inner_cuts", "pause_max_ms"} <= set(log[0])


async def test_request_log_reads_ns_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"".join(_pcm_chunks(0.3)), headers={"X-Non-Streaming": "1"})

    _, session = await _speak_session(_settings(), httpx.MockTransport(handler), ("네.",))
    assert session.request_log[0]["ns"] == 1


async def test_request_log_outcome_on_transport_error() -> None:
    session = await QwenRealtimeTTSClient(
        _settings(), transport=_chunk_transport(_pcm_chunks(1.0)[:3], httpx.ReadError("boom"))
    ).open()
    await session.begin()

    async def source():
        yield "안녕하세요."

    with pytest.raises(QwenTTSUnavailableError):
        async for _ in session.stream(source()):
            pass
    await session.aclose()
    assert [r["outcome"] for r in session.request_log] == ["error"]


async def test_request_log_outcome_on_runaway() -> None:
    from app.services.qwen_tts import QwenTTSRunawayError

    session = await QwenRealtimeTTSClient(_settings(), transport=_chunk_transport(_pcm_chunks(8.0))).open()
    await session.begin()

    async def source():
        yield "네."

    with pytest.raises(QwenTTSRunawayError):
        async for _ in session.stream(source()):
            pass
    await session.aclose()
    assert [r["outcome"] for r in session.request_log] == ["runaway"]


async def test_request_log_outcome_when_consumer_stops_early() -> None:
    session = await QwenRealtimeTTSClient(_settings(), transport=_chunk_transport(_pcm_chunks(1.0))).open()
    await session.begin()

    async def source():
        yield "안녕하세요."

    stream = session.stream(source())
    await stream.__anext__()
    await stream.aclose()
    await session.aclose()
    assert [r["outcome"] for r in session.request_log] == ["aborted"]


async def test_request_log_times_with_injected_clock() -> None:
    ticks = iter([0.0, 0.2, 0.25, 0.9, 1.0])
    client = httpx.AsyncClient(transport=_chunk_transport([b"\x01\x00" * 160] * 3))
    session = qwen_mod.QwenRealtimeTTSSession(client, "http://tts.test", clock=lambda: next(ticks))
    await session.begin()

    async def source():
        yield "네."

    _ = [c async for c in session.stream(source())]
    await session.aclose()
    rec = session.request_log[0]
    assert (rec["ttfb_ms"], rec["stall_max_ms"], rec["wall_ms"]) == (200.0, 650.0, 1000.0)


def test_turn_metrics_aggregate_requests() -> None:
    log = [
        {"raw_ms": 1000.0, "ttfb_ms": 100.0, "wall_ms": 600.0, "stall_max_ms": 30.0, "decay_ms": 8.0,
         "cut": True, "hard": False, "trail_ms": 50.0, "lead_ms": 100.0},
        {"raw_ms": 1000.0, "ttfb_ms": 300.0, "wall_ms": 800.0, "stall_max_ms": 80.0, "decay_ms": 20.0,
         "cut": False, "hard": False, "trail_ms": 900.0, "lead_ms": 700.0},
    ]
    got = qwen_mod._turn_metrics(log, 1.0)
    assert got == {
        "tts_req": 2, "tts_req_cut": 1, "tts_req_hard": 0, "tts_last_cut": False, "tts_tail_short": 1,
        "tts_raw_ms": 2000.0, "tts_gen_rtf": 0.5, "tts_ttfb_max_ms": 300.0, "tts_stall_max_ms": 80.0,
        "tts_gap_max_ms": 750.0, "tts_gap_over700": 1,
    }
    assert qwen_mod._turn_metrics(log, 1.5)["tts_gap_max_ms"] == 500.0
    assert qwen_mod._turn_metrics([], 1.0) == {}


def _shaped_req(trail: float, lead: float, **extra) -> dict:
    return {"raw_ms": 1000.0, "ttfb_ms": 100.0, "wall_ms": 600.0, "stall_max_ms": 0.0, "decay_ms": 20.0,
            "trail_ms": trail, "lead_ms": lead, **extra}


def test_gap_uses_compressed_trail_for_compressed_requests() -> None:
    on = [_shaped_req(900.0, 40.0, e="on", out_trail_ms=180.0), _shaped_req(900.0, 40.0, e="on", out_trail_ms=180.0)]
    got = qwen_mod._turn_metrics(on, 1.0)
    assert (got["tts_gap_max_ms"], got["tts_gap_over700"]) == (220.0, 0)
    assert qwen_mod._turn_metrics(on, 1.25)["tts_gap_max_ms"] == 176.0
    then_off = [_shaped_req(900.0, 40.0, e="on", out_trail_ms=180.0), _shaped_req(900.0, 40.0, e="no_header")]
    assert qwen_mod._turn_metrics(then_off, 1.0)["tts_gap_max_ms"] == 220.0


@pytest.mark.parametrize(
    "first",
    [
        {}, {"e": "no_header"}, {"e": "comp_error", "out_trail_ms": 180.0},
        {"e": "on", "out_trail_ms": None}, {"e": "on"},
    ],
    ids=["no_e", "no_header", "comp_error", "out_trail_none", "out_trail_missing"],
)
def test_gap_keeps_raw_trail_when_the_request_was_not_compressed(first) -> None:
    log = [_shaped_req(900.0, 40.0, **first), _shaped_req(900.0, 40.0, e="on", out_trail_ms=180.0)]
    got = qwen_mod._turn_metrics(log, 1.0)
    assert (got["tts_gap_max_ms"], got["tts_gap_over700"]) == (940.0, 1)


def test_gap_is_taken_per_boundary_and_counts_every_long_one() -> None:
    log = [
        _shaped_req(800.0, 40.0),
        _shaped_req(900.0, 40.0, e="comp_error", out_trail_ms=180.0),
        _shaped_req(900.0, 40.0, e="on", out_trail_ms=180.0),
        _shaped_req(900.0, 40.0, e="on", out_trail_ms=180.0),
    ]
    got = qwen_mod._turn_metrics(log, 1.0)
    assert (got["tts_gap_max_ms"], got["tts_gap_over700"]) == (940.0, 2)


async def test_request_raw_ms_counts_worker_bytes_before_speed_change() -> None:
    chunks = _pcm_chunks(1.0)
    out, session = await _speak_session(_settings(speed=1.1), _chunk_transport(chunks))
    assert sum(map(len, out)) < 2 * sum(map(len, chunks))
    assert all(r["raw_ms"] == round(sum(map(len, chunks)) / 32, 1) for r in session.request_log)


async def test_recorder_failure_does_not_break_the_stream(monkeypatch) -> None:
    def _boom(self, text, voice, outcome):
        raise RuntimeError("boom")

    monkeypatch.setattr(qwen_mod._RequestRecorder, "finish", _boom)
    chunks = _pcm_chunks(1.0)
    out, session = await _speak_session(_settings(), _chunk_transport(chunks))
    assert out == chunks + chunks and session.request_log == []


async def test_excluded_worker_stays_available_while_a_bake_waits() -> None:
    settings = _settings(urls="http://w1.test,http://w2.test")
    call, _ = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    mine = call._worker_url
    other = ({"http://w1.test", "http://w2.test"} - {mine}).pop()
    batch = QwenTTSClient(settings, transport=_ready_transport())

    async def _bake():
        async with batch.connect(exclude={other}) as session:
            return session._session_url

    waiter = asyncio.ensure_future(_bake())
    await asyncio.sleep(0.05)
    assert has_free_worker(settings)
    second, reason = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    assert second is not None and second._worker_url == other
    call.release_slot()
    assert await asyncio.wait_for(waiter, 2) == mine
    second.release_slot()


async def test_excluded_wait_still_times_out(monkeypatch) -> None:
    monkeypatch.setattr(qwen_mod, "_SLOT_WAIT_TIMEOUT", 0.3)
    settings = _settings(urls="http://w1.test,http://w2.test")
    call, _ = await try_acquire_realtime_tts(settings, transport=_ready_transport())
    other = ({"http://w1.test", "http://w2.test"} - {call._worker_url}).pop()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(QwenTTSUnavailableError):
        async with QwenTTSClient(settings, transport=_ready_transport()).connect(exclude={other}):
            pass
    assert 0.25 < loop.time() - t0 < 1.0
    assert has_free_worker(settings)
    call.release_slot()
