import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from app.schemas.llm import AiEmotion
from app.services import csm_tts as csm_mod
from app.services.csm_tts import (
    CsmRealtimeTTSClient,
    try_acquire_realtime_csm,
    worker_urls,
)
from app.services.qwen_tts import QwenTTSUnavailableError


def _settings(urls: str | None = "http://csm-a.test:8020", enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        csm_tts_urls=urls,
        csm_tts_realtime_enabled=enabled,
        csm_tts_voice="ai",
        csm_tts_health_timeout=1.0,
    )


def _worker_transport(
    calls: list[dict], *, ready: bool = True, fail_paths: set[str] = frozenset()
) -> httpx.MockTransport:

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"ready": ready})
        body = json.loads(request.read()) if request.content else {}
        calls.append({"path": path, "host": request.url.host, **body})
        if path in fail_paths:
            return httpx.Response(503, json={"detail": "down"})
        if path == "/v1/session/speak":

            async def _chunks():
                yield b"\x01\x00"
                yield b"\x02\x00"

            return httpx.Response(200, content=_chunks(), headers={"X-Sample-Rate": "16000"})
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


async def _turn(session, *chunks: str) -> list[bytes]:
    async def source():
        for chunk in chunks:
            yield chunk

    return [pcm async for pcm in session.stream(source())]


@pytest.fixture(autouse=True)
def _fresh_pool():
    csm_mod._worker_pools.clear()
    yield
    csm_mod._worker_pools.clear()


def test_worker_urls_parses_comma_list() -> None:
    assert worker_urls(_settings("http://a:8020, http://b:8020/ ,")) == ["http://a:8020", "http://b:8020"]
    assert worker_urls(_settings(None)) == []


async def test_open_once_per_call_then_speak_per_sentence() -> None:
    calls: list[dict] = []
    client = CsmRealtimeTTSClient(_settings(), transport=_worker_transport(calls))

    first = await client.open()
    await first.begin(AiEmotion.NEUTRAL)
    pcm = await _turn(first, "네, 알겠습니다. ", "잠시만요.")
    await first.aclose()

    second = await client.open()
    await second.begin(AiEmotion.FRIENDLY)
    await _turn(second, "네.")
    await second.aclose()

    paths = [c["path"] for c in calls]
    assert paths.count("/v1/session/open") == 1, "워커 세션은 통화당 한 번만 연다"
    assert calls[0] == {
        "path": "/v1/session/open", "host": "csm-a.test",
        "session_id": client._session_id, "voice": "ai",
    }
    speaks = [c for c in calls if c["path"] == "/v1/session/speak"]
    assert [c["text"] for c in speaks] == ["네, 알겠습니다.", "잠시만요.", "네."]
    assert all(c["session_id"] == client._session_id for c in speaks)
    assert pcm == [b"\x01\x00", b"\x02\x00"] * 2, "문장마다 청크 스트림"
    assert first.chars_sent == len("네, 알겠습니다.") + len("잠시만요.")


async def test_begin_posts_user_turn_as_base64_pcm() -> None:
    calls: list[dict] = []
    client = CsmRealtimeTTSClient(_settings(), transport=_worker_transport(calls))
    session = await client.open()
    user_pcm = b"\x10\x00" * 160

    await session.begin(AiEmotion.NEUTRAL, user_turn=(user_pcm, "여보세요"))

    user = [c for c in calls if c["path"] == "/v1/session/user"]
    assert len(user) == 1
    assert user[0]["text"] == "여보세요"
    assert base64.b64decode(user[0]["pcm_b64"]) == user_pcm
    assert user[0]["session_id"] == client._session_id


async def test_begin_skips_empty_user_turn_and_survives_user_post_failure() -> None:
    calls: list[dict] = []
    client = CsmRealtimeTTSClient(
        _settings(), transport=_worker_transport(calls, fail_paths={"/v1/session/user"})
    )
    session = await client.open()

    await session.begin(user_turn=(b"", "여보세요"))
    await session.begin(user_turn=(b"\x00\x00", "   "))
    assert not [c for c in calls if c["path"] == "/v1/session/user"], "빈 PCM·빈 전사는 안 보낸다"

    await session.begin(user_turn=(b"\x01\x00" * 8, "여보세요"))
    assert await _turn(session, "네.") == [b"\x01\x00", b"\x02\x00"]


async def test_speak_failure_raises_unavailable() -> None:
    calls: list[dict] = []
    client = CsmRealtimeTTSClient(
        _settings(), transport=_worker_transport(calls, fail_paths={"/v1/session/speak"})
    )
    session = await client.open()
    await session.begin()
    with pytest.raises(QwenTTSUnavailableError):
        await _turn(session, "네.")


async def test_open_failure_raises_unavailable() -> None:
    calls: list[dict] = []
    client = CsmRealtimeTTSClient(
        _settings(), transport=_worker_transport(calls, fail_paths={"/v1/session/open"})
    )
    with pytest.raises(QwenTTSUnavailableError):
        await client.open()


async def test_stream_before_begin_is_an_error() -> None:
    client = CsmRealtimeTTSClient(_settings(), transport=_worker_transport([]))
    session = await client.open()
    with pytest.raises(RuntimeError):
        await _turn(session, "네.")


async def test_acquire_takes_one_worker_per_call_and_release_closes_session() -> None:
    calls: list[dict] = []
    transport = _worker_transport(calls)
    settings = _settings("http://csm-a.test:8020,http://csm-b.test:8020")

    a, why_a = await try_acquire_realtime_csm(settings, transport=transport)
    b, why_b = await try_acquire_realtime_csm(settings, transport=transport)
    c, why_c = await try_acquire_realtime_csm(settings, transport=transport)
    assert a is not None and b is not None and (why_a, why_b) == (None, None)
    assert {a._base_url, b._base_url} == {"http://csm-a.test:8020", "http://csm-b.test:8020"}
    assert (c, why_c) == (None, "busy")
    assert a.engine_name == "csm"

    session = await a.open()
    await session.begin()
    await session.aclose()
    a.release_slot()
    assert len(csm_mod._background_tasks) == 1, "close 태스크는 끝날 때까지 강한 참조로 잡아 둔다"
    await asyncio.gather(*csm_mod._background_tasks)
    assert not csm_mod._background_tasks, "끝난 태스크는 집합에서 빠진다"
    closes = [c for c in calls if c["path"] == "/v1/session/close"]
    assert [c["session_id"] for c in closes] == [a._session_id]

    d, why_d = await try_acquire_realtime_csm(settings, transport=transport)
    assert d is not None and why_d is None and d._base_url == a._base_url, "반납된 워커를 다시 쓴다"
    a.release_slot()
    assert csm_mod._pool(settings).qsize() == 0


async def test_acquire_reports_disabled_and_unhealthy() -> None:
    assert await try_acquire_realtime_csm(_settings(enabled=False)) == (None, "disabled")
    assert await try_acquire_realtime_csm(_settings(urls=None)) == (None, "disabled")

    calls: list[dict] = []
    settings = _settings()
    got = await try_acquire_realtime_csm(settings, transport=_worker_transport(calls, ready=False))
    assert got == (None, "unhealthy")
    assert csm_mod._pool(settings).qsize() == 1, "건강하지 않은 워커도 풀에 되돌린다"
