import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import jwt
import pytest
import uvicorn
import websockets
from starlette.websockets import WebSocket

import app.api.v1.websocket as ws_mod
import app.deps.auth as auth_mod
from app.core.config import get_settings
from app.deps.redis import get_redis
from app.main import app

_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI3IiwidHlwZSI6IkFDQ0VTUyJ9.c2lnbmF0dXJlLXZhbHVl"
_PLAYBACK = json.dumps({"type": "playback_capabilities", "completion_ack": True})
_RECV_TIMEOUT = 3.0


def _auth(token: object = _JWT, **extra: object) -> str:
    return json.dumps({"type": "auth", "token": token, **extra})


class _Wired:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.metrics: list[tuple[str, dict]] = []


class _EchoPipeline:

    def __init__(self, ws, *args, **kwargs) -> None:
        self._ws = ws

    async def run(self) -> None:
        msg = await self._ws.receive()
        if msg.get("type") == "websocket.disconnect":
            return
        got_bytes = msg.get("bytes") is not None
        await self._ws.send_json({"type": "pipeline_got", "text": msg.get("text"), "bytes": got_bytes})
        await self._ws.close()


@pytest.fixture
def wired(monkeypatch):
    state = _Wired()

    async def _fake_auth(ws, token):
        state.tokens.append(token)
        return 7, "USER"

    async def _fake_session(ws, redis, session_id, user_id):
        return {"scenario": {"aiRole": "직원"}}

    monkeypatch.setattr(ws_mod, "authenticate_ws", _fake_auth)
    monkeypatch.setattr(ws_mod, "authenticate_session", _fake_session)
    monkeypatch.setattr(ws_mod, "VoicePipeline", _EchoPipeline)
    monkeypatch.setattr(ws_mod, "log_metric", lambda name, **fields: state.metrics.append((name, fields)))
    app.dependency_overrides[get_redis] = lambda: None
    yield state
    app.dependency_overrides.clear()


@pytest.fixture
def real_decode(wired, monkeypatch):
    monkeypatch.setattr(ws_mod, "authenticate_ws", auth_mod.authenticate_ws_user)
    db_calls: list[int] = []

    def _no_db():
        db_calls.append(1)
        raise AssertionError("토큰 해독 전에 DB 를 열면 안 된다")

    monkeypatch.setattr(auth_mod, "AsyncSessionLocal", _no_db)
    wired.db_calls = db_calls
    return wired


@asynccontextmanager
async def _serve():
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_config=None))
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            assert not task.done(), "uvicorn 이 뜨지 못함"
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws/voice/s1"
    finally:
        server.should_exit = True
        await task


async def _recv_json(ws) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), _RECV_TIMEOUT))


async def _close_of(ws) -> tuple[int | None, str]:
    try:
        while True:
            await asyncio.wait_for(ws.recv(), _RECV_TIMEOUT)
    except websockets.ConnectionClosed as exc:
        return (exc.rcvd.code, exc.rcvd.reason) if exc.rcvd else (None, "")


async def _wait_metric(state: _Wired) -> None:
    deadline = time.monotonic() + _RECV_TIMEOUT
    while not state.metrics:
        assert time.monotonic() < deadline, "metric=ws_auth 가 안 남음"
        await asyncio.sleep(0.01)


def _ws_auth(state: _Wired) -> dict:
    assert [name for name, _ in state.metrics] == ["ws_auth"]
    return state.metrics[0][1]


async def test_frame_auth_uses_frame_token_and_next_frame_reaches_pipeline(wired) -> None:
    async with _serve() as url, websockets.connect(url) as ws:
        await ws.send(_auth())
        assert await _recv_json(ws) == {"type": "scenario_info", "aiRole": "직원"}
        await ws.send(_PLAYBACK)
        assert await _recv_json(ws) == {"type": "pipeline_got", "text": _PLAYBACK, "bytes": False}

    assert wired.tokens == [_JWT]
    assert _ws_auth(wired) == {"session_id": "s1", "method": "frame", "result": "ok"}


async def test_frame_auth_ignores_extra_keys(wired) -> None:
    async with _serve() as url, websockets.connect(url) as ws:
        await ws.send(_auth(app_version="1.4.0"))
        assert (await _recv_json(ws))["type"] == "scenario_info"

    assert wired.tokens == [_JWT]


async def test_query_token_still_works_without_consuming_first_frame(wired) -> None:
    async with _serve() as url, websockets.connect(f"{url}?token=legacy") as ws:
        await ws.send(_PLAYBACK)
        assert (await _recv_json(ws))["type"] == "scenario_info"
        assert await _recv_json(ws) == {"type": "pipeline_got", "text": _PLAYBACK, "bytes": False}

    assert wired.tokens == ["legacy"]
    assert _ws_auth(wired) == {"session_id": "s1", "method": "query", "result": "ok"}


async def test_query_token_wins_over_auth_frame(wired) -> None:
    async with _serve() as url, websockets.connect(f"{url}?token=legacy") as ws:
        await ws.send(_auth("frame-token"))
        assert (await _recv_json(ws))["type"] == "scenario_info"
        assert (await _recv_json(ws))["text"] == _auth("frame-token")

    assert wired.tokens == ["legacy"]


async def test_empty_query_token_falls_back_to_frame(wired) -> None:
    async with _serve() as url, websockets.connect(f"{url}?token=") as ws:
        await ws.send(_auth())
        assert (await _recv_json(ws))["type"] == "scenario_info"

    assert wired.tokens == [_JWT]
    assert _ws_auth(wired)["method"] == "frame"


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(b"\x00\x01" * 160, id="binary-audio"),
        pytest.param(_PLAYBACK, id="playback-capabilities"),
        pytest.param("not json", id="not-json"),
        pytest.param(json.dumps(["auth", _JWT]), id="json-list"),
        pytest.param(json.dumps("auth"), id="json-string"),
        pytest.param(json.dumps({"type": "auth"}), id="missing-token"),
        pytest.param(_auth(""), id="empty-token"),
        pytest.param(_auth(123), id="non-str-token"),
        pytest.param(_auth("\ud800"), id="lone-surrogate-token"),
        pytest.param(_auth("토큰"), id="non-ascii-token"),
        pytest.param(json.dumps({"type": "ping", "token": _JWT}), id="wrong-type"),
    ],
)
async def test_bad_first_frame_closes_auth_required(wired, first) -> None:
    async with _serve() as url, websockets.connect(url) as ws:
        await ws.send(first)
        assert await _close_of(ws) == (1008, "AUTH_REQUIRED")

    assert wired.tokens == []
    assert _ws_auth(wired) == {"session_id": "s1", "method": "frame", "result": "bad_frame"}


async def test_auth_frame_size_limit(wired) -> None:
    def frame_of(size: int) -> str:
        frame = _auth("t" * size)
        return _auth("t" * (size - (len(frame) - size)))

    at_limit, over_limit = frame_of(8192), frame_of(8193)
    assert (len(at_limit), len(over_limit)) == (8192, 8193)

    async with _serve() as url:
        async with websockets.connect(url) as ws:
            await ws.send(at_limit)
            assert (await _recv_json(ws))["type"] == "scenario_info"
        async with websockets.connect(url) as ws:
            await ws.send(over_limit)
            assert await _close_of(ws) == (1008, "AUTH_REQUIRED")

    assert len(wired.tokens) == 1


async def test_no_auth_frame_times_out(wired, monkeypatch) -> None:
    settings = get_settings().model_copy(update={"ws_auth_timeout_seconds": 0.2})
    monkeypatch.setattr(ws_mod, "get_settings", lambda: settings)

    async with _serve() as url, websockets.connect(url) as ws:
        started = time.monotonic()
        assert await _close_of(ws) == (1008, "AUTH_TIMEOUT")
        assert time.monotonic() - started < 2.0

    assert wired.tokens == []
    assert _ws_auth(wired) == {"session_id": "s1", "method": "frame", "result": "timeout"}


def test_auth_timeout_default_is_ten_seconds() -> None:
    assert get_settings().ws_auth_timeout_seconds == 10.0


async def test_client_leaving_before_auth_frame_is_quiet(wired, caplog, monkeypatch) -> None:
    closes: list[tuple] = []
    original_close = WebSocket.close

    async def _spy_close(self, *args, **kwargs):
        closes.append((args, kwargs))
        return await original_close(self, *args, **kwargs)

    monkeypatch.setattr(WebSocket, "close", _spy_close)
    caplog.set_level(logging.INFO)
    async with _serve() as url:
        ws = await websockets.connect(url)
        await ws.close()
        await _wait_metric(wired)

    assert wired.tokens == []
    assert _ws_auth(wired) == {"session_id": "s1", "method": "frame", "result": "disconnected"}
    assert closes == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize(
    ("claims", "reason"),
    [
        pytest.param({"sub": "7", "type": "ACCESS", "exp": 1}, "TOKEN_EXPIRED", id="expired"),
        pytest.param({"sub": "7", "type": "REFRESH", "exp": 4102444800}, "NOT_ACCESS_TOKEN", id="refresh"),
        pytest.param(None, "TOKEN_INVALID", id="garbage"),
    ],
)
async def test_frame_token_rejections_keep_close_reasons(real_decode, claims, reason) -> None:
    settings = get_settings()
    token = _JWT if claims is None else jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    async with _serve() as url, websockets.connect(url) as ws:
        await ws.send(_auth(token))
        assert await _close_of(ws) == (1008, reason)

    assert real_decode.db_calls == []
    assert _ws_auth(real_decode) == {"session_id": "s1", "method": "frame", "result": "rejected"}


async def test_query_token_rejection_keeps_close_reason(real_decode) -> None:
    async with _serve() as url, websockets.connect(f"{url}?token={_JWT}") as ws:
        assert await _close_of(ws) == (1008, "TOKEN_INVALID")

    assert _ws_auth(real_decode) == {"session_id": "s1", "method": "query", "result": "rejected"}


async def test_token_and_frame_never_logged(real_decode, caplog) -> None:
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.DEBUG, logger="app")
    bad_frame = json.dumps({"type": "nope", "token": _JWT})

    async with _serve() as url:
        async with websockets.connect(url) as ws:
            await ws.send(_auth(_JWT))
            assert await _close_of(ws) == (1008, "TOKEN_INVALID")
        async with websockets.connect(url) as ws:
            await ws.send(bad_frame)
            assert await _close_of(ws) == (1008, "AUTH_REQUIRED")

    accepted = [r.getMessage() for r in caplog.records if "[accepted]" in r.getMessage()]
    assert accepted and all('"WebSocket /ws/voice/s1"' in line for line in accepted)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert _JWT not in logged
    assert '"nope"' not in logged
    assert all(_JWT not in str(vars(r)) for r in caplog.records)
    assert all(_JWT not in str(fields) for _, fields in real_decode.metrics)


async def test_unexpected_auth_error_still_closes_socket(wired, monkeypatch) -> None:
    async def _boom(ws, token):
        raise ValueError("boom")

    monkeypatch.setattr(ws_mod, "authenticate_ws", _boom)
    async with _serve() as url, websockets.connect(url) as ws:
        await ws.send(_auth())
        assert (await _close_of(ws))[0] == 1011

    assert _ws_auth(wired) == {"session_id": "s1", "method": "frame", "result": "rejected"}


@pytest.mark.parametrize("legacy", [False, True], ids=["frame", "query"])
async def test_metric_logged_once_even_when_session_check_fails(wired, monkeypatch, legacy) -> None:
    async def _no_session(ws, redis, session_id, user_id):
        await ws.close(code=1008, reason="SESSION_NOT_FOUND")
        raise RuntimeError("no session")

    monkeypatch.setattr(ws_mod, "authenticate_session", _no_session)
    async with _serve() as url, websockets.connect(f"{url}?token=legacy" if legacy else url) as ws:
        if not legacy:
            await ws.send(_auth())
        assert await _close_of(ws) == (1008, "SESSION_NOT_FOUND")

    assert _ws_auth(wired) == {"session_id": "s1", "method": "query" if legacy else "frame", "result": "ok"}


async def test_metric_session_id_cannot_forge_log_lines(wired) -> None:
    async with _serve() as url:
        forged = url.replace("/s1", "/x%20method=query%20result=ok%0Ametric=ws_auth")
        async with websockets.connect(forged) as ws:
            await ws.send("nope")
            assert await _close_of(ws) == (1008, "AUTH_REQUIRED")

    assert _ws_auth(wired) == {"session_id": "invalid", "method": "frame", "result": "bad_frame"}


async def test_uuid_session_id_kept_in_metric(wired) -> None:
    sid = "0b6f4c1e-3a5d-4b8e-9f21-7c2d1e0a9b34"
    async with _serve() as url, websockets.connect(url.replace("/s1", f"/{sid}")) as ws:
        await ws.send("nope")
        assert await _close_of(ws) == (1008, "AUTH_REQUIRED")

    assert _ws_auth(wired)["session_id"] == sid


async def test_token_not_logged_on_successful_frame_auth(wired, caplog) -> None:
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.DEBUG, logger="app")
    async with _serve() as url, websockets.connect(url) as ws:
        await ws.send(_auth())
        assert (await _recv_json(ws))["type"] == "scenario_info"
        await ws.send(_PLAYBACK)
        assert (await _recv_json(ws))["type"] == "pipeline_got"

    assert caplog.records
    assert all(_JWT not in str(vars(r)) for r in caplog.records)
