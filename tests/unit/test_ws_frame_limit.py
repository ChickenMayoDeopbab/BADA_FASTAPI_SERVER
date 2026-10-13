import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import uvicorn
import websockets
from uvicorn.main import main as uvicorn_cli

import app.api.v1.websocket as ws_mod
from app.deps.redis import get_redis
from app.main import app

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"
_RECV_TIMEOUT = 3.0
_LARGEST_APP_FRAME = 8192


def _dockerfile_cmd() -> list[str]:
    cmds = [line for line in DOCKERFILE.read_text().splitlines() if line.startswith("CMD ")]
    assert len(cmds) == 1, f"Dockerfile CMD 가 정확히 하나여야 한다: {cmds}"
    return json.loads(cmds[0].removeprefix("CMD "))


def _launch_params() -> dict:
    cmd = _dockerfile_cmd()
    assert cmd[:2] == ["uvicorn", "app.main:app"], f"운영 기동 명령이 바뀌었다: {cmd}"
    return uvicorn_cli.make_context("uvicorn", cmd[1:]).params


def test_cmd_caps_ws_frame_size() -> None:
    assert _launch_params()["ws_max_size"] == 64 * 1024, (
        "uvicorn 기본 16 MiB 면 인증 전 클라가 deflate 로 잘 압축되는 프레임을 max_queue(32)개 쌓아 "
        "연결 하나로 ~512 MiB 를 잡는다(계획 0062 실측)"
    )


def test_cap_fits_legit_frames() -> None:
    cap = _launch_params()["ws_max_size"]
    assert cap >= _LARGEST_APP_FRAME
    assert cap >= ws_mod._AUTH_FRAME_MAX_CHARS * 4, (
        "인증 프레임 글자 상한이 UTF-8 최악(4바이트)에서도 프레임 상한 안에 들어야 한다"
    )


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
        await self._ws.send_json({"type": "pipeline_got", "bytes": len(msg.get("bytes") or b"")})
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


@asynccontextmanager
async def _serve():
    p = _launch_params()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        lifespan="off",
        log_config=None,
        ws=p["ws"],
        ws_max_size=p["ws_max_size"],
        ws_max_queue=p["ws_max_queue"],
        ws_per_message_deflate=p["ws_per_message_deflate"],
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            assert not task.done(), "uvicorn 이 뜨지 못함"
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws/voice/s1", p["ws_max_size"]
    finally:
        server.should_exit = True
        await task


async def _close_after(ws, frame: str | bytes) -> tuple[int | None, str]:
    try:
        await ws.send(frame)
        while True:
            await asyncio.wait_for(ws.recv(), _RECV_TIMEOUT)
    except websockets.ConnectionClosed as exc:
        return (exc.rcvd.code, exc.rcvd.reason) if exc.rcvd else (None, "")


async def _ws_auth_result(state: _Wired) -> str:
    deadline = time.monotonic() + _RECV_TIMEOUT
    while not state.metrics:
        assert time.monotonic() < deadline, "metric=ws_auth 가 안 남음"
        await asyncio.sleep(0.01)
    assert [name for name, _ in state.metrics] == ["ws_auth"]
    return state.metrics[0][1]["result"]


async def test_oversized_compressible_frame_before_auth_closes_1009(wired) -> None:
    async with _serve() as (url, cap), websockets.connect(url) as ws:
        assert ws.protocol.extensions, "deflate 가 협상돼야 압축 해제 뒤 크기로 막는지 볼 수 있다"
        assert await _close_after(ws, bytes(cap + 1)) == (1009, "")

    assert wired.tokens == []
    assert await _ws_auth_result(wired) == "disconnected"


async def test_frame_at_cap_still_reaches_app(wired) -> None:
    async with _serve() as (url, cap), websockets.connect(url) as ws:
        assert await _close_after(ws, "x" * cap) == (1008, "AUTH_REQUIRED")

    assert await _ws_auth_result(wired) == "bad_frame"


@pytest.mark.parametrize(
    ("size", "largest_ok"),
    [pytest.param(_LARGEST_APP_FRAME, True, id="largest-app-frame"), pytest.param(None, False, id="cap+1")],
)
async def test_frame_limit_after_auth(wired, size, largest_ok) -> None:
    async with _serve() as (url, cap), websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "auth", "token": "t"}))
        assert json.loads(await asyncio.wait_for(ws.recv(), _RECV_TIMEOUT))["type"] == "scenario_info"
        frame = os.urandom(size or cap + 1)
        if largest_ok:
            await ws.send(frame)
            assert json.loads(await asyncio.wait_for(ws.recv(), _RECV_TIMEOUT)) == {
                "type": "pipeline_got",
                "bytes": len(frame),
            }
        else:
            assert await _close_after(ws, frame) == (1009, "")

    assert wired.tokens == ["t"]
