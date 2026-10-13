import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager, suppress

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket

from tests.unit.test_avti_save import _pipeline
from tests.unit.test_ws_frame_limit import PROD_WS, _launch_params

_TEARDOWN_WORK_S = 1.0
_MIC_CHUNK, _MIC_INTERVAL_S = 2048, 0.02
_WAIT_S = 25.0


def _teardown_app(result: dict, *, never_close: bool) -> FastAPI:
    api = FastAPI()

    @api.websocket("/t")
    async def _route(ws: WebSocket) -> None:
        await ws.accept()
        p = _pipeline(seconds=0, turns=[], spans=[])
        p._ws = ws

        async def _slow_notify(*args, **kwargs) -> bool:
            await asyncio.sleep(_TEARDOWN_WORK_S)
            result["spring"] = True
            return True

        p._spring.notify_session_closed = _slow_notify
        if never_close:
            async def _no_close(*args, **kwargs) -> None:
                return None

            p._ws.close = _no_close
        started = time.monotonic()
        await p._teardown()
        result["teardown_s"] = time.monotonic() - started

    return api


@asynccontextmanager
async def _serve(result: dict, *, never_close: bool = False):
    p = _launch_params()
    server = uvicorn.Server(
        uvicorn.Config(
            _teardown_app(result, never_close=never_close),
            host="127.0.0.1",
            port=0,
            lifespan="off",
            log_config=None,
            ws=PROD_WS,
            ws_max_size=p["ws_max_size"],
            ws_max_queue=p["ws_max_queue"],
            ws_per_message_deflate=p["ws_per_message_deflate"],
        )
    )
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            assert not task.done(), "uvicorn 이 뜨지 못함"
            await asyncio.sleep(0.01)
        yield f"ws://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}/t"
    finally:
        server.should_exit = True
        await task


async def _wait_for(result: dict, key: str) -> None:
    deadline = time.monotonic() + _WAIT_S
    while key not in result:
        assert time.monotonic() < deadline, f"{key} 가 안 남음: {result}"
        await asyncio.sleep(0.02)


async def test_mic_streaming_during_teardown_does_not_stall_close() -> None:
    result: dict = {}
    async with _serve(result) as url, websockets.connect(url) as ws:

        async def _mic() -> None:
            while True:
                await ws.send(os.urandom(_MIC_CHUNK))
                await asyncio.sleep(_MIC_INTERVAL_S)

        mic = asyncio.create_task(_mic())
        try:
            while json.loads(await asyncio.wait_for(ws.recv(), _WAIT_S)).get("type") != "end":
                pass
        finally:
            mic.cancel()
            with suppress(asyncio.CancelledError, websockets.ConnectionClosed):
                await mic
        await _wait_for(result, "teardown_s")

    assert result["teardown_s"] < _TEARDOWN_WORK_S + 2.0, (
        f"종료 처리 {_TEARDOWN_WORK_S}s 인데 _teardown 이 {result['teardown_s']:.1f}s — "
        "쌓인 프레임 때문에 클라의 close 를 못 읽어 close_timeout 을 기다렸다"
    )


async def test_client_leaving_mid_teardown_still_finishes(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app")
    result: dict = {}
    async with _serve(result) as url:
        async with websockets.connect(url) as ws:
            for _ in range(10):
                await ws.send(os.urandom(_MIC_CHUNK))
        await _wait_for(result, "teardown_s")

    assert result["spring"] is True
    assert result["teardown_s"] < _TEARDOWN_WORK_S + 2.0
    sends = [r for r in caplog.records if "ws 송신 중단" in r.getMessage()]
    assert not sends, "떠난 클라에게 end 를 보내려 하면 안 된다"


async def test_teardown_never_waits_on_the_drain() -> None:
    result: dict = {}
    async with _serve(result, never_close=True) as url, websockets.connect(url) as ws:
        while json.loads(await asyncio.wait_for(ws.recv(), _WAIT_S)).get("type") != "end":
            pass
        await _wait_for(result, "teardown_s")

    assert result["teardown_s"] < _TEARDOWN_WORK_S + 2.0
