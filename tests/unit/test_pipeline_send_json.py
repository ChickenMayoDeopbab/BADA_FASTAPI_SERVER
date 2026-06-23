import logging

import pytest
from fastapi import WebSocketDisconnect

from app.services.pipeline import VoicePipeline


def _make_pipeline(ws) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = ws
    p._session_id = "sess-test"
    p._ws_alive = True
    return p


class _RaisingWS:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def send_json(self, payload: dict) -> None:
        raise self._exc


@pytest.mark.asyncio
async def test_send_json_disconnect_is_quiet(caplog) -> None:
    """예상된 끊김 예외는 한 줄 로그."""
    p = _make_pipeline(_RaisingWS(WebSocketDisconnect(code=1006)))
    with caplog.at_level(logging.INFO, logger="app.services.pipeline"):
        await p._send_json({"type": "ping"})

    assert p._ws_alive is False
    recs = [r for r in caplog.records if r.name == "app.services.pipeline"]
    assert len(recs) == 1
    assert recs[0].exc_info is None
    assert "WebSocketDisconnect" in recs[0].getMessage()


@pytest.mark.asyncio
async def test_send_json_runtime_error_is_quiet(caplog) -> None:
    """이미 닫힌 ws 송신도 예상된 끊김으로 조용히 처리"""
    p = _make_pipeline(_RaisingWS(RuntimeError("Cannot call send once closed")))
    with caplog.at_level(logging.INFO, logger="app.services.pipeline"):
        await p._send_json({"type": "ping"})

    assert p._ws_alive is False
    recs = [r for r in caplog.records if r.name == "app.services.pipeline"]
    assert len(recs) == 1
    assert recs[0].exc_info is None


@pytest.mark.asyncio
async def test_send_json_unexpected_keeps_traceback(caplog) -> None:
    """예상 못 한 예외는 traceback 유지."""
    p = _make_pipeline(_RaisingWS(ValueError("boom")))
    with caplog.at_level(logging.WARNING, logger="app.services.pipeline"):
        await p._send_json({"type": "ping"})

    assert p._ws_alive is False
    recs = [r for r in caplog.records if r.name == "app.services.pipeline"]
    assert len(recs) == 1
    assert recs[0].levelno >= logging.WARNING
    assert recs[0].exc_info is not None


@pytest.mark.asyncio
async def test_send_json_noop_when_already_dead(caplog) -> None:
    """ws_alive=False 면 송신 시도 안함"""
    p = _make_pipeline(_RaisingWS(ValueError("should not be called")))
    p._ws_alive = False
    with caplog.at_level(logging.DEBUG, logger="app.services.pipeline"):
        await p._send_json({"type": "ping"})

    assert p._ws_alive is False
    assert [r for r in caplog.records if r.name == "app.services.pipeline"] == []
