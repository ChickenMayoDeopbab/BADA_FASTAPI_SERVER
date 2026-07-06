from fastapi.testclient import TestClient

import app.api.v1.websocket as ws_mod
from app.deps.redis import get_redis
from app.main import app


class _BoomPipeline:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("init boom")


class _MarkerPipeline:

    def __init__(self, ws, *args, **kwargs) -> None:
        self._ws = ws

    async def run(self) -> None:
        await self._ws.send_json({"type": "pipeline_marker"})


def _patch_auth(monkeypatch, session: dict | None = None) -> None:
    async def _fake_auth(ws, token):
        return 1, "ROLE_USER"

    async def _fake_session(ws, redis, session_id, user_id):
        return session if session is not None else {}

    monkeypatch.setattr(ws_mod, "authenticate_ws", _fake_auth)
    monkeypatch.setattr(ws_mod, "authenticate_session", _fake_session)


def _first_frame(monkeypatch, session: dict) -> dict:
    _patch_auth(monkeypatch, session)
    monkeypatch.setattr(ws_mod, "VoicePipeline", _MarkerPipeline)
    app.dependency_overrides[get_redis] = lambda: None
    try:
        client = TestClient(app)
        with client.websocket_connect("/ws/voice/s1?token=t") as ws:
            return ws.receive_json()
    finally:
        app.dependency_overrides.clear()


def test_connect_sends_scenario_info_with_ai_role(monkeypatch) -> None:
    session = {"scenario": {"aiRole": "병원 접수 데스크 직원"}}
    frame = _first_frame(monkeypatch, session)
    assert frame == {"type": "scenario_info", "aiRole": "병원 접수 데스크 직원"}


def test_connect_sends_empty_ai_role_when_scenario_missing(monkeypatch) -> None:
    frame = _first_frame(monkeypatch, {})
    assert frame == {"type": "scenario_info", "aiRole": ""}


def test_connect_sends_empty_ai_role_when_scenario_not_dict(monkeypatch) -> None:
    frame = _first_frame(monkeypatch, {"scenario": "broken"})
    assert frame == {"type": "scenario_info", "aiRole": ""}


def test_init_failure_sends_error_frame_then_1011(monkeypatch) -> None:
    _patch_auth(monkeypatch)
    monkeypatch.setattr(ws_mod, "VoicePipeline", _BoomPipeline)
    app.dependency_overrides[get_redis] = lambda: None
    try:
        client = TestClient(app)
        with client.websocket_connect("/ws/voice/s1?token=t") as ws:
            frame = ws.receive_json()
            assert frame == {"type": "error", "code": "PIPELINE_INIT_FAILED"}
            closed = ws.receive()
            assert closed["type"] == "websocket.close"
            assert closed["code"] == 1011
    finally:
        app.dependency_overrides.clear()
