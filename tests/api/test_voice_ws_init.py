from fastapi.testclient import TestClient

import app.api.v1.websocket as ws_mod
from app.deps.redis import get_redis
from app.main import app


class _BoomPipeline:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("init boom")


def _patch_auth(monkeypatch) -> None:
    async def _fake_auth(ws, token):
        return 1, "ROLE_USER"

    async def _fake_session(ws, redis, session_id, user_id):
        return {}

    monkeypatch.setattr(ws_mod, "authenticate_ws", _fake_auth)
    monkeypatch.setattr(ws_mod, "authenticate_session", _fake_session)


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
