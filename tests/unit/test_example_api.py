from types import SimpleNamespace

import httpx
from anthropic import APIError
from fastapi import FastAPI
from websockets.exceptions import WebSocketException

import app.api.v1.scenario as scenario_api
from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.schemas.scenario import ExampleConversationResponse, ExampleTurn
from app.services.example_service import ScenarioNotFoundError


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(scenario_api.router)
    app.dependency_overrides[get_current_user_id] = lambda: 7

    async def _fake_db():
        yield SimpleNamespace()

    app.dependency_overrides[get_db] = _fake_db
    return app


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


def _svc_returning(response: ExampleConversationResponse):
    async def _svc(db, scenario_id, user_id):
        return response

    return _svc


def _svc_raising(exc: Exception):
    async def _svc(db, scenario_id, user_id):
        raise exc

    return _svc


async def test_example_returns_dialogue_and_audio_url(monkeypatch) -> None:
    response = ExampleConversationResponse(
        scenario_id=1,
        dialogue=[
            ExampleTurn(speaker="ai", text="네, 바다레스토랑입니다."),
            ExampleTurn(speaker="user", text="예약하려고요."),
        ],
        audio_url="https://bucket.s3.test/examples/1-abcd1234.wav",
    )
    monkeypatch.setattr(scenario_api, "svc_get_example_conversation", _svc_returning(response))

    resp = await _get(_make_app(), "/api/v1/scenario/1/example")

    assert resp.status_code == 200
    body = resp.json()
    assert body["scenario_id"] == 1
    assert body["dialogue"][0] == {"speaker": "ai", "text": "네, 바다레스토랑입니다."}
    assert body["audio_url"] == "https://bucket.s3.test/examples/1-abcd1234.wav"


async def test_example_not_found_maps_404(monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_api, "svc_get_example_conversation", _svc_raising(ScenarioNotFoundError("없음"))
    )

    resp = await _get(_make_app(), "/api/v1/scenario/999/example")

    assert resp.status_code == 404


async def test_example_llm_error_maps_502(monkeypatch) -> None:
    exc = APIError("LLM 실패", httpx.Request("POST", "http://api.test"), body=None)
    monkeypatch.setattr(scenario_api, "svc_get_example_conversation", _svc_raising(exc))

    resp = await _get(_make_app(), "/api/v1/scenario/42/example")

    assert resp.status_code == 502


async def test_example_tts_error_maps_502(monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_api, "svc_get_example_conversation", _svc_raising(WebSocketException("TTS 실패"))
    )

    resp = await _get(_make_app(), "/api/v1/scenario/1/example")

    assert resp.status_code == 502


async def test_example_parse_error_maps_500(monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_api, "svc_get_example_conversation", _svc_raising(ValueError("파싱 오류"))
    )

    resp = await _get(_make_app(), "/api/v1/scenario/42/example")

    assert resp.status_code == 500
