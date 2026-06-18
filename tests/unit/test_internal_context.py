from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app.api.v1 import internal
from app.core import security
from app.core.security import require_internal_secret
from app.deps.db import get_db


class _FakeSession:
    """ScenarioORM 한 행만 돌려주는 가짜 비동기 세션."""

    def __init__(self, row: object | None) -> None:
        self._row = row

    async def get(self, _model: object, _pk: int) -> object | None:
        return self._row


def _make_app(db_row: object | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(internal.router, prefix="/internal/v1")
    app.dependency_overrides[require_internal_secret] = lambda: None

    async def _fake_db():
        yield _FakeSession(db_row)

    app.dependency_overrides[get_db] = _fake_db
    return app


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def test_preset_context_returns_camelcase_with_script() -> None:
    resp = await _get(_make_app(), "/internal/v1/scenarios/1/context")

    assert resp.status_code == 200
    body = resp.json()
    assert body["title"]
    assert body["aiRole"]
    assert len(body["script"]) > 0
    first = body["script"][0]
    assert set(first.keys()) == {"step", "aiGoal", "hint"}
    assert first["step"] == 1


async def test_custom_context_returns_stored_script() -> None:
    row = SimpleNamespace(
        title="구청 민원 문의",
        call_target="구청 민원실 직원",
        script=[
            {"step": 2, "ai_goal": "필요 서류를 안내한다", "hint": "어떤 서류가 필요한지 물어보세요"},
            {"step": 1, "ai_goal": "전화를 응대하고 용건을 확인한다", "hint": "민원을 문의한다고 말하세요"},
        ],
    )
    resp = await _get(_make_app(db_row=row), "/internal/v1/scenarios/9999/context")

    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "구청 민원 문의"
    assert body["aiRole"] == "구청 민원실 직원"
    assert len(body["script"]) == 2
    assert set(body["script"][0].keys()) == {"step", "aiGoal", "hint"}


async def test_custom_context_without_script_is_empty() -> None:
    row = SimpleNamespace(title="커스텀 시나리오", call_target="구청 민원실 직원", script=None)
    resp = await _get(_make_app(db_row=row), "/internal/v1/scenarios/9999/context")

    assert resp.status_code == 200
    assert resp.json()["script"] == []


async def test_unknown_scenario_returns_404() -> None:
    resp = await _get(_make_app(db_row=None), "/internal/v1/scenarios/9999/context")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "SCENARIO_NOT_FOUND"


async def test_require_internal_secret_rejects_wrong(monkeypatch) -> None:
    monkeypatch.setattr(
        security, "get_settings", lambda: SimpleNamespace(internal_secret="expected")
    )
    with pytest.raises(HTTPException) as exc:
        await require_internal_secret("wrong")
    assert exc.value.status_code == 403


async def test_require_internal_secret_accepts_match(monkeypatch) -> None:
    monkeypatch.setattr(
        security, "get_settings", lambda: SimpleNamespace(internal_secret="expected")
    )
    assert await require_internal_secret("expected") is None
