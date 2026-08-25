from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.api.v1 import internal
from app.api.v1.scenario import router as scenario_router
from app.core.enums import ScenarioCategory
from app.core.preset_scenarios import PRESET_MAP, PRESET_SCENARIOS
from app.core.security import require_internal_secret
from app.db.seed import seed_preset_scenarios
from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.services import example_service
from app.services.example_service import ScenarioNotFoundError, get_example_conversation
from app.services.scenario_service import delete_custom_scenario, get_scenarios
from tests.unit.test_scenario_visibility import _FakeSeedDB
from tests.unit.test_scenario_visibility import _scenario_orm as _seed_scenario_orm


class _FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list:
        return self._rows


class _FakeDB:
    def __init__(self, rows: list) -> None:
        self.rows = {row.scenario_id: row for row in rows}
        self.commits = 0
        self.locked_gets = 0

    async def get(
        self, _model: object, key: int, with_for_update: bool = False
    ) -> object | None:
        if with_for_update:
            self.locked_gets += 1
        return self.rows.get(key)

    async def commit(self) -> None:
        self.commits += 1

    async def execute(self, _stmt: object) -> _FakeResult:
        return _FakeResult(list(self.rows.values()))


def _custom_row(scenario_id: int, user_id: int | None, **kw) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=scenario_id,
        title=kw.get("title", f"커스텀 {scenario_id}"),
        content="설명",
        category=kw.get("category", ScenarioCategory.OTHER.value),
        scenario_image=None,
        tts_voice_id=None,
        ai_prompt="prompt",
        user_id=user_id,
        is_custom=True,
        is_warmup=kw.get("is_warmup", False),
        call_target="상대",
        call_purpose="목적",
        script=[{"step": 1, "ai_goal": "인사", "hint": "인사하세요"}],
        created_at=kw.get("created_at", datetime(2026, 1, 1)),
        deleted_at=kw.get("deleted_at"),
    )


def _preset_row(scenario_id: int, **kw) -> SimpleNamespace:
    seed = PRESET_MAP[scenario_id]
    return SimpleNamespace(
        scenario_id=scenario_id,
        title=seed["title"],
        content=seed["content"],
        category=seed["category"].value,
        scenario_image=seed["scenario_image"],
        tts_voice_id=seed["tts_voice_id"],
        ai_prompt=seed["ai_prompt"],
        user_id=None,
        is_custom=False,
        is_warmup=False,
        call_target=seed["call_target"],
        call_purpose=seed["call_purpose"],
        script=seed["script"],
        created_at=datetime(2026, 1, 1),
        deleted_at=kw.get("deleted_at"),
    )


def _ids(response) -> list[int]:
    return [s.scenario_id for s in response.scenarios]



async def test_owner_soft_deletes_own_custom() -> None:
    row = _custom_row(101, user_id=1)
    db = _FakeDB([row])

    assert await delete_custom_scenario(db, 101, user_id=1) is True
    assert row.deleted_at is not None
    assert db.commits == 1
    assert 101 in db.rows
    assert db.locked_gets == 1


async def test_delete_other_users_custom_refused() -> None:
    row = _custom_row(101, user_id=2)
    db = _FakeDB([row])

    assert await delete_custom_scenario(db, 101, user_id=1) is False
    assert row.deleted_at is None
    assert db.commits == 0


async def test_delete_preset_refused() -> None:
    row = _preset_row(1)
    db = _FakeDB([row])

    assert await delete_custom_scenario(db, 1, user_id=1) is False
    assert row.deleted_at is None
    assert db.commits == 0


async def test_delete_missing_refused() -> None:
    db = _FakeDB([])

    assert await delete_custom_scenario(db, 999, user_id=1) is False
    assert db.commits == 0


async def test_delete_already_deleted_refused() -> None:
    stamp = datetime(2026, 8, 1)
    row = _custom_row(101, user_id=1, deleted_at=stamp)
    db = _FakeDB([row])

    assert await delete_custom_scenario(db, 101, user_id=1) is False
    assert row.deleted_at == stamp
    assert db.commits == 0


async def test_owner_can_delete_own_warmup() -> None:
    row = _custom_row(101, user_id=1, is_warmup=True)
    db = _FakeDB([row])

    assert await delete_custom_scenario(db, 101, user_id=1) is True
    assert row.deleted_at is not None



async def test_deleted_custom_hidden_from_owner_list() -> None:
    rows = [
        _custom_row(101, user_id=1, deleted_at=datetime(2026, 8, 1)),
        _custom_row(102, user_id=1),
    ]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert _ids(resp) == [102]


async def test_preset_row_deleted_at_ignored_in_list() -> None:
    first_id = PRESET_SCENARIOS[0]["scenario_id"]
    rows = [
        _preset_row(s["scenario_id"], deleted_at=datetime(2026, 8, 1) if s["scenario_id"] == first_id else None)
        for s in PRESET_SCENARIOS
    ]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert _ids(resp) == [s["scenario_id"] for s in PRESET_SCENARIOS]



async def test_seed_relocation_preserves_soft_delete() -> None:
    stamp = datetime(2026, 8, 1)
    conflicting = _seed_scenario_orm(
        1, title="삭제된 커스텀", user_id=7, is_custom=True,
        call_target="상대", call_purpose="목적",
    )
    conflicting.deleted_at = stamp
    db = _FakeSeedDB([conflicting])

    await seed_preset_scenarios(db)

    moved = [row for row in db.scenarios.values() if row.is_custom]
    assert len(moved) == 1
    assert moved[0].deleted_at == stamp



def _make_app(rows: list, user_id: int | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(scenario_router)
    db = _FakeDB(rows)

    async def _fake_db():
        yield db

    app.dependency_overrides[get_db] = _fake_db
    if user_id is not None:
        app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.state.fake_db = db
    return app


async def _delete(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.delete(path)


async def test_delete_endpoint_returns_204_and_soft_deletes() -> None:
    row = _custom_row(101, user_id=1)
    app = _make_app([row], user_id=1)

    resp = await _delete(app, "/api/v1/scenario/custom/101")

    assert resp.status_code == 204
    assert resp.content == b""
    assert row.deleted_at is not None


async def test_delete_endpoint_404_for_others_preset_missing_deleted() -> None:
    rows = [
        _custom_row(101, user_id=2),
        _preset_row(1),
        _custom_row(103, user_id=1, deleted_at=datetime(2026, 8, 1)),
    ]
    app = _make_app(rows, user_id=1)

    for scenario_id in (101, 1, 999, 103):
        resp = await _delete(app, f"/api/v1/scenario/custom/{scenario_id}")
        assert resp.status_code == 404, scenario_id


async def test_delete_endpoint_requires_auth() -> None:
    app = _make_app([_custom_row(101, user_id=1)])

    resp = await _delete(app, "/api/v1/scenario/custom/101")

    assert resp.status_code in (401, 403)



async def test_example_of_deleted_custom_not_found(monkeypatch) -> None:
    monkeypatch.setattr(
        example_service,
        "get_settings",
        lambda: SimpleNamespace(
            elevenlabs_voice_id="voice-default",
            s3_bucket=None,
            anthropic_api_key="k",
            llm_analysis_model="m",
        ),
    )
    row = _custom_row(42, user_id=7, deleted_at=datetime(2026, 8, 1))

    with pytest.raises(ScenarioNotFoundError):
        await get_example_conversation(_FakeDB([row]), 42, user_id=7)



def _make_internal_app(rows: list) -> FastAPI:
    app = FastAPI()
    app.include_router(internal.router, prefix="/internal/v1")
    app.dependency_overrides[require_internal_secret] = lambda: None
    db = _FakeDB(rows)

    async def _fake_db():
        yield db

    app.dependency_overrides[get_db] = _fake_db
    return app


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def test_internal_context_404_for_deleted_custom() -> None:
    row = _custom_row(999, user_id=1, deleted_at=datetime(2026, 8, 1))
    app = _make_internal_app([row])

    resp = await _get(app, "/internal/v1/scenarios/999/context")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "SCENARIO_NOT_FOUND"


async def test_internal_context_preset_fallback_when_deleted_row_shares_id() -> None:
    row = _custom_row(1, user_id=1, deleted_at=datetime(2026, 8, 1))
    app = _make_internal_app([row])

    resp = await _get(app, "/internal/v1/scenarios/1/context")

    assert resp.status_code == 200
    assert resp.json()["title"] == PRESET_MAP[1]["title"]
