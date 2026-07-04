import json
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from app.api.v1.scenario import router as scenario_router
from app.core.enums import ScenarioCategory
from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.schemas.scenario import CustomSessionRequest
from app.services import scenario_service
from app.services.scenario_service import create_custom_scenario, get_scenarios


class _FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list:
        return self._rows


class _FakeDB:
    """execute→scalars→all 체인만 흉내내는 읽기 전용 가짜 세션."""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    async def execute(self, _stmt: object) -> _FakeResult:
        return _FakeResult(self._rows)


def _custom_row(scenario_id: int, user_id: int | None, **kw) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=scenario_id,
        title=kw.get("title", f"커스텀 {scenario_id}"),
        content="설명",
        scenario_image=None,
        tts_voice_id=None,
        ai_prompt="prompt",
        user_id=user_id,
        is_custom=True,
        is_warmup=kw.get("is_warmup", False),
    )


def _preset_row(scenario_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=scenario_id,
        title="음식점 예약",
        content="설명",
        scenario_image=None,
        tts_voice_id=None,
        ai_prompt="prompt",
        user_id=None,
        is_custom=False,
        is_warmup=False,
    )


def _ids(response) -> list[int]:
    return [s.scenario_id for s in response.scenarios]


async def test_list_shows_only_own_customs() -> None:
    rows = [_custom_row(101, user_id=1), _custom_row(102, user_id=2)]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert _ids(resp) == [101]


async def test_null_owner_custom_hidden_from_everyone() -> None:
    rows = [_custom_row(101, user_id=None)]

    for requester in (1, 2):
        resp = await get_scenarios(_FakeDB(rows), None, user_id=requester)
        assert _ids(resp) == []


async def test_presets_visible_to_all_users() -> None:
    rows = [_preset_row(1), _custom_row(101, user_id=2)]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert _ids(resp) == [1]
    assert resp.scenarios[0].is_custom is False


async def test_custom_category_filter_combines_with_owner() -> None:
    rows = [
        _preset_row(1),
        _custom_row(101, user_id=1),
        _custom_row(102, user_id=2),
    ]

    resp = await get_scenarios(
        _FakeDB(rows), ScenarioCategory.CUSTOM, user_id=1
    )

    assert _ids(resp) == [101]


def _make_app(rows: list, user_id: int | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(scenario_router)

    async def _fake_db():
        yield _FakeDB(rows)

    app.dependency_overrides[get_db] = _fake_db
    if user_id is not None:
        app.dependency_overrides[get_current_user_id] = lambda: user_id
    return app


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def test_list_requires_auth() -> None:
    resp = await _get(_make_app(rows=[]), "/api/v1/scenario/scenarios")

    assert resp.status_code in (401, 403)


async def test_empty_db_falls_back_to_presets() -> None:
    resp = await _get(_make_app(rows=[], user_id=1), "/api/v1/scenario/scenarios")

    assert resp.status_code == 200
    scenarios = resp.json()["scenarios"]
    assert len(scenarios) > 0
    assert all(s["is_custom"] is False for s in scenarios)


async def test_warmup_custom_hidden_even_from_owner() -> None:
    rows = [
        _custom_row(101, user_id=1, is_warmup=True),
        _custom_row(102, user_id=1),
    ]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert _ids(resp) == [102]


class _FakeWriteDB:
    def __init__(self) -> None:
        self.added: object | None = None

    def add(self, obj: object) -> None:
        self.added = obj

    async def commit(self) -> None:
        pass

    async def refresh(self, obj: object) -> None:
        obj.scenario_id = 123


class _FakeAnthropicMessages:
    async def create(self, **_kw) -> SimpleNamespace:
        payload = {
            "content": "집주인에게 월세 문의 전화",
            "ai_prompt": "You are a landlord.",
            "script": [{"step": 1, "ai_goal": "용건을 묻는다", "hint": "인사하세요"}],
        }
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


class _FakeAnthropicClient:
    def __init__(self, api_key: str) -> None:
        self.messages = _FakeAnthropicMessages()


async def test_create_custom_stores_warmup_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "get_settings",
        lambda: SimpleNamespace(anthropic_api_key="k", llm_analysis_model="m"),
    )
    monkeypatch.setattr(scenario_service, "AsyncAnthropic", _FakeAnthropicClient)
    db = _FakeWriteDB()

    request = CustomSessionRequest(
        title="집주인 통화 연습",
        call_target="집주인",
        call_purpose="월세 납부일 조정 문의",
        is_warmup=True,
    )
    await create_custom_scenario(db, request, user_id=1)

    assert db.added is not None
    assert db.added.is_warmup is True
    assert db.added.user_id == 1

    db2 = _FakeWriteDB()
    request2 = CustomSessionRequest(
        title="일반 커스텀",
        call_target="레스토랑 직원",
        call_purpose="창가 자리로 예약 요청",
    )
    await create_custom_scenario(db2, request2, user_id=1)
    assert db2.added.is_warmup is False
