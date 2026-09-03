import json
from datetime import datetime
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from sqlalchemy.sql.dml import Update
from sqlalchemy.sql.selectable import Select

from app.api.v1.scenario import router as scenario_router
from app.core.enums import ScenarioCategory
from app.core.preset_scenarios import PRESET_MAP, PRESET_SCENARIOS
from app.db.models import FeedbackORM, ScenarioORM
from app.db.seed import seed_preset_scenarios
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

    def __init__(self, rows: list, practice: list[tuple[int, int]] | None = None) -> None:
        self._rows = rows
        self._practice = practice or []

    async def execute(self, stmt: object) -> _FakeResult:
        # get_scenarios는 시나리오 조회와 연습 횟수 집계 두 번을 부른다.
        if FeedbackORM.__tablename__ in str(stmt):
            return _FakeResult(self._practice)
        return _FakeResult(self._rows)


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
        call_target=kw.get("call_target", "상대"),
        call_purpose=kw.get("call_purpose", "목적"),
        script=kw.get("script", []),
        created_at=kw.get("created_at", datetime(2026, 1, 1)),
        origin_scenario_id=kw.get("origin_scenario_id"),
    )


def _preset_row(scenario_id: int) -> SimpleNamespace:
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
    )


def _preset_rows() -> list[SimpleNamespace]:
    return [_preset_row(s["scenario_id"]) for s in PRESET_SCENARIOS]


def _preset_ids() -> list[int]:
    return [s["scenario_id"] for s in PRESET_SCENARIOS]


def _ids(response) -> list[int]:
    return [s.scenario_id for s in response.scenarios]


async def test_list_shows_presets_and_only_own_customs() -> None:
    rows = [
        *_preset_rows(),
        _custom_row(101, user_id=1),
        _custom_row(102, user_id=2),
    ]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert _ids(resp) == [*_preset_ids(), 101]
    assert all(s.is_custom is False for s in resp.scenarios[:len(PRESET_SCENARIOS)])
    assert resp.scenarios[-1].is_custom is True


async def test_null_owner_custom_hidden_from_everyone() -> None:
    rows = [*_preset_rows(), _custom_row(101, user_id=None)]

    for requester in (1, 2):
        resp = await get_scenarios(_FakeDB(rows), None, user_id=requester)
        assert _ids(resp) == _preset_ids()


async def test_presets_visible_to_all_users() -> None:
    rows = [*_preset_rows(), _custom_row(101, user_id=2)]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert _ids(resp) == _preset_ids()
    assert all(s.is_custom is False for s in resp.scenarios)


async def test_daily_filter_includes_presets_and_own_daily_custom() -> None:
    rows = [
        *_preset_rows(),
        _custom_row(101, user_id=1, category=ScenarioCategory.DAILY.value),
        _custom_row(102, user_id=1, category=ScenarioCategory.OTHER.value),
        _custom_row(103, user_id=2, category=ScenarioCategory.DAILY.value),
    ]

    response = await get_scenarios(
        _FakeDB(rows), ScenarioCategory.DAILY, user_id=1
    )

    assert _ids(response) == [*_preset_ids(), 101]
    assert all(
        scenario.category == ScenarioCategory.DAILY
        for scenario in response.scenarios
    )


async def test_work_filter_returns_matching_custom_only() -> None:
    rows = [
        *_preset_rows(),
        _custom_row(101, user_id=1, category=ScenarioCategory.WORK.value),
        _custom_row(102, user_id=1, category=ScenarioCategory.SCHOOL.value),
    ]

    response = await get_scenarios(
        _FakeDB(rows), ScenarioCategory.WORK, user_id=1
    )

    assert _ids(response) == [101]
    assert response.scenarios[0].is_custom is True
    assert response.scenarios[0].category == ScenarioCategory.WORK


async def test_invalid_stored_custom_category_falls_back_to_other() -> None:
    rows = [
        *_preset_rows(),
        _custom_row(101, user_id=1, category="legacy-invalid"),
    ]

    response = await get_scenarios(
        _FakeDB(rows), ScenarioCategory.OTHER, user_id=1
    )

    assert _ids(response) == [101]
    assert response.scenarios[0].category == ScenarioCategory.OTHER


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


async def test_legacy_category_query_is_rejected() -> None:
    app = _make_app(rows=[], user_id=1)

    response = await _get(
        app,
        "/api/v1/scenario/scenarios?category=hospital",
    )

    assert response.status_code == 422


def test_custom_create_defaults_missing_category_to_other() -> None:
    request = CustomSessionRequest(
        title="교수님께 과제 문의",
        call_target="담당 교수님",
        call_purpose="과제 제출 기한 연장 가능 여부 문의",
    )

    assert request.category == ScenarioCategory.OTHER


async def test_warmup_custom_hidden_even_from_owner() -> None:
    rows = [
        *_preset_rows(),
        _custom_row(101, user_id=1, is_warmup=True),
        _custom_row(102, user_id=1),
    ]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert _ids(resp) == [*_preset_ids(), 102]


class _SeedScalarResult:
    def __init__(self, value: int | None) -> None:
        self._value = value

    def scalar_one(self) -> int | None:
        return self._value


class _FakeSeedDB:
    def __init__(self, scenarios: list[ScenarioORM] | None = None, feedbacks: list | None = None) -> None:
        self.scenarios = {row.scenario_id: row for row in scenarios or []}
        self.feedbacks = feedbacks or []
        self.commits = 0

    async def get(self, model: type, key: int) -> ScenarioORM | None:
        if model is ScenarioORM:
            return self.scenarios.get(key)
        return None

    def add(self, row: ScenarioORM) -> None:
        self.scenarios[row.scenario_id] = row

    async def flush(self) -> None:
        pass

    async def delete(self, row: ScenarioORM) -> None:
        self.scenarios.pop(row.scenario_id, None)

    async def execute(self, stmt: object) -> _SeedScalarResult:
        if isinstance(stmt, Select):
            return _SeedScalarResult(max(self.scenarios, default=0))
        if isinstance(stmt, Update):
            params = stmt.compile().params
            old_id = params["scenario_id_1"]
            new_id = params["scenario_id"]
            for feedback in self.feedbacks:
                if feedback.scenario_id == old_id:
                    feedback.scenario_id = new_id
            return _SeedScalarResult(None)
        return _SeedScalarResult(None)

    def get_bind(self) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1


def _scenario_orm(scenario_id: int, **kw) -> ScenarioORM:
    seed = PRESET_MAP.get(scenario_id, PRESET_MAP[1])
    is_custom = kw.get("is_custom", False)
    return ScenarioORM(
        scenario_id=scenario_id,
        title=kw.get("title", seed["title"]),
        content=kw.get("content", seed["content"]),
        category=kw.get(
            "category",
            ScenarioCategory.OTHER.value if is_custom else seed["category"].value,
        ),
        scenario_image=kw.get("scenario_image", seed["scenario_image"]),
        tts_voice_id=kw.get("tts_voice_id", seed["tts_voice_id"]),
        ai_prompt=kw.get("ai_prompt", seed["ai_prompt"]),
        user_id=kw.get("user_id"),
        is_custom=is_custom,
        is_warmup=kw.get("is_warmup", False),
        call_target=kw.get("call_target", seed["call_target"]),
        call_purpose=kw.get("call_purpose", seed["call_purpose"]),
        script=kw.get("script", seed["script"]),
        created_at=kw.get("created_at", datetime(2026, 1, 1)),
        origin_scenario_id=kw.get("origin_scenario_id"),
    )


async def test_seed_preset_scenarios_inserts_missing_presets() -> None:
    db = _FakeSeedDB()

    await seed_preset_scenarios(db)

    assert sorted(db.scenarios) == _preset_ids()
    assert all(row.is_custom is False for row in db.scenarios.values())
    assert all(row.category == ScenarioCategory.DAILY.value for row in db.scenarios.values())
    assert db.commits == 1


async def test_seed_preset_scenarios_is_idempotent_and_syncs_existing_presets() -> None:
    original_created_at = datetime(2025, 1, 1)
    existing = _scenario_orm(
        1,
        title="오래된 제목",
        content="오래된 설명",
        scenario_image="scenario_profile/legacy-image",
        is_custom=False,
        created_at=original_created_at,
    )
    db = _FakeSeedDB([existing])

    await seed_preset_scenarios(db)
    await seed_preset_scenarios(db)

    assert sorted(db.scenarios) == _preset_ids()
    assert db.scenarios[1].title == PRESET_MAP[1]["title"]
    assert db.scenarios[1].content == PRESET_MAP[1]["content"]
    assert db.scenarios[1].scenario_image == PRESET_MAP[1]["scenario_image"]
    assert db.scenarios[1].created_at == original_created_at
    assert db.commits == 2


async def test_seed_preset_scenarios_moves_conflicting_custom_and_feedback() -> None:
    custom = _scenario_orm(
        1,
        title="커스텀 충돌",
        user_id=7,
        is_custom=True,
        category=ScenarioCategory.WORK.value,
        call_target="구청 직원",
        call_purpose="민원 문의",
    )
    feedback = SimpleNamespace(scenario_id=1)
    db = _FakeSeedDB([custom], [feedback])

    await seed_preset_scenarios(db)

    moved_customs = [row for row in db.scenarios.values() if row.is_custom]
    assert sorted(db.scenarios) == [*_preset_ids(), 9]
    assert len(moved_customs) == 1
    assert moved_customs[0].scenario_id == 9
    assert moved_customs[0].title == "커스텀 충돌"
    assert moved_customs[0].user_id == 7
    assert moved_customs[0].category == ScenarioCategory.WORK.value
    assert db.scenarios[1].is_custom is False
    assert db.scenarios[1].title == PRESET_MAP[1]["title"]
    assert feedback.scenario_id == 9


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
            "script": [
                {"step": 1, "ai_goal": "용건을 묻는다", "hint": "인사하세요"},
                {"step": 2, "ai_goal": "내용을 확인한다", "hint": "용건을 말하세요"},
                {"step": 3, "ai_goal": "통화를 마무리한다", "hint": "감사 인사를 하세요"},
            ],
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
        category=ScenarioCategory.OTHER,
        call_target="집주인",
        call_purpose="월세 납부일 조정 문의",
        is_warmup=True,
    )
    await create_custom_scenario(db, request, user_id=1)

    assert db.added is not None
    assert db.added.is_warmup is True
    assert db.added.user_id == 1
    assert db.added.category == ScenarioCategory.OTHER.value

    db2 = _FakeWriteDB()
    request2 = CustomSessionRequest(
        title="일반 커스텀",
        category=ScenarioCategory.DAILY,
        call_target="레스토랑 직원",
        call_purpose="창가 자리로 예약 요청",
    )
    await create_custom_scenario(db2, request2, user_id=1)
    assert db2.added.is_warmup is False
    assert db2.added.category == ScenarioCategory.DAILY.value
