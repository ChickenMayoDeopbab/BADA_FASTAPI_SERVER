import os
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1 import internal
from app.core.default_scenarios import LEGACY_VARIATIONS, PIZZA_TITLE, new_preset_scenarios
from app.core.enums import ScenarioCategory
from app.core.preset_scenarios import PRESET_MAP
from app.core.timeutil import now_utc
from app.db import seed
from app.db.base import Base
from app.db.models import FeedbackORM, ScenarioORM
from app.services import example_service, scenario_service

_EXPECTED_IDS = [84, 1, 5, 85, 86, 87, 88, 89]


@pytest.mark.parametrize("rollback", [False, True])
def test_configured_catalog_preserves_lookup_and_controls_visibility(rollback: bool) -> None:
    environment = {
        **os.environ,
        "DEFAULT_SCENARIO_START_ID": "84",
        "LEGACY_DEFAULT_SCENARIO_LIST": str(rollback),
    }
    expected_ids = list(range(1, 9)) if rollback else _EXPECTED_IDS
    expected_variations = set() if rollback else set(_EXPECTED_IDS)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.core.preset_scenarios import ACTIVE_PRESET_MAP, PRESET_MAP, SESSION_VARIATIONS\n"
            f"assert list(ACTIVE_PRESET_MAP) == {expected_ids!r}\n"
            "assert set(PRESET_MAP) == set(range(1, 9)) | set(range(84, 90))\n"
            f"assert set(SESSION_VARIATIONS) == {expected_variations!r}\n",
        ],
        env=environment,
        check=True,
    )


@pytest.fixture
def refreshed_catalog(monkeypatch) -> list[dict]:
    new_presets = new_preset_scenarios(84)
    legacy_presets = [PRESET_MAP[index] for index in range(1, 9)]
    catalog = {preset["scenario_id"]: preset for preset in [*legacy_presets, *new_presets]}
    active = {scenario_id: catalog[scenario_id] for scenario_id in _EXPECTED_IDS}
    variations = {**LEGACY_VARIATIONS, **{p["scenario_id"]: p["variation"] for p in new_presets}}
    monkeypatch.setattr(seed, "PRESET_SCENARIOS", list(catalog.values()))
    monkeypatch.setattr(seed, "NEW_PRESET_SCENARIOS", new_presets)
    monkeypatch.setattr(scenario_service, "ACTIVE_PRESET_SCENARIOS", list(active.values()))
    monkeypatch.setattr(scenario_service, "ACTIVE_PRESET_MAP", active)
    storage = SimpleNamespace(presigned_url=lambda key, **_: f"https://example.com/{key}")
    monkeypatch.setattr(scenario_service, "RecordingStorageService", lambda _: storage)
    monkeypatch.setattr(internal, "PRESET_MAP", catalog)
    monkeypatch.setattr(internal, "SESSION_VARIATIONS", variations)
    monkeypatch.setattr(example_service, "PRESET_MAP", catalog)
    monkeypatch.setattr(example_service, "get_settings", lambda: SimpleNamespace(s3_bucket=None))
    return new_presets


@pytest.fixture
async def preset_db():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.execute(text("CREATE TABLE training_records (scenario_id BIGINT NOT NULL)"))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            yield session
    finally:
        await engine.dispose()


async def test_seed_uses_past_reference_and_preserves_legacy_and_image(preset_db, refreshed_catalog) -> None:
    legacy = seed._preset_values(PRESET_MAP[1], now_utc())
    preset_db.add(ScenarioORM(**{**legacy, "title": "운영 DB에 저장된 기존 제목"}))
    await preset_db.execute(text("INSERT INTO training_records VALUES (83)"))
    await preset_db.commit()

    assert await seed.next_preset_start_id(preset_db) == 84
    await seed.seed_preset_scenarios(preset_db)
    pizza = await preset_db.get(ScenarioORM, 84)
    assert pizza.title == PIZZA_TITLE
    pizza.scenario_image = "scenario-images/84-pizza.png"
    await preset_db.commit()
    await seed.seed_preset_scenarios(preset_db)

    assert (await preset_db.get(ScenarioORM, 1)).title == "운영 DB에 저장된 기존 제목"
    assert (await preset_db.get(ScenarioORM, 84)).scenario_image == "scenario-images/84-pizza.png"
    scenarios = await scenario_service.get_scenarios(preset_db, None, 7)
    assert scenarios.scenarios[0].scenario_image == "https://example.com/scenario-images/84-pizza.png"
    assert list(await preset_db.scalars(select(ScenarioORM.scenario_id).order_by(ScenarioORM.scenario_id))) == [
        *range(1, 9), *range(84, 90),
    ]


@pytest.mark.parametrize("conflict", ["custom", "historic_reference"])
async def test_seed_rejects_conflict_before_any_insert(preset_db, refreshed_catalog, conflict: str) -> None:
    if conflict == "custom":
        values = seed._preset_values(refreshed_catalog[0], now_utc())
        preset_db.add(ScenarioORM(**{**values, "title": "이미 등록된 커스텀", "is_custom": True, "user_id": 7}))
    else:
        await preset_db.execute(text("INSERT INTO training_records VALUES (84)"))
    await preset_db.commit()

    with pytest.raises(RuntimeError, match="conflict|must be 85"):
        await seed.seed_preset_scenarios(preset_db)

    assert await preset_db.get(ScenarioORM, 1) is None
    assert await preset_db.get(ScenarioORM, 85) is None
    if conflict == "custom":
        assert (await preset_db.get(ScenarioORM, 84)).title == "이미 등록된 커스텀"


async def test_active_catalog_order_category_history_and_legacy_lookup(preset_db, refreshed_catalog) -> None:
    empty = await scenario_service.get_scenarios(preset_db, None, 7)
    assert [scenario.scenario_id for scenario in empty.scenarios] == _EXPECTED_IDS

    await preset_db.execute(text("INSERT INTO training_records VALUES (83)"))
    await preset_db.commit()
    await seed.seed_preset_scenarios(preset_db)
    for scenario_id in _EXPECTED_IDS:
        preset_db.add(FeedbackORM(
            feedback_id=scenario_id, session_id=f"session-{scenario_id}", user_id=7, scenario_id=scenario_id,
            shake_count=0, silence_duration=0, highlights="", created_at=now_utc(),
        ))
    await preset_db.commit()

    result = await scenario_service.get_scenarios(preset_db, None, 7)
    assert [scenario.scenario_id for scenario in result.scenarios] == _EXPECTED_IDS
    assert [scenario.practice_count for scenario in result.scenarios] == [1] * 8
    assert Counter(s.category for s in result.scenarios) == {
        ScenarioCategory.DAILY: 3,
        ScenarioCategory.WORK: 2,
        ScenarioCategory.SCHOOL: 2,
        ScenarioCategory.OTHER: 1,
    }
    school = await scenario_service.get_scenarios(preset_db, ScenarioCategory.SCHOOL, 7)
    assert [scenario.scenario_id for scenario in school.scenarios] == [87, 88]
    recommendation = await scenario_service.get_recommended_scenario(preset_db, 7)
    assert recommendation.scenario.scenario_id in _EXPECTED_IDS

    example = await example_service.get_example_conversation(preset_db, 2, 7)
    assert [turn.model_dump() for turn in example.dialogue] == PRESET_MAP[2]["example_dialogue"]
    assert (await internal.get_scenario_context(2, preset_db)).ai_prompt == PRESET_MAP[2]["ai_prompt"]


@pytest.mark.parametrize("draw, has_variation", [(0.199, True), (0.20, False)])
async def test_context_selects_variation_once_without_changing_examples(
    monkeypatch, preset_db, refreshed_catalog, draw: float, has_variation: bool,
) -> None:
    calls = []

    def draw_once() -> float:
        calls.append(draw)
        return draw

    monkeypatch.setattr(internal, "random", draw_once)

    context = await internal.get_scenario_context(84, preset_db)
    example = await example_service.get_example_conversation(preset_db, 84, 7)

    assert calls == [draw]
    assert ("품절을 안내" in context.ai_prompt) is has_variation
    assert ("별도의 돌발 상황을 만들지 않는다" in context.ai_prompt) is not has_variation
    assert [turn.model_dump() for turn in example.dialogue] == refreshed_catalog[0]["example_dialogue"]
    assert all("품절" not in turn.text for turn in example.dialogue)
