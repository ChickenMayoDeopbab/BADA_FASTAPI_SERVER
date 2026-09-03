from datetime import UTC, date, datetime

from app.core.enums import ALL_DIFFICULTIES, ALL_PERSONALITIES, ScenarioCategory
from app.schemas.scenario import ScenarioInfo, ScenarioListResponse
from app.services import scenario_service


class _HistoryResult:
    def __init__(self, rows: list[tuple[int, datetime]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[int, datetime]]:
        return self._rows


class _FakeDB:
    def __init__(self, history: list[tuple[int, datetime]]) -> None:
        self._history = history

    async def execute(self, _stmt: object) -> _HistoryResult:
        return _HistoryResult(self._history)


def _scenario(
    scenario_id: int,
    *,
    is_custom: bool = False,
    practice_count: int = 0,
    scenario_image: str | None = None,
) -> ScenarioInfo:
    return ScenarioInfo(
        scenario_id=scenario_id,
        title=f"시나리오 {scenario_id}",
        content="설명",
        category=ScenarioCategory.OTHER if is_custom else ScenarioCategory.DAILY,
        difficulties=ALL_DIFFICULTIES,
        personalities=ALL_PERSONALITIES,
        scenario_image=scenario_image,
        ai_prompt="prompt",
        is_custom=is_custom,
        practice_count=practice_count,
    )


def _stub_candidates(monkeypatch, candidates: list[ScenarioInfo]) -> None:
    async def _get_scenarios(_db, _category, _user_id) -> ScenarioListResponse:
        return ScenarioListResponse(scenarios=candidates)

    monkeypatch.setattr(scenario_service, "get_scenarios", _get_scenarios)


def _patch_icon_storage(monkeypatch, *, fail: bool = False) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    class _Storage:
        def __init__(self, _settings) -> None:
            pass

        def presigned_url(self, key: str, expires_in: int) -> str:
            calls.append((key, expires_in))
            if fail:
                raise RuntimeError("signature failed")
            return f"https://s3.example.com/{key}?signed"

    monkeypatch.setattr(scenario_service, "RecordingStorageService", _Storage)
    return calls


def test_all_categories_use_uploaded_icon_keys(monkeypatch) -> None:
    calls = _patch_icon_storage(monkeypatch)
    expected = {
        ScenarioCategory.WORK: "scenario_profile/9c59b8ee-46d0-4207-bed0-ab7136104fef",
        ScenarioCategory.DAILY: "scenario_profile/0c11a382-99b6-457d-80c1-4c00915c5e6c",
        ScenarioCategory.SCHOOL: "scenario_profile/78b33292-2156-4665-86b7-80e0ca3535d5",
        ScenarioCategory.OTHER: "scenario_profile/29bdac11-0f65-4689-8ad8-f64d06f3d7b6",
    }

    for category, key in expected.items():
        candidate = _scenario(1)
        candidate.category = category

        response = scenario_service._recommendation_response(candidate, "NOT_PRACTICED")

        assert response.category_icon_url == f"https://s3.example.com/{key}?signed"

    assert calls == [(key, scenario_service._IMAGE_URL_TTL_SEC) for key in expected.values()]


async def test_unpracticed_custom_is_recommended_first(monkeypatch) -> None:
    candidates = [_scenario(1), _scenario(102, is_custom=True), _scenario(101, is_custom=True)]
    _stub_candidates(monkeypatch, candidates)

    recommendation = await scenario_service.get_recommended_scenario(
        _FakeDB([]),
        user_id=7,
        recommendation_date=date(2026, 9, 1),
    )

    assert recommendation is not None
    assert recommendation.scenario.scenario_id == 102
    assert recommendation.reason == "CUSTOM_NOT_PRACTICED"


async def test_recommendation_returns_presigned_category_icon(monkeypatch) -> None:
    scenario_image = "https://s3.example.com/scenario-images/88-example.png?signed"
    candidate = _scenario(1, scenario_image=scenario_image)
    _stub_candidates(monkeypatch, [candidate])
    calls = _patch_icon_storage(monkeypatch)

    recommendation = await scenario_service.get_recommended_scenario(
        _FakeDB([]),
        user_id=7,
        recommendation_date=date(2026, 9, 1),
    )

    key = scenario_service._CATEGORY_ICON_KEYS[ScenarioCategory.DAILY]
    assert recommendation is not None
    assert recommendation.scenario.scenario_image == scenario_image
    assert recommendation.category_icon_url == f"https://s3.example.com/{key}?signed"
    assert recommendation.scenario.scenario_image != recommendation.category_icon_url
    assert calls == [(key, scenario_service._IMAGE_URL_TTL_SEC)]


async def test_icon_signing_failure_keeps_recommendation_available(monkeypatch) -> None:
    _stub_candidates(monkeypatch, [_scenario(1)])
    _patch_icon_storage(monkeypatch, fail=True)

    recommendation = await scenario_service.get_recommended_scenario(
        _FakeDB([]),
        user_id=7,
        recommendation_date=date(2026, 9, 1),
    )

    assert recommendation is not None
    assert recommendation.category_icon_url is None


async def test_practiced_custom_does_not_block_other_unpracticed_scenario(monkeypatch) -> None:
    practiced_at = datetime(2026, 8, 31, tzinfo=UTC)
    candidates = [_scenario(1), _scenario(2), _scenario(101, is_custom=True)]
    _stub_candidates(monkeypatch, candidates)

    recommendation = await scenario_service.get_recommended_scenario(
        _FakeDB([(101, practiced_at)]),
        user_id=7,
        recommendation_date=date(2026, 9, 1),
    )

    assert recommendation is not None
    assert recommendation.scenario.scenario_id in {1, 2}
    assert recommendation.reason == "NOT_PRACTICED"


async def test_daily_pick_is_stable_for_same_candidates(monkeypatch) -> None:
    candidates = [_scenario(3), _scenario(1), _scenario(2)]
    _stub_candidates(monkeypatch, candidates)
    recommendation_date = date(2026, 9, 1)

    first = await scenario_service.get_recommended_scenario(
        _FakeDB([]),
        user_id=7,
        recommendation_date=recommendation_date,
    )

    _stub_candidates(monkeypatch, list(reversed(candidates)))
    second = await scenario_service.get_recommended_scenario(
        _FakeDB([]),
        user_id=7,
        recommendation_date=recommendation_date,
    )

    assert first is not None
    assert second is not None
    assert first.scenario.scenario_id == second.scenario.scenario_id


async def test_oldest_practiced_scenario_is_recommended(monkeypatch) -> None:
    candidates = [_scenario(1), _scenario(2), _scenario(3)]
    _stub_candidates(monkeypatch, candidates)
    history = [
        (1, datetime(2026, 8, 31, tzinfo=UTC)),
        (2, datetime(2026, 7, 1, tzinfo=UTC)),
        (3, datetime(2026, 8, 1, tzinfo=UTC)),
    ]

    recommendation = await scenario_service.get_recommended_scenario(
        _FakeDB(history),
        user_id=7,
        recommendation_date=date(2026, 9, 1),
    )

    assert recommendation is not None
    assert recommendation.scenario.scenario_id == 2
    assert recommendation.reason == "LONGEST_ABSENT"


async def test_no_visible_candidate_returns_none(monkeypatch) -> None:
    _stub_candidates(monkeypatch, [])

    recommendation = await scenario_service.get_recommended_scenario(
        _FakeDB([]),
        user_id=7,
        recommendation_date=date(2026, 9, 1),
    )

    assert recommendation is None


async def test_recommendation_carries_practice_count(monkeypatch) -> None:
    _patch_icon_storage(monkeypatch)
    _stub_candidates(monkeypatch, [_scenario(1, practice_count=4)])

    recommendation = await scenario_service.get_recommended_scenario(
        _FakeDB([(1, datetime(2026, 8, 1, tzinfo=UTC))]),
        user_id=7,
        recommendation_date=date(2026, 9, 1),
    )

    assert recommendation is not None
    assert recommendation.scenario.practice_count == 4
