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


def _scenario(scenario_id: int, *, is_custom: bool = False) -> ScenarioInfo:
    return ScenarioInfo(
        scenario_id=scenario_id,
        title=f"시나리오 {scenario_id}",
        content="설명",
        category=ScenarioCategory.OTHER if is_custom else ScenarioCategory.DAILY,
        difficulties=ALL_DIFFICULTIES,
        personalities=ALL_PERSONALITIES,
        ai_prompt="prompt",
        is_custom=is_custom,
    )


def _stub_candidates(monkeypatch, candidates: list[ScenarioInfo]) -> None:
    async def _get_scenarios(_db, _category, _user_id) -> ScenarioListResponse:
        return ScenarioListResponse(scenarios=candidates)

    monkeypatch.setattr(scenario_service, "get_scenarios", _get_scenarios)


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
