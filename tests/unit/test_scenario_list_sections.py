from datetime import datetime

import pytest

from app.core.enums import ScenarioCategory
from app.services.scenario_service import get_scenarios
from tests.unit.test_scenario_visibility import (
    _custom_row,
    _FakeDB,
    _preset_ids,
    _preset_rows,
)


def _mine(scenario_id: int, day: int, user_id: int = 1):
    return _custom_row(
        scenario_id, user_id, title=f"내가 만든 {scenario_id}",
        created_at=datetime(2026, 1, day),
    )


def _copied(scenario_id: int, day: int, origin: int, user_id: int = 1):
    row = _custom_row(
        scenario_id, user_id, title=f"가져온 {scenario_id}",
        created_at=datetime(2026, 1, day),
    )
    row.origin_scenario_id = origin
    return row


def _customs(resp) -> list:
    return [s for s in resp.scenarios if s.is_custom]


@pytest.mark.asyncio
async def test_copied_scenarios_are_marked() -> None:
    rows = [*_preset_rows(), _mine(101, day=1), _copied(102, day=2, origin=500)]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    flags = {s.title: s.is_copied for s in _customs(resp)}
    assert flags == {"내가 만든 101": False, "가져온 102": True}


@pytest.mark.asyncio
async def test_newest_comes_first_within_a_section() -> None:
    rows = [
        *_preset_rows(),
        _mine(101, day=1),
        _mine(102, day=5),
        _mine(103, day=3),
    ]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert [s.scenario_id for s in _customs(resp)] == [102, 103, 101]


@pytest.mark.asyncio
async def test_sections_come_in_order_preset_mine_copied() -> None:
    rows = [
        *_preset_rows(),
        _copied(201, day=9, origin=500),
        _mine(101, day=1),
        _copied(202, day=8, origin=501),
        _mine(102, day=2),
    ]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    kinds = [(s.is_custom, s.is_copied) for s in resp.scenarios]
    presets = [k for k in kinds if k == (False, False)]
    assert kinds[: len(presets)] == presets, "프리셋이 앞에 모여있지 않다"

    customs = [s.scenario_id for s in _customs(resp)]
    assert customs == [102, 101, 201, 202], "내것(최신순) → 가져온것(최신순) 순서가 아니다"


@pytest.mark.asyncio
async def test_presets_keep_their_own_order() -> None:
    rows = [*_preset_rows(), _mine(101, day=1)]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    preset_ids = [s.scenario_id for s in resp.scenarios if not s.is_custom]
    assert preset_ids == sorted(preset_ids)
    assert all(s.is_copied is False for s in resp.scenarios if not s.is_custom)


@pytest.mark.asyncio
async def test_category_filter_still_applies_to_both_sections() -> None:
    rows = [
        *_preset_rows(),
        _mine(101, day=1),
        _copied(201, day=2, origin=500),
    ]
    rows[-1].category = ScenarioCategory.WORK.value
    rows[-2].category = ScenarioCategory.SCHOOL.value

    resp = await get_scenarios(_FakeDB(rows), ScenarioCategory.WORK, user_id=1)

    assert [s.scenario_id for s in _customs(resp)] == [201]


@pytest.mark.asyncio
async def test_row_without_created_at_does_not_break_sorting() -> None:
    rows = [*_preset_rows(), _mine(101, day=1), _mine(102, day=2)]
    del rows[-1].created_at

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert {s.scenario_id for s in _customs(resp)} == {101, 102}


# --- 연습 횟수(practice_count) ---

def _counting_db(rows: list, practice: list[tuple[int, int]] | None = None):
    """execute 호출 횟수를 세는 _FakeDB."""

    class _CountingDB(_FakeDB):
        def __init__(self) -> None:
            super().__init__(rows, practice)
            self.execute_calls = 0

        async def execute(self, stmt: object):
            self.execute_calls += 1
            return await super().execute(stmt)

    return _CountingDB()


@pytest.mark.asyncio
async def test_practice_count_is_filled_for_presets_and_customs() -> None:
    preset_id = _preset_ids()[0]
    rows = [*_preset_rows(), _mine(101, day=1)]
    db = _FakeDB(rows, practice=[(preset_id, 2), (101, 5)])

    resp = await get_scenarios(db, None, user_id=1)

    counts = {s.scenario_id: s.practice_count for s in resp.scenarios}
    assert counts[preset_id] == 2
    assert counts[101] == 5


@pytest.mark.asyncio
async def test_scenario_without_feedback_counts_zero() -> None:
    rows = [*_preset_rows(), _mine(101, day=1)]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert all(s.practice_count == 0 for s in resp.scenarios)


@pytest.mark.asyncio
async def test_history_for_invisible_scenario_is_ignored() -> None:
    rows = [*_preset_rows(), _mine(101, day=1)]
    db = _FakeDB(rows, practice=[(101, 3), (999, 7)])

    resp = await get_scenarios(db, None, user_id=1)

    assert 999 not in {s.scenario_id for s in resp.scenarios}
    assert {s.scenario_id: s.practice_count for s in resp.scenarios}[101] == 3


@pytest.mark.asyncio
async def test_preset_fallback_is_zero_without_extra_query() -> None:
    db = _counting_db([])

    resp = await get_scenarios(db, None, user_id=1)

    assert resp.scenarios, "DB가 비면 프리셋으로 폴백해야 한다"
    assert all(s.practice_count == 0 for s in resp.scenarios)
    assert db.execute_calls == 1, "폴백 경로에서 집계 쿼리가 나가면 안 된다"
