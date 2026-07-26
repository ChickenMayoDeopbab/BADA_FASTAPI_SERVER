import json
import logging

import pytest

from app.services import feedback_service
from app.services.feedback_service import save_feedback
from app.services.pipeline import VoicePipeline


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeSession:
    """AsyncSessionLocal 대역. 실행된 statement 와 commit 여부를 기록한다."""

    def __init__(self, store: dict, rowcount: int = 1, raise_on_execute: bool = False) -> None:
        self._store = store
        self._rowcount = rowcount
        self._raise = raise_on_execute

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def execute(self, stmt):
        if self._raise:
            raise RuntimeError("DB 연결 끊김")
        self._store["values"] = stmt.compile().params
        return _FakeResult(self._rowcount)

    async def commit(self) -> None:
        self._store["committed"] = True


def _factory(store: dict, **kwargs):
    return lambda: _FakeSession(store, **kwargs)


async def _save(store: dict, *, silence_total: float = 0.0, **kwargs) -> bool:
    params = {
        "session_id": "sess-1",
        "user_id": 7,
        "scenario_id": 3,
        "shake_count": 2,
        "silence_total": silence_total,
        "good_segments": [{"start": 0.0, "end": 1.0, "good_point": "또박또박 말했어요"}],
    }
    params.update(kwargs)
    return await save_feedback(session_factory=_factory(store, **{}), **params)


# --- 반올림 -------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [(0.0, 0), (12.4, 12), (12.6, 13)])
@pytest.mark.asyncio
async def test_silence_duration_is_rounded_seconds(raw: float, expected: int) -> None:
    store: dict = {}
    await _save(store, silence_total=raw)
    assert store["values"]["silence_duration"] == expected


# --- 저장 내용 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_saves_all_fields_and_commits() -> None:
    store: dict = {}
    assert await _save(store) is True

    values = store["values"]
    assert values["session_id"] == "sess-1"
    assert values["user_id"] == 7
    assert values["scenario_id"] == 3
    assert values["shake_count"] == 2
    assert store["committed"] is True


@pytest.mark.asyncio
async def test_highlights_serialized_as_utf8_json() -> None:
    store: dict = {}
    await _save(store)

    highlights = json.loads(store["values"]["highlights"])
    assert highlights[0]["good_point"] == "또박또박 말했어요"
    # ensure_ascii=False 라 한글이 이스케이프되지 않는다
    assert "또박또박" in store["values"]["highlights"]


@pytest.mark.asyncio
async def test_empty_good_segments_saved_as_empty_list() -> None:
    store: dict = {}
    await _save(store, good_segments=None)
    assert store["values"]["highlights"] == "[]"


# --- 실패/멱등 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_db_failure_is_swallowed(caplog) -> None:
    """저장 실패가 세션 종료 흐름을 깨면 안 된다."""
    store: dict = {}
    with caplog.at_level(logging.ERROR, logger="app.services.feedback_service"):
        saved = await save_feedback(
            session_id="sess-1",
            user_id=7,
            scenario_id=3,
            shake_count=0,
            silence_total=0.0,
            good_segments=[],
            session_factory=_factory(store, raise_on_execute=True),
        )

    assert saved is False
    assert "피드백 저장 실패" in caplog.text


@pytest.mark.asyncio
async def test_duplicate_session_is_skipped() -> None:
    """ON CONFLICT DO NOTHING 으로 rowcount 0 → 중복 저장 아님."""
    store: dict = {}
    saved = await save_feedback(
        session_id="sess-1",
        user_id=7,
        scenario_id=3,
        shake_count=0,
        silence_total=0.0,
        good_segments=[],
        session_factory=_factory(store, rowcount=0),
    )
    assert saved is False


# --- 파이프라인 연결 ----------------------------------------------------


def _pipeline(session: dict) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-1"
    p._session = session
    p._silence_total = 3.2
    return p


@pytest.mark.asyncio
async def test_pipeline_passes_session_identifiers(monkeypatch) -> None:
    calls: list[dict] = []

    async def _fake_save(**kwargs) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.save_feedback", _fake_save)

    p = _pipeline({"userId": 42, "scenarioId": 5})
    await p._save_feedback(3, [{"start": 0.0, "end": 1.0}])

    assert calls[0]["user_id"] == 42
    assert calls[0]["scenario_id"] == 5
    assert calls[0]["session_id"] == "sess-1"
    assert calls[0]["silence_total"] == 3.2
    assert calls[0]["shake_count"] == 3


@pytest.mark.asyncio
async def test_pipeline_reads_nested_scenario_id(monkeypatch) -> None:
    """scenario 안에 들어와도 읽는다."""
    calls: list[dict] = []

    async def _fake_save(**kwargs) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.save_feedback", _fake_save)

    p = _pipeline({"userId": 42, "scenario": {"scenarioId": 9}})
    await p._save_feedback(0, [])

    assert calls[0]["scenario_id"] == 9


@pytest.mark.parametrize(
    "session",
    [
        {"userId": 42},  # Spring 이 아직 scenarioId 를 안 넣는 상태
        {"scenarioId": 5},
        {"userId": "abc", "scenarioId": 5},
        {},
    ],
)
@pytest.mark.asyncio
async def test_pipeline_skips_without_identifiers(monkeypatch, caplog, session: dict) -> None:
    """식별자가 없으면 저장을 건너뛰되 예외 없이 종료 흐름을 이어간다."""
    called = False

    async def _fake_save(**kwargs) -> bool:
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("app.services.pipeline.save_feedback", _fake_save)

    p = _pipeline(session)
    with caplog.at_level(logging.WARNING, logger="app.services.pipeline"):
        await p._save_feedback(0, [])

    assert called is False
    assert "피드백 저장 스킵" in caplog.text


@pytest.mark.asyncio
async def test_warmup_session_is_saved(monkeypatch) -> None:
    """워밍업 세션도 저장 대상."""
    calls: list[dict] = []

    async def _fake_save(**kwargs) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.save_feedback", _fake_save)

    p = _pipeline({"userId": 42, "scenarioId": 5, "type": "WARMUP"})
    await p._save_feedback(0, [])

    assert len(calls) == 1


def test_module_uses_postgres_insert() -> None:
    """ON CONFLICT 를 쓰려면 postgres 방언 insert 여야 한다."""
    assert feedback_service.pg_insert.__module__.startswith("sqlalchemy.dialects.postgresql")
