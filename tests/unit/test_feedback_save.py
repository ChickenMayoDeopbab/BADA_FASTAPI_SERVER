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


@pytest.mark.asyncio
async def test_turn_metrics_saved_as_json() -> None:
    store: dict = {}
    await _save(store, turn_metrics={"turn_sec": {1: [0.0, 4.2]}, "syllables_per_sec": {1: 6.67}})

    saved = json.loads(store["values"]["turn_metrics"])
    assert saved == {"turn_sec": {"1": [0.0, 4.2]}, "syllables_per_sec": {"1": 6.67}}


@pytest.mark.asyncio
async def test_missing_turn_metrics_saved_as_null() -> None:
    """못 잰 통화를 빈 값으로 채우면 '재 봤는데 없었다'와 구분이 안 된다."""
    store: dict = {}
    await _save(store)
    assert store["values"]["turn_metrics"] is None


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


@pytest.mark.asyncio
async def test_pipeline_passes_the_turn_measurements(monkeypatch) -> None:
    """통화가 끝날 때 잰 턴별 값이 턴 구간과 함께 저장으로 넘어간다."""
    calls: list[dict] = []

    async def _fake_save(**kwargs) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.save_feedback", _fake_save)

    p = _pipeline({"userId": 42, "scenarioId": 5})
    p._user_turn_intervals = [(0.0, 4.2), (6.0, 11.5)]
    p._note_turn_metric("loudness_db", {1: -21.4, 2: -27.9})
    p._note_turn_metric("syllables_per_sec", {2: 6.67})
    await p._save_feedback(0, [])

    assert calls[0]["turn_metrics"] == {
        "turn_sec": {1: [0.0, 4.2], 2: [6.0, 11.5]},
        "loudness_db": {1: -21.4, 2: -27.9},
        "syllables_per_sec": {2: 6.67},
    }


@pytest.mark.asyncio
async def test_pipeline_passes_no_measurements_when_none_were_taken(monkeypatch) -> None:
    calls: list[dict] = []

    async def _fake_save(**kwargs) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.save_feedback", _fake_save)

    await _pipeline({"userId": 42, "scenarioId": 5})._save_feedback(0, [])

    assert calls[0]["turn_metrics"] is None


class _Tremor:
    def __init__(self, voiced_spans) -> None:
        self.voiced_spans = voiced_spans


def test_judgements_are_kept_next_to_the_measurements() -> None:
    """잰 값만 남기면 기준값을 바꾼 뒤에는 그때 무엇을 짚었는지 알 수 없다. 판정과 고른 구간도 같이 남긴다."""
    p = _pipeline({"userId": 42, "scenarioId": 5})
    p._user_turn_intervals = [(0.0, 5.0), (6.0, 16.0)]
    p._user_turn_texts = ["예약하려고 전화드렸어요", "토요일 저녁 여섯 시에 네 명이요"]
    p._heard_at_by_turn = {2: 7.0}
    tremor = _Tremor([(0.5, 4.5), (9.0, 11.0), (13.2, 15.0)])   # 2번 턴: 2초 뒤 대답, 중간에 2.2초 멈춤

    pauses = p._long_pauses_by_turn(tremor)
    late = p._late_by_turn(tremor)
    p._pick_segments([(0.5, 4.5)], {0: 3.3, 1: 2.0}, pauses, {}, late)
    stored = p._turn_metrics_for_storage()

    assert stored["pause_sec"] == {2: [2.2]}
    assert stored["answer_delay_sec"] == stored["late_answer_sec"] == {2: 2.0}
    assert stored["avti"] == {1: 2.0}          # 0번(통화 전체)은 턴이 아니라 뺀다
    # 2번 턴: 뜸 2.0/1.2 이 멈춤 2.2/1.5 보다 기준을 더 넘어 대표 이유가 된다.
    assert stored["picked"] == [
        {"turn": 1, "type": "GOOD", "reason": "voice", "reasons": [], "start": 0.0, "end": 5.0},
        {"turn": 2, "type": "IMPROVE", "reason": "late", "reasons": ["late", "pause"], "start": 6.0, "end": 16.0},
    ]
