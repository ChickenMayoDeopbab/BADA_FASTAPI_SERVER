import asyncio
import logging

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core import usage as usage_core
from app.core.usage import emit, log_llm_usage
from app.db.base import Base
from app.db.models import UsageEventORM
from app.services import usage_service
from app.services.usage_service import build_row, db_sink, record_event
from tests.unit.test_usage_realtime import _closing, _metrics


@pytest.fixture(autouse=True)
def _isolated_sinks():
    saved = list(usage_core._SINKS)
    usage_core._SINKS.clear()
    yield
    usage_core._SINKS[:] = saved


class _Recorder:
    def __init__(self) -> None:
        self.seen: list[tuple[str, dict]] = []

    def __call__(self, kind: str, fields: dict) -> None:
        self.seen.append((kind, fields))


class _FakeSession:
    def __init__(self, store: dict, *, commit_error: Exception | None = None) -> None:
        self._store = store
        self._commit_error = commit_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def add(self, obj: object) -> None:
        self._store.setdefault("added", []).append(obj)

    async def commit(self) -> None:
        if self._commit_error is not None:
            raise self._commit_error
        self._store["commits"] = self._store.get("commits", 0) + 1


_SESSION_FIELDS = {
    "session_id": "sess-1", "user_id": "77", "scenario_id": 3, "reason": "USER_END",
    "stt_engine": "chirp", "llm_model": "gemini-3.5-flash-lite", "turns": 4,
    "stt_sec": 12.5, "llm_prompt_tokens": 4000, "tts_chars_eleven": 120,
}
_LLM_FIELDS = {
    "provider": "anthropic", "model": "claude-sonnet-4-6", "purpose": "scenario_gen",
    "user_id": 7, "scenario_id": 42, "attempt": 2, "ok": True,
    "input_tokens": 700, "output_tokens": 220, "images": 0,
}
_TTS_FIELDS = {
    "engine": "qwen", "model": "qwen", "purpose": "example_audio", "user_id": None,
    "scenario_id": 1, "trigger": "prebake", "chars": 33, "audio_sec": 4.2, "turns": 2,
}


def test_emit_logs_metric_and_calls_every_sink_even_if_one_raises(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")

    def _boom(kind, fields):
        raise RuntimeError("싱크 버그")

    recorder = _Recorder()
    usage_core.register_sink(_boom)
    usage_core.register_sink(recorder)

    emit("llm_usage", provider="anthropic", user_id=1)

    [rec] = _metrics(caplog, "llm_usage")
    assert rec.provider == "anthropic"
    assert recorder.seen == [("llm_usage", {"provider": "anthropic", "user_id": 1})]


def test_log_llm_usage_reaches_registered_sink() -> None:
    recorder = _Recorder()
    usage_core.register_sink(recorder)

    log_llm_usage("anthropic", "m", "scenario_gen", usage=None, user_id=3)

    [(kind, fields)] = recorder.seen
    assert kind == "llm_usage"
    assert fields["user_id"] == 3 and fields["input_tokens"] == 0 and fields["purpose"] == "scenario_gen"


def test_register_sink_is_idempotent_and_unregister_removes() -> None:
    recorder = _Recorder()
    usage_core.register_sink(recorder)
    usage_core.register_sink(recorder)
    emit("tts_usage", engine="qwen")
    assert len(recorder.seen) == 1
    usage_core.unregister_sink(recorder)
    emit("tts_usage", engine="qwen")
    assert len(recorder.seen) == 1


def test_build_row_maps_each_kind() -> None:
    session = build_row("session_usage", _SESSION_FIELDS)
    assert session["kind"] == "session" and session["provider"] is None
    assert session["session_id"] == "sess-1" and session["user_id"] == 77
    assert session["payload"]["stt_sec"] == 12.5 and session["created_at"] is not None

    llm = build_row("llm_usage", _LLM_FIELDS)
    assert llm["kind"] == "llm" and llm["provider"] == "anthropic"
    assert llm["model"] == "claude-sonnet-4-6" and llm["purpose"] == "scenario_gen"
    assert llm["user_id"] == 7 and llm["scenario_id"] == 42 and llm["session_id"] is None

    tts = build_row("tts_usage", _TTS_FIELDS)
    assert tts["kind"] == "tts" and tts["provider"] == "qwen", "tts 는 engine 이 provider"
    assert tts["user_id"] is None and tts["scenario_id"] == 1


def test_build_row_rejects_unknown_kind_and_bad_ids() -> None:
    with pytest.raises(ValueError):
        build_row("voice_turn", {})
    row = build_row("llm_usage", {"user_id": "abc", "scenario_id": True})
    assert row["user_id"] is None and row["scenario_id"] is None


@pytest.mark.asyncio
async def test_record_event_adds_row_and_commits() -> None:
    store: dict = {}
    ok = await record_event("llm_usage", _LLM_FIELDS, session_factory=lambda: _FakeSession(store))

    assert ok is True and store["commits"] == 1
    [row] = store["added"]
    assert isinstance(row, UsageEventORM)
    assert row.kind == "llm" and row.user_id == 7 and row.payload["output_tokens"] == 220


@pytest.mark.asyncio
async def test_record_event_swallows_failures() -> None:
    boom = await record_event(
        "llm_usage", _LLM_FIELDS, session_factory=lambda: _FakeSession({}, commit_error=RuntimeError("DB down"))
    )
    dup = await record_event(
        "session_usage", _SESSION_FIELDS,
        session_factory=lambda: _FakeSession({}, commit_error=IntegrityError("stmt", {}, Exception("dup"))),
    )
    assert boom is False and dup is False
    assert await record_event("voice_turn", {}, session_factory=lambda: _FakeSession({})) is False


@pytest.mark.asyncio
async def test_sqlite_round_trip_and_session_row_uniqueness() -> None:
    assert "usage_event" in Base.metadata.tables, "create_all 대상에 들어가야 배포 때 자동 생성된다"
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)

        assert await record_event("session_usage", _SESSION_FIELDS, session_factory=sessions) is True
        assert await record_event("session_usage", _SESSION_FIELDS, session_factory=sessions) is False, "세션 행 중복"
        assert await record_event("llm_usage", _LLM_FIELDS, session_factory=sessions) is True
        assert await record_event("tts_usage", _TTS_FIELDS, session_factory=sessions) is True

        async with sessions() as db:
            rows = (await db.execute(select(UsageEventORM).order_by(UsageEventORM.event_id))).scalars().all()
        assert [r.kind for r in rows] == ["session", "llm", "tts"]
        assert rows[0].payload == dict(_SESSION_FIELDS)
        assert rows[1].provider == "anthropic" and rows[2].provider == "qwen"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_db_sink_schedules_a_task_and_the_row_lands(monkeypatch) -> None:
    store: dict = {}
    monkeypatch.setattr(usage_service, "AsyncSessionLocal", lambda: _FakeSession(store))

    db_sink("llm_usage", dict(_LLM_FIELDS))
    assert usage_service._running, "태스크가 떼어져 있어야 한다"
    await usage_service.drain()

    assert store.get("commits") == 1 and store["added"][0].kind == "llm"
    assert not usage_service._running


def test_db_sink_outside_an_event_loop_is_a_noop() -> None:
    db_sink("llm_usage", dict(_LLM_FIELDS))
    assert not usage_service._running


def test_install_registers_db_sink_once() -> None:
    usage_service.install()
    usage_service.install()
    assert usage_core._SINKS.count(db_sink) == 1
    usage_service.uninstall()
    assert db_sink not in usage_core._SINKS


@pytest.mark.asyncio
async def test_pipeline_close_delivers_session_usage_to_sink(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    recorder = _Recorder()
    usage_core.register_sink(recorder)
    p = _closing(monkeypatch, seconds=2)

    await p._teardown()

    [(kind, fields)] = [s for s in recorder.seen if s[0] == "session_usage"]
    assert fields["user_id"] == 77 and fields["scenario_id"] == 3
    assert fields["stt_sec"] == 2.0
    assert asyncio.iscoroutinefunction(record_event)


@pytest.mark.asyncio
async def test_db_sink_limits_concurrent_db_sessions(monkeypatch) -> None:
    state = {"active": 0, "max_active": 0, "commits": 0}

    class _SlowSession:
        async def __aenter__(self):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            return self

        async def __aexit__(self, *exc) -> bool:
            state["active"] -= 1
            return False

        def add(self, obj: object) -> None:
            pass

        async def commit(self) -> None:
            await asyncio.sleep(0.01)
            state["commits"] += 1

    monkeypatch.setattr(usage_service, "AsyncSessionLocal", lambda: _SlowSession())
    for i in range(10):
        db_sink("llm_usage", {**_LLM_FIELDS, "attempt": i})
    await usage_service.drain()

    assert state["commits"] == 10
    assert state["max_active"] <= usage_service._WRITE_MAX_CONCURRENCY
