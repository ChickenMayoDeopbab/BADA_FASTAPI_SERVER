import gzip
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.v1.biosignal import get_biosignal_storage, router
from app.core.config import get_settings
from app.core.timeutil import now_utc
from app.db.base import Base
from app.db.external import external_metadata, users_table
from app.db.models import CallBiosignalORM, FeedbackORM
from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.schemas.biosignal import MAX_BODY_BYTES
from app.services import biosignal_service as svc
from app.services import recording_crypto
from app.services.recording_storage import RecordingStorageService
from app.services.training_data_cleanup import delete_training_data
from app.services.user_data_cleanup import delete_user_data
from tests.unit.fake_kms import FakeKms
from tests.unit.test_biosignal_cardiac import W2P, _beats, _hr, _ibi_rows, _markers, _timeline
from tests.unit.test_biosignal_eda_temp import _eda, _temps
from tests.unit.test_recording_storage import _S3Client, _settings

AGREED = datetime(2026, 11, 1, tzinfo=UTC)
USERS = (
    {"user_id": 7, "name": "동의", "role": "USER", "status": "ACTIVE", "biosignal_agreed_at": AGREED},
    {"user_id": 8, "name": "미동의", "role": "USER", "status": "ACTIVE"},
    {"user_id": 9, "name": "다른 사람", "role": "USER", "status": "ACTIVE", "biosignal_agreed_at": AGREED},
    {
        "user_id": 10, "name": "철회", "role": "USER", "status": "ACTIVE",
        "biosignal_agreed_at": AGREED, "biosignal_withdrawn_at": AGREED,
    },
)
AVTI = {"1": 2.0, "2": 6.5, "3": 2.2, "4": 3.0, "5": 7.0, "6": 1.5}


class _Storage:
    def __init__(self, *, fail: bool = False) -> None:
        self.uploads: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.fail = fail

    def upload_encrypted(self, key: str, data: bytes) -> str:
        if self.fail:
            raise RuntimeError("kms down")
        self.uploads[key] = data
        return key

    def delete(self, key: str) -> bool:
        self.deleted.append(key)
        return True


@dataclass
class Env:
    client: httpx.AsyncClient
    sessions: async_sessionmaker
    storage: _Storage
    current: dict


@asynccontextmanager
async def _env(*, enabled: bool = True, raw_users: str = "", storage: _Storage | None = None) -> AsyncIterator[Env]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(external_metadata.create_all)
        await conn.execute(text("ALTER TABLE file ADD COLUMN user_id BIGINT"))
        for user in USERS:
            await conn.execute(users_table.insert().values(**user))
    sessions = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    current = {"user_id": 7}
    fake = storage or _Storage()
    settings = get_settings().model_copy(update={"biosignal_enabled": enabled, "biosignal_raw_user_ids": raw_users})
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_current_user_id] = lambda: current["user_id"]
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_biosignal_storage] = lambda: fake
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield Env(client=client, sessions=sessions, storage=fake, current=current)
    finally:
        await engine.dispose()


async def _seed(env: Env, session_id: str = "sess-1", user_id: int = 7, *, turn_metrics: dict | None = None, **fields):
    row = {
        "session_id": session_id,
        "user_id": user_id,
        "scenario_id": 3,
        "session_type": "SCENARIO",
        "status": "AWAITING_DATA",
        "timeline": _timeline(),
        "created_at": now_utc(),
    }
    row.update(fields)
    async with env.sessions() as session:
        session.add(CallBiosignalORM(**row))
        if turn_metrics is not None:
            session.add(FeedbackORM(
                feedback_id=1, session_id=session_id, user_id=user_id, scenario_id=3, shake_count=0, silence_duration=0,
                highlights="[]", turn_metrics=json.dumps(turn_metrics), created_at=now_utc(),
            ))
        await session.commit()


async def _row(env: Env, session_id: str = "sess-1") -> CallBiosignalORM:
    async with env.sessions() as session:
        stmt = select(CallBiosignalORM).where(CallBiosignalORM.session_id == session_id)
        return (await session.execute(stmt)).scalar_one()


def _payload(**overrides) -> dict:
    times, rr = _beats()
    body = {
        "v": 1,
        "source": "samsung_sensor_sdk",
        "device": {"model": "SM-TEST", "sdk": "1.4.1"},
        "clock": {"watch_minus_phone_ms": W2P},
        "markers": _markers(),
        "hr": _hr(lag=3000),
        "ibi": _ibi_rows(times, rr),
        "eda": _eda(lag=3000),
        "skin_temp": _temps(),
        "acc": [],
    }
    body.update(overrides)
    return body


def _no_rest_markers() -> dict:
    return {key: value for key, value in _markers().items() if key != "rest_baseline"}


def _baseline_row(
    session_id: str, value: float, at: datetime, *, user_id: int = 7, source: str = "rest", status: str = "MEASURED"
) -> dict:
    baseline = {"status": status, "value": value, "source": source}
    return {
        "session_id": session_id, "user_id": user_id, "status": "ANALYZED", "created_at": at,
        "metrics": {"cardiac": {"baseline": baseline}},
    }


def _baseline_history(now: datetime) -> list[dict]:
    start = now - timedelta(days=1)
    rests = [
        _baseline_row(f"sess-rest-{k}", value, start + timedelta(hours=k))
        for k, value in enumerate((60.0, 70.0, 71.0, 69.0, 73.0, 90.0))
    ]
    without_rest = [
        _baseline_row(f"sess-norest-{k}", 99.0, start + timedelta(hours=6 + k), source="previous_sessions")
        for k in range(5)
    ]
    ignored = [
        _baseline_row("sess-short-rest", 120.0, start + timedelta(hours=11), status="INSUFFICIENT_DATA"),
        _baseline_row("sess-other-user", 150.0, start + timedelta(hours=12), user_id=9),
        _baseline_row("sess-later", 160.0, now + timedelta(hours=1)),
    ]
    return rests + without_rest + ignored


LATEST_FIVE_RESTS = [70.0, 71.0, 69.0, 73.0, 90.0]


async def test_disabled_feature_is_not_found() -> None:
    async with _env(enabled=False) as env:
        await _seed(env)

        assert (await env.client.post("/api/v1/biosignals/sess-1", json=_payload())).status_code == 404
        assert (await env.client.get("/api/v1/biosignals/sess-1")).status_code == 404


async def test_upload_analyzes_and_stores_the_session() -> None:
    async with _env() as env:
        await _seed(env, turn_metrics={"turn_sec": {k: [0, 1] for k in AVTI}, "avti": AVTI})

        response = await env.client.post("/api/v1/biosignals/sess-1", json=_payload())

        assert response.status_code == 200
        body = response.json()
        assert (body["status"], body["source"]) == ("ANALYZED", "samsung_sensor_sdk")
        metrics = body["metrics"]
        assert set(metrics) == {"cardiac", "eda", "skin_temp", "fusion", "ppg"}
        assert metrics["ppg"]["status"] == "UNAVAILABLE"
        assert (metrics["cardiac"]["baseline"]["value"], metrics["cardiac"]["baseline"]["source"]) == (70.0, "rest")
        assert metrics["cardiac"]["hrv"]["baseline"]["status"] == "MEASURED"
        assert {t["change_vs_baseline_pct"] for t in metrics["eda"]["turns"].values()} == {40.0}
        assert metrics["skin_temp"]["status"] == "MEASURED"
        assert [r["turn"] for r in metrics["fusion"]["turns"]] == [1, 2, 3, 4, 5, 6]
        assert metrics["fusion"]["quadrants"]["body_high_voice_steady"] == 4

        row = await _row(env)
        assert (row.status, row.source, row.device, row.raw_key) == (
            "ANALYZED", "samsung_sensor_sdk", {"model": "SM-TEST", "sdk": "1.4.1"}, None,
        )
        assert row.analyzed_at is not None
        assert row.metrics == metrics
        assert (await env.client.get("/api/v1/biosignals/sess-1")).json() == body


async def test_get_reports_waiting_and_guards_the_owner() -> None:
    async with _env() as env:
        await _seed(env)
        await _seed(env, "sess-9", 9)

        assert (await env.client.get("/api/v1/biosignals/sess-1")).json() == {
            "session_id": "sess-1", "status": "AWAITING_DATA",
        }
        assert (await env.client.get("/api/v1/biosignals/sess-9")).status_code == 403
        assert (await env.client.get("/api/v1/biosignals/missing")).status_code == 404


async def test_upload_conflicts_and_ownership() -> None:
    async with _env() as env:
        await _seed(env)
        await _seed(env, "sess-9", 9)

        missing = await env.client.post("/api/v1/biosignals/missing", json=_payload())
        foreign = await env.client.post("/api/v1/biosignals/sess-9", json=_payload())
        first = await env.client.post("/api/v1/biosignals/sess-1", json=_payload())
        again = await env.client.post("/api/v1/biosignals/sess-1", json=_payload())

        assert (missing.status_code, missing.json()["detail"]) == (409, "NO_TIMELINE")
        assert (foreign.status_code, foreign.json()["detail"]) == (403, "FORBIDDEN")
        assert first.status_code == 200
        assert (again.status_code, again.json()["detail"]) == (409, "ALREADY_ANALYZED")
        assert (await _row(env, "sess-9")).status == "AWAITING_DATA"


@pytest.mark.parametrize("user_id", [8, 10])
async def test_upload_without_consent_rejects_and_destroys_the_timeline(user_id: int) -> None:
    async with _env() as env:
        await _seed(env, f"sess-{user_id}", user_id)
        env.current["user_id"] = user_id

        response = await env.client.post(f"/api/v1/biosignals/sess-{user_id}", json=_payload())

        assert (response.status_code, response.json()["detail"]) == (403, "BIOSIGNAL_CONSENT_REQUIRED")
        row = await _row(env, f"sess-{user_id}")
        assert (row.status, row.timeline, row.metrics) == ("REJECTED", {}, None)


async def test_too_large_or_invalid_payloads_leave_the_timeline_waiting() -> None:
    async with _env() as env:
        await _seed(env)

        large = await env.client.post(
            "/api/v1/biosignals/sess-1",
            content=b"x" * (MAX_BODY_BYTES + 1),
            headers={"content-type": "application/json"},
        )
        wrong_version = await env.client.post("/api/v1/biosignals/sess-1", json=_payload(v=2))
        extra_field = await env.client.post("/api/v1/biosignals/sess-1", json=_payload(spo2=[]))
        wrong_source = await env.client.post("/api/v1/biosignals/sess-1", json=_payload(source="fitbit"))

        assert (large.status_code, large.json()["detail"]) == (413, "PAYLOAD_TOO_LARGE")
        assert {wrong_version.status_code, extra_field.status_code, wrong_source.status_code} == {422}
        assert (await _row(env)).status == "AWAITING_DATA"


async def test_stale_timelines_expire_and_are_destroyed_on_upload() -> None:
    async with _env() as env:
        await _seed(env, "sess-old", created_at=now_utc() - timedelta(hours=25))
        await _seed(env, "sess-9-old", 9, created_at=now_utc() - timedelta(hours=30))
        await _seed(env, "sess-fresh", 9, created_at=now_utc() - timedelta(hours=23))

        response = await env.client.post("/api/v1/biosignals/sess-old", json=_payload())

        assert (response.status_code, response.json()["detail"]) == (409, "EXPIRED")
        for session_id in ("sess-old", "sess-9-old"):
            row = await _row(env, session_id)
            assert (row.status, row.timeline) == ("EXPIRED", {})
        assert (await _row(env, "sess-fresh")).status == "AWAITING_DATA"


async def test_previous_rest_baselines_fill_in_when_rest_is_missing() -> None:
    def metrics(value: float, source: str = "rest") -> dict:
        return {"cardiac": {"baseline": {"status": "MEASURED", "value": value, "source": source}}}

    async with _env() as env:
        earlier = now_utc() - timedelta(days=1)
        await _seed(env, "sess-a", status="ANALYZED", metrics=metrics(72.0), created_at=earlier - timedelta(hours=3))
        await _seed(env, "sess-b", status="ANALYZED", metrics=metrics(74.0), created_at=earlier - timedelta(hours=2))
        await _seed(env, "sess-c", status="ANALYZED", metrics=metrics(90.0, "previous_sessions"), created_at=earlier)
        await _seed(env, "sess-d", 9, status="ANALYZED", metrics=metrics(60.0), created_at=earlier)
        await _seed(env)

        no_rest = {key: value for key, value in _markers().items() if key != "rest_baseline"}
        response = await env.client.post("/api/v1/biosignals/sess-1", json=_payload(markers=no_rest))

        baseline = response.json()["metrics"]["cardiac"]["baseline"]
        assert (baseline["source"], baseline["value"], baseline["sessions"]) == ("previous_sessions", 73.0, 2)


async def test_one_rest_baseline_outlasts_later_sessions_without_rest() -> None:
    async with _env() as env:
        start = now_utc() - timedelta(days=1)
        await _seed(env, **_baseline_row("sess-rest", 72.0, start))
        for k in range(6):
            row = _baseline_row(f"sess-norest-{k}", 99.0, start + timedelta(hours=k + 1), source="previous_sessions")
            await _seed(env, **row)
        await _seed(env)

        response = await env.client.post("/api/v1/biosignals/sess-1", json=_payload(markers=_no_rest_markers()))

        baseline = response.json()["metrics"]["cardiac"]["baseline"]
        assert (baseline["source"], baseline.get("value"), baseline.get("sessions")) == ("previous_sessions", 72.0, 1)


async def test_previous_baselines_use_the_latest_five_rest_sessions() -> None:
    async with _env() as env:
        for row in _baseline_history(now_utc()):
            await _seed(env, **row)
        await _seed(env)

        response = await env.client.post("/api/v1/biosignals/sess-1", json=_payload(markers=_no_rest_markers()))

        baseline = response.json()["metrics"]["cardiac"]["baseline"]
        assert (baseline["source"], baseline.get("value"), baseline.get("sessions")) == ("previous_sessions", 71.0, 5)
        async with env.sessions() as session:
            assert await svc.previous_baselines(session, 7, now_utc()) == LATEST_FIVE_RESTS


async def test_raw_signals_are_kept_encrypted_only_for_allowed_users() -> None:
    async with _env(raw_users="3, 7") as env:
        await _seed(env)
        payload = _payload()

        await env.client.post("/api/v1/biosignals/sess-1", json=payload)

        assert list(env.storage.uploads) == ["biosignal-raw/sess-1.json.gz"]
        stored = json.loads(gzip.decompress(env.storage.uploads["biosignal-raw/sess-1.json.gz"]))
        assert stored["source"] == "samsung_sensor_sdk"
        assert (len(stored["hr"]), len(stored["ibi"])) == (len(payload["hr"]), len(payload["ibi"]))
        row = await _row(env)
        assert row.raw_key == "biosignal-raw/sess-1.json.gz"
        expires = row.raw_expires_at.replace(tzinfo=UTC) - now_utc()
        assert timedelta(days=89, hours=23) < expires <= timedelta(days=90)

    async with _env(raw_users="9") as env:
        await _seed(env)
        await env.client.post("/api/v1/biosignals/sess-1", json=_payload())
        assert env.storage.uploads == {}
        assert (await _row(env)).raw_key is None


async def test_raw_upload_failure_still_saves_the_analysis(caplog) -> None:
    async with _env(raw_users="7", storage=_Storage(fail=True)) as env:
        await _seed(env)

        response = await env.client.post("/api/v1/biosignals/sess-1", json=_payload())

        assert response.json()["status"] == "ANALYZED"
        assert (await _row(env)).raw_key is None
        assert any("원시 생체신호 보관 실패" in r.getMessage() for r in caplog.records)


async def test_expired_raw_signals_are_swept_on_upload() -> None:
    async with _env() as env:
        await _seed(
            env, "sess-x", 9, status="ANALYZED", metrics={},
            raw_key="biosignal-raw/sess-x.json.gz", raw_expires_at=now_utc() - timedelta(days=1),
        )
        await _seed(
            env, "sess-y", 9, status="ANALYZED", metrics={},
            raw_key="biosignal-raw/sess-y.json.gz", raw_expires_at=now_utc() + timedelta(days=1),
        )
        await _seed(env)

        await env.client.post("/api/v1/biosignals/sess-1", json=_payload())

        assert env.storage.deleted == ["biosignal-raw/sess-x.json.gz"]
        assert (await _row(env, "sess-x")).raw_key is None
        assert (await _row(env, "sess-y")).raw_key == "biosignal-raw/sess-y.json.gz"


def test_upload_encrypted_locks_data_with_kms() -> None:
    s3, kms = _S3Client(), FakeKms()
    storage = RecordingStorageService(_settings(), client=s3, kms_client=kms)

    key = storage.upload_encrypted("biosignal-raw/s.json.gz", b"raw")

    blob = s3.objects[key]
    assert recording_crypto.is_encrypted(blob)
    assert recording_crypto.decrypt(blob, s3_key=key, kms=kms) == b"raw"

    unlocked = _settings().model_copy(update={"recording_kms_key_id": None})
    no_key = RecordingStorageService(unlocked, client=s3, kms_client=kms)
    with pytest.raises(RuntimeError):
        no_key.upload_encrypted("biosignal-raw/s.json.gz", b"raw")


async def test_record_and_account_deletion_remove_raw_objects() -> None:
    async with _env() as env:
        await _seed(env, "sess-1", status="ANALYZED", metrics={}, raw_key="biosignal-raw/sess-1.json.gz")
        await _seed(env, "sess-2", status="ANALYZED", metrics={}, raw_key="biosignal-raw/sess-2.json.gz")
        await _seed(env, "sess-9", 9, status="ANALYZED", metrics={}, raw_key="biosignal-raw/sess-9.json.gz")

        async with env.sessions() as session:
            await delete_training_data(session, "sess-1", storage=env.storage)
        assert env.storage.deleted == ["biosignal-raw/sess-1.json.gz"]

        async with env.sessions() as session:
            await delete_user_data(session, 7, storage=env.storage)
            remaining = list((await session.execute(select(CallBiosignalORM.session_id))).scalars())

        assert env.storage.deleted == ["biosignal-raw/sess-1.json.gz", "biosignal-raw/sess-2.json.gz"]
        assert remaining == ["sess-9"]


class _CapturingDb:
    def __init__(self) -> None:
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        return self

    def scalar_one_or_none(self) -> None:
        return None


async def test_find_owned_locks_the_row_only_when_asked() -> None:
    db = _CapturingDb()

    await svc.find_owned(db, "sess-1", 7, for_update=True)
    await svc.find_owned(db, "sess-1", 7)

    locked, plain = (str(stmt.compile(dialect=postgresql.dialect())) for stmt in db.statements)
    assert locked.rstrip().endswith("FOR UPDATE")
    assert "FOR UPDATE" not in plain


async def test_upload_locks_the_row_and_get_does_not(monkeypatch) -> None:
    locks = []
    original = svc.find_owned

    async def spy(db, session_id, user_id, **kwargs):
        locks.append(kwargs.get("for_update", False))
        return await original(db, session_id, user_id, **kwargs)

    monkeypatch.setattr(svc, "find_owned", spy)
    async with _env() as env:
        await _seed(env)

        assert (await env.client.post("/api/v1/biosignals/sess-1", json=_payload())).status_code == 200
        assert (await env.client.get("/api/v1/biosignals/sess-1")).status_code == 200

    assert locks == [True, False]


async def test_a_failing_raw_sweep_does_not_fail_a_committed_upload(monkeypatch) -> None:
    async def broken_sweep(*args, **kwargs):
        raise RuntimeError("sweep down")

    monkeypatch.setattr(svc, "sweep_expired_raw", broken_sweep)
    async with _env() as env:
        await _seed(env)

        response = await env.client.post("/api/v1/biosignals/sess-1", json=_payload())

        assert (response.status_code, response.json()["status"]) == (200, "ANALYZED")
        assert (await _row(env)).status == "ANALYZED"


def test_the_app_serves_the_biosignal_routes() -> None:
    from app.main import create_app

    paths = {(route.path, method) for route in create_app().routes for method in getattr(route, "methods", ())}

    assert {("/api/v1/biosignals/{session_id}", "POST"), ("/api/v1/biosignals/{session_id}", "GET")} <= paths
