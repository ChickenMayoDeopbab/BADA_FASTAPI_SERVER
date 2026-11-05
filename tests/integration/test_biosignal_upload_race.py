from __future__ import annotations

import asyncio
import os
import re
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.v1.biosignal import get_biosignal_storage, router
from app.core.config import get_settings
from app.core.timeutil import now_utc
from app.db.base import Base
from app.db.external import external_metadata, users_table
from app.db.models import CallBiosignalORM
from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.services import biosignal_service as svc
from tests.unit.test_biosignal_api import USERS, _payload, _Storage
from tests.unit.test_biosignal_cardiac import _timeline

_ENV = Path(__file__).resolve().parents[2] / ".env"
_TEMP_DB = f"bada_biosignal_race_{os.getpid()}"
_URL = "/api/v1/biosignals/sess-1"
_RAW_KEY = "biosignal-raw/sess-1.json.gz"


def _admin_dsn() -> str | None:
    if not _ENV.exists():
        return None
    found = re.search(r"^DATABASE_URL=(.+)$", _ENV.read_text(), re.M)
    return found.group(1).strip() if found else None


_DSN = _admin_dsn()

pytestmark = pytest.mark.skipif(
    _DSN is None, reason=".env 에 DATABASE_URL 이 없어 로컬 Postgres 로 검증할 수 없다"
)


async def _admin(*statements: str) -> None:
    admin = create_async_engine(_DSN, isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            for statement in statements:
                await conn.execute(text(statement))
    finally:
        await admin.dispose()


class _CountingStorage(_Storage):
    def __init__(self) -> None:
        super().__init__()
        self.keys: list[str] = []

    def upload_encrypted(self, key: str, data: bytes) -> str:
        self.keys.append(key)
        return super().upload_encrypted(key, data)


class _GatedAnalysis:
    """첫 호출은 release 까지 붙잡고(fail_first 면 그 뒤 실패), 나머지는 바로 끝나는 분석 대역"""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail_first = fail_first
        self._lock = threading.Lock()

    def __call__(self, timeline: dict, payload: dict, previous: list[float], turn_metrics: dict | None) -> dict:
        with self._lock:
            self.calls += 1
            first = self.calls == 1
        if first:
            self.started.set()
            self.release.wait(10)
            if self.fail_first:
                raise RuntimeError("분석 실패")
        return {"cardiac": {"status": "MEASURED"}}


@dataclass
class RaceEnv:
    client: httpx.AsyncClient
    sessions: async_sessionmaker
    storage: _CountingStorage

    async def row(self) -> CallBiosignalORM:
        async with self.sessions() as session:
            stmt = select(CallBiosignalORM).where(CallBiosignalORM.session_id == "sess-1")
            return (await session.execute(stmt)).scalar_one()


@pytest.fixture
async def race_env() -> AsyncIterator[RaceEnv]:
    drop = f'DROP DATABASE IF EXISTS "{_TEMP_DB}" WITH (FORCE)'
    try:
        await _admin(drop, f'CREATE DATABASE "{_TEMP_DB}"')
    except Exception as exc:
        pytest.skip(f"로컬 Postgres 에 붙을 수 없다: {exc}")

    engine = create_async_engine(_DSN.rsplit("/", 1)[0] + "/" + _TEMP_DB)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(external_metadata.create_all)
            await conn.execute(users_table.insert().values(**USERS[0]))
        sessions = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with sessions() as session:
            session.add(CallBiosignalORM(
                session_id="sess-1", user_id=7, scenario_id=3, session_type="SCENARIO",
                status="AWAITING_DATA", timeline=_timeline(), created_at=now_utc(),
            ))
            await session.commit()

        async def _get_db() -> AsyncIterator[AsyncSession]:
            async with sessions() as session:
                yield session

        storage = _CountingStorage()
        settings = get_settings().model_copy(update={"biosignal_enabled": True, "biosignal_raw_user_ids": "7"})
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = _get_db
        app.dependency_overrides[get_current_user_id] = lambda: 7
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_biosignal_storage] = lambda: storage
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield RaceEnv(client=client, sessions=sessions, storage=storage)
    finally:
        await engine.dispose()
        await _admin(drop)


async def _race(env: RaceEnv, gate: _GatedAnalysis) -> list[httpx.Response]:
    first = asyncio.create_task(env.client.post(_URL, json=_payload()))
    second = None
    try:
        assert await asyncio.to_thread(gate.started.wait, 10)
        second = asyncio.create_task(env.client.post(_URL, json=_payload()))
        await asyncio.sleep(0.5)
        assert not second.done()
        assert gate.calls == 1
    finally:
        gate.release.set()
        results = await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    return list(results)


async def test_concurrent_uploads_of_one_call_analyze_once(race_env: RaceEnv, monkeypatch) -> None:
    gate = _GatedAnalysis()
    monkeypatch.setattr(svc, "analyze_upload", gate)

    first, second = await _race(race_env, gate)

    assert (first.status_code, second.status_code) == (200, 409)
    assert second.json()["detail"] == "ALREADY_ANALYZED"
    assert gate.calls == 1
    assert race_env.storage.keys == [_RAW_KEY]
    row = await race_env.row()
    assert (row.status, row.raw_key) == ("ANALYZED", _RAW_KEY)


async def test_failed_upload_rolls_back_and_the_waiting_upload_analyzes(race_env: RaceEnv, monkeypatch) -> None:
    gate = _GatedAnalysis(fail_first=True)
    monkeypatch.setattr(svc, "analyze_upload", gate)

    first, second = await _race(race_env, gate)

    assert (first.status_code, second.status_code) == (500, 200)
    assert second.json()["status"] == "ANALYZED"
    assert gate.calls == 2
    assert race_env.storage.keys == [_RAW_KEY]
    assert (await race_env.row()).status == "ANALYZED"
