from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.timeutil import now_utc
from app.db.base import Base
from app.db.models import CallBiosignalORM
from app.services import biosignal_service as svc
from tests.integration.test_biosignal_upload_race import _DSN, _admin
from tests.unit.test_biosignal_api import LATEST_FIVE_RESTS, _baseline_history

_TEMP_DB = f"bada_biosignal_baseline_{os.getpid()}"

pytestmark = pytest.mark.skipif(
    _DSN is None, reason=".env 에 DATABASE_URL 이 없어 로컬 Postgres 로 검증할 수 없다"
)


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker]:
    drop = f'DROP DATABASE IF EXISTS "{_TEMP_DB}" WITH (FORCE)'
    try:
        await _admin(drop, f'CREATE DATABASE "{_TEMP_DB}"')
    except Exception as exc:
        pytest.skip(f"로컬 Postgres 에 붙을 수 없다: {exc}")

    engine = create_async_engine(_DSN.rsplit("/", 1)[0] + "/" + _TEMP_DB)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()
        await _admin(drop)


async def test_previous_baselines_pick_the_latest_five_rest_rows_in_postgres(sessions: async_sessionmaker) -> None:
    now = now_utc()
    async with sessions() as session:
        session.add_all(
            CallBiosignalORM(scenario_id=3, session_type="SCENARIO", timeline={}, **row)
            for row in _baseline_history(now)
        )
        await session.commit()

        assert await svc.previous_baselines(session, 7, now) == LATEST_FIVE_RESTS
