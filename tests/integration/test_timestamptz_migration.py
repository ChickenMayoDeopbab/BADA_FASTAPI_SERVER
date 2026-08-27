from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.base import migrate_naive_utc_to_timestamptz

_ENV = Path(__file__).resolve().parents[2] / ".env"
_KST = ZoneInfo("Asia/Seoul")
_TEMP_DB = "bada_tz_migration_test"

_LEGACY_NAIVE_UTC = datetime(2026, 8, 25, 4, 55, 13, 738841)
_EXPECTED_KST = datetime(2026, 8, 25, 13, 55, 13, 738841, tzinfo=_KST)


def _admin_dsn() -> str | None:
    if not _ENV.exists():
        return None
    found = re.search(r"^DATABASE_URL=(.+)$", _ENV.read_text(), re.M)
    return found.group(1).strip() if found else None


_DSN = _admin_dsn()

pytestmark = pytest.mark.skipif(
    _DSN is None, reason=".env 에 DATABASE_URL 이 없어 로컬 Postgres 로 검증할 수 없다"
)


def _swap_database(dsn: str, name: str) -> str:
    return dsn.rsplit("/", 1)[0] + "/" + name


async def _recreate_temp_database() -> None:
    admin = create_async_engine(_DSN, isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{_TEMP_DB}"'))
            await conn.execute(text(f'CREATE DATABASE "{_TEMP_DB}"'))
    finally:
        await admin.dispose()


async def _drop_temp_database() -> None:
    admin = create_async_engine(_DSN, isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{_TEMP_DB}"'))
    finally:
        await admin.dispose()


@pytest.fixture
async def legacy_engine():
    try:
        await _recreate_temp_database()
    except Exception as exc:
        pytest.skip(f"로컬 Postgres 에 붙을 수 없다: {exc}")

    engine = create_async_engine(_swap_database(_DSN, _TEMP_DB))
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                CREATE TABLE post (
                    post_id    BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    deleted_at TIMESTAMP WITHOUT TIME ZONE NULL
                )
                """
            )
        )
        await conn.execute(
            text("INSERT INTO post (created_at, updated_at) VALUES (:v, :v)"),
            {"v": _LEGACY_NAIVE_UTC},
        )
    try:
        yield engine
    finally:
        await engine.dispose()
        await _drop_temp_database()


async def _column_type(engine, column: str) -> str:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = 'post' AND column_name = :c"
                ),
                {"c": column},
            )
        ).scalar_one()


async def _created_at(engine) -> datetime:
    async with engine.connect() as conn:
        return (await conn.execute(text("SELECT created_at FROM post"))).scalar_one()


async def test_migration_changes_column_type(legacy_engine) -> None:
    assert await _column_type(legacy_engine, "created_at") == "timestamp without time zone"

    async with legacy_engine.begin() as conn:
        await migrate_naive_utc_to_timestamptz(conn)

    assert await _column_type(legacy_engine, "created_at") == "timestamp with time zone"
    assert await _column_type(legacy_engine, "deleted_at") == "timestamp with time zone"


async def test_existing_row_keeps_the_same_instant(legacy_engine) -> None:
    async with legacy_engine.begin() as conn:
        await migrate_naive_utc_to_timestamptz(conn)

    stored = await _created_at(legacy_engine)
    assert stored.tzinfo is not None
    assert stored == _EXPECTED_KST
    assert stored.astimezone(_KST).hour == 13, (
        f"9시간 밀렸다: {stored.astimezone(_KST).isoformat()}"
    )


async def test_second_run_is_a_no_op(legacy_engine) -> None:
    async with legacy_engine.begin() as conn:
        first = await migrate_naive_utc_to_timestamptz(conn)
    after_first = await _created_at(legacy_engine)

    async with legacy_engine.begin() as conn:
        second = await migrate_naive_utc_to_timestamptz(conn)
    after_second = await _created_at(legacy_engine)

    assert first, "첫 실행은 컬럼을 옮겼어야 한다"
    assert second == [], f"두 번째 실행이 컬럼을 또 건드렸다: {second}"
    assert after_second == after_first == _EXPECTED_KST


async def test_new_rows_land_on_the_right_instant(legacy_engine) -> None:
    async with legacy_engine.begin() as conn:
        await migrate_naive_utc_to_timestamptz(conn)

    await legacy_engine.dispose()

    now = datetime.now(UTC)
    async with legacy_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO post (created_at, updated_at) VALUES (:v, :v)"), {"v": now}
        )
        stored = (
            await conn.execute(text("SELECT created_at FROM post ORDER BY post_id DESC LIMIT 1"))
        ).scalar_one()

    assert stored == now


async def test_concurrent_migrations_do_not_double_apply(legacy_engine) -> None:
    first_done = asyncio.Event()

    async def slow_first() -> list[str]:
        async with legacy_engine.begin() as conn:
            moved = await migrate_naive_utc_to_timestamptz(conn)
            first_done.set()
            await asyncio.sleep(0.3)
            return moved

    async def second() -> list[str]:
        await first_done.wait()
        async with legacy_engine.begin() as conn:
            return await migrate_naive_utc_to_timestamptz(conn)

    first, other = await asyncio.gather(slow_first(), second())

    assert first, "첫 번째는 컬럼을 옮겼어야 한다"

    stored = await _created_at(legacy_engine)
    assert stored == _EXPECTED_KST, (
        f"동시 실행으로 값이 밀렸다: {stored.astimezone(_KST).isoformat()}"
    )
    assert other == [], f"두 번째가 이미 옮긴 컬럼을 또 건드렸다: {other}"
