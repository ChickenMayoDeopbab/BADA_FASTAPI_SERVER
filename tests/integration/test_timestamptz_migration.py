"""naive UTC → timestamptz 마이그레이션 회귀 테스트 (실제 Postgres 필요).

두 가지를 지킨다.

1. **캐스팅 방향.** 기존 행은 naive UTC 로 들어있다. 그냥 `::timestamptz` 로 캐스팅하면
   Postgres 가 세션 TimeZone(KST) 으로 해석해서 값이 9시간 밀린다.
   반드시 `AT TIME ZONE 'UTC'` 를 거쳐야 한다.
2. **멱등성.** `init_db()` 는 매 부팅 실행된다. 이미 timestamptz 인 컬럼에
   `AT TIME ZONE 'UTC'` 를 또 걸면 naive 로 되돌아가므로, 재부팅마다 9시간씩 밀린다.
   두 번째 실행은 아무것도 건드리지 않아야 한다.
"""

from __future__ import annotations

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

# 로컬 DB 의 실제 행과 같은 값. 04:55 UTC 는 KST 로 13:55 여야 한다.
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
    """전환 이전 스키마(naive) 로 만들어진 post 테이블에 기존 행 하나를 심어둔다."""
    try:
        await _recreate_temp_database()
    except Exception as exc:  # noqa: BLE001 - 로컬에 Postgres 가 없으면 건너뛴다
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
    """04:55 naive UTC 는 13:55 KST 여야 한다. 04:55 KST 로 나오면 캐스팅이 틀린 것."""
    async with legacy_engine.begin() as conn:
        await migrate_naive_utc_to_timestamptz(conn)

    stored = await _created_at(legacy_engine)
    assert stored.tzinfo is not None
    assert stored == _EXPECTED_KST
    assert stored.astimezone(_KST).hour == 13, (
        f"9시간 밀렸다: {stored.astimezone(_KST).isoformat()}"
    )


async def test_second_run_is_a_no_op(legacy_engine) -> None:
    """재부팅마다 9시간씩 밀리는 사고를 막는 테스트."""
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

    # init_db 가 전환 직후 하는 것과 같다. asyncpg 는 컬럼 타입을 박아 statement 를
    # 캐시하므로, 풀을 비우지 않으면 전환 전에 준비된 statement 가 남아
    # 이후 INSERT 가 DataError 로 터진다.
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
