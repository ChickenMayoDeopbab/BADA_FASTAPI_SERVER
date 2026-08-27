import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    get_settings().database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


_TZ_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scenario", ("created_at", "deleted_at")),
    ("feedback", ("created_at",)),
    ("voice_tremor_metric", ("created_at",)),
    ("post", ("created_at", "updated_at", "deleted_at")),
    ("post_comment", ("created_at", "updated_at", "deleted_at")),
    ("post_reaction", ("created_at",)),
    ("post_attachment", ("created_at",)),
)


async def migrate_naive_utc_to_timestamptz(conn: AsyncConnection) -> list[str]:
    """timestamptz로 마이그레이션 하는 코드"""
    naive = {
        (row.table_name, row.column_name)
        for row in (
            await conn.execute(
                text(
                    """
                    SELECT table_name, column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND data_type = 'timestamp without time zone'
                    """
                )
            )
        ).all()
    }

    migrated: list[str] = []
    for table, columns in _TZ_COLUMNS:
        for column in columns:
            if (table, column) not in naive:
                continue
            await conn.execute(
                text(
                    f'ALTER TABLE "{table}" ALTER COLUMN "{column}" '
                    f"TYPE timestamptz USING \"{column}\" AT TIME ZONE 'UTC'"
                )
            )
            migrated.append(f"{table}.{column}")
    return migrated


async def init_db() -> None:
    """FastAPI 소유 테이블 공유 DB에 생성"""
    from app.db import models  # noqa: F401  (ScenarioORM/FeedbackORM 을 metadata 에 등록)
    from app.db.seed import seed_preset_scenarios

    moved: list[str] = []
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        if engine.dialect.name == "postgresql":
            await conn.execute(
                text(
                    "ALTER TABLE scenario "
                    "ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ NULL"
                )
            )

            await conn.execute(
                text(
                    "ALTER TABLE scenario "
                    "ADD COLUMN IF NOT EXISTS category VARCHAR(20) NULL"
                )
            )

            await conn.execute(
                text(
                    "ALTER TABLE scenario "
                    "ADD COLUMN IF NOT EXISTS origin_scenario_id BIGINT NULL"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_scenario_origin_user "
                    "ON scenario (origin_scenario_id, user_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_scenario_origin_user_alive "
                    "ON scenario (origin_scenario_id, user_id) "
                    "WHERE origin_scenario_id IS NOT NULL AND deleted_at IS NULL"
                )
            )

            await conn.execute(
                text(
                    """
                    UPDATE scenario
                    SET category = CASE
                        WHEN is_custom = TRUE THEN 'other'
                        ELSE 'daily'
                    END
                    WHERE category IS NULL
                       OR category NOT IN ('work', 'daily', 'school', 'other')
                    """
                )
            )

            await conn.execute(
                text(
                    "ALTER TABLE scenario "
                    "ALTER COLUMN category SET NOT NULL"
                )
            )

            moved = await migrate_naive_utc_to_timestamptz(conn)
            if moved:
                logger.info("timestamptz 로 전환한 컬럼: %s", ", ".join(moved))

    if moved:
        await engine.dispose()

    async with AsyncSessionLocal() as session:
        await seed_preset_scenarios(session)
