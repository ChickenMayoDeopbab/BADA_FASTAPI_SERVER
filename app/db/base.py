from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

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


async def init_db() -> None:
    """FastAPI 소유 테이블 공유 DB에 생성"""
    from app.db import models  # noqa: F401  (ScenarioORM/FeedbackORM 을 metadata 에 등록)
    from app.db.seed import seed_preset_scenarios

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if engine.dialect.name == "postgresql":
            await conn.execute(text(
                "ALTER TABLE scenario ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP NULL"
            ))

    async with AsyncSessionLocal() as session:
        await seed_preset_scenarios(session)
