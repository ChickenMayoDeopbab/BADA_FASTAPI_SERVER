from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

engine = create_async_engine(
    get_settings().database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass

async def init_db() -> None:
    from app.core.preset_scenarios import PRESET_SCENARIOS
    from app.db.models import ScenarioORM

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ScenarioORM.scenario_id))
        existing_ids = set(result.scalars().all())

        for scenario in PRESET_SCENARIOS:
            if scenario["scenario_id"] in existing_ids:
                continue

            session.add(
                ScenarioORM(
                    scenario_id=scenario["scenario_id"],
                    title=scenario["title"],
                    content=scenario["content"],
                    scenario_image=scenario["scenario_image"],
                    tts_voice_id=scenario["tts_voice_id"],
                    ai_prompt=scenario["ai_prompt"],
                    user_id=None,
                    is_custom=scenario["is_custom"],
                    call_target=scenario["call_target"],
                    call_purpose=scenario["call_purpose"],
                    created_at=datetime.utcnow(),
                )
            )

        await session.flush()
        await session.execute(
            text(
                "SELECT setval("
                "pg_get_serial_sequence('scenario', 'scenario_id'), "
                "(SELECT COALESCE(MAX(scenario_id), 1) FROM scenario)"
                ")"
            )
        )
        await session.commit()
