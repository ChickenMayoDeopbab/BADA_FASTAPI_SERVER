from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.timeutil import now_utc
from app.db.base import AsyncSessionLocal
from app.db.models import CallBiosignalORM

STATUS_AWAITING_DATA = "AWAITING_DATA"


async def save_call_timeline(
    *,
    session_id: str,
    user_id: int,
    scenario_id: int | None,
    session_type: str | None,
    difficulty: str | None,
    timeline: dict,
    session_factory=AsyncSessionLocal,
) -> bool:
    """통화 사건표를 생체신호 업로드 대기 행으로 저장"""
    async with session_factory() as db:
        stmt = (
            pg_insert(CallBiosignalORM)
            .values(
                session_id=session_id,
                user_id=user_id,
                scenario_id=scenario_id,
                session_type=session_type,
                difficulty=difficulty,
                status=STATUS_AWAITING_DATA,
                timeline=timeline,
                created_at=now_utc(),
            )
            .on_conflict_do_nothing(index_elements=["session_id"])
        )
        result = await db.execute(stmt)
        await db.commit()
    return bool(result.rowcount)
