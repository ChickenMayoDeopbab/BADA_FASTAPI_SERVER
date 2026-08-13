from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import AsyncSessionLocal
from app.db.models import FeedbackORM

logger = logging.getLogger(__name__)


async def delete_feedback(db: AsyncSession, session_id: str) -> bool:
    """세션 피드백 행을 삭제"""
    result = await db.execute(
        delete(FeedbackORM).where(FeedbackORM.session_id == session_id)
    )
    await db.commit()
    return bool(result.rowcount)


async def save_feedback(
    *,
    session_id: str,
    user_id: int,
    scenario_id: int,
    shake_count: int,
    silence_total: float,
    good_segments: list[dict] | None,
    session_factory=AsyncSessionLocal,
) -> bool:
    """세션당 1행 저장. 이미 저장된 세션이면 조용히 건너뛴다."""
    highlights = json.dumps(good_segments or [], ensure_ascii=False)

    try:
        async with session_factory() as db:
            stmt = (
                pg_insert(FeedbackORM)
                .values(
                    session_id=session_id,
                    user_id=user_id,
                    scenario_id=scenario_id,
                    shake_count=shake_count,
                    silence_duration=round(silence_total),
                    highlights=highlights,
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                )
                # 재연결로 파이프라인이 두 번 떠도 중복 저장 안 되게
                .on_conflict_do_nothing(index_elements=["session_id"])
            )
            result = await db.execute(stmt)
            await db.commit()
    except Exception:
        logger.error(
            "피드백 저장 실패",
            exc_info=True,
            extra={"session_id": session_id, "scenario_id": scenario_id},
        )
        return False

    if not result.rowcount:
        logger.info(
            "피드백 이미 저장된 세션, 건너뜀",
            extra={"session_id": session_id},
        )
        return False

    logger.info(
        "피드백 저장 완료",
        extra={
            "session_id": session_id,
            "scenario_id": scenario_id,
            "shake_count": shake_count,
        },
    )
    return True
