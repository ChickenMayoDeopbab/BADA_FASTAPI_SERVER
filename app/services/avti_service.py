from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.base import AsyncSessionLocal
from app.db.models import VoiceTremorMetricORM
from app.services.avti import SCRIPT_VERSION, AvtiPart

logger = logging.getLogger(__name__)


async def save_avti_metrics(
    *,
    session_id: str,
    parts: list[AvtiPart],
    session_factory=AsyncSessionLocal,
) -> int:
    """파트별 측정을 한 번에 저장. 재분석으로 두 번 들어와도 중복되지 않는다."""
    if not parts:
        return 0

    now = datetime.now(UTC).replace(tzinfo=None)
    rows = [
        {
            "session_id": session_id,
            "part_index": part.part_index,
            "start_sec": part.start_sec,
            "end_sec": part.end_sec,
            "status": str(part.status),
            "avti": part.avti,
            "ftrcip": part.ftrcip,
            "atri": part.atri,
            "fcohnr": part.fcohnr,
            "fcom": part.fcom,
            "script_version": SCRIPT_VERSION,
            "created_at": now,
        }
        for part in parts
    ]

    try:
        async with session_factory() as db:
            stmt = pg_insert(VoiceTremorMetricORM).values(rows).on_conflict_do_nothing(
                index_elements=["session_id", "part_index"]
            )
            result = await db.execute(stmt)
            await db.commit()
    except Exception:
        logger.error(
            "AVTI 저장 실패",
            exc_info=True,
            extra={"session_id": session_id, "parts": len(rows)},
        )
        return 0

    saved = result.rowcount or 0
    logger.info(
        "AVTI 저장 완료",
        extra={
            "session_id": session_id,
            "parts": len(rows),
            "saved": saved,
            "ok": sum(1 for p in parts if p.avti is not None),
        },
    )
    return saved
