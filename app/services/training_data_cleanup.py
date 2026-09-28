from __future__ import annotations

import asyncio
import logging

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.enums import AttachmentKind
from app.db.external import training_records_table
from app.db.models import FeedbackORM, PostAttachmentORM, VoiceTremorMetricORM
from app.services.morphed_recording import build_storage, morphed_key
from app.services.recording_storage import RecordingStorageService

logger = logging.getLogger(__name__)


async def delete_training_data(
    db: AsyncSession,
    session_id: str,
    *,
    storage: RecordingStorageService | None = None,
) -> None:
    """훈련 기록에 종속된 FastAPI 데이터와 공개용 음성 변조본을 삭제한다."""
    record_stmt = select(
        training_records_table.c.record_id,
        training_records_table.c.recording_key,
    ).where(training_records_table.c.session_id == session_id)
    record = (await db.execute(record_stmt)).first()

    try:
        if record is not None and record.recording_key:
            recording_storage = storage or build_storage(get_settings())
            await asyncio.to_thread(
                recording_storage.delete,
                morphed_key(record.recording_key),
            )

        await db.execute(delete(FeedbackORM).where(FeedbackORM.session_id == session_id))
        await db.execute(
            delete(VoiceTremorMetricORM).where(
                VoiceTremorMetricORM.session_id == session_id
            )
        )
        if record is not None:
            await db.execute(
                delete(PostAttachmentORM).where(
                    PostAttachmentORM.kind == AttachmentKind.TRAINING_RECORD.value,
                    PostAttachmentORM.ref_id == record.record_id,
                )
            )
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("훈련 데이터 삭제 실패", extra={"session_id": session_id})
        raise
