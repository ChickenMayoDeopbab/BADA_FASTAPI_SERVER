from __future__ import annotations

import asyncio
import logging

from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.enums import AttachmentKind
from app.db.external import files_table, training_records_table
from app.db.models import (
    FeedbackORM,
    FileORM,
    PostAttachmentORM,
    PostCommentORM,
    PostORM,
    PostReactionORM,
    ScenarioORM,
    VoiceTremorMetricORM,
)
from app.services.morphed_recording import build_storage, morphed_key
from app.services.recording_storage import RecordingStorageService

logger = logging.getLogger(__name__)


async def delete_user_data(
    db: AsyncSession,
    user_id: int,
    *,
    storage: RecordingStorageService | None = None,
) -> None:
    """회원 탈퇴 대상 데이터만 삭제하고 법정·정책상 보존 대상은 유지한다."""
    training_rows = (
        await db.execute(
            select(
                training_records_table.c.record_id,
                training_records_table.c.session_id,
                training_records_table.c.recording_key,
            ).where(training_records_table.c.user_id == user_id)
        )
    ).all()
    scenario_rows = (
        await db.execute(
            select(
                ScenarioORM.scenario_id,
                ScenarioORM.scenario_image,
                ScenarioORM.example_audio_url,
            ).where(ScenarioORM.user_id == user_id)
        )
    ).all()
    post_ids = list(
        (
            await db.execute(select(PostORM.post_id).where(PostORM.user_id == user_id))
        ).scalars()
    )
    comment_ids = list(
        (
            await db.execute(
                select(PostCommentORM.comment_id).where(PostCommentORM.user_id == user_id)
            )
        ).scalars()
    )
    file_ids = list(
        (
            await db.execute(
                select(files_table.c.file_id).where(files_table.c.user_id == user_id)
            )
        ).scalars()
    )

    await _delete_owned_s3_objects(training_rows, scenario_rows, storage)

    training_ids = [row.record_id for row in training_rows]
    session_ids = [row.session_id for row in training_rows]
    scenario_ids = [row.scenario_id for row in scenario_rows]
    scenario_keys = {
        key
        for row in scenario_rows
        for key in (row.scenario_image, row.example_audio_url)
        if key
    }

    try:
        if comment_ids:
            await db.execute(
                update(PostCommentORM)
                .where(PostCommentORM.parent_comment_id.in_(comment_ids))
                .values(parent_comment_id=None)
            )

        attachment_conditions = []
        if post_ids:
            attachment_conditions.append(PostAttachmentORM.post_id.in_(post_ids))
        if training_ids:
            attachment_conditions.append(
                (PostAttachmentORM.kind == AttachmentKind.TRAINING_RECORD.value)
                & PostAttachmentORM.ref_id.in_(training_ids)
            )
        if scenario_ids:
            attachment_conditions.append(
                (PostAttachmentORM.kind == AttachmentKind.SCENARIO.value)
                & PostAttachmentORM.ref_id.in_(scenario_ids)
            )
        if file_ids:
            attachment_conditions.append(
                (PostAttachmentORM.kind == AttachmentKind.FILE.value)
                & PostAttachmentORM.ref_id.in_(file_ids)
            )
        if attachment_conditions:
            await db.execute(
                delete(PostAttachmentORM).where(or_(*attachment_conditions))
            )

        reaction_conditions = [PostReactionORM.user_id == user_id]
        if post_ids:
            reaction_conditions.append(PostReactionORM.post_id.in_(post_ids))
        await db.execute(delete(PostReactionORM).where(or_(*reaction_conditions)))

        comment_conditions = [PostCommentORM.user_id == user_id]
        if post_ids:
            comment_conditions.append(PostCommentORM.post_id.in_(post_ids))
        await db.execute(delete(PostCommentORM).where(or_(*comment_conditions)))

        if post_ids:
            await db.execute(delete(PostORM).where(PostORM.post_id.in_(post_ids)))

        await db.execute(delete(FeedbackORM).where(FeedbackORM.user_id == user_id))
        if session_ids:
            await db.execute(
                delete(VoiceTremorMetricORM).where(
                    VoiceTremorMetricORM.session_id.in_(session_ids)
                )
            )

        if scenario_keys:
            await db.execute(delete(FileORM).where(FileORM.s3_key.in_(scenario_keys)))
        if scenario_ids:
            await db.execute(
                delete(ScenarioORM).where(ScenarioORM.scenario_id.in_(scenario_ids))
            )

        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("회원 데이터 삭제 실패", extra={"user_id": user_id})
        raise


async def _delete_owned_s3_objects(
    training_rows,
    scenario_rows,
    storage: RecordingStorageService | None,
) -> None:
    keys = {
        morphed_key(row.recording_key)
        for row in training_rows
        if row.recording_key
    }
    keys.update(
        key
        for row in scenario_rows
        for key in (row.scenario_image, row.example_audio_url)
        if key
    )
    if not keys:
        return

    recording_storage = storage or build_storage(get_settings())
    for key in sorted(keys):
        await asyncio.to_thread(recording_storage.delete, key)
