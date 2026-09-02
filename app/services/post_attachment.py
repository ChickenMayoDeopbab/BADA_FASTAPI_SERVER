from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.enums import AttachmentKind
from app.core.timeutil import now_utc
from app.db.external import training_records_table
from app.db.models import PostAttachmentORM, ScenarioORM
from app.schemas.community import (
    AttachedScenario,
    AttachedTrainingRecord,
    AttachmentRequest,
    PostAttachment,
)
from app.services.morphed_recording import build_storage, morphed_url
from app.services.recording_storage import RecordingStorageService
from app.workers.voice_morph_worker import schedule_morph

logger = logging.getLogger(__name__)

_IMAGE_URL_TTL_SEC = 3600


class AttachmentInvalidError(Exception):
    """붙이면 안되는거 붙임"""


async def commit_translating_conflicts(db: AsyncSession) -> None:
    """디비 에러 시 에러 반환, 아니면 커밋"""
    try:
        await db.commit()
    except IntegrityError as e:
        await db.rollback()
        if "post_attachment" in str(e.orig):
            raise AttachmentInvalidError("같은 종류는 하나만 첨부할 수 있습니다.") from e
        raise


async def _check_scenario(db: AsyncSession, ref_id: int, user_id: int) -> None:
    """시나리오 예외 분기"""
    row = await db.get(ScenarioORM, ref_id)
    if row is None or row.deleted_at is not None:
        raise AttachmentInvalidError("없거나 삭제된 시나리오입니다.")
    if not row.is_custom:
        raise AttachmentInvalidError("기본 시나리오는 첨부할 수 없습니다.")
    if row.is_warmup:
        raise AttachmentInvalidError("워밍업 시나리오는 첨부할 수 없습니다.")
    if row.user_id != user_id:
        raise AttachmentInvalidError("내가 만든 시나리오만 첨부할 수 있습니다.")


async def _check_training_record(db: AsyncSession, ref_id: int, user_id: int) -> None:
    stmt = select(training_records_table.c.user_id).where(
        training_records_table.c.record_id == ref_id
    )
    owner = (await db.execute(stmt)).scalar_one_or_none()
    if owner is None:
        raise AttachmentInvalidError("없는 훈련 기록입니다.")
    if owner != user_id:
        raise AttachmentInvalidError("내 훈련 기록만 첨부할 수 있습니다.")


_CHECKS = {
    AttachmentKind.SCENARIO: _check_scenario,
    AttachmentKind.TRAINING_RECORD: _check_training_record,
}


async def build_rows(
    db: AsyncSession, post_id: int, requested: list[AttachmentRequest], user_id: int
) -> list[PostAttachmentORM]:
    """검증 통과한것만 행 만들기"""
    now = now_utc()
    rows: list[PostAttachmentORM] = []
    for item in requested:
        await _CHECKS[item.kind](db, item.ref_id, user_id)
        rows.append(
            PostAttachmentORM(
                post_id=post_id, kind=item.kind.value, ref_id=item.ref_id, created_at=now
            )
        )
    return rows


async def current_pairs(db: AsyncSession, post_id: int) -> set[tuple[str, int]]:
    """수정 요청이 실제로 다른지 볼 때 확인용"""
    stmt = select(PostAttachmentORM.kind, PostAttachmentORM.ref_id).where(
        PostAttachmentORM.post_id == post_id
    )
    return {(kind, ref_id) for kind, ref_id in (await db.execute(stmt)).all()}


async def kinds_by_post(db: AsyncSession, post_ids: list[int]) -> dict[int, list[str]]:
    """목록용 정렬"""
    if not post_ids:
        return {}

    stmt = (
        select(PostAttachmentORM.post_id, PostAttachmentORM.kind)
        .where(PostAttachmentORM.post_id.in_(post_ids))
        .order_by(PostAttachmentORM.attachment_id)
    )
    found: dict[int, list[str]] = {}
    for post_id, kind in (await db.execute(stmt)).all():
        found.setdefault(post_id, []).append(kind)
    return found


def _image_url(storage: RecordingStorageService, image_key: str) -> str | None:
    """썸네일 재생 URL. 서명이 실패해도 첨부 전체를 날리지는 않는다."""
    try:
        return storage.presigned_url(image_key, expires_in=_IMAGE_URL_TTL_SEC)
    except Exception as e:
        logger.warning(
            "썸네일 URL 서명 실패, 이미지 없이 응답: %s: %s",
            type(e).__name__,
            e,
            extra={"s3_key": image_key},
        )
        return None


async def _scenario_block(
    db: AsyncSession, ref_id: int, viewer_id: int
) -> AttachedScenario | None:
    row = await db.get(ScenarioORM, ref_id)
    if row is None:
        return None

    # 컬럼에 담긴 건 URL 이 아니라 S3 키라 매번 새로 서명해서 내려준다.
    image = None
    if row.scenario_image:
        image = await asyncio.to_thread(
            _image_url, build_storage(get_settings()), row.scenario_image
        )

    return AttachedScenario(
        title=row.title,
        content=row.content,
        category=row.category,
        scenario_image=image,
        is_available=row.deleted_at is None,
        is_mine=row.user_id == viewer_id,
    )


async def recording_key_of(db: AsyncSession, record_id: int) -> str | None:
    """변조 예약에만 씀"""
    stmt = select(training_records_table.c.recording_key).where(
        training_records_table.c.record_id == record_id
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _training_record_block(
    db: AsyncSession, ref_id: int, viewer_id: int
) -> AttachedTrainingRecord | None:
    t = training_records_table.c
    stmt = select(
        t.scenario_name, t.session_type, t.started_at,
        t.duration_seconds, t.anxiety_score, t.recording_key,
    ).where(t.record_id == ref_id)
    row = (await db.execute(stmt)).first()
    if row is None:
        return AttachedTrainingRecord(is_available=False)

    # 변조본이 준비됐을 때만 URL 을 준다. 원본 키는 절대 싣지 않는다.
    if not row.recording_key:
        url, status = None, "none"
    else:
        url = await asyncio.to_thread(
            morphed_url, build_storage(get_settings()), row.recording_key
        )
        status = "ready" if url else "processing"

    return AttachedTrainingRecord(
        scenario_name=row.scenario_name,
        session_type=row.session_type,
        started_at=row.started_at,
        duration_seconds=row.duration_seconds,
        anxiety_score=row.anxiety_score,
        audio_url=url,
        audio_status=status,
    )


async def detail_for_post(
    db: AsyncSession, post_id: int, viewer_id: int
) -> list[PostAttachment]:
    stmt = (
        select(PostAttachmentORM)
        .where(PostAttachmentORM.post_id == post_id)
        .order_by(PostAttachmentORM.attachment_id)
    )
    rows = (await db.execute(stmt)).scalars().all()

    out: list[PostAttachment] = []
    for row in rows:
        kind = AttachmentKind(row.kind)
        item = PostAttachment(kind=kind, ref_id=row.ref_id)
        if kind is AttachmentKind.SCENARIO:
            item.scenario = await _scenario_block(db, row.ref_id, viewer_id)
        else:
            item.training_record = await _training_record_block(db, row.ref_id, viewer_id)
        out.append(item)
    return out


async def schedule_morphs(db: AsyncSession, rows: list[PostAttachmentORM]) -> None:
    """훈련 기록에 대한 변조본 생성을 백그라운드로 올림"""
    settings = get_settings()
    for row in rows:
        if row.kind != AttachmentKind.TRAINING_RECORD.value:
            continue
        schedule_morph(
            settings=settings, recording_key=await recording_key_of(db, row.ref_id)
        )
