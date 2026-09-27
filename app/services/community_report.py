from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import CommunityReportReason, CommunityReportStatus, CommunityReportTargetType
from app.core.timeutil import now_utc
from app.db.models import CommunityReportORM, PostAttachmentORM, PostCommentORM, PostORM
from app.schemas.community import CommunityReportResponse

_REPORT_REVIEW_WINDOW = timedelta(hours=24)


class ReportTargetNotFoundError(Exception):
    """없거나 삭제된 신고 대상."""


class OwnContentReportError(Exception):
    """본인 콘텐츠 신고 시도."""


class DuplicateReportError(Exception):
    """동일 사용자가 이미 신고한 대상."""


async def _post_snapshot(db: AsyncSession, post_id: int) -> tuple[int, dict]:
    post = await db.get(PostORM, post_id)
    if post is None or post.deleted_at is not None:
        raise ReportTargetNotFoundError

    stmt = (
        select(PostAttachmentORM.kind, PostAttachmentORM.ref_id)
        .where(PostAttachmentORM.post_id == post_id)
        .order_by(PostAttachmentORM.attachment_id)
    )
    attachments = [{"kind": kind, "ref_id": ref_id} for kind, ref_id in (await db.execute(stmt)).all()]
    return post.user_id, {"title": post.title, "content": post.content, "attachments": attachments}


async def _comment_snapshot(db: AsyncSession, comment_id: int) -> tuple[int, dict]:
    comment = await db.get(PostCommentORM, comment_id)
    if comment is None or comment.deleted_at is not None:
        raise ReportTargetNotFoundError

    post = await db.get(PostORM, comment.post_id)
    if post is None or post.deleted_at is not None:
        raise ReportTargetNotFoundError

    return comment.user_id, {"content": comment.content, "post_id": comment.post_id}


async def _target_snapshot(
    db: AsyncSession, target_type: CommunityReportTargetType, target_id: int
) -> tuple[int, dict]:
    if target_type is CommunityReportTargetType.POST:
        return await _post_snapshot(db, target_id)
    return await _comment_snapshot(db, target_id)


def _to_response(row: CommunityReportORM) -> CommunityReportResponse:
    return CommunityReportResponse(
        report_id=row.report_id,
        target_type=CommunityReportTargetType(row.target_type),
        target_id=row.target_id,
        reason=CommunityReportReason(row.reason),
        status=CommunityReportStatus(row.status),
        created_at=row.created_at,
        due_at=row.due_at,
    )


async def create_report(
    db: AsyncSession,
    *,
    reporter_user_id: int,
    target_type: CommunityReportTargetType,
    target_id: int,
    reason: CommunityReportReason,
) -> CommunityReportResponse:
    reported_user_id, snapshot = await _target_snapshot(db, target_type, target_id)
    if reported_user_id == reporter_user_id:
        raise OwnContentReportError

    created_at = now_utc()
    row = CommunityReportORM(
        reporter_user_id=reporter_user_id,
        reported_user_id=reported_user_id,
        target_type=target_type.value,
        target_id=target_id,
        reason=reason.value,
        content_snapshot=snapshot,
        status=CommunityReportStatus.PENDING.value,
        created_at=created_at,
        due_at=created_at + _REPORT_REVIEW_WINDOW,
    )
    db.add(row)
    try:
        await db.commit()
    except IntegrityError as error:
        await db.rollback()
        stmt = select(CommunityReportORM.report_id).where(
            CommunityReportORM.reporter_user_id == reporter_user_id,
            CommunityReportORM.target_type == target_type.value,
            CommunityReportORM.target_id == target_id,
        )
        if (await db.execute(stmt)).scalar_one_or_none() is not None:
            raise DuplicateReportError from error
        raise

    return _to_response(row)
