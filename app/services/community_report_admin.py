from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import (
    CommunityReportReason,
    CommunityReportResolutionAction,
    CommunityReportStatus,
    CommunityReportTargetType,
    UserModerationStatus,
)
from app.core.timeutil import ensure_utc, now_utc
from app.db.external import users_table
from app.db.models import CommunityReportORM, PostCommentORM, PostORM
from app.schemas.community_admin import (
    AdminCommunityReportListResponse,
    AdminCommunityReportResolutionRequest,
    AdminCommunityReportResponse,
    AdminUserModerationRequest,
)
from app.services.spring_client import SpringInternalClient


class CommunityReportNotFoundError(Exception):
    """존재하지 않는 신고."""


class InvalidReportResolutionError(Exception):
    """신고 처리 동작과 제재 기간이 맞지 않음."""


class UserModerationFailedError(Exception):
    """Spring 사용자 제재 호출 실패."""


def _to_response(
    row: CommunityReportORM, reporter_name: str | None, reported_user_name: str | None
) -> AdminCommunityReportResponse:
    status = CommunityReportStatus(row.status)
    return AdminCommunityReportResponse(
        report_id=row.report_id,
        reporter_user_id=row.reporter_user_id,
        reporter_name=reporter_name,
        reported_user_id=row.reported_user_id,
        reported_user_name=reported_user_name,
        target_type=CommunityReportTargetType(row.target_type),
        target_id=row.target_id,
        reason=CommunityReportReason(row.reason),
        content_snapshot=row.content_snapshot,
        status=status,
        created_at=row.created_at,
        due_at=row.due_at,
        is_overdue=status is CommunityReportStatus.PENDING and ensure_utc(row.due_at) < now_utc(),
        resolved_at=row.resolved_at,
        resolved_by_user_id=row.resolved_by_user_id,
        resolution_note=row.resolution_note,
    )


def _report_with_users():
    reporter = users_table.alias("reporter")
    reported = users_table.alias("reported")
    return (
        select(CommunityReportORM, reporter.c.name, reported.c.name)
        .outerjoin(reporter, reporter.c.user_id == CommunityReportORM.reporter_user_id)
        .outerjoin(reported, reported.c.user_id == CommunityReportORM.reported_user_id)
    )


async def list_reports(
    db: AsyncSession, *, report_status: CommunityReportStatus, page: int, size: int
) -> AdminCommunityReportListResponse:
    condition = CommunityReportORM.status == report_status.value
    total = await db.scalar(select(func.count()).select_from(CommunityReportORM).where(condition))
    stmt = (
        _report_with_users()
        .where(condition)
        .order_by(CommunityReportORM.due_at, CommunityReportORM.report_id)
        .offset((page - 1) * size)
        .limit(size + 1)
    )
    rows = (await db.execute(stmt)).all()
    has_next = len(rows) > size
    reports = [_to_response(*row) for row in rows[:size]]
    return AdminCommunityReportListResponse(
        reports=reports, page=page, size=size, total=total or 0, has_next=has_next
    )


async def get_report(db: AsyncSession, report_id: int) -> AdminCommunityReportResponse:
    row = (await db.execute(_report_with_users().where(CommunityReportORM.report_id == report_id))).first()
    if row is None:
        raise CommunityReportNotFoundError
    return _to_response(*row)


def _sanction_values(
    action: CommunityReportResolutionAction, suspended_until: datetime | None
) -> tuple[UserModerationStatus, datetime | None]:
    if action is CommunityReportResolutionAction.SUSPEND:
        if suspended_until is None or ensure_utc(suspended_until) <= now_utc():
            raise InvalidReportResolutionError
        return UserModerationStatus.SUSPENDED, suspended_until
    if action is CommunityReportResolutionAction.BAN:
        if suspended_until is not None:
            raise InvalidReportResolutionError
        return UserModerationStatus.BANNED, None
    raise InvalidReportResolutionError


async def _soft_delete_target(db: AsyncSession, report: CommunityReportORM, deleted_at: datetime) -> None:
    if report.target_type == CommunityReportTargetType.POST.value:
        post = await db.get(PostORM, report.target_id)
        if post is not None and post.deleted_at is None:
            post.deleted_at = deleted_at
        return

    comment = await db.get(PostCommentORM, report.target_id)
    if comment is None or comment.deleted_at is not None:
        return
    comment.deleted_at = deleted_at
    if comment.parent_comment_id is None:
        await db.execute(
            update(PostCommentORM)
            .where(PostCommentORM.parent_comment_id == comment.comment_id, PostCommentORM.deleted_at.is_(None))
            .values(deleted_at=deleted_at)
        )


async def resolve_report(
    db: AsyncSession,
    spring: SpringInternalClient,
    *,
    report_id: int,
    admin_user_id: int,
    request: AdminCommunityReportResolutionRequest,
) -> AdminCommunityReportResponse:
    stmt = select(CommunityReportORM).where(CommunityReportORM.report_id == report_id).with_for_update()
    report = (await db.execute(stmt)).scalar_one_or_none()
    if report is None:
        raise CommunityReportNotFoundError
    if report.status != CommunityReportStatus.PENDING.value:
        return await get_report(db, report_id)

    processed_at = now_utc()
    if request.action is CommunityReportResolutionAction.DISMISS:
        report.status = CommunityReportStatus.DISMISSED.value
    else:
        moderation_status, suspended_until = _sanction_values(request.action, request.suspended_until)
        await _soft_delete_target(db, report, processed_at)
        succeeded = await spring.update_user_moderation_status(
            report.reported_user_id,
            moderation_status=moderation_status,
            suspended_until=suspended_until,
            reason=request.note,
            actor_user_id=admin_user_id,
        )
        if not succeeded:
            await db.rollback()
            raise UserModerationFailedError
        report.status = CommunityReportStatus.RESOLVED.value

    report.resolved_at = processed_at
    report.resolved_by_user_id = admin_user_id
    report.resolution_note = request.note
    await db.commit()
    return await get_report(db, report_id)


async def update_user_moderation_status(
    spring: SpringInternalClient,
    *,
    user_id: int,
    admin_user_id: int,
    request: AdminUserModerationRequest,
) -> None:
    if request.status is UserModerationStatus.SUSPENDED:
        if request.suspended_until is None or ensure_utc(request.suspended_until) <= now_utc():
            raise InvalidReportResolutionError
    elif request.suspended_until is not None:
        raise InvalidReportResolutionError
    if request.status is not UserModerationStatus.ACTIVE and (request.reason is None or not request.reason.strip()):
        raise InvalidReportResolutionError

    succeeded = await spring.update_user_moderation_status(
        user_id,
        moderation_status=request.status,
        suspended_until=request.suspended_until,
        reason=request.reason.strip() if request.reason is not None else None,
        actor_user_id=admin_user_id,
    )
    if not succeeded:
        raise UserModerationFailedError
