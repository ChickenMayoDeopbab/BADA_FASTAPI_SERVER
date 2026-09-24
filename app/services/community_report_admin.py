from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import CommunityReportReason, CommunityReportStatus, CommunityReportTargetType
from app.core.timeutil import ensure_utc, now_utc
from app.db.external import users_table
from app.db.models import CommunityReportORM
from app.schemas.community_admin import AdminCommunityReportListResponse, AdminCommunityReportResponse


class CommunityReportNotFoundError(Exception):
    """존재하지 않는 신고."""


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
