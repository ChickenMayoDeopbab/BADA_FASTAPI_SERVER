from pydantic import BaseModel

from app.core.enums import CommunityReportReason, CommunityReportStatus, CommunityReportTargetType
from app.core.timeutil import KstDatetime


class AdminCommunityReportResponse(BaseModel):
    report_id: int
    reporter_user_id: int
    reporter_name: str | None = None
    reported_user_id: int
    reported_user_name: str | None = None
    target_type: CommunityReportTargetType
    target_id: int
    reason: CommunityReportReason
    content_snapshot: dict
    status: CommunityReportStatus
    created_at: KstDatetime
    due_at: KstDatetime
    is_overdue: bool
    resolved_at: KstDatetime | None = None
    resolved_by_user_id: int | None = None
    resolution_note: str | None = None


class AdminCommunityReportListResponse(BaseModel):
    reports: list[AdminCommunityReportResponse]
    page: int
    size: int
    total: int
    has_next: bool
