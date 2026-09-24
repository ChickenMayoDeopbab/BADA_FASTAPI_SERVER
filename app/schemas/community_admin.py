from pydantic import AwareDatetime, BaseModel, Field

from app.core.enums import (
    CommunityReportReason,
    CommunityReportResolutionAction,
    CommunityReportStatus,
    CommunityReportTargetType,
    UserModerationStatus,
)
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


class AdminCommunityReportResolutionRequest(BaseModel):
    action: CommunityReportResolutionAction
    note: str = Field(min_length=1, max_length=500)
    suspended_until: AwareDatetime | None = None


class AdminUserModerationRequest(BaseModel):
    status: UserModerationStatus
    suspended_until: AwareDatetime | None = None
    reason: str | None = Field(default=None, max_length=500)
