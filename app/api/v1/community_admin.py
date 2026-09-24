from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import CommunityReportStatus
from app.deps.admin import require_admin_user_id
from app.deps.db import get_db
from app.schemas.community_admin import AdminCommunityReportListResponse, AdminCommunityReportResponse
from app.services.community_report_admin import CommunityReportNotFoundError
from app.services.community_report_admin import get_report as svc_get_report
from app.services.community_report_admin import list_reports as svc_list_reports

router = APIRouter(prefix="/api/v1/admin/community", tags=["community-admin"])


@router.get("/reports", response_model=AdminCommunityReportListResponse, summary="관리자 신고 목록 조회")
async def list_reports(
    report_status: CommunityReportStatus = Query(CommunityReportStatus.PENDING, alias="status"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: int = Depends(require_admin_user_id),
) -> AdminCommunityReportListResponse:
    return await svc_list_reports(db, report_status=report_status, page=page, size=size)


@router.get("/reports/{report_id}", response_model=AdminCommunityReportResponse, summary="관리자 신고 상세 조회")
async def get_report(
    report_id: int,
    db: AsyncSession = Depends(get_db),
    _: int = Depends(require_admin_user_id),
) -> AdminCommunityReportResponse:
    try:
        return await svc_get_report(db, report_id)
    except CommunityReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="신고를 찾을 수 없습니다.") from exc
