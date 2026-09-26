from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import CommunityReportStatus
from app.deps.admin import require_admin_user_id
from app.deps.db import get_db
from app.deps.spring import get_spring_client
from app.schemas.community_admin import (
    AdminCommunityReportListResponse,
    AdminCommunityReportResolutionRequest,
    AdminCommunityReportResponse,
    AdminUserModerationRequest,
)
from app.services.community_report_admin import (
    CommunityReportNotFoundError,
    InvalidReportResolutionError,
    UserModerationFailedError,
)
from app.services.community_report_admin import get_report as svc_get_report
from app.services.community_report_admin import list_reports as svc_list_reports
from app.services.community_report_admin import resolve_report as svc_resolve_report
from app.services.community_report_admin import update_user_moderation_status as svc_update_user_status
from app.services.spring_client import SpringInternalClient

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


@router.post(
    "/reports/{report_id}/resolve", response_model=AdminCommunityReportResponse, summary="관리자 신고 처리"
)
async def resolve_report(
    report_id: int,
    body: AdminCommunityReportResolutionRequest,
    db: AsyncSession = Depends(get_db),
    admin_user_id: int = Depends(require_admin_user_id),
    spring: SpringInternalClient = Depends(get_spring_client),
) -> AdminCommunityReportResponse:
    try:
        return await svc_resolve_report(
            db, spring, report_id=report_id, admin_user_id=admin_user_id, request=body
        )
    except CommunityReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="신고를 찾을 수 없습니다.") from exc
    except InvalidReportResolutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="신고 처리 요청이 유효하지 않습니다."
        ) from exc
    except UserModerationFailedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="사용자 제재 처리에 실패했습니다."
        ) from exc


@router.patch(
    "/users/{user_id}/moderation-status",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="관리자 사용자 제재",
)
async def update_user_moderation_status(
    user_id: int,
    body: AdminUserModerationRequest,
    admin_user_id: int = Depends(require_admin_user_id),
    spring: SpringInternalClient = Depends(get_spring_client),
) -> None:
    try:
        await svc_update_user_status(spring, user_id=user_id, admin_user_id=admin_user_id, request=body)
    except InvalidReportResolutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="사용자 제재 요청이 유효하지 않습니다."
        ) from exc
    except UserModerationFailedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="사용자 제재 처리에 실패했습니다."
        ) from exc
