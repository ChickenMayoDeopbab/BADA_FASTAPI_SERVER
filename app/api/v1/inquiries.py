from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.deps.inquiry import get_inquiry_service
from app.schemas.inquiry import InquiryCreateRequest, InquiryResponse
from app.services.inquiry import InquiryDeliveryError, InquiryService

router = APIRouter(prefix="/api/v1/inquiries", tags=["inquiries"])


@router.post("", response_model=InquiryResponse, summary="문의·피드백 전송")
async def send_inquiry(
    body: InquiryCreateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    service: InquiryService = Depends(get_inquiry_service),
) -> InquiryResponse:
    try:
        await service.send(db, user_id=user_id, request=body)
    except InquiryDeliveryError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="문의를 전송하지 못했습니다. 잠시 후 다시 시도해 주세요.",
        ) from error
    return InquiryResponse()
