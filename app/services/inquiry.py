import logging

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.http import shared_ssl_context
from app.db.external import users_table
from app.schemas.inquiry import InquiryCreateRequest

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 5.0


class InquiryDeliveryError(Exception):
    """문의 웹훅이 설정되지 않았거나 전송에 실패한 경우."""


class InquiryService:
    def __init__(self, webhook_url: str | None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._webhook_url = webhook_url.strip() if webhook_url else None
        self._transport = transport

    async def send(self, db: AsyncSession, *, user_id: int, request: InquiryCreateRequest) -> None:
        if not self._webhook_url:
            raise InquiryDeliveryError

        email = (
            await db.execute(select(users_table.c.email).where(users_table.c.user_id == user_id))
        ).scalar_one()
        payload = {
            "allowed_mentions": {"parse": []},
            "embeds": [
                {
                    "title": f"[앱 문의] {request.title}",
                    "description": request.content,
                    "fields": [
                        {"name": "사용자 ID", "value": str(user_id), "inline": True},
                        {"name": "이메일", "value": email or "미등록", "inline": True},
                    ],
                }
            ],
        }
        try:
            async with httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT_SECONDS, transport=self._transport, verify=shared_ssl_context()
            ) as client:
                # 문의를 저장하지 않으므로 Discord의 메시지 생성 확인 후에만 성공 처리한다.
                response = await client.post(self._webhook_url, params={"wait": "true"}, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as error:
            # 예외 문구에는 비밀 값인 웹훅 URL이 포함될 수 있어 오류 종류만 기록한다.
            logger.error("문의 전송 실패: %s", type(error).__name__, extra={"user_id": user_id})
            raise InquiryDeliveryError from None
