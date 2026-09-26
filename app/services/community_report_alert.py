import logging

import httpx

from app.core.timeutil import as_kst
from app.schemas.community import CommunityReportResponse

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 5.0


class CommunityReportAlertService:
    def __init__(self, webhook_url: str | None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._webhook_url = webhook_url.strip() if webhook_url else None
        self._transport = transport

    async def notify_report_created(self, report: CommunityReportResponse) -> bool:
        if self._webhook_url is None:
            return False

        due_at = as_kst(report.due_at).strftime("%Y-%m-%d %H:%M:%S KST")
        text = (
            "새 커뮤니티 신고가 접수되었습니다.\n"
            f"신고 ID: {report.report_id}\n"
            f"대상: {report.target_type.value} #{report.target_id}\n"
            f"사유: {report.reason.value}\n"
            f"처리 기한: {due_at}\n"
            "관리자 신고 목록에서 24시간 이내 처리해 주세요."
        )
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS, transport=self._transport) as client:
                response = await client.post(self._webhook_url, params={"wait": "true"}, json={"content": text})
                response.raise_for_status()
            return True
        except Exception as error:
            # Discord webhook URL은 비밀 값이므로 예외 문구를 로그에 남기지 않는다.
            logger.error(
                "커뮤니티 신고 운영 알림 전송 실패: %s", type(error).__name__, extra={"report_id": report.report_id}
            )
            return False
