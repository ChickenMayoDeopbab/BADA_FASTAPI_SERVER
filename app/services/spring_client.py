import logging

import httpx

from app.core.config import Settings
from app.schemas.frames import EndReason

logger = logging.getLogger(__name__)

class SpringInternalClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.spring_boot_internal_url.rstrip("/")
        self._secret = settings.internal_secret

    async def notify_session_closed(
        self,
        session_id: str,
        *,
        reason: EndReason,
        transcript: list[dict],
        silence_total: float,
        shake_count: int = 0,
        good_segments: list[dict] | None = None,
        recording_url: str | None = None,
    ) -> None:
        url = f"{self._base_url}/internal/v1/sessions/{session_id}/closed"
        payload = {
            "reason": reason.value,
            "transcript": transcript,
            "silence_total": silence_total,
            "shake_count": shake_count,
            "good_segments": good_segments,
            "recording_url": recording_url,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"X-Internal-Secret": self._secret},
                )
                resp.raise_for_status()
            logger.info(
                "세션 종료 콜백 성공",
                extra={"session_id": session_id, "reason": reason.value},
            )
        except httpx.HTTPError as e:
            logger.warning(
                "세션 종료 콜백 실패: %s",
                e,
                extra={"session_id": session_id, "reason": reason.value},
            )
