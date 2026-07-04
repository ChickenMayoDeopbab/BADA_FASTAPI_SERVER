import asyncio
import logging

import httpx

from app.core.config import Settings
from app.schemas.frames import EndReason

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECONDS = 0.5


class SpringInternalClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = settings.spring_boot_internal_url.rstrip("/")
        self._secret = settings.internal_secret
        self._transport = transport

    async def notify_session_closed(
        self,
        session_id: str,
        *,
        reason: EndReason,
        transcript: list[dict],
        silence_total: float,
        shake_count: int = 0,
        good_segments: list[dict] | None = None,
    ) -> None:
        url = f"{self._base_url}/internal/v1/sessions/{session_id}/closed"
        payload = {
            "reason": reason.value,
            "transcript": transcript,
            "silence_total": silence_total,
            "shake_count": shake_count,
            "good_segments": good_segments,
        }

        last_error: httpx.HTTPError | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                async with httpx.AsyncClient(
                    timeout=5.0, transport=self._transport
                ) as client:
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
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    logger.error(
                        "세션 종료 콜백 400번대 에러, 재시도 안 함: %s",
                        e,
                        extra={"session_id": session_id, "reason": reason.value},
                    )
                    return
                last_error = e
            except httpx.HTTPError as e:
                last_error = e

            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RETRY_BASE_DELAY_SECONDS * (2**attempt)
                logger.warning(
                    "세션 종료 콜백 실패(%d/%d), %.1fs 후 재시도: %s",
                    attempt + 1,
                    _RETRY_ATTEMPTS,
                    delay,
                    last_error,
                    extra={"session_id": session_id, "reason": reason.value},
                )
                await asyncio.sleep(delay)

        logger.error(
            "세션 종료 콜백 최종 실패(%d회) — transcript/피드백 유실: %s",
            _RETRY_ATTEMPTS,
            last_error,
            extra={"session_id": session_id, "reason": reason.value},
        )
