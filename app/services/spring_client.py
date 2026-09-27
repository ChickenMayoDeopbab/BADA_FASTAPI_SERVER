import asyncio
import logging
from datetime import datetime

import httpx

from app.core.config import Settings
from app.core.enums import CommunityNotificationType, ReactionKind, UserModerationStatus
from app.schemas.frames import EndReason
from app.schemas.training_analysis import TrainingAnalysisPayload

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

    async def update_user_moderation_status(
        self,
        user_id: int,
        *,
        moderation_status: UserModerationStatus,
        suspended_until: datetime | None,
        reason: str | None,
        actor_user_id: int,
    ) -> bool:
        url = f"{self._base_url}/internal/v1/users/{user_id}/moderation-status"
        payload = {
            "status": moderation_status,
            "suspendedUntil": suspended_until.isoformat() if suspended_until is not None else None,
            "reason": reason,
            "actorUserId": actor_user_id,
        }
        last_error: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=5.0, transport=self._transport) as client:
                    response = await client.patch(
                        url, json=payload, headers={"X-Internal-Secret": self._secret}
                    )
                    response.raise_for_status()
                return True
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    logger.error("사용자 제재 콜백 400번대 에러, 재시도 안 함: %s", error)
                    return False
                last_error = error
            except httpx.HTTPError as error:
                last_error = error
            except Exception as error:
                logger.exception("사용자 제재 콜백 처리 중 예외")
                last_error = error
                break

            if attempt < _RETRY_ATTEMPTS - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))

        logger.error("사용자 제재 콜백 최종 실패(%d회): %s", _RETRY_ATTEMPTS, last_error)
        return False

    async def notify_session_closed(
        self,
        session_id: str,
        *,
        reason: EndReason,
        transcript: list[dict],
        silence_total: float,
        shake_count: int = 0,
        good_segments: list[dict] | None = None,
        recording_key: str | None = None,
        session_type: str | None = None,
        analysis: TrainingAnalysisPayload | None = None,
    ) -> bool:
        url = f"{self._base_url}/internal/v1/sessions/{session_id}/closed"
        payload = {
            "type": session_type,
            "reason": reason.value,
            "transcript": transcript,
            "silence_total": silence_total,
            "shake_count": shake_count,
            "good_segments": good_segments,
            "recording_key": recording_key,
            "analysis": (
                analysis.model_dump(mode="json")
                if analysis is not None
                else None
            ),
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
                return True
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    logger.error(
                        "세션 종료 콜백 400번대 에러, 재시도 안 함: %s",
                        e,
                        extra={"session_id": session_id, "reason": reason.value},
                    )
                    return False
                last_error = e
            except httpx.HTTPError as e:
                last_error = e
            except Exception:
                logger.exception(
                    "세션 종료 콜백 처리 중 예외",
                    extra={"session_id": session_id, "reason": reason.value},
                )
                return False

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
        return False

    async def notify_community_notification(
        self,
        *,
        notification_type: CommunityNotificationType,
        recipient_user_id: int,
        actor_user_id: int,
        post_id: int,
        comment_id: int | None = None,
        reaction_id: int | None = None,
        reaction_kind: ReactionKind | None = None,
    ) -> None:
        url = f"{self._base_url}/internal/v1/notifications/community"
        # Spring CommunityNotificationRequest가 Jackson 기본 camelCase 필드를 사용한다.
        payload = {
            "type": notification_type,
            "recipientUserId": recipient_user_id,
            "actorUserId": actor_user_id,
            "postId": post_id,
            "commentId": comment_id,
            "reactionId": reaction_id,
            "reactionKind": reaction_kind,
        }
        log_context = {
            "notification_type": notification_type,
            "recipient_user_id": recipient_user_id,
            "actor_user_id": actor_user_id,
            "post_id": post_id,
            "comment_id": comment_id,
            "reaction_id": reaction_id,
            "reaction_kind": reaction_kind,
        }

        last_error: httpx.HTTPError | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                async with httpx.AsyncClient(
                    timeout=5.0, transport=self._transport
                ) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={"X-Internal-Secret": self._secret},
                    )
                    response.raise_for_status()
                logger.info("커뮤니티 알림 콜백 성공", extra=log_context)
                return
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    logger.error(
                        "커뮤니티 알림 콜백 400번대 에러, 재시도 안 함: %s",
                        error,
                        extra=log_context,
                    )
                    return
                last_error = error
            except httpx.HTTPError as error:
                last_error = error

            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RETRY_BASE_DELAY_SECONDS * (2**attempt)
                logger.warning(
                    "커뮤니티 알림 콜백 실패(%d/%d), %.1fs 후 재시도: %s",
                    attempt + 1,
                    _RETRY_ATTEMPTS,
                    delay,
                    last_error,
                    extra=log_context,
                )
                await asyncio.sleep(delay)

        logger.error(
            "커뮤니티 알림 콜백 최종 실패(%d회): %s",
            _RETRY_ATTEMPTS,
            last_error,
            extra=log_context,
        )
