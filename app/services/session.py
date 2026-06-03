import json
import logging
from typing import Any

from fastapi import WebSocket, status
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

SESSION_KEY_PREFIX = "session:"
OWNER_FIELD = "userId"

async def load_session(
        redis: Redis,
        session_id: str
) -> dict[str, Any] | None:
    """session: {session_id} 조회, 없으면 None"""
    raw = await redis.get(f"{SESSION_KEY_PREFIX}{session_id}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error(
            "세션 파싱 실패",
            extra={"session_id": session_id},
            exc_info=True,
        )
        return None

def is_session_owner(session: dict[str, Any], user_id: int) -> bool:
    """spring이 박은 userId(int)와 토큰 user_id(int) 일치 확인"""
    owner = session.get(OWNER_FIELD)
    if owner is None:
        logger.error(
            "세션에 user 필드 없음",
            extra={"token_user_id": user_id},
        )
        return False
    try:
        return int(owner) == user_id
    except (TypeError, ValueError):
        logger.error(
            "세션 userId 타입 불일치",
            extra={"token_user_id": user_id, "owner_type": type(owner).__name__}
        )
        return False

async def authenticate_session(
        ws: WebSocket,
        redis: Redis,
        session_id: str,
        user_id: int,
) -> dict[str, Any]:
    """세션 조회랑 소유자 검증"""
    session = await load_session(redis, session_id)

    if session is None:
        logger.warning(
            "세션 없음",
            extra={"session_id": session_id, "user_id": user_id},
        )
        await ws.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="SESSION_NOT_FOUND"
        )
        raise RuntimeError("SESSION_NOT_FOUND")

    if not is_session_owner(session, user_id):
        logger.warning(
            "세션 소유자 불일치",
            extra={"session_id": session_id, "user_id": user_id}
        )
        await ws.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="SESSION_FORBIDDEN"
        )
        raise RuntimeError("SESSION_FORBIDDEN")

    return session