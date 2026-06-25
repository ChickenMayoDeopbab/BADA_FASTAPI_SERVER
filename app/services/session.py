import json
import logging
from typing import Any

from fastapi import WebSocket, status
from redis.asyncio import Redis

from app.schemas.llm import AiPersonality, ScenarioTurn, TurnContext

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


def _parse_script(raw_script: Any) -> list[ScenarioTurn]:
    """세션의 리스트 딕셔너리를 ScenarioTurn 리스트로."""
    turns: list[ScenarioTurn] = []
    if not isinstance(raw_script, list):
        return turns
    for item in raw_script:
        if not isinstance(item, dict):
            continue
        try:
            turns.append(
                ScenarioTurn(
                    step=int(item["step"]),
                    ai_goal=str(item["aiGoal"]),
                    hint=str(item.get("hint", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("시나리오 턴 파싱 실패, 건너뜀: %r", item)
    return sorted(turns, key=lambda t: t.step)


def build_turn_context(
    session: dict[str, Any],
    *,
    current_step: int,
    history: list[dict],
    user_utterance: str,
) -> TurnContext:
    raw_personality = session.get("aiPersonality", AiPersonality.NORMAL.value)
    try:
        personality = AiPersonality(str(raw_personality).upper())
    except ValueError:
        logger.warning("알 수 없는 aiPersonality, NORMAL로 대체: %r", raw_personality)
        personality = AiPersonality.NORMAL

    scenario = session.get("scenario") or {}
    if not isinstance(scenario, dict):
        scenario = {}

    return TurnContext(
        personality=personality,
        scenario_title=str(scenario.get("title", "")),
        scenario_role=str(scenario.get("aiRole", "")),
        script=_parse_script(scenario.get("script")),
        current_step=current_step,
        history=history,
        user_utterance=user_utterance,
        scenario_prompt=str(scenario.get("aiPrompt", "")),
    )
