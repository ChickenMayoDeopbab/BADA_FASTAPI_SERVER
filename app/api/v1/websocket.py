import logging
from contextlib import suppress

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status
from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.security import authenticate_ws
from app.deps.redis import get_redis
from app.schemas.frames import error_frame, scenario_info_frame
from app.services.pipeline import VoicePipeline
from app.services.session import authenticate_session
from app.services.spring_client import SpringInternalClient

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/voice/{session_id}")
async def voice_stream(
        ws: WebSocket,
        session_id: str,
        token: str = Query(..., description="JWT access token"),
        redis: Redis = Depends(get_redis),
) -> None:
    """웹소켓 연결: 인증/세션 검증 후 STT/LLM/TTS 파이프라인 가동."""
    await ws.accept()

    try:
        user_id, role = await authenticate_ws(ws, token)
    except Exception:
        # authenticate_ws가 예외 처리(ws.close) 해줌
        return

    try:
        session = await authenticate_session(ws, redis, session_id, user_id)
    except RuntimeError:
        return

    logger.info(
        "웹소켓 연결됨",
        extra={"session_id": session_id, "user_id": user_id, "role": role},
    )

    settings = get_settings()
    try:
        pipeline = VoicePipeline(
            ws,
            session_id,
            session,
            settings=settings,
            spring=SpringInternalClient(settings),
        )
    except Exception:
        logger.exception(
            "파이프라인 초기화 실패",
            extra={"session_id": session_id, "user_id": user_id},
        )
        with suppress(Exception):
            await ws.send_json(error_frame("PIPELINE_INIT_FAILED"))
        with suppress(Exception):
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    scenario = session.get("scenario") or {}
    if not isinstance(scenario, dict):
        scenario = {}
    try:
        await ws.send_json(scenario_info_frame(str(scenario.get("aiRole", ""))))
    except (WebSocketDisconnect, RuntimeError):
        return

    try:
        await pipeline.run()
    except WebSocketDisconnect:
        logger.info(
            "웹소켓 끊어짐",
            extra={"session_id": session_id, "user_id": user_id},
        )
    except Exception:
        logger.exception(
            "파이프라인 예상치 못한 에러",
            extra={"session_id": session_id, "user_id": user_id},
        )
        with suppress(Exception):
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
