import asyncio
import json
import logging
import re
from contextlib import suppress

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status
from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.metrics import log_metric
from app.deps.auth import authenticate_ws_user as authenticate_ws
from app.deps.redis import get_redis
from app.schemas.frames import error_frame, scenario_info_frame
from app.services.pipeline import VoicePipeline
from app.services.session import authenticate_session, parse_script_level
from app.services.spring_client import SpringInternalClient

logger = logging.getLogger(__name__)
router = APIRouter()

_AUTH_FRAME_MAX_CHARS = 8192
_LOGGABLE_SESSION_ID = re.compile(r"[0-9A-Za-z_-]{1,64}")


async def _receive_auth_frame(ws: WebSocket, timeout: float) -> tuple[str | None, str]:
    """첫 프레임 {"type": "auth", "token": ...} 에서 토큰을 꺼냄. 못 꺼내면 닫음"""
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout)
    except TimeoutError:
        with suppress(Exception):
            await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="AUTH_TIMEOUT")
        return None, "timeout"
    if msg.get("type") == "websocket.disconnect":
        return None, "disconnected"

    text = msg.get("text")
    data = None
    if isinstance(text, str) and len(text) <= _AUTH_FRAME_MAX_CHARS:
        with suppress(ValueError):
            data = json.loads(text)
    token = data.get("token") if isinstance(data, dict) and data.get("type") == "auth" else None
    if not isinstance(token, str) or not token or not token.isascii():
        with suppress(Exception):
            await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="AUTH_REQUIRED")
        return None, "bad_frame"
    return token, "ok"


@router.websocket("/voice/{session_id}")
async def voice_stream(
        ws: WebSocket,
        session_id: str,
        token: str | None = Query(None, description="구버전 앱 전용. 새 클라는 첫 프레임 {type: auth, token}"),
        redis: Redis = Depends(get_redis),
) -> None:
    """웹소켓 연결: 인증/세션 검증 후 STT/LLM/TTS 파이프라인 가동."""
    await ws.accept()
    settings = get_settings()

    metric_sid = session_id if _LOGGABLE_SESSION_ID.fullmatch(session_id) else "invalid"
    method = "query" if token else "frame"
    if not token:
        token, result = await _receive_auth_frame(ws, settings.ws_auth_timeout_seconds)
        if token is None:
            log_metric("ws_auth", session_id=metric_sid, method=method, result=result)
            return

    try:
        user_id, role = await authenticate_ws(ws, token)
    except Exception:
        with suppress(Exception):
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        log_metric("ws_auth", session_id=metric_sid, method=method, result="rejected")
        return
    log_metric("ws_auth", session_id=metric_sid, method=method, result="ok")

    try:
        session = await authenticate_session(ws, redis, session_id, user_id)
    except RuntimeError:
        return

    logger.info(
        "웹소켓 연결됨",
        extra={"session_id": session_id, "user_id": user_id, "role": role},
    )

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
        await ws.send_json(
            scenario_info_frame(
                str(scenario.get("aiRole", "")),
                script_level=parse_script_level(session.get("scriptLevel")),
            )
        )
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
