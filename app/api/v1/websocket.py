import logging
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from fastapi import Depends
from redis.asyncio import Redis

from app.deps.redis import get_redis
from app.services.session import authenticate_session
from app.core.security import authenticate_ws

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/voice/{session_id}")
async def voice_stream(
        ws: WebSocket,
        session_id: str,
        token: str = Query(..., description="JWT access token"),
        redis: Redis = Depends(get_redis),
) -> None:
    """웹소켓 연결"""
    await ws.accept()

    try:
        user_id, role = await authenticate_ws(ws, token)
    except Exception as e:
        # authenticate_ws가 예외 처리 해줌
        return

    try:
        session = await authenticate_session(ws, redis, session_id, user_id)
    except RuntimeError:
        return

    logging.info(
        "웹소켓 연결됨",
        extra={"session_id": session_id, "user_id": user_id, "role": role}
    )

    # TODO: Redis에서 session:{session_id} 조회
    #  -> STT/LLM/TTS 파이프라인 가동
    #  -> 실전 워밍업이면 30초 타이머

    try:
        while True:
            msg = await ws.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                audio_chunk: bytes = msg["bytes"]
                _ = audio_chunk
                continue

            if "text" in msg and msg["text"] is not None:
                # TODO: JSON 파싱 후 type별 분기
                # {"type":"end"} -> break
                # {"type":"mute","muted":true} -> 무시 또는 STT 일시정지
                # {"type":"ping"} -> {"type":"pong"} 응답
                pass
    except WebSocketDisconnect:
        logger.info(
            "웹소켓 끊어짐",
            extra={"session_id": session_id, "user_id": user_id},
        )
    except Exception:
        logger.exception(
            "웹소켓 예상치 못한 에러",
            extra={"session_id": session_id, "user_id": user_id},
        )
        await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        # TODO: Spring에 비정상 종료 콜백