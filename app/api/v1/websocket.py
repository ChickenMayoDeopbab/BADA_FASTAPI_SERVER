from fastapi import APIRouter, Query, WebSocket

router = APIRouter()

@router.websocket("/voice/{session_id}")
async def voice_stream(
        ws: WebSocket,
        session_id: str,
        token: str = Query(...)
) -> None:
    # TODO: JWT 검증
    #  -> Redis에서 session:{session_id} 조회
    #  -> STT/LLM/TTS 파이프라인 가동
    #  -> 실전 워밍업이면 30초 타이머
    await ws.accept()
    await ws.close()