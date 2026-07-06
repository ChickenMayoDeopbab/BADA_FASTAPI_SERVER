import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import WebSocketException

from app.core.config import Settings
from app.schemas.llm import AiEmotion

logger = logging.getLogger(__name__)

_EMOTION_VOICE_OVERRIDES: dict[AiEmotion, dict] = {
    AiEmotion.NEUTRAL:    {},
    AiEmotion.FRIENDLY:   {"stability": 0.40, "style": 0.30, "speed": 1.03},
    AiEmotion.ANNOYED:    {"stability": 0.45, "style": 0.20, "speed": 1.05},
    AiEmotion.ANGRY:      {"stability": 0.30, "style": 0.50, "speed": 1.08},
    AiEmotion.APOLOGETIC: {"stability": 0.60, "style": 0.10, "speed": 0.96},
}

_SENTENCE_BOUNDARIES = ".?!…\n"
_CLOSING_AFTER_BOUNDARY = "\"'」』)"


class _SentenceBuffer:
    """LLM이 준거 문장 단위로 짜름"""

    def __init__(self, max_chars: int = 80) -> None:
        self._max_chars = max_chars
        self._buf = ""

    def feed(self, chunk: str) -> str | None:
        self._buf += chunk
        boundary = self._last_boundary()
        if boundary is not None:
            return self._cut(boundary + 1)
        if len(self._buf) > self._max_chars:
            ws = self._buf.rfind(" ")
            return self._cut(ws if ws > 0 else len(self._buf))
        return None

    def flush(self) -> str | None:
        emit = self._buf.strip()
        self._buf = ""
        return emit + " " if emit else None

    def _last_boundary(self) -> int | None:
        for i in range(len(self._buf) - 1, -1, -1):
            if self._buf[i] not in _SENTENCE_BOUNDARIES:
                continue
            if not self._is_sentence_end(i):
                continue
            idx = i
            while idx + 1 < len(self._buf) and self._buf[idx + 1] in _CLOSING_AFTER_BOUNDARY:
                idx += 1
            return idx
        return None

    def _is_sentence_end(self, idx: int) -> bool:
        if self._buf[idx] != "." or idx == 0 or not self._buf[idx - 1].isdigit():
            return True
        if idx + 1 == len(self._buf):
            return False
        return not self._buf[idx + 1].isdigit()

    def _cut(self, n: int) -> str | None:
        emit = self._buf[:n].rstrip()
        self._buf = self._buf[n:].lstrip()
        return emit + " " if emit else None


class TTSSession:
    """턴 1번마다 ElevenLabs ws 연결"""

    def __init__(self, ws: ClientConnection, voice_settings: dict[str, object]) -> None:
        self._ws = ws
        self._base_voice_settings = voice_settings
        self._inited = False
        self._closed = False

    def _voice_settings_for(self, emotion: AiEmotion) -> dict:
        settings = dict(self._base_voice_settings)
        settings.update(_EMOTION_VOICE_OVERRIDES.get(emotion, {}))
        return settings

    async def begin(self, emotion: AiEmotion = AiEmotion.NEUTRAL) -> None:
        if self._inited:
            return
        await self._ws.send(
            json.dumps({"text": " ", "voice_settings": self._voice_settings_for(emotion)})
        )
        self._inited = True

    async def _send_text(self, text_source: AsyncIterator[str]) -> None:
        buf = _SentenceBuffer()
        async for chunk in text_source:
            if not chunk:
                continue
            payload = buf.feed(chunk)
            if payload:
                await self._ws.send(json.dumps({"text": payload}))
        rest = buf.flush()
        if rest:
            await self._ws.send(json.dumps({"text": rest}))
        await self._ws.send(json.dumps({"text": ""}))

    async def stream(self, text_source: AsyncIterator[str]) -> AsyncIterator[bytes]:
        if not self._inited:
            raise RuntimeError("TTSSession.begin()을 먼저 호출해야 합니다.")

        sender = asyncio.create_task(self._send_text(text_source))
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                audio_b64 = msg.get("audio")
                if audio_b64:
                    yield base64.b64decode(audio_b64)
                if msg.get("isFinal"):
                    break
        finally:
            if not sender.done():
                sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.close()
        except Exception:
            logger.debug("TTS ws close 중 무시된 예외", exc_info=True)


class ElevenLabsTTSClient:
    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.elevenlabs_api_key
        self._voice_id = settings.elevenlabs_voice_id
        self._model = settings.elevenlabs_model
        self._output_format = settings.elevenlabs_output_format
        self._language = settings.elevenlabs_language_code
        self._auto_mode = settings.elevenlabs_auto_mode
        self._ws_host = settings.elevenlabs_ws_host
        self._text_normalization = settings.elevenlabs_apply_text_normalization
        self._voice_settings = {
            "stability": settings.elevenlabs_stability,
            "similarity_boost": settings.elevenlabs_similarity_boost,
            "style": settings.elevenlabs_style,
            "use_speaker_boost": settings.elevenlabs_speaker_boost,
            "speed": settings.elevenlabs_speed,
        }

    def _build_uri(self, voice_id: str | None = None) -> str:
        vid = voice_id or self._voice_id
        return (
            f"{self._ws_host}/v1/text-to-speech/{vid}/stream-input"
            f"?model_id={self._model}"
            f"&output_format={self._output_format}"
            f"&language_code={self._language}"
            f"&auto_mode={'true' if self._auto_mode else 'false'}"
            f"&apply_text_normalization={self._text_normalization}"
        )

    async def open(self, voice_id: str | None = None) -> TTSSession:
        """ws 연결"""
        try:
            ws = await websockets.connect(
                self._build_uri(voice_id),
                additional_headers={"xi-api-key": self._api_key},
            )
        except WebSocketException:
            logger.exception("TTS WebSocket 연결 실패")
            raise
        return TTSSession(ws, self._voice_settings)

    async def stream(
        self,
        text_source: AsyncIterator[str],
        emotion: AiEmotion = AiEmotion.NEUTRAL,
    ) -> AsyncIterator[bytes]:
        session = await self.open()
        try:
            await session.begin(emotion)
            async for pcm in session.stream(text_source):
                yield pcm
        finally:
            await session.aclose()
