import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator

import websockets
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

class ElevenLabsTTSClient:
    """TTS 클라이언트, Elevenlabs Flash 2.5 모델 사용, 턴당 ws 연결"""
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
            "speed": settings.elevenlabs_speed
        }

    def _build_uri(self) -> str:
        return (
            f"{self._ws_host}/v1/text-to-speech/{self._voice_id}/stream-input"
            f"?model_id={self._model}"
            f"&output_format={self._output_format}"
            f"&language_code={self._language}"
            f"&auto_mode={'true' if self._auto_mode else 'false'}"
            f"&apply_text_normalization={self._text_normalization}"
        )

    def _voice_settings_for(self, emotion: AiEmotion):
        settings = dict(self._voice_settings)
        settings.update(_EMOTION_VOICE_OVERRIDES.get(emotion, {}))
        return settings

    def _build_init_message(self, emotion: AiEmotion) -> dict:
        return {"text": " ", "voice_settings": self._voice_settings_for(emotion)}

    async def _send_text(self, ws, text_source: AsyncIterator[str]) -> None:
        async for chunk in text_source:
            if not chunk:
                continue
            payload = chunk if chunk.endswith(" ") else chunk + " "
            await ws.send(json.dumps({"text": payload}))
        await ws.send(json.dumps({"text": ""}))

    async def stream(
            self,
            text_source: AsyncIterator[str],
            emotion: AiEmotion = AiEmotion.NEUTRAL
    ) -> AsyncIterator[bytes]:
        uri = self._build_uri()
        headers = {"xi-api-key": self._api_key}

        try:
            async with websockets.connect(uri, additional_headers=headers) as ws:
                await ws.send(json.dumps(self._build_init_message(emotion)))

                sender = asyncio.create_task(self._send_text(ws, text_source))
                try:
                    async for raw in ws:
                        msg = json.loads(raw)

                        audio_b64 = msg.get("audio")
                        if audio_b64:
                            yield base64.b64decode(audio_b64)

                        if msg.get("isFinal"):
                            break
                finally:
                    if not sender.done():
                        sender.cancel()
                    try:
                        await sender
                    except asyncio.CancelledError:
                        pass

        except asyncio.CancelledError:
            raise
        except WebSocketException:
            logger.exception("TTS WebSocket 에러")
            raise