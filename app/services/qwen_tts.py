from __future__ import annotations

import asyncio
import logging

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class QwenTTSUnavailableError(Exception):
    """합성 실패"""


class QwenTTSClient:
    """Qwen3-TTS 서버 클라이언트"""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = (settings.qwen_tts_url or "").rstrip("/")
        self._synth_timeout = settings.qwen_tts_timeout
        self._health_timeout = settings.qwen_tts_health_timeout
        self._transport = transport
        self._semaphore = asyncio.Semaphore(1)

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def healthy(self) -> bool:
        """합성을 시작하기 전 헬스체크"""
        if not self.enabled:
            return False
        try:
            async with self._client(self._health_timeout) as client:
                response = await client.get(f"{self._base_url}/health")
                response.raise_for_status()
                return bool(response.json().get("ready"))
        except Exception:
            logger.info("Qwen TTS 헬스체크 실패 — ElevenLabs로 폴백", exc_info=True)
            return False

    async def synth(self, voice: str, text: str) -> bytes:
        """한 턴을 합성해 PCM 16kHz 바이트로 반환"""
        if not self.enabled:
            raise QwenTTSUnavailableError("qwen_tts_url 미설정")
        async with self._semaphore:
            try:
                async with self._client(self._synth_timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/v1/tts", json={"voice": voice, "text": text}
                    )
                    response.raise_for_status()
                    pcm = response.content
            except Exception as exc:
                raise QwenTTSUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        if not pcm:
            raise QwenTTSUnavailableError("빈 응답")
        return pcm
