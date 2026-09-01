from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


_semaphores: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()


_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = _semaphores[loop] = asyncio.Semaphore(1)
    return sem


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
        self._session: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[QwenTTSClient]:
        """여러 턴을 합성할 때 커넥션 재사용"""
        async with _semaphore():
            try:
                client = self._client(self._synth_timeout)
                await client.__aenter__()
            except Exception as exc:
                raise QwenTTSUnavailableError(f"connect: {type(exc).__name__}: {exc}") from exc
            self._session = client
            try:
                yield self
            finally:
                self._session = None
                try:
                    await client.aclose()
                except Exception:
                    logger.info("Qwen TTS 커넥션 종료 실패(무시)", exc_info=True)

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

    async def _post(self, client: httpx.AsyncClient, voice: str, text: str) -> bytes:
        response = await client.post(
            f"{self._base_url}/v1/tts", json={"voice": voice, "text": text}
        )
        response.raise_for_status()
        return response.content

    async def _attempt(self, client: httpx.AsyncClient, voice: str, text: str) -> bytes:
        """일시적 실패는 재시도"""
        last: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                pcm = await self._post(client, voice, text)
                if not pcm:
                    raise QwenTTSUnavailableError("빈 응답")
                return pcm
            except QwenTTSUnavailableError:
                raise
            except httpx.TimeoutException as exc:
                raise QwenTTSUnavailableError(f"{type(exc).__name__}: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise QwenTTSUnavailableError(f"HTTP {exc.response.status_code}") from exc
                last = exc
            except Exception as exc:
                last = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))
        raise QwenTTSUnavailableError(f"{type(last).__name__}: {last}") from last

    async def synth(self, voice: str, text: str) -> bytes:
        """한 턴을 합성해 PCM 16kHz 바이트로 반환"""
        if not self.enabled:
            raise QwenTTSUnavailableError("qwen_tts_url 미설정")
        if self._session is not None:
            return await self._attempt(self._session, voice, text)
        async with _semaphore(), self._client(self._synth_timeout) as client:
            return await self._attempt(client, voice, text)
