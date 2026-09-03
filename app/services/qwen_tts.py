from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.core.config import Settings
from app.schemas.llm import AiEmotion
from app.services.tts import _SentenceBuffer

logger = logging.getLogger(__name__)


_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5

_SLOT_WAIT_TIMEOUT = 90.0

# 실시간 경로 전용 — 첫 청크 실측 ~181ms 라 read 갭 10초면 행업 감지로 충분
_REALTIME_VOICE = "ai"
_REALTIME_CONNECT_TIMEOUT = 3.0
_REALTIME_READ_TIMEOUT = 10.0


class QwenTTSUnavailableError(Exception):
    """합성 실패"""


_worker_pools: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Queue[str]] = (
    weakref.WeakKeyDictionary()
)


def worker_urls(settings: Settings) -> list[str]:
    """워커 URL 목록"""
    raw = settings.qwen_tts_urls or settings.qwen_tts_url or ""
    return [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]


def _pool(settings: Settings) -> asyncio.Queue[str]:
    """이벤트 루프별 가용 워커 큐"""
    loop = asyncio.get_running_loop()
    pool = _worker_pools.get(loop)
    if pool is None:
        pool = _worker_pools[loop] = asyncio.Queue()
        for url in worker_urls(settings):
            pool.put_nowait(url)
    return pool


def has_free_worker(settings: Settings) -> bool:
    """빈 워커가 있는지 확인"""
    if not worker_urls(settings):
        return False
    try:
        return not _pool(settings).empty()
    except RuntimeError:
        return False


@asynccontextmanager
async def _hold_worker(settings: Settings) -> AsyncIterator[str]:
    """워커 하나를 잡고 그 URL을 반환"""
    if not worker_urls(settings):
        raise QwenTTSUnavailableError("qwen_tts_url 미설정")
    pool = _pool(settings)
    try:
        url = await asyncio.wait_for(pool.get(), timeout=_SLOT_WAIT_TIMEOUT)
    except TimeoutError:
        raise QwenTTSUnavailableError("GPU 워커 대기 시간 초과") from None
    try:
        yield url
    finally:
        pool.put_nowait(url)


class QwenTTSClient:
    """Qwen3-TTS 서버 클라이언트"""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str | None = None,
    ) -> None:
        self._settings = settings
        urls = worker_urls(settings)
        self._base_url = (base_url or (urls[0] if urls else "")).rstrip("/")
        self._synth_timeout = settings.qwen_tts_timeout
        self._health_timeout = settings.qwen_tts_health_timeout
        self._transport = transport
        self._session: httpx.AsyncClient | None = None
        self._session_url: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[QwenTTSClient]:
        """여러 턴을 합성할 때 커넥션 재사용"""
        async with _hold_worker(self._settings) as url:
            try:
                client = self._client(self._synth_timeout)
                await client.__aenter__()
            except Exception as exc:
                raise QwenTTSUnavailableError(f"connect: {type(exc).__name__}: {exc}") from exc
            self._session = client
            self._session_url = url
            try:
                yield self
            finally:
                self._session = None
                self._session_url = None
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

    async def _post(
        self, client: httpx.AsyncClient, base_url: str, voice: str, text: str
    ) -> bytes:
        response = await client.post(
            f"{base_url}/v1/tts", json={"voice": voice, "text": text}
        )
        response.raise_for_status()
        return response.content

    async def _attempt(
        self, client: httpx.AsyncClient, base_url: str, voice: str, text: str
    ) -> bytes:
        """일시적 실패는 재시도"""
        last: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                pcm = await self._post(client, base_url, voice, text)
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
            return await self._attempt(
                self._session, self._session_url or self._base_url, voice, text
            )
        async with (
            _hold_worker(self._settings) as url,
            self._client(self._synth_timeout) as client,
        ):
            return await self._attempt(client, url, voice, text)


class QwenRealtimeTTSSession:
    """실시간 한 턴"""

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url
        self._inited = False
        self._closed = False

    async def begin(self, emotion: AiEmotion = AiEmotion.NEUTRAL) -> None:
        """감정 지시는 클론 경로에 없음"""
        self._inited = True

    async def stream(self, text_source: AsyncIterator[str]) -> AsyncIterator[bytes]:
        if not self._inited:
            raise RuntimeError("QwenRealtimeTTSSession.begin()을 먼저 호출해야 합니다.")
        buf = _SentenceBuffer()
        async for chunk in text_source:
            if not chunk:
                continue
            sentence = buf.feed(chunk)
            if sentence:
                async for pcm in self._synth_sentence(sentence):
                    yield pcm
        rest = buf.flush()
        if rest:
            async for pcm in self._synth_sentence(rest):
                yield pcm

    async def _synth_sentence(self, text: str) -> AsyncIterator[bytes]:
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/tts/stream",
                json={"voice": _REALTIME_VOICE, "text": text},
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except Exception as exc:
            raise QwenTTSUnavailableError(f"{type(exc).__name__}: {exc}") from exc

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._client.aclose()
        except Exception:
            logger.info("Qwen 실시간 세션 종료 실패(무시)", exc_info=True)


class QwenRealtimeTTSClient:
    """실시간 통화용 클라이언트
    open() → begin() → stream() → aclose()
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str | None = None,
    ) -> None:
        self._settings = settings
        urls = worker_urls(settings)
        self._base_url = (base_url or (urls[0] if urls else "")).rstrip("/")
        self._transport = transport
        self._worker_url: str | None = None

    async def open(self, voice_id: str | None = None) -> QwenRealtimeTTSSession:
        """턴마다 호출"""
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                _REALTIME_READ_TIMEOUT, connect=_REALTIME_CONNECT_TIMEOUT
            ),
            transport=self._transport,
        )
        return QwenRealtimeTTSSession(client, self._base_url)

    def release_slot(self) -> None:
        """통화 종료, 잡고 있던 워커를 풀에 되돌림"""
        url = self._worker_url
        if url is None:
            return
        self._worker_url = None
        _pool(self._settings).put_nowait(url)


async def try_acquire_realtime_tts(
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[QwenRealtimeTTSClient | None, str | None]:
    """통화 시작 시 워커 하나 통화에 배치"""
    if not (settings.qwen_tts_realtime_enabled and worker_urls(settings)):
        return None, "disabled"
    pool = _pool(settings)
    try:
        url = pool.get_nowait()
    except asyncio.QueueEmpty:
        return None, "busy"
    try:
        healthy = await QwenTTSClient(settings, transport, base_url=url).healthy()
    except BaseException:
        pool.put_nowait(url)
        raise
    if not healthy:
        pool.put_nowait(url)
        return None, "unhealthy"
    client = QwenRealtimeTTSClient(settings, transport, base_url=url)
    client._worker_url = url
    return client, None
