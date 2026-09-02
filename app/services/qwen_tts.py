from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.core.concurrency import loop_semaphore
from app.core.config import Settings
from app.schemas.llm import AiEmotion
from app.services.tts import _SentenceBuffer

logger = logging.getLogger(__name__)


_semaphore = loop_semaphore(1)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5

# 배치 경로 슬롯 대기 상한 — 실시간 통화가 슬롯을 길게 점유할 수 있어 무한 대기를 막는다
_SLOT_WAIT_TIMEOUT = 90.0

# 실시간 경로 전용 — 첫 청크 실측 ~181ms 라 read 갭 10초면 행업 감지로 충분
_REALTIME_VOICE = "ai"
_REALTIME_CONNECT_TIMEOUT = 3.0
_REALTIME_READ_TIMEOUT = 10.0


class QwenTTSUnavailableError(Exception):
    """합성 실패"""


class _RealtimeSlot:
    """실시간 통화의 GPU 슬롯 점유 상태"""

    __slots__ = ("held",)

    def __init__(self) -> None:
        self.held = False


_realtime_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _RealtimeSlot] = (
    weakref.WeakKeyDictionary()
)


def _realtime_slot() -> _RealtimeSlot:
    loop = asyncio.get_running_loop()
    slot = _realtime_slots.get(loop)
    if slot is None:
        slot = _realtime_slots[loop] = _RealtimeSlot()
    return slot


def realtime_slot_active() -> bool:
    """실시간 통화가 슬롯을 잡고 있는지"""
    return _realtime_slot().held


@asynccontextmanager
async def _hold_slot() -> AsyncIterator[None]:
    """배치 경로의 슬롯 획득"""
    semaphore = _semaphore()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=_SLOT_WAIT_TIMEOUT)
    except TimeoutError:
        raise QwenTTSUnavailableError("GPU 슬롯 대기 시간 초과") from None
    try:
        yield
    finally:
        semaphore.release()


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
        async with _hold_slot():
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
        async with _hold_slot(), self._client(self._synth_timeout) as client:
            return await self._attempt(client, voice, text)


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
    ) -> None:
        self._base_url = (settings.qwen_tts_url or "").rstrip("/")
        self._transport = transport
        self._slot_held = False

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
        """통화 종료"""
        if not self._slot_held:
            return
        self._slot_held = False
        _realtime_slot().held = False
        _semaphore().release()


async def try_acquire_realtime_tts(
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[QwenRealtimeTTSClient | None, str | None]:
    """통화 시작 시 1회"""
    if not (settings.qwen_tts_realtime_enabled and settings.qwen_tts_url):
        return None, "disabled"
    semaphore = _semaphore()
    if semaphore.locked():
        return None, "busy"
    await semaphore.acquire()
    try:
        healthy = await QwenTTSClient(settings, transport).healthy()
    except BaseException:
        semaphore.release()
        raise
    if not healthy:
        semaphore.release()
        return None, "unhealthy"
    client = QwenRealtimeTTSClient(settings, transport)
    client._slot_held = True
    _realtime_slot().held = True
    return client, None
