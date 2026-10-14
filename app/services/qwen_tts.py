from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.core.audio_stats import BYTES_PER_MS
from app.core.config import Settings
from app.core.metrics import log_metric
from app.schemas.llm import AiEmotion
from app.services.pcm_stretch import PcmStretcher, clamp_speed
from app.services.tts import _SentenceBuffer

logger = logging.getLogger(__name__)


_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5

_SLOT_WAIT_TIMEOUT = 90.0

# 실시간 경로 전용 — 첫 청크 실측 ~181ms 라 read 갭 10초면 행업 감지로 충분
_REALTIME_VOICE = "ai"
_REALTIME_CONNECT_TIMEOUT = 3.0
_REALTIME_READ_TIMEOUT = 10.0
_RUNAWAY_FLOOR_MS = 3000
_RUNAWAY_MS_PER_CHAR = 300
_SPEED_WARMUP_MS = 180  # 턴 첫 문장 앞 부분은 원래 속도


class QwenTTSUnavailableError(Exception):
    """합성 실패"""


class QwenTTSRunawayError(QwenTTSUnavailableError):
    """문장 길이에 비해 오디오가 너무 길다"""

    def __init__(
        self, *, worker: str, chars: int, pcm_bytes: int, cap_ms: int, wall_ms: float
    ) -> None:
        self.worker = worker
        self.chars = chars
        self.pcm_bytes = pcm_bytes
        self.audio_ms = round(pcm_bytes / BYTES_PER_MS, 1)
        self.cap_ms = cap_ms
        self.wall_ms = round(wall_ms, 1)
        super().__init__(f"runaway chars={chars} audio_ms={self.audio_ms} cap_ms={cap_ms}")


def runaway_cap_ms(text: str) -> int:
    """문장 하나에 허용하는 오디오 길이"""
    return _RUNAWAY_FLOOR_MS + len(text.strip()) * _RUNAWAY_MS_PER_CHAR


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

    async def health_state(self) -> str:
        """ready / busy(락 잡힘) / warming(ready:false) / down(연결 거부) / unhealthy(응답 이상)"""
        if not self.enabled:
            return "unhealthy"
        try:
            async with self._client(self._health_timeout) as client:
                response = await client.get(f"{self._base_url}/health")
                response.raise_for_status()
                body = response.json()
        except httpx.ConnectError:
            # 워밍업 중인 워커 — 박스 다운(타임아웃)과 구분
            logger.info("Qwen TTS 헬스체크 연결 거부", exc_info=True)
            return "down"
        except Exception:
            logger.info("Qwen TTS 헬스체크 실패", exc_info=True)
            return "unhealthy"
        if not isinstance(body, dict):
            return "unhealthy"
        if not body.get("ready"):
            return "warming"
        return "busy" if body.get("busy") else "ready"

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

    def __init__(self, client: httpx.AsyncClient, base_url: str, speed: float = 1.0) -> None:
        self._client = client
        self._base_url = base_url
        self._speed = speed
        self._sentences = 0
        self._inited = False
        self._closed = False
        self.chars_sent = 0

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
        self.chars_sent += len(text)
        cap_ms = runaway_cap_ms(text)
        received = 0
        started = time.monotonic()
        stretcher = PcmStretcher(
            self._speed, warmup_ms=0 if self._sentences else _SPEED_WARMUP_MS
        )
        self._sentences += 1
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/tts/stream",
                json={"voice": _REALTIME_VOICE, "text": text},
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > cap_ms * BYTES_PER_MS:
                        raise QwenTTSRunawayError(
                            worker=self._base_url,
                            chars=len(text.strip()),
                            pcm_bytes=received,
                            cap_ms=cap_ms,
                            wall_ms=(time.monotonic() - started) * 1000,
                        )
                    out = stretcher.push(chunk)
                    if out:
                        yield out
                tail = stretcher.flush()
                if tail:
                    yield tail
        except QwenTTSUnavailableError:
            raise
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
        self.speed = clamp_speed(getattr(settings, "qwen_tts_speed", 1.0))

    async def open(self, voice_id: str | None = None) -> QwenRealtimeTTSSession:
        """턴마다 호출"""
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                _REALTIME_READ_TIMEOUT, connect=_REALTIME_CONNECT_TIMEOUT
            ),
            transport=self._transport,
        )
        return QwenRealtimeTTSSession(client, self._base_url, self.speed)

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
    skipped: list[tuple[str, str]] = []
    try:
        for _ in range(len(worker_urls(settings))):
            try:
                url = pool.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                state = await QwenTTSClient(settings, transport, base_url=url).health_state()
            except BaseException:
                pool.put_nowait(url)
                raise
            if state == "ready":
                client = QwenRealtimeTTSClient(settings, transport, base_url=url)
                client._worker_url = url
                return client, None
            if state == "unhealthy":
                pool.put_nowait(url)
                return None, "unhealthy"
            skipped.append((url, state))
    finally:
        for url, state in skipped:
            pool.put_nowait(url)
            log_metric("realtime_tts_worker_skip", url=url, state=state)
    if not skipped:
        return None, "busy"
    if any(state == "busy" for _, state in skipped):
        return None, "worker_busy"
    return None, "unhealthy"
