from __future__ import annotations

import asyncio
import base64
import logging
import uuid
import weakref
from collections.abc import AsyncIterator

import httpx

from app.core.config import Settings
from app.schemas.llm import AiEmotion
from app.services.qwen_tts import QwenTTSUnavailableError
from app.services.tts import _SentenceBuffer

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 3.0
_READ_TIMEOUT = 10.0

_worker_pools: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Queue[str]] = (
    weakref.WeakKeyDictionary()
)
# 통화 종료 뒤 워커 세션을 닫는 백그라운드 태스크. 강한 참조가 없으면 완료 전에 GC 될 수 있다(asyncio 문서).
_background_tasks: set[asyncio.Task[None]] = set()


def worker_urls(settings: Settings) -> list[str]:
    raw = getattr(settings, "csm_tts_urls", None) or ""
    return [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]


def _pool(settings: Settings) -> asyncio.Queue[str]:
    loop = asyncio.get_running_loop()
    pool = _worker_pools.get(loop)
    if pool is None:
        pool = _worker_pools[loop] = asyncio.Queue()
        for url in worker_urls(settings):
            pool.put_nowait(url)
    return pool


class CsmRealtimeTTSSession:
    """실시간 한 턴"""

    accepts_user_turn = True

    def __init__(self, client: httpx.AsyncClient, base_url: str, session_id: str) -> None:
        self._client = client
        self._base_url = base_url
        self._session_id = session_id
        self._inited = False
        self._closed = False
        self.chars_sent = 0

    async def begin(
        self,
        emotion: AiEmotion = AiEmotion.NEUTRAL,
        user_turn: tuple[bytes, str] | None = None,
    ) -> None:
        """감정을 제외하고 문맥을 넣음"""
        self._inited = True
        if not user_turn:
            return
        pcm, text = user_turn
        if len(pcm) < 2 or not text.strip():
            return
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/session/user",
                json={
                    "session_id": self._session_id,
                    "text": text[:300],
                    "pcm_b64": base64.b64encode(pcm).decode("ascii"),
                },
            )
            response.raise_for_status()
        except Exception:
            logger.warning("CSM 사용자 턴 전송 실패(문맥 없이 계속)", exc_info=True)

    async def stream(self, text_source: AsyncIterator[str]) -> AsyncIterator[bytes]:
        if not self._inited:
            raise RuntimeError("CsmRealtimeTTSSession.begin()을 먼저 호출해야 합니다.")
        buf = _SentenceBuffer()
        async for chunk in text_source:
            if not chunk:
                continue
            sentence = buf.feed(chunk)
            if sentence:
                async for pcm in self._speak(sentence):
                    yield pcm
        rest = buf.flush()
        if rest:
            async for pcm in self._speak(rest):
                yield pcm

    async def _speak(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        self.chars_sent += len(text)
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/session/speak",
                json={"session_id": self._session_id, "text": text[:300]},
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except Exception as exc:
            raise QwenTTSUnavailableError(f"{type(exc).__name__}: {exc}") from exc

    async def aclose(self) -> None:
        """턴 종료, 세션은 유지"""
        if self._closed:
            return
        self._closed = True
        try:
            await self._client.aclose()
        except Exception:
            logger.info("CSM 실시간 세션 종료 실패(무시)", exc_info=True)


class CsmRealtimeTTSClient:
    """통화 하나는 워커 세션 하나 차지, open() 은 턴마다하고 워커의 /v1/session/open 은 통화당 한 번 호출"""

    engine_name = "csm"

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
        self._session_id = f"call-{uuid.uuid4().hex[:12]}"
        self._opened = False

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(_READ_TIMEOUT, connect=_CONNECT_TIMEOUT),
            transport=self._transport,
        )

    async def open(self, voice_id: str | None = None) -> CsmRealtimeTTSSession:
        """턴마다 호출"""
        client = self._new_client()
        if not self._opened:
            voice = getattr(self._settings, "csm_tts_voice", "ai")
            try:
                response = await client.post(
                    f"{self._base_url}/v1/session/open",
                    json={"session_id": self._session_id, "voice": voice},
                )
                response.raise_for_status()
            except Exception as exc:
                await client.aclose()
                raise QwenTTSUnavailableError(f"{type(exc).__name__}: {exc}") from exc
            self._opened = True
        return CsmRealtimeTTSSession(client, self._base_url, self._session_id)

    async def _close_session(self) -> None:
        try:
            async with self._new_client() as client:
                await client.post(
                    f"{self._base_url}/v1/session/close",
                    json={"session_id": self._session_id},
                )
        except Exception:
            logger.info("CSM 워커 세션 닫기 실패(무시)", exc_info=True)

    def release_slot(self) -> None:
        """통화 종료"""
        url = self._worker_url
        if url is None:
            return
        self._worker_url = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("CSM release_slot 이 이벤트 루프 밖에서 불림 — 워커 %s 를 풀에 되돌리지 못함", url)
            return
        task = loop.create_task(self._close_session())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        _pool(self._settings).put_nowait(url)


async def _healthy(base_url: str, settings: Settings, transport) -> bool:
    timeout = httpx.Timeout(getattr(settings, "csm_tts_health_timeout", 1.0), connect=1.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.get(f"{base_url}/health")
            response.raise_for_status()
            return bool(response.json().get("ready"))
    except Exception:
        return False


async def try_acquire_realtime_csm(
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[CsmRealtimeTTSClient | None, str | None]:
    """통화 시작 시 CSM 워커 하나를 통화에 배치"""
    if not (getattr(settings, "csm_tts_realtime_enabled", False) and worker_urls(settings)):
        return None, "disabled"
    pool = _pool(settings)
    try:
        url = pool.get_nowait()
    except asyncio.QueueEmpty:
        return None, "busy"
    try:
        healthy = await _healthy(url, settings, transport)
    except BaseException:
        pool.put_nowait(url)
        raise
    if not healthy:
        pool.put_nowait(url)
        return None, "unhealthy"
    client = CsmRealtimeTTSClient(settings, transport, base_url=url)
    client._worker_url = url
    return client, None
