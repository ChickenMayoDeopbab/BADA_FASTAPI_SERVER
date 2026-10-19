from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import weakref
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from app.core.audio_shape import RequestMeter
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


class QwenVoiceMissingError(QwenTTSUnavailableError):
    """워커에 그 보이스가 없음(404)"""

    def __init__(self, voice: str) -> None:
        super().__init__(f"HTTP 404 voice={voice}")
        self.voice = voice


class QwenCapHitError(QwenTTSUnavailableError):
    """워커 생성 상한(502)"""


class QwenStalledError(QwenTTSUnavailableError):
    """요청 시간 초과"""


class QwenUnreachableError(QwenTTSUnavailableError):
    """워커 연결 실패"""

    def __init__(self, message: str, worker: str | None = None) -> None:
        super().__init__(message)
        self.worker = worker


NS_HEADER = "x-non-streaming"
VOICE_REF_HEADER = "x-voice-ref"
_REF_RE = re.compile(r"[0-9a-f]{8}")
_UNREACHABLE = object()
_CAP_BASE_S, _CAP_PER_CHAR_S = 5.0, 0.4  # 워커 max_frames_for 와 같은 값


@dataclass(frozen=True)
class NsResult:
    pcm: bytes
    ns: bool
    ref: str | None
    worker: str


@dataclass(frozen=True)
class RefsState:
    state: str  # ok / down / no_refs / rolling / missing
    refs: dict[str, str]
    unreachable: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


def is_voice_ref(value: object) -> bool:
    return isinstance(value, str) and _REF_RE.fullmatch(value) is not None


def ns_timeout(base: float, text: str) -> float:
    """워커 상한 502 가 클라이언트 타임아웃보다 먼저 오도록 하는 read 타임아웃"""
    return max(base, (_CAP_BASE_S + _CAP_PER_CHAR_S * len(text)) * 0.8 + 10)


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
async def _hold_worker(settings: Settings, exclude: Iterable[str] = ()) -> AsyncIterator[str]:
    """워커 하나를 잡고 그 URL을 반환, exclude 에 든 워커는 건너뜀"""
    urls = worker_urls(settings)
    if not urls:
        raise QwenTTSUnavailableError("qwen_tts_url 미설정")
    skip = set(exclude)
    if skip and not set(urls) - skip:
        raise QwenTTSUnavailableError("쓸 수 있는 워커 없음(모두 제외)")
    pool = _pool(settings)
    skipped: list[str] = []
    try:
        while True:
            try:
                url = await asyncio.wait_for(pool.get(), timeout=_SLOT_WAIT_TIMEOUT)
            except TimeoutError:
                raise QwenTTSUnavailableError("GPU 워커 대기 시간 초과") from None
            if url not in skip:
                break
            skipped.append(url)
    finally:
        for other in skipped:
            pool.put_nowait(other)
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
        self.voices: frozenset[str] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    @asynccontextmanager
    async def connect(self, exclude: Iterable[str] = ()) -> AsyncIterator[QwenTTSClient]:
        """여러 턴을 합성할 때 커넥션 재사용"""
        async with _hold_worker(self._settings, exclude) as url:
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
        voices = body.get("voices")
        self.voices = frozenset(v for v in voices if isinstance(v, str)) if isinstance(voices, list) else None
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
                if exc.response.status_code == 404:
                    raise QwenVoiceMissingError(voice) from exc
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

    async def synth_ns(self, voice: str, text: str) -> NsResult:
        """non_streaming_mode 로 한 번만 합성, 재시도는 부르는 쪽이 한다"""
        if not self.enabled:
            raise QwenTTSUnavailableError("qwen_tts_url 미설정")
        timeout = ns_timeout(self._synth_timeout, text)
        if self._session is not None:
            return await self._post_ns(self._session, self._session_url or self._base_url, voice, text, timeout)
        async with (
            _hold_worker(self._settings) as url,
            self._client(self._synth_timeout) as client,
        ):
            return await self._post_ns(client, url, voice, text, timeout)

    async def _post_ns(
        self, client: httpx.AsyncClient, base_url: str, voice: str, text: str, timeout: float
    ) -> NsResult:
        try:
            response = await client.post(
                f"{base_url}/v1/tts",
                json={"voice": voice, "text": text, "non_streaming_mode": True},
                timeout=httpx.Timeout(self._synth_timeout, read=timeout),
            )
        except httpx.TimeoutException as exc:
            raise QwenStalledError(f"{type(exc).__name__}: {exc}") from exc
        except httpx.ConnectError as exc:
            raise QwenUnreachableError(f"{type(exc).__name__}: {exc}", worker=base_url) from exc
        except httpx.HTTPError as exc:
            raise QwenTTSUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        status = response.status_code
        if status == 404:
            raise QwenVoiceMissingError(voice)
        if status == 502:
            raise QwenCapHitError(f"HTTP 502 voice={voice}")
        if status >= 400:
            raise QwenTTSUnavailableError(f"HTTP {status}")
        if not response.content:
            raise QwenTTSUnavailableError("빈 응답")
        return NsResult(
            pcm=response.content,
            ns=response.headers.get(NS_HEADER) == "1",
            ref=response.headers.get(VOICE_REF_HEADER),
            worker=base_url,
        )

    async def health_refs(self, needed: Iterable[str]) -> RefsState:
        """모든 워커의 ref 지문을 모아 판정"""
        urls = worker_urls(self._settings)
        found = await asyncio.gather(*(self._refs_of(url) for url in urls))
        reachable = [(url, refs) for url, refs in zip(urls, found, strict=True) if refs is not _UNREACHABLE]
        unreachable = tuple(url for url, refs in zip(urls, found, strict=True) if refs is _UNREACHABLE)
        if not reachable:
            return RefsState("down", {}, unreachable)
        if any(refs is None for _, refs in reachable):
            return RefsState("no_refs", {}, unreachable)
        merged: dict[str, str] = {}
        missing: list[str] = []
        rolling = False
        for voice in dict.fromkeys(needed):
            values = {refs.get(voice) for _, refs in reachable}
            if None in values:
                missing.append(voice)
            values.discard(None)
            if len(values) > 1:
                rolling = True
            elif values:
                merged[voice] = values.pop()
        if rolling:
            return RefsState("rolling", merged, unreachable)
        if missing:
            return RefsState("missing", merged, unreachable, tuple(missing))
        return RefsState("ok", merged, unreachable)

    async def _refs_of(self, url: str) -> dict[str, str] | None | object:
        try:
            async with self._client(self._health_timeout) as client:
                response = await client.get(f"{url}/health")
                response.raise_for_status()
                body = response.json()
        except Exception:
            logger.info("Qwen TTS refs 조회 실패 %s", url, exc_info=True)
            return _UNREACHABLE
        refs = body.get("refs") if isinstance(body, dict) else None
        if not isinstance(refs, dict):
            return None
        return {k: v for k, v in refs.items() if isinstance(k, str) and is_voice_ref(v)}


_SENTENCE_END = re.compile(r"[.?!…]")


class _RequestRecorder:
    """실시간 요청 하나의 시간·헤더·끝 모양"""

    def __init__(self, clock) -> None:
        self._clock = clock
        self._t0 = clock()
        self._first: float | None = None
        self._last: float | None = None
        self._stall = 0.0
        self._raw = 0
        self._ns: int | None = None
        self._meter = RequestMeter()

    def header(self, response: httpx.Response) -> None:
        with contextlib.suppress(Exception):
            value = response.headers.get(NS_HEADER)
            self._ns = int(value) if value in ("0", "1") else None

    def chunk(self, data: bytes) -> None:
        try:
            now = self._clock()
            if self._first is None:
                self._first = now
            elif self._last is not None:
                self._stall = max(self._stall, now - self._last)
            self._last = now
            self._raw += len(data)
            self._meter.push(data)
        except Exception:
            logger.info("요청 계측 실패(무시)", exc_info=True)

    def finish(self, text: str, voice: str, outcome: str) -> dict:
        end = self._clock()
        shape = self._meter.result()
        return {
            "chars": len(text.strip()),
            "sents": len(_SENTENCE_END.findall(text)),
            "voice": voice,
            "ns": self._ns,
            "outcome": outcome,
            "raw_ms": round(self._raw / BYTES_PER_MS, 1),
            "ttfb_ms": None if self._first is None else round((self._first - self._t0) * 1000, 1),
            "wall_ms": round((end - self._t0) * 1000, 1),
            "stall_max_ms": round(self._stall * 1000, 1),
            **shape,
        }


def _turn_metrics(log: list[dict], speed: float) -> dict:
    if not log:
        return {}
    shaped = [r for r in log if "decay_ms" in r]
    raw = sum(r["raw_ms"] for r in log)
    gen = sum(r["wall_ms"] - (r["ttfb_ms"] or 0.0) for r in log if r["ttfb_ms"] is not None)
    gaps = [
        (a.get("trail_ms", 0.0) + b.get("lead_ms", 0.0)) / max(speed, 1e-6)
        for a, b in zip(shaped, shaped[1:], strict=False)
    ]
    ttfbs = [r["ttfb_ms"] for r in log if r["ttfb_ms"] is not None]
    return {
        "tts_req": len(log),
        "tts_req_cut": sum(bool(r.get("cut")) for r in log),
        "tts_req_hard": sum(bool(r.get("hard")) for r in log),
        "tts_last_cut": bool(log[-1].get("cut")),
        "tts_tail_short": sum(r.get("trail_ms", 0.0) < 100 for r in shaped if r.get("decay_ms") is not None),
        "tts_raw_ms": round(raw, 1),
        "tts_gen_rtf": round(gen / raw, 3) if raw > 0 else None,
        "tts_ttfb_max_ms": max(ttfbs) if ttfbs else None,
        "tts_stall_max_ms": max(r["stall_max_ms"] for r in log),
        "tts_gap_max_ms": round(max(gaps), 1) if gaps else None,
        "tts_gap_over700": sum(g > 700 for g in gaps),
    }


class QwenRealtimeTTSSession:
    """실시간 한 턴"""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        speed: float = 1.0,
        voice: str = _REALTIME_VOICE,
        *,
        req_metric: bool = True,
        clock=time.monotonic,
    ) -> None:
        self._client = client
        self._base_url = base_url
        self._speed = speed
        self._voice = voice
        self._audio_started = False
        self._inited = False
        self._closed = False
        self.chars_sent = 0
        self._req_metric = req_metric
        self._clock = clock
        self.request_log: list[dict] = []

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
                async with contextlib.aclosing(self._synth_sentence(sentence)) as gen:
                    async for pcm in gen:
                        yield pcm
        rest = buf.flush()
        if rest:
            async with contextlib.aclosing(self._synth_sentence(rest)) as gen:
                async for pcm in gen:
                    yield pcm

    async def _synth_sentence(self, text: str) -> AsyncIterator[bytes]:
        self.chars_sent += len(text)
        cap_ms = runaway_cap_ms(text)
        received = 0
        started = time.monotonic()
        stretcher = PcmStretcher(
            self._speed, warmup_ms=0 if self._audio_started else _SPEED_WARMUP_MS
        )
        rec = _RequestRecorder(self._clock) if self._req_metric else None
        outcome = "aborted"
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/tts/stream",
                json={"voice": self._voice, "text": text},
            ) as response:
                if rec is not None:
                    rec.header(response)
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
                    if rec is not None:
                        rec.chunk(chunk)
                    out = stretcher.push(chunk)
                    if out:
                        self._audio_started = True
                        yield out
                tail = stretcher.flush()
                if tail:
                    self._audio_started = True
                    yield tail
            outcome = "eos"
        except QwenTTSRunawayError:
            outcome = "runaway"
            raise
        except QwenTTSUnavailableError:
            outcome = "error"
            raise
        except Exception as exc:
            outcome = "error"
            raise QwenTTSUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            if rec is not None:
                with contextlib.suppress(Exception):
                    self.request_log.append(rec.finish(text, self._voice, outcome))

    def turn_metrics(self) -> dict:
        """voice_turn 에 붙일 요청 집계"""
        try:
            return _turn_metrics(self.request_log, self._speed)
        except Exception:
            logger.info("요청 집계 실패(무시)", exc_info=True)
            return {}

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
        self.voices: frozenset[str] | None = None

    async def open(self, voice_id: str | None = None) -> QwenRealtimeTTSSession:
        """턴마다 호출"""
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                _REALTIME_READ_TIMEOUT, connect=_REALTIME_CONNECT_TIMEOUT
            ),
            transport=self._transport,
        )
        return QwenRealtimeTTSSession(
            client,
            self._base_url,
            self.speed,
            voice_id or _REALTIME_VOICE,
            req_metric=getattr(self._settings, "qwen_tts_req_metric", True),
        )

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
            checker = QwenTTSClient(settings, transport, base_url=url)
            try:
                state = await checker.health_state()
            except BaseException:
                pool.put_nowait(url)
                raise
            if state == "ready":
                client = QwenRealtimeTTSClient(settings, transport, base_url=url)
                client._worker_url = url
                client.voices = checker.voices
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
