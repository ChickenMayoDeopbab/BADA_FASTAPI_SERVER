import asyncio
import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum

from fastapi import WebSocket, WebSocketDisconnect
from google.api_core.exceptions import GoogleAPICallError

from app.core.config import Settings
from app.core.enums import SessionType
from app.core.metrics import log_metric, now_ms
from app.schemas.frames import (
    EndReason,
    emotion_frame,
    end_frame,
    error_frame,
    interrupt_frame,
    speaking_end_frame,
)
from app.schemas.llm import AiEmotion, LLMEventType
from app.services.llm import LLMClient
from app.services.session import build_turn_context
from app.services.spring_client import SpringInternalClient
from app.services.stt import (
    AUDIO_EOS,
    GoogleSTTClient,
    STTEventType,
    STTIdleTimeoutError,
)
from app.services.tremor import TremorAnalyzer
from app.services.tts import ElevenLabsTTSClient, TTSSession

logger = logging.getLogger(__name__)

# barge-in off 해뒀음
# OFF면 AI가 발화중에 입력 무시 나중에 제대로 구현 시 True로 고민
_BARGE_IN_ENABLED = False
_BARGE_IN_MIN_CHARS = 2
_STT_RECYCLE_SECONDS = 240.0
_SAFETY_FALLBACK = "죄송해요, 그 부분은 지금 답하기 어렵네요."


def _clip(text: str, limit: int = 120) -> str:
    """진단 로그용 텍스트 길이 상한"""
    return text if len(text) <= limit else text[:limit] + "…"


class _State(StrEnum):
    LISTENING = "LISTENING" # 듣는중
    THINKING = "THINKING" # 생각중
    SPEAKING = "SPEAKING" # 말하는중
    CLOSING = "CLOSING" # 끝


@dataclass
class _TurnTimings:
    """한 음성 턴의 단계별 시점(monotonic ms). 미도달 시점은 None(부분 턴)."""

    final_at: float                      # STT FINAL 수신
    last_audio_at: float | None = None   # 직전 사용자 오디오 청크 수신
    ctx_sent_at: float | None = None     # LLM 스트림 시작
    first_token_at: float | None = None  # 첫 텍스트 토큰
    last_token_at: float | None = None   # 마지막 텍스트 토큰
    first_pcm_at: float | None = None    # 첫 PCM 클라 송출
    last_pcm_at: float | None = None     # 마지막 PCM 클라 송출

    @staticmethod
    def _delta(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round(end - start, 1)

    def as_metrics(self) -> dict[str, float | None]:
        """지표 정의대로 구간 지연(ms) 산출. 스트리밍이라 구간은 시간상 겹침"""
        return {
            "stt_ms": self._delta(self.last_audio_at, self.final_at),
            "llm_ttft_ms": self._delta(self.ctx_sent_at, self.first_token_at),
            "llm_total_ms": self._delta(self.ctx_sent_at, self.last_token_at),
            "tts_ttfb_ms": self._delta(self.first_token_at, self.first_pcm_at),
            "tts_total_ms": self._delta(self.first_token_at, self.last_pcm_at),
            "response_ms": self._delta(self.final_at, self.first_pcm_at),
            "turn_total_ms": self._delta(self.final_at, self.last_pcm_at),
        }


class VoicePipeline:
    def __init__(
        self,
        ws: WebSocket,
        session_id: str,
        session: dict,
        *,
        settings: Settings,
        spring: SpringInternalClient,
    ) -> None:
        self._ws = ws
        self._session_id = session_id
        self._session = session
        self._settings = settings
        self._spring = spring

        self._stt = GoogleSTTClient(
            project_id=settings.google_project_id,
            location=settings.google_stt_location,
            model=settings.google_stt_model,
            language=settings.google_stt_language,
        )
        self._llm = LLMClient()
        self._tts = ElevenLabsTTSClient(settings)

        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._state = _State.LISTENING
        self._history: list[dict] = []
        self._current_step = 1
        self._muted = False
        self._ws_alive = True
        self._time_up = False
        self._last_audio_at: float | None = None

        self._turn_task: asyncio.Task | None = None
        self._closing = asyncio.Event()
        self._end_reason: EndReason | None = None

        self._listening_since: float | None = time.monotonic()
        self._silence_total: float = 0
        self._tremor = TremorAnalyzer()
        self._tremor_buf = bytearray()

        self._user_turn_intervals: list[tuple[float, float]] = []
        self._turn_open_at: float | None = 0.0

        max_duration = session.get("maxDurationSeconds")
        if max_duration is None and session.get("type") == SessionType.WARMUP:
            max_duration = 30  # Spring이 안 박아도 워밍업은 30초 보장
        self._max_duration = int(max_duration) if max_duration else None

        scenario = session.get("scenario") or {}
        script = scenario.get("script") if isinstance(scenario, dict) else None
        self._script_len = len(script) if isinstance(script, list) else 0

    async def run(self) -> None:
        """진입점"""
        tasks: list[asyncio.Task] = [
            asyncio.create_task(self._recv_loop()),
            asyncio.create_task(self._stt_consumer()),
        ]
        if self._max_duration:
            tasks.append(asyncio.create_task(self._max_duration_timer()))
        closing = asyncio.create_task(self._closing.wait())

        try:
            done, _ = await asyncio.wait(
                {*tasks, closing}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task is closing or task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.error(
                        "백그라운드 태스크 비정상 종료",
                        extra={"session_id": self._session_id},
                        exc_info=exc,
                    )
            if not self._closing.is_set():
                await self._close(EndReason.ERROR)
        finally:
            closing.cancel()
            with suppress(asyncio.CancelledError):
                await closing
            await self._teardown(*tasks)

    async def _recv_loop(self) -> None:
        # 수신
        try:
            while not self._closing.is_set():
                msg = await self._ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    self._ws_alive = False
                    await self._close(EndReason.USER_END)
                    return
                if msg.get("bytes") is not None:
                    if not self._muted:
                        self._last_audio_at = now_ms()
                        self._tremor_buf.extend(msg["bytes"])
                        await self._audio_queue.put(msg["bytes"])
                elif msg.get("text") is not None:
                    await self._handle_client_text(msg["text"])
        except (WebSocketDisconnect, RuntimeError):
            self._ws_alive = False
            await self._close(EndReason.USER_END)

    async def _handle_client_text(self, raw: str) -> None:
        """클라이언트 텍스트 핸들링"""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("클라 텍스트 프레임 파싱 실패", extra={"session_id": self._session_id})
            return
        mtype = data.get("type")
        if mtype == "end":
            await self._close(EndReason.USER_END)
        elif mtype == "ping":
            await self._send_json({"type": "pong"})
        elif mtype == "mute":
            self._muted = bool(data.get("muted", False))

    async def _stt_consumer(self) -> None:
        """stt 소비 발화마다 스트림 재오픈함"""
        try:
            while not self._closing.is_set():
                await self._consume_one_stream(self._audio_queue)
        except STTIdleTimeoutError:
            logger.info(
                "오디오 미수신으로 STT 스트림 종료",
                extra={"session_id": self._session_id},
            )
            await self._close(EndReason.NO_AUDIO)
        except GoogleAPICallError:
            logger.exception("STT 스트리밍 실패", extra={"session_id": self._session_id})
            await self._close(EndReason.ERROR)
        except asyncio.CancelledError:
            raise

    async def _consume_one_stream(self, queue: "asyncio.Queue[bytes | None]") -> None:
        """한 스트림을 FINAL까지 소비하고 닫음"""
        recycle_at = time.monotonic() + _STT_RECYCLE_SECONDS
        stream = self._stt.stream(queue)
        try:
            async for event in stream:
                await self._handle_stt_event(event)
                if event.type == STTEventType.FINAL or time.monotonic() >= recycle_at:
                    break
        finally:
            if not self._closing.is_set() and self._audio_queue is queue:
                self._audio_queue = asyncio.Queue()
            with suppress(Exception):
                queue.put_nowait(AUDIO_EOS)
            with suppress(Exception):
                await stream.aclose()  # 강제 종료

    async def _handle_stt_event(self, event) -> None:
        if self._state == _State.CLOSING:
            return
        if event.type == STTEventType.INTERIM:
            if (
                _BARGE_IN_ENABLED
                and self._state in (_State.THINKING, _State.SPEAKING)
                and len(event.text.strip()) >= _BARGE_IN_MIN_CHARS
            ):
                await self._barge_in()
        elif event.type == STTEventType.FINAL:
            text = event.text.strip()
            logger.info(
                "STT FINAL 수신 state=%s text=%r",
                self._state.value,
                _clip(text),
                extra={"session_id": self._session_id},
            )
            if text and self._state == _State.LISTENING and not self._time_up:
                self._start_turn(text, final_at=now_ms())
        elif event.type == STTEventType.SPEECH_BEGIN and (
                self._listening_since is not None and
                time.monotonic() - self._listening_since > 1.5 and
                self._state == _State.LISTENING):
            self._silence_total += time.monotonic() - self._listening_since
            self._listening_since = None

    def _start_turn(self, user_utterance: str, *, final_at: float) -> None:
        """LLM -> TTS"""
        self._close_user_turn()
        self._state = _State.THINKING
        timings = _TurnTimings(final_at=final_at, last_audio_at=self._last_audio_at)
        self._turn_task = asyncio.create_task(self._run_turn(user_utterance, timings))

    async def _run_turn(self, user_utterance: str, timings: _TurnTimings) -> None:
        """턴 시작"""
        ctx = build_turn_context(
            self._session,
            current_step=self._current_step,
            history=self._history,
            user_utterance=user_utterance,
        )
        first_goal = ctx.script[0].ai_goal if ctx.script else ""
        logger.info(
            "LLM 턴 컨텍스트 생성 step=%s title=%r role=%r goal=%r",
            self._current_step,
            ctx.scenario_title,
            ctx.scenario_role,
            first_goal,
            extra={"session_id": self._session_id},
        )

        connect_task = asyncio.create_task(self._tts.open())
        text_q: asyncio.Queue[str | None] = asyncio.Queue()
        emotion_ready = asyncio.Event()
        ai_parts: list[str] = []
        flags = {"step_done": False, "end_call": False, "error": False}
        box: dict[str, TTSSession | None] = {"tts": None}

        async def ensure_tts(emotion: AiEmotion) -> None:
            if box["tts"] is not None:
                return
            session = await connect_task
            await session.begin(emotion)
            box["tts"] = session
            await self._send_json(emotion_frame(emotion))
            emotion_ready.set()

        async def produce() -> None:
            """이벤트 타입따라 분기"""
            timings.ctx_sent_at = now_ms()
            try:
                async for ev in self._llm.stream(ctx):
                    if ev.type == LLMEventType.EMOTION_RESOLVED:
                        await ensure_tts(ev.emotion or AiEmotion.NEUTRAL)
                    elif ev.type == LLMEventType.TEXT_DELTA:
                        if timings.first_token_at is None:
                            timings.first_token_at = now_ms()
                        timings.last_token_at = now_ms()
                        ai_parts.append(ev.text)
                        await text_q.put(ev.text)
                    elif ev.type == LLMEventType.STEP_DONE:
                        flags["step_done"] = True
                    elif ev.type == LLMEventType.END_CALL:
                        flags["end_call"] = True
                    elif ev.type == LLMEventType.SAFETY_BLOCK:
                        await ensure_tts(AiEmotion.NEUTRAL)
                        ai_parts.append(_SAFETY_FALLBACK)
                        await text_q.put(_SAFETY_FALLBACK)
                    elif ev.type == LLMEventType.ERROR:
                        flags["error"] = True
            finally:
                await text_q.put(None)
                emotion_ready.set()

        async def consume() -> None:
            await emotion_ready.wait()
            tts = box["tts"]
            if tts is None:
                return

            async def text_source():
                while True:
                    chunk = await text_q.get()
                    if chunk is None:
                        return
                    yield chunk

            self._state = _State.SPEAKING
            async for pcm in tts.stream(text_source()):
                if not self._ws_alive:
                    break
                try:
                    await self._ws.send_bytes(pcm)
                except Exception:
                    self._ws_alive = False
                    break
                if timings.first_pcm_at is None:
                    timings.first_pcm_at = now_ms()
                timings.last_pcm_at = now_ms()
            await self._send_json(speaking_end_frame())

        try:
            await asyncio.gather(produce(), consume())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("턴 실행 중 예외", extra={"session_id": self._session_id})
            flags["error"] = True
        finally:
            await self._cleanup_turn_tts(connect_task, box["tts"])

        await self._finalize_turn(user_utterance, ai_parts, flags, timings)

    async def _cleanup_turn_tts(
        self, connect_task: asyncio.Task, used: TTSSession | None
    ) -> None:
        """턴이 끝나거나 취소되면 TTS ws 닫기"""
        if used is not None:
            await used.aclose()
            return
        if connect_task.done():
            if not connect_task.cancelled():
                with suppress(Exception):
                    session = connect_task.result()
                    await session.aclose()
        else:
            connect_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                session = await connect_task
                await session.aclose()

    async def _finalize_turn(
        self,
        user_utterance: str,
        ai_parts: list[str],
        flags: dict,
        timings: _TurnTimings,
    ) -> None:
        log_metric(
            "voice_turn",
            session_id=self._session_id,
            step=self._current_step,
            error=flags["error"],
            **timings.as_metrics(),
        )
        ai_text = "".join(ai_parts).strip()
        logger.info(
            "턴 완료 step=%s user=%r ai=%r",
            self._current_step,
            _clip(user_utterance),
            _clip(ai_text),
            extra={"session_id": self._session_id},
        )
        self._history.append({"role": "user", "text": user_utterance})
        if ai_text:
            self._history.append({"role": "assistant", "text": ai_text})

        if flags["error"]:
            await self._close(EndReason.ERROR)
            return
        if flags["end_call"]:
            await self._close(EndReason.END_CALL)
            return
        if flags["step_done"]:
            if self._current_step < self._script_len:
                self._current_step += 1
            elif self._script_len:
                await self._close(EndReason.SCENARIO_DONE)
                return

        self._turn_task = None
        self._state = _State.LISTENING
        self._listening_since = time.monotonic()
        self._open_user_turn()

    async def _barge_in(self) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        self._turn_task = None
        await self._send_json(interrupt_frame())
        self._state = _State.LISTENING
        self._open_user_turn()
        logger.info("barge-in 처리", extra={"session_id": self._session_id})

    async def _max_duration_timer(self) -> None:
        """종료"""
        assert self._max_duration is not None
        await asyncio.sleep(self._max_duration)
        self._time_up = True
        logger.info(
            "최대 시간 도달, 현재 턴 마무리 후 종료",
            extra={"session_id": self._session_id},
        )
        turn = self._turn_task
        if turn is not None and not turn.done():
            with suppress(asyncio.CancelledError, Exception):
                await turn
        await self._close(EndReason.TIMEOUT)

    async def _close(self, reason: EndReason) -> None:
        if self._closing.is_set():
            return
        if self._state == _State.LISTENING and self._listening_since is not None:
            self._silence_total += time.monotonic() - self._listening_since
            self._listening_since = None
        self._end_reason = reason
        self._state = _State.CLOSING
        self._closing.set()
        logger.info(
            "파이프라인 종료 트리거",
            extra={"session_id": self._session_id, "reason": reason.value},
        )

    async def _teardown(self, *tasks: asyncio.Task | None) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._turn_task

        for task in tasks:
            if task is not None:
                task.cancel()
        for task in tasks:
            if task is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await task

        reason = self._end_reason or EndReason.USER_END

        self._close_user_turn()

        shake_count = 0
        good_segments: list[dict] = []
        if self._tremor_buf:
            try:
                result = await asyncio.to_thread(
                    self._tremor.analyze, bytes(self._tremor_buf)
                )
                shake_count = result.shake_count
                good_segments = self._pick_good_segments(result.good_candidates)
            except Exception:
                logger.warning(
                    "떨림 분석 실패",
                    extra={"session_id": self._session_id},
                    exc_info=True,
                )
            finally:
                self._tremor_buf.clear()

        feedback = {
            "shake_count": shake_count,
            "silence_total": self._silence_total,
            "good_segments": good_segments,
        }

        if self._ws_alive:
            if reason == EndReason.ERROR:
                await self._send_json(error_frame("PIPELINE_ERROR"))
            else:
                await self._send_json(end_frame(reason, feedback))
            with suppress(Exception):
                await self._ws.close()

        await self._spring.notify_session_closed(
            self._session_id,
            reason=reason,
            transcript=self._history,
            silence_total=self._silence_total,
            shake_count=shake_count,
            good_segments=good_segments,
        )

    async def _send_json(self, payload: dict) -> None:
        """송신 도움"""
        if not self._ws_alive:
            return
        try:
            await self._ws.send_json(payload)
        except (WebSocketDisconnect, RuntimeError) as e:
            self._ws_alive = False
            logger.info(
                "ws 송신 중단(클라이언트 끊김): %s",
                type(e).__name__,
                extra={"session_id": self._session_id},
            )
        except Exception:
            self._ws_alive = False
            logger.warning(
                "프레임 송신 실패",
                exc_info=True,
                extra={"session_id": self._session_id},
            )

    def _buf_sec(self) -> float:
        return len(self._tremor_buf) / (16000 * 2)

    def _open_user_turn(self) -> None:
        self._turn_open_at = self._buf_sec()

    def _close_user_turn(self) -> None:
        if self._turn_open_at is not None:
            self._user_turn_intervals.append((self._turn_open_at, self._buf_sec()))
            self._turn_open_at = None

    @staticmethod
    def _intersect(a, b):
        """구간 리스트 a, b의 겹치는 부분만 반환."""
        out = []
        for s, e in a:
            for bs, be in b:
                lo, hi = max(s, bs), min(e, be)
                if hi > lo:
                    out.append((lo, hi))
        return out

    def _pick_good_segments(self, candidates) -> list[dict]:
        """잘한 구간 후보(발화−떨림) ∩ 사용자 턴 → 가장 긴 3개 (시간순)."""
        good = self._intersect(candidates, self._user_turn_intervals)
        good = [(s, e) for s, e in good if e - s >= 1.0]   # min_good_sec
        good.sort(key=lambda se: se[1] - se[0], reverse=True)
        good = good[:3]
        good.sort(key=lambda se: se[0])
        return [{"start": round(s, 2), "end": round(e, 2)} for s, e in good]
