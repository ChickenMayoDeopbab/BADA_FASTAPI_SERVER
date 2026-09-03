import asyncio
import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from itertools import zip_longest

from fastapi import WebSocket, WebSocketDisconnect
from google.api_core.exceptions import GoogleAPICallError

from app.core.config import Settings
from app.core.enums import SessionType
from app.core.metrics import log_metric, now_ms
from app.core.tts_voices import resolve_voice
from app.schemas.frames import (
    EndReason,
    NoticeCode,
    TranscriptRole,
    emotion_frame,
    end_frame,
    error_frame,
    interrupt_frame,
    notice_frame,
    script_hint_frame,
    speaking_end_frame,
    transcript_frame,
)
from app.schemas.llm import AiEmotion, LLMEventType, TurnContext
from app.services.feedback_points import Band, is_safe, voice_band
from app.services.feedback_service import save_feedback
from app.services.llm import LLMClient
from app.services.qwen_tts import (
    QwenRealtimeTTSClient,
    QwenRealtimeTTSSession,
    QwenTTSUnavailableError,
    try_acquire_realtime_tts,
)
from app.services.recording_storage import RecordingStorageService
from app.services.session import (
    build_turn_context,
    parse_difficulty,
    parse_scenario_id,
    parse_user_id,
)
from app.services.spring_client import SpringInternalClient
from app.services.stt import (
    AUDIO_EOS,
    GoogleSTTClient,
    STTEventType,
    STTIdleTimeoutError,
    STTStreamAbortedError,
)
from app.services.training_analysis import TrainingPerformanceAnalyzer
from app.services.tremor import TremorAnalyzer
from app.services.tts import ElevenLabsTTSClient, TTSSession
from app.workers.avti_worker import run_avti

logger = logging.getLogger(__name__)

# barge-in off 해뒀음
# OFF면 AI가 발화중에 입력 무시 나중에 제대로 구현 시 True로 고민
_BARGE_IN_ENABLED = False
_BARGE_IN_MIN_CHARS = 2
_STT_RECYCLE_SECONDS = 240.0
# 첫 오디오 청크 대기 상한. 초과 시 NO_AUDIO 정상 종료 (Google 의존이던 무오디오 감지를 서버 주도로 대체).
# 긴 TTS 재생(~20s) 중 클라 mute 를 견디도록 여유 있게 잡음. 최후 방어는 세션 최대시간 타이머.
_STT_NO_AUDIO_TIMEOUT = 60.0
_STT_NO_AUDIO_WARNING_LEAD = 30.0
_NO_AUDIO_WARNING_TEXT = "목소리가 들리지 않아요. 계속 조용하면 통화가 잠시 후 종료돼요."
_SAFETY_FALLBACK = "죄송해요, 그 부분은 지금 답하기 어렵네요."
_TURN_WATCHDOG_SECONDS = 30.0
_TURN_FALLBACK_TEXT = "죄송해요, 다시 한번 말씀해 주시겠어요?"
_FALLBACK_TTS_TIMEOUT = 8.0
_AVTI_WAIT_SECONDS = 20.0
# 구간 피드백
_MAX_SEGMENTS = 3
_MIN_GOOD_SEC = 1.0
_SEG_GOOD = "GOOD"
_SEG_IMPROVE = "IMPROVE"
# 구간을 고르는 데만 쓰는 키. 밖으로 나가기 전에 털어낸다.
_INTERNAL_SEGMENT_KEYS = ("turn", "avti", "reason")


def _segment_fallback(kind: str) -> tuple[str, str]:
    """LLM 이 못 쓰거나 걸러졌을 때 구간에 남길 최소 문구."""
    if kind == _SEG_IMPROVE:
        return ("이 구간을 다시 들어봐요", "말이 흔들린 부분이에요. 다음엔 여기서 한 호흡 쉬고 말해봐요.")
    return ("잘 이야기했어요", "이 구간은 흔들림 없이 이어서 말했어요.")


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
        self._difficulty = parse_difficulty(session)
        self._tts = ElevenLabsTTSClient(settings, self._difficulty)
        self._qwen_tts: QwenRealtimeTTSClient | None = None
        self._recording_storage = RecordingStorageService(settings)

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
        self._ai_pcm_bytes = 0
        self._server_wait_duration_ms = 0
        self._completed_script_steps = 0

        self._user_turn_intervals: list[tuple[float, float]] = []
        self._user_turn_texts: list[str] = []
        self._turn_open_at: float | None = 0.0

        max_duration = session.get("maxDurationSeconds")
        if max_duration is None and session.get("type") == SessionType.WARMUP:
            max_duration = 30  # Spring이 안 박아도 워밍업은 30초 보장
        self._max_duration = int(max_duration) if max_duration else None

        scenario = session.get("scenario") or {}
        script = scenario.get("script") if isinstance(scenario, dict) else None
        self._script_len = len(script) if isinstance(script, list) else 0

        raw_voice = scenario.get("ttsVoiceId") if isinstance(scenario, dict) else None
        assigned_voice = (
            raw_voice if isinstance(raw_voice, str) and raw_voice.strip() else None
        )
        self._voice_id_override = resolve_voice(assigned_voice, self._difficulty)

    async def run(self) -> None:
        """진입점"""
        await self._init_qwen_tts()
        warmup_task = asyncio.create_task(self._llm.warmup())
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
            await self._teardown(*tasks, warmup_task)
            self._release_qwen_slot()

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
                try:
                    await self._consume_one_stream(self._audio_queue)
                except STTStreamAbortedError:
                    # 무요청 ABORTED는 복구 가능 — 세션 유지, 스트림만 재오픈
                    logger.info(
                        "STT 무요청 ABORTED — 스트림 재오픈",
                        extra={"session_id": self._session_id},
                    )
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
        """한 스트림을 FINAL까지 소비하고 닫음. 첫 오디오 도착 전엔 gRPC 스트림을 열지 않는다(지연 오픈)."""
        try:
            first_chunk = await self._wait_first_chunk(queue)
        except TimeoutError:
            raise STTIdleTimeoutError from None
        if first_chunk is AUDIO_EOS:
            return

        recycle_at = time.monotonic() + _STT_RECYCLE_SECONDS
        stream = self._stt.stream(queue, first_chunk=first_chunk)
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

    async def _wait_first_chunk(
        self, queue: "asyncio.Queue[bytes | None]"
    ) -> bytes | None:
        """첫 청크 대기"""
        warn_after = _STT_NO_AUDIO_TIMEOUT - _STT_NO_AUDIO_WARNING_LEAD
        if warn_after <= 0:
            return await asyncio.wait_for(queue.get(), _STT_NO_AUDIO_TIMEOUT)
        try:
            return await asyncio.wait_for(queue.get(), warn_after)
        except TimeoutError:
            pass
        if self._state == _State.LISTENING:
            await self._send_json(
                notice_frame(NoticeCode.NO_AUDIO_WARNING, _NO_AUDIO_WARNING_TEXT)
            )
        return await asyncio.wait_for(queue.get(), _STT_NO_AUDIO_WARNING_LEAD)

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
                await self._send_json(transcript_frame(TranscriptRole.USER, text))
                self._start_turn(text, final_at=now_ms())
        elif event.type == STTEventType.SPEECH_BEGIN and (
                self._listening_since is not None and
                time.monotonic() - self._listening_since > 1.5 and
                self._state == _State.LISTENING):
            self._silence_total += time.monotonic() - self._listening_since
            self._listening_since = None

    def _start_turn(self, user_utterance: str, *, final_at: float) -> None:
        """LLM -> TTS"""
        self._close_user_turn(user_utterance)
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

        connect_task = asyncio.create_task(self._open_tts())
        text_q: asyncio.Queue[str | None] = asyncio.Queue()
        emotion_ready = asyncio.Event()
        ai_parts: list[str] = []
        flags = {
            "step_done": False,
            "end_call": False,
            "error": False,
            "tts_engine": "qwen" if getattr(self, "_qwen_tts", None) else "eleven",
        }
        usage: dict[str, int | None] = {"prompt": None, "cached": None}
        box: dict[str, TTSSession | QwenRealtimeTTSSession | None] = {"tts": None}
        suggestion_box: dict[str, str | None] = {"text": None}

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
                    elif ev.type == LLMEventType.TURN_END:
                        usage["prompt"] = ev.prompt_tokens
                        usage["cached"] = ev.cached_tokens
                    elif ev.type == LLMEventType.STEP_DONE:
                        flags["step_done"] = True
                    elif ev.type == LLMEventType.END_CALL:
                        flags["end_call"] = True
                    elif ev.type == LLMEventType.SUGGESTION:
                        suggestion_box["text"] = ev.text
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
                    self._ai_pcm_bytes += len(pcm)
                except Exception:
                    self._ws_alive = False
                    break
                if timings.first_pcm_at is None:
                    timings.first_pcm_at = now_ms()
                timings.last_pcm_at = now_ms()
            await self._send_json(speaking_end_frame())

        produce_task = asyncio.create_task(produce())
        consume_task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(
                asyncio.gather(produce_task, consume_task),
                timeout=_TURN_WATCHDOG_SECONDS,
            )
        except TimeoutError:
            flags["watchdog"] = True
            logger.warning(
                "AI 응답이 %.0f초 안에 안 와서 이번 턴은 건너뛰고 듣기 상태로 전환",
                _TURN_WATCHDOG_SECONDS,
                extra={"session_id": self._session_id},
            )
        except asyncio.CancelledError:
            flags["cancelled"] = True
            raise
        except QwenTTSUnavailableError as exc:
            flags["tts_failed"] = True
            self._switch_to_eleven("synth_failed")
            logger.warning(
                "Qwen TTS 실패 — 이번 턴은 건너뛰고 ElevenLabs로 전환: %s",
                exc,
                extra={"session_id": self._session_id},
            )
        except Exception:
            logger.exception("턴 실행 중 예외", extra={"session_id": self._session_id})
            flags["error"] = True
        finally:
            for task in (produce_task, consume_task):
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task
            await self._cleanup_turn_tts(connect_task, box["tts"])
            if flags.get("cancelled"):
                await self._salvage_cancelled_turn(user_utterance, ai_parts)

        await self._finalize_turn(
            user_utterance,
            ai_parts,
            flags,
            timings,
            usage,
            ctx=ctx,
            suggestion=suggestion_box["text"],
        )

    async def _salvage_cancelled_turn(
        self, user_utterance: str, ai_parts: list[str]
    ) -> None:
        """취소된 턴(세션 종료/barge-in)도 나눈 대화는 히스토리에 남김"""
        self._history.append({"role": "user", "text": user_utterance})
        ai_text = "".join(ai_parts).strip()
        if ai_text:
            self._history.append({"role": "assistant", "text": ai_text})
            await self._send_json(transcript_frame(TranscriptRole.AI, ai_text))

    async def _init_qwen_tts(self) -> None:
        """elevenlabs 갈지 qwen 쓸지"""
        client, skip_reason = await try_acquire_realtime_tts(self._settings)
        self._qwen_tts = client
        log_metric(
            "realtime_tts_engine",
            session_id=self._session_id,
            engine="qwen" if client is not None else "eleven",
            skip_reason=skip_reason,
        )

    def _switch_to_eleven(self, reason: str) -> None:
        """통화 중 Qwen 장애 시 elevenlabs로 전환"""
        qwen = getattr(self, "_qwen_tts", None)
        if qwen is None:
            return
        qwen.release_slot()
        self._qwen_tts = None
        log_metric(
            "realtime_tts_switch", session_id=self._session_id, reason=reason
        )

    def _release_qwen_slot(self) -> None:
        qwen = getattr(self, "_qwen_tts", None)
        if qwen is not None:
            qwen.release_slot()
            self._qwen_tts = None

    async def _open_tts(self) -> TTSSession | QwenRealtimeTTSSession:
        """시나리오 보이스 연결"""
        qwen = getattr(self, "_qwen_tts", None)
        if qwen is not None:
            try:
                return await qwen.open()
            except QwenTTSUnavailableError:
                logger.warning(
                    "Qwen TTS 세션 열기 실패 — ElevenLabs 로 전환",
                    exc_info=True,
                    extra={"session_id": self._session_id},
                )
                self._switch_to_eleven("open_failed")
        voice_id_override = getattr(self, "_voice_id_override", None)
        if voice_id_override is None:
            return await self._tts.open()
        try:
            return await self._tts.open(voice_id_override)
        except Exception:
            logger.warning(
                "시나리오 보이스 연결 실패, 기본 보이스 폴백: %s",
                voice_id_override,
            )
            return await self._tts.open()

    async def _cleanup_turn_tts(
        self,
        connect_task: asyncio.Task,
        used: TTSSession | QwenRealtimeTTSSession | None,
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
        usage: dict[str, int | None],
        *,
        ctx: TurnContext,
        suggestion: str | None = None,
    ) -> None:
        ai_text = "".join(ai_parts).strip()

        if timings.first_pcm_at is not None:
            response_wait_ms = max(
                0,
                int(round(
                    timings.first_pcm_at
                    - timings.final_at
                )),
            )

            self._server_wait_duration_ms += response_wait_ms

        if flags["step_done"] and self._script_len:
            self._completed_script_steps = min(
                self._completed_script_steps + 1,
                self._script_len,
            )

        scenario_finished = bool(
            flags["step_done"]
            and self._script_len
            and self._current_step >= self._script_len
        )
        will_close = flags["error"] or flags["end_call"] or scenario_finished
        fallback = not will_close and (
            flags.get("watchdog", False)
            or flags.get("tts_failed", False)
            or not ai_text
        )
        log_metric(
            "voice_turn",
            session_id=self._session_id,
            step=self._current_step,
            error=flags["error"],
            watchdog=flags.get("watchdog", False),
            fallback=fallback,
            tts_engine=flags.get("tts_engine"),
            tts_failed=flags.get("tts_failed", False),
            llm_prompt_tokens=usage["prompt"],
            llm_cached_tokens=usage["cached"],
            **timings.as_metrics(),
        )
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
            await self._send_json(transcript_frame(TranscriptRole.AI, ai_text))

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

        if fallback:
            await self._play_turn_fallback()
        if not flags.get("watchdog"):
            await self._maybe_send_script_hint(ctx, suggestion)

        self._turn_task = None
        self._state = _State.LISTENING
        self._listening_since = time.monotonic()
        self._open_user_turn()

    async def _play_turn_fallback(self) -> None:
        await self._send_json(
            notice_frame(NoticeCode.TURN_FALLBACK, _TURN_FALLBACK_TEXT)
        )
        try:
            await asyncio.wait_for(self._speak_fallback(), _FALLBACK_TTS_TIMEOUT)
        except TimeoutError:
            logger.warning(
                "폴백 멘트 TTS 시간 초과 — 멘트 없이 진행",
                extra={"session_id": self._session_id},
            )
        except Exception:
            logger.warning(
                "폴백 멘트 TTS 실패 — 멘트 없이 진행",
                exc_info=True,
                extra={"session_id": self._session_id},
            )
        if self._history and self._history[-1]["role"] == "assistant":
            self._history[-1]["text"] += " " + _TURN_FALLBACK_TEXT
        else:
            self._history.append({"role": "assistant", "text": _TURN_FALLBACK_TEXT})
        await self._send_json(transcript_frame(TranscriptRole.AI, _TURN_FALLBACK_TEXT))
        await self._send_json(speaking_end_frame())

    async def _speak_fallback(self) -> None:
        session = await self._open_tts()
        try:
            await session.begin(AiEmotion.NEUTRAL)
            await self._send_json(emotion_frame(AiEmotion.NEUTRAL))

            async def _text():
                yield _TURN_FALLBACK_TEXT

            self._state = _State.SPEAKING
            async for pcm in session.stream(_text()):
                if not self._ws_alive:
                    break
                try:
                    await self._ws.send_bytes(pcm)
                    self._ai_pcm_bytes += len(pcm)
                except Exception:
                    self._ws_alive = False
                    break
        finally:
            with suppress(Exception):
                await session.aclose()

    async def _maybe_send_script_hint(
        self, ctx: TurnContext, suggestion: str | None
    ) -> None:
        if ctx.script_level not in (1, 2):
            return
        text = (suggestion or "").strip()
        if not text:
            text = next(
                (t.hint.strip() for t in ctx.script if t.step == self._current_step),
                "",
            )
        if not text:
            return
        await self._send_json(script_hint_frame(ctx.script_level, text))

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
        good_candidates: list = []
        recording_key: str | None = None
        recording_pcm = b""
        sustained_spans: list = []
        tremor_result = None
        analysis_failed = False
        if self._tremor_buf:
            recording_pcm = bytes(self._tremor_buf)
            try:
                recording_key = await asyncio.to_thread(
                    self._recording_storage.upload_pcm,
                    self._session_id,
                    recording_pcm,
                )
            except Exception:
                logger.warning(
                    "녹음본 업로드 실패",
                    extra={"session_id": self._session_id},
                    exc_info=True,
                )

            try:
                tremor_result = await asyncio.to_thread(
                    self._tremor.analyze, recording_pcm
                )
                shake_count = tremor_result.shake_count
                good_candidates = tremor_result.good_candidates
                sustained_spans = tremor_result.sustained_spans
            except Exception:
                analysis_failed = True
                logger.warning(
                    "떨림 분석 실패",
                    extra={"session_id": self._session_id},
                    exc_info=True,
                )
            finally:
                self._tremor_buf.clear()

        # 턴별 AVTI 를 먼저 잰다. 어느 구간을 칭찬하고 어느 구간을 짚을지가
        # 이 값으로 갈리기 때문에 end 프레임보다 앞에 있어야 한다(약 0.2초).
        avti_by_turn = await self._measure_avti(recording_pcm, sustained_spans)
        good_segments = self._pick_segments(good_candidates, avti_by_turn)

        analysis = None
        try:
            analysis = TrainingPerformanceAnalyzer().analyze(
                session_type=self._session.get("type"),
                reason=reason,
                user_turn_intervals=getattr(
                    self,
                    "_user_turn_intervals",
                    [],
                ),
                user_turn_texts=getattr(
                    self,
                    "_user_turn_texts",
                    [],
                ),
                tremor_result=tremor_result,
                completed_script_steps=self._completed_script_steps,
                script_step_count=self._script_len,
                ai_pcm_bytes=self._ai_pcm_bytes,
                server_wait_duration_ms=self._server_wait_duration_ms,
                analyzer_failed=analysis_failed,
            )
        except Exception:
            logger.warning(
                "훈련 성과 분석 실패 - 분석 없이 종료 콜백 진행",
                extra={"session_id": self._session_id},
                exc_info=True,
            )

        feedback = {
            "shake_count": shake_count,
            "silence_total": self._silence_total,
            # 문구는 아직 없다(end 프레임 뒤에 채운다). 판단 재료(turn·avti·reason)는
            # 내부용이라 클라이언트로 내보내지 않는다.
            "good_segments": [
                {"start": seg["start"], "end": seg["end"], "type": seg["type"]}
                for seg in good_segments
            ],
        }

        # Spring의 훈련 기록에 완성된 구간 피드백을 포함하기 위해 콜백 전에 채운다.
        await self._write_segment_feedback(good_segments)

        await self._save_feedback(shake_count, good_segments)

        session_closed = await self._spring.notify_session_closed(
            self._session_id,
            reason=reason,
            transcript=self._history,
            silence_total=self._silence_total,
            shake_count=shake_count,
            good_segments=good_segments,
            recording_key=recording_key,
            session_type=self._session.get("type"),
            analysis=analysis,
        )

        # 정상 종료 이벤트는 Spring 트랜잭션이 완료된 뒤에만 보낸다. 앱은 이
        # 이벤트를 기준으로 training_record 의 후속 API를 호출한다.
        if self._ws_alive:
            if not session_closed:
                await self._send_json(error_frame("SESSION_CLOSE_FAILED"))
            elif reason == EndReason.ERROR:
                await self._send_json(error_frame("PIPELINE_ERROR"))
            else:
                await self._send_json(end_frame(reason, feedback))
            with suppress(Exception):
                await self._ws.close()

    async def _measure_avti(
        self, recording_pcm: bytes, sustained_spans: list
    ) -> dict[int, float]:
        """턴별 AVTI 측정. 늦어지거나 터져도 피드백은 그대로 나간다.

        {0: 세션 전체, 1..n: 사용자 턴}. 못 잰 턴은 키가 없다 —
        열 턴에 한 번쯤만 값이 나오므로 없는 게 기본이라고 보면 된다.
        """
        if not recording_pcm:
            return {}
        parts = [
            (0.0, len(recording_pcm) / (16000 * 2)),  # 0번은 세션 전체
            *self._user_turn_intervals,
        ]
        try:
            return await asyncio.wait_for(
                run_avti(
                    settings=self._settings,
                    session_id=self._session_id,
                    pcm=recording_pcm,
                    parts=parts,
                    sustained_spans=sustained_spans,
                ),
                timeout=_AVTI_WAIT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.warning("AVTI 측정 지연 — 목소리 근거 없이 진행",
                           extra={"session_id": self._session_id})
        except Exception:
            logger.warning("AVTI 측정 실패 — 목소리 근거 없이 진행",
                           exc_info=True, extra={"session_id": self._session_id})
        return {}

    async def _write_segment_feedback(self, segments: list[dict]) -> None:
        """구간마다 title / content 를 채운다. 실패해도 구간 자체는 남는다."""
        if not segments:
            return

        scenario = self._session.get("scenario") or {}
        if not isinstance(scenario, dict):
            scenario = {}

        items = [
            {
                "type": seg["type"],
                "utterance": self._utterance_for(seg["start"], seg["end"]),
                "avti": seg.get("avti"),
                "reason": seg.get("reason"),
            }
            for seg in segments
        ]

        try:
            written = await self._llm.segment_feedback(
                items,
                scenario_title=str(scenario.get("title", "")),
                call_target=str(scenario.get("callTarget", "")),
                call_purpose=str(scenario.get("callPurpose", "")),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("구간 피드백 생성 실패", exc_info=True,
                           extra={"session_id": self._session_id})
            written = []

        for seg, pair in zip_longest(segments, written, fillvalue=None):
            if seg is None:
                break
            title, content = pair if pair else ("", "")
            if not is_safe(title, content):
                title, content = _segment_fallback(seg["type"])
            seg["title"] = title
            seg["content"] = content
            seg["good_point"] = content  # 기존 Spring 매핑 유지
            for internal in _INTERNAL_SEGMENT_KEYS:
                seg.pop(internal, None)

    async def _save_feedback(
        self,
        shake_count: int,
        good_segments: list[dict],
    ) -> None:
        scenario_id = parse_scenario_id(self._session)
        user_id = parse_user_id(self._session)
        if scenario_id is None or user_id is None:
            # Spring 이 세션에 scenarioId 를 넣기 전이면 여기로 온다. 배포 순서 무관하게 안전.
            logger.warning(
                "피드백 저장 스킵 — 세션에 scenarioId/userId 없음",
                extra={
                    "session_id": self._session_id,
                    "has_scenario_id": scenario_id is not None,
                    "has_user_id": user_id is not None,
                },
            )
            return

        await save_feedback(
            session_id=self._session_id,
            user_id=user_id,
            scenario_id=scenario_id,
            shake_count=shake_count,
            silence_total=self._silence_total,
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

    def _close_user_turn(self, text: str = "") -> None:
        if self._turn_open_at is not None:
            self._user_turn_intervals.append((self._turn_open_at, self._buf_sec()))
            self._user_turn_texts.append(text)
            self._turn_open_at = None

    def _utterance_for(self, start: float, end: float) -> str:
        """[start, end] 구간과 가장 많이 겹치는 사용자 발화 텍스트를 찾는다."""
        best_text, best_overlap = "", 0.0
        for (s, e), text in zip(
            self._user_turn_intervals, self._user_turn_texts, strict=True
        ):
            overlap = min(end, e) - max(start, s)
            if overlap > best_overlap:
                best_overlap, best_text = overlap, text
        return best_text

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

    def _spoken_turns(
        self, avti_by_turn: dict[int, float]
    ) -> list[tuple[int, tuple[float, float]]]:
        """말한 턴만 (턴 번호, 구간) 으로. 번호는 AVTI 키와 맞춰 그대로 둔다.

        STT 가 아무것도 못 받아적은 턴은 마이크만 열려 있었을 뿐이라 뺀다.
        예외는 AVTI 가 나온 턴 — 그 값은 3초 넘게 이어진 목소리에서만 나오므로
        말은 했는데 문장이 못 넘어온 경우다(끝에서 세션이 끊길 때).
        """
        turns: list[tuple[int, tuple[float, float]]] = []
        for index, (start, end) in enumerate(self._user_turn_intervals, start=1):
            if end - start <= 0:
                continue
            said = ""
            if index - 1 < len(self._user_turn_texts):
                said = self._user_turn_texts[index - 1].strip()
            if not said and avti_by_turn.get(index) is None:
                logger.info(
                    "발화 없는 턴 - 구간 피드백 후보에서 제외",
                    extra={
                        "session_id": self._session_id,
                        "turn": index,
                        "start": round(start, 2),
                        "end": round(end, 2),
                    },
                )
                continue
            turns.append((index, (start, end)))
        return turns

    def _pick_segments(self, candidates, avti_by_turn: dict[int, float]) -> list[dict]:
        """칭찬할 구간과 아쉬운 구간을 함께 고른다. 발화 없는 턴은 제외한다."""
        turns = self._spoken_turns(avti_by_turn)
        intervals = [span for _, span in turns]
        clean = self._intersect(candidates, intervals)
        picked: list[dict] = []

        for index, (start, end) in turns:
            avti = avti_by_turn.get(index)
            band = voice_band(avti)
            if band is Band.NEEDS_WORK:
                kind, reason = _SEG_IMPROVE, "voice"
            elif band is Band.GOOD:
                kind, reason = _SEG_GOOD, "voice"
            elif band is Band.MIDDLE:
                continue  # 애매한 값에서는 이 턴을 아예 안 고른다
            else:
                # AVTI 가 없는 턴 — 떨림 없이 이어 말한 구간이 있으면 칭찬 후보
                overlap = sum(
                    max(0.0, min(end, ce) - max(start, cs)) for cs, ce in clean
                )
                if overlap < _MIN_GOOD_SEC:
                    continue
                kind, reason = _SEG_GOOD, "clean"
            picked.append({
                "turn": index,
                "start": round(start, 2),
                "end": round(end, 2),
                "type": kind,
                "reason": reason,
                "avti": avti,
            })

        # 아쉬운 구간을 먼저 살리고, 남는 자리를 긴 칭찬 구간으로 채운다.
        improve = [p for p in picked if p["type"] == _SEG_IMPROVE]
        good = sorted(
            (p for p in picked if p["type"] == _SEG_GOOD),
            key=lambda p: p["end"] - p["start"],
            reverse=True,
        )
        chosen = (improve + good)[:_MAX_SEGMENTS]
        if not chosen:
            chosen = [
                {**seg, "type": _SEG_GOOD, "reason": "fallback", "avti": None}
                for seg in self._pick_good_segments(candidates, intervals)
            ]
        chosen.sort(key=lambda p: p["start"])
        return chosen

    def _pick_good_segments(self, candidates, turn_intervals=None) -> list[dict]:
        """마지막 보루. turn_intervals 를 주면 그 턴들 안에서만 고른다."""
        turn_intervals = (
            self._user_turn_intervals if turn_intervals is None else turn_intervals
        )
        overlap = self._intersect(candidates, turn_intervals)
        overlap.sort(key=lambda se: se[1] - se[0], reverse=True)
        long_enough = [se for se in overlap if se[1] - se[0] >= 1.0]  # min_good_sec
        fragments = [se for se in overlap if se[1] - se[0] < 1.0]
        turns = sorted(turn_intervals, key=lambda se: se[1] - se[0], reverse=True)

        good: list[tuple[float, float]] = []
        for pool in (long_enough, fragments, turns):
            for s, e in pool:
                if len(good) == 3:
                    break
                if e - s <= 0:
                    continue
                if any(min(e, ge) > max(s, gs) for gs, ge in good):
                    continue
                good.append((s, e))

        good.sort(key=lambda se: se[0])
        return [{"start": round(s, 2), "end": round(e, 2)} for s, e in good]
