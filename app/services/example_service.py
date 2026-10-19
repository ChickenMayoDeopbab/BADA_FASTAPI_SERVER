from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import time
import weakref
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from app.core.concurrency import loop_semaphore
from app.core.config import Settings, get_settings
from app.core.enums import Difficulty
from app.core.metrics import log_metric, now_ms
from app.core.preset_scenarios import PRESET_MAP
from app.core.tts_voices import el_voice_for, pick_example_user_voice, profile_for, qwen_voice_for, resolve_voice
from app.core.usage import log_llm_usage, log_tts_usage
from app.db.base import AsyncSessionLocal
from app.db.models import ScenarioORM, is_deleted
from app.schemas.scenario import ExampleConversationResponse, ExampleTurn
from app.services import example_audio
from app.services.qwen_tts import (
    QwenCapHitError,
    QwenTTSClient,
    QwenTTSUnavailableError,
    QwenUnreachableError,
    QwenVoiceMissingError,
    has_free_worker,
    is_voice_ref,
    worker_urls,
)
from app.services.recording_storage import RecordingStorageService
from app.services.tts import ElevenLabsTTSClient

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_TURN_GAP_MS = 400
_TURN_GAP_PCM = b"\x00" * (_SAMPLE_RATE * _TURN_GAP_MS // 1000 * 2)
_TTS_MAX_CONCURRENCY = 3
# 대본 생성(LLM) + 12턴 합성(실측 ~60초)에 여유를 둔 상한
_PREBAKE_TIMEOUT_SEC = 180.0
_prebake_semaphore = loop_semaphore(1)

# 사용 중인 요청(지역변수)만 락을 강참조 → 요청이 끝나면 GC가 엔트리를 자동 제거
_generation_locks: weakref.WeakValueDictionary[int, asyncio.Lock] = weakref.WeakValueDictionary()


class ScenarioNotFoundError(Exception):
    """시나리오가 없거나 요청자가 접근할 수 없는 경우"""


class QwenNonStreamingMissingError(QwenTTSUnavailableError):
    """워커가 non_streaming_mode 를 적용하지 않음"""


class QwenVoiceRefError(QwenTTSUnavailableError):
    """워커가 알린 ref 지문이 없거나 기대와 다름"""


_NS_METHOD = "ns1"


_EXAMPLE_GEN_SYSTEM = """You are writing a model example phone conversation for a Korean \
phone call training application.
The user provides the scenario (title, call purpose, call target) and step-by-step script goals.
Write one successful example call in natural, polite spoken Korean (normal personality).
You must output ONLY a valid JSON array. No markdown, no code fences, no extra text.
Each element: {"speaker": "ai" | "user", "text": "Korean utterance"}
Rules:
- "ai" is the callee (the call target answering the phone), "user" is the caller (trainee).
- The first turn is "ai" answering the phone; speakers strictly alternate.
- 8 to 12 turns total, following the script steps in order, ending with a natural closing.
- Plain spoken dialogue only. No stage directions, no parentheses.
- Use concrete example values (dates, times, names) where natural.
- In "ai" lines, write the honorific people counter 분 with Korean number words ('네 분', '열 분'), \
never digits like '4분' — TTS reads '4분' as minutes. The "user" counts their own party with 명 ('4명').
"""


def _normalize_dialogue(raw: object) -> list[dict]:
    """LLM이 생성한 예시 대화를 {speaker, text} 형태로 정규화"""
    if not isinstance(raw, list):
        raise ValueError("예시 대화 응답이 JSON 배열이 아닙니다.")
    turns: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker", "")).strip().lower()
        text = str(item.get("text", "")).strip()
        if speaker not in ("ai", "user") or not text:
            continue
        turns.append({"speaker": speaker, "text": text})
    if len(turns) < 2:
        raise ValueError("예시 대화 턴이 부족합니다.")
    return turns


async def _generate_custom_dialogue(settings: Settings, row: ScenarioORM) -> list[dict]:
    """커스텀 시나리오의 예시 대화를 LLM으로 생성"""
    steps = "\n".join(
        f"{t.get('step')}. {t.get('ai_goal', '')} (힌트: {t.get('hint', '')})"
        for t in (row.script or [])
        if isinstance(t, dict)
    )
    user_msg = (
        f"Scenario title: {row.title}\n"
        f"Call purpose: {row.call_purpose}\n"
        f"Call target: {row.call_target}\n"
        f"Script steps:\n{steps}"
    )
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    ai_response = await client.messages.create(
        model=settings.llm_analysis_model,
        max_tokens=1500,
        system=_EXAMPLE_GEN_SYSTEM,
        messages=[MessageParam(role="user", content=user_msg)],
    )
    log_llm_usage(
        "anthropic",
        settings.llm_analysis_model,
        "example_dialogue",
        usage=getattr(ai_response, "usage", None),
        user_id=getattr(row, "user_id", None),
        scenario_id=getattr(row, "scenario_id", None),
    )
    raw = ai_response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return _normalize_dialogue(json.loads(raw.strip()))


async def _one_text(text: str) -> AsyncIterator[str]:
    yield text


async def _synth_turn(
    tts_client: ElevenLabsTTSClient,
    voice_id: str,
    text: str,
    chars: list[int] | None = None,
) -> bytes:
    session = await tts_client.open(voice_id)
    chunks: list[bytes] = []
    try:
        await session.begin()
        async for pcm in session.stream(_one_text(text)):
            chunks.append(pcm)
    finally:
        await session.aclose()
        if chars is not None:
            sent = getattr(session, "chars_sent", None)
            chars.append(sent if isinstance(sent, int) else len(text))
    return b"".join(chunks)


async def _synthesize(
    tts_client: ElevenLabsTTSClient,
    dialogue: list[dict],
    ai_voice: str,
    user_voice: str,
    chars: list[int] | None = None,
) -> bytes:
    semaphore = asyncio.Semaphore(_TTS_MAX_CONCURRENCY)

    async def _synth_limited(turn: dict) -> bytes:
        voice = ai_voice if turn["speaker"] == "ai" else user_voice
        async with semaphore:
            return await _synth_turn(tts_client, voice, turn["text"], chars)

    parts = await asyncio.gather(*(_synth_limited(turn) for turn in dialogue))
    return _TURN_GAP_PCM.join(parts)


async def _synthesize_qwen(qwen_client: QwenTTSClient, dialogue: list[dict], ai_voice: str = "ai") -> bytes:
    """GPU 서버로 합성"""
    parts: list[bytes] = []
    async with qwen_client.connect() as session:  # 턴마다 TCP 재연결하지 않게
        for turn in dialogue:
            voice = ai_voice if turn["speaker"] == "ai" else "user"
            parts.append(await session.synth(voice, turn["text"]))
    return _TURN_GAP_PCM.join(parts)


def _audio_key(
    scenario_id: int,
    dialogue: list[dict],
    ai_voice: str,
    user_voice: str,
    *,
    qwen: bool = False,
) -> str:
    """대본이나 보이스가 바뀌면 키가 달라져 캐시가 자동 무효화"""
    voices = [f"qwen-{ai_voice}", f"qwen-{user_voice}"] if qwen else [ai_voice, user_voice]
    payload = json.dumps([dialogue, *voices], ensure_ascii=False)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    suffix = "-q" if qwen else ""
    return f"examples/{scenario_id}-{digest}{suffix}.wav"


def _ns_d8(dialogue: list[dict]) -> str:
    """대본 digest"""
    return hashlib.sha1(json.dumps([dialogue, "ns"], ensure_ascii=False).encode()).hexdigest()[:8]


def _ns_v8(voices: dict[str, str], refs: dict[str, str], levels: dict[str, float]) -> str:
    """렌더 설정 digest: 방식, 보이스 이름과 ref 지문, 크기 목표"""
    payload = {
        "method": _NS_METHOD,
        "ai": [voices["ai"], refs["ai"]],
        "user": [voices["user"], refs["user"]],
        "level": [levels["ai"], levels["user"]],
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8]


def _ns_key(scenario_id: int, dialogue: list[dict], v8: str) -> str:
    return f"examples/{scenario_id}-{_ns_d8(dialogue)}-{v8}-qn.wav"


def _ns_cached(scenario_id: int, dialogue: list[dict], cached: str | None) -> str | None:
    """대본이 같은 ns 키면 렌더 설정과 상관없이 인정"""
    if not cached:
        return None
    found = re.search(rf"examples/{scenario_id}-{_ns_d8(dialogue)}-[0-9a-f]{{8}}-qn\.wav$", cached)
    return found.group(0) if found else None


def _levels(settings: Settings) -> dict[str, float]:
    return {
        "ai": getattr(settings, "example_level_ai_db", example_audio.LEVEL_DB["ai"]),
        "user": getattr(settings, "example_level_user_db", example_audio.LEVEL_DB["user"]),
    }


async def _iter_turns(turns: Iterable[dict] | AsyncIterator[dict]) -> AsyncIterator[dict]:
    if hasattr(turns, "__aiter__"):
        async for turn in turns:
            yield turn
    else:
        for turn in turns:
            yield turn


def _check_ns(result, role: str, refs: dict[str, str], expect_refs: dict[str, str] | None) -> None:
    if not result.ns:
        raise QwenNonStreamingMissingError(f"non_streaming 미적용 worker={result.worker}")
    if not is_voice_ref(result.ref):
        raise QwenVoiceRefError(f"ref 지문 없음 worker={result.worker}")
    if role not in refs:
        if expect_refs is not None and expect_refs.get(role) != result.ref:
            raise QwenVoiceRefError(f"ref 불일치 {role}={result.ref} worker={result.worker}")
        refs[role] = result.ref
    elif refs[role] != result.ref:
        raise QwenVoiceRefError(f"굽는 중 ref 바뀜 {role} worker={result.worker}")


async def _synthesize_qwen_ns(
    qwen_client: QwenTTSClient,
    turns: Iterable[dict] | AsyncIterator[dict],
    voices: dict[str, str],
    *,
    levels: dict[str, float],
    expect_refs: dict[str, str] | None = None,
    on_render: Callable[[str, str, int, bytes], None] | None = None,
    on_turn: Callable[[float], None] | None = None,
    exclude: Iterable[str] = (),
) -> tuple[bytes, dict, dict[str, str]]:
    """non_streaming_mode 로 턴마다 다시 합성해 고르고 다듬어 크기를 맞춘다"""
    parts: list[bytes] = []
    gains: list[float] = []
    refs: dict[str, str] = {}
    info: dict = {"attempts": 0, "chars": 0, "retried_turns": 0, "residual_cuts": 0, "length_fails": 0,
                  "tail_after_rms_s_max": 0.0, "worker": None}
    async with qwen_client.connect(exclude=exclude) as session:
        async for turn in _iter_turns(turns):
            role, text = turn["speaker"], turn["text"]
            turn_started = time.monotonic()
            renders: list[example_audio.Render] = []
            tried = 0
            for tried in range(1, example_audio.MAX_ATTEMPTS + 1):
                info["attempts"] += 1
                info["chars"] += len(text)
                try:
                    result = await session.synth_ns(voices[role], text)
                except QwenCapHitError:
                    continue
                _check_ns(result, role, refs, expect_refs)
                info["worker"] = result.worker
                if on_render is not None:
                    on_render(role, text, tried, result.pcm)
                render = example_audio.evaluate(result.pcm, text)
                info["length_fails"] += render.long
                renders.append(render)
                if example_audio.acceptable(render):
                    break
            if not renders:
                raise QwenCapHitError(f"{tried}번 모두 상한 voice={voices[role]}")
            info["retried_turns"] += tried > 1
            pcm, gain, chosen = example_audio.finish(renders, levels[role])
            info["residual_cuts"] += chosen.cut
            info["tail_after_rms_s_max"] = round(
                max(info["tail_after_rms_s_max"], example_audio.tail_after_rms_s(chosen.pcm)), 3
            )
            parts.append(pcm)
            gains.append(gain)
            if on_turn is not None:
                on_turn(time.monotonic() - turn_started)
    if not parts:
        raise QwenTTSUnavailableError("굽을 턴 없음")
    info["gain_db_min"] = round(min(gains), 2) if gains else None
    info["gain_db_max"] = round(max(gains), 2) if gains else None
    return example_audio.join_turns(parts), info, refs


async def _resolve_dialogue(
    db: AsyncSession,
    settings: Settings,
    row: ScenarioORM | None,
    seed: dict | None,
) -> list[dict]:
    if seed is not None:  # 프리셋: 코드가 단일 원천
        return seed["example_dialogue"]
    if row.example_dialogue:
        return row.example_dialogue
    dialogue = await _generate_custom_dialogue(settings, row)
    row.example_dialogue = dialogue
    await db.commit()
    return dialogue


async def get_example_conversation(
    db: AsyncSession,
    scenario_id: int,
    user_id: int,
) -> ExampleConversationResponse:
    """시나리오별 예시 대화(대본 및 캐시된 TTS 오디오 URL)를 반환"""
    settings = get_settings()

    row = await db.get(ScenarioORM, scenario_id)
    if row is not None and row.is_custom:
        if row.user_id != user_id or is_deleted(row):
            raise ScenarioNotFoundError(f"시나리오 {scenario_id} 없음")
        seed = None
    else:
        seed = PRESET_MAP.get(scenario_id)
        if seed is None:
            raise ScenarioNotFoundError(f"시나리오 {scenario_id} 없음")

    storage = RecordingStorageService(settings) if settings.s3_bucket else None
    dialogue, stored_key = await _resolve_and_bake(
        db, settings, row, seed, scenario_id, storage, trigger="request"
    )
    turns = [ExampleTurn(**t) for t in dialogue]
    audio_url = storage.presigned_url(stored_key) if storage and stored_key else None
    return ExampleConversationResponse(scenario_id=scenario_id, dialogue=turns, audio_url=audio_url)


async def _resolve_and_bake(
    db: AsyncSession,
    settings: Settings,
    row: ScenarioORM | None,
    seed: dict | None,
    scenario_id: int,
    storage: RecordingStorageService | None,
    *,
    trigger: str,
) -> tuple[list[dict], str | None]:
    """시나리오 생성 직후 오디오도 만들어 지도록"""
    if storage is not None and row is not None and row.example_audio_url:
        dialogue = seed["example_dialogue"] if seed is not None else row.example_dialogue
        if dialogue:
            hit = _cache_hit(settings, row, scenario_id, dialogue)
            if hit:
                return dialogue, hit
    if trigger == "request" and storage is not None and _ns_target(settings, scenario_id, row):
        return await _request_ns(db, settings, row, seed, scenario_id)
    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    async with lock:
        if row is not None:
            await db.refresh(row, ["deleted_at", "example_dialogue", "example_audio_url"])
            if is_deleted(row):
                return row.example_dialogue or [], None
        dialogue = await _resolve_dialogue(db, settings, row, seed)
        if storage is None:
            return dialogue, None
        key = await _bake_locked(
            db, settings, row, scenario_id, dialogue, storage, trigger=trigger
        )
        return dialogue, key


@dataclass(frozen=True)
class _PlainKeys:
    qwen_ai: str
    ai_voice: str
    user_voice: str
    qwen: str
    eleven: str
    qwen_legacy: str
    eleven_legacy: str


def _plain_keys(settings: Settings, row: ScenarioORM | None, scenario_id: int, dialogue: list[dict]) -> _PlainKeys:
    stored = row.tts_voice_id if row is not None else None
    voice = resolve_voice(stored, Difficulty.LOW)
    ai_voice = el_voice_for(voice) or settings.elevenlabs_voice_id
    user_voice = pick_example_user_voice(ai_voice)
    qwen_ai = qwen_voice_for(voice, Difficulty.LOW) if getattr(settings, "qwen_scenario_voices", False) else "ai"
    slot = profile_for(stored)
    legacy_ai = (None if slot is not None and slot.voice_id == stored else stored) or settings.elevenlabs_voice_id
    return _PlainKeys(
        qwen_ai=qwen_ai,
        ai_voice=ai_voice,
        user_voice=user_voice,
        qwen=_audio_key(scenario_id, dialogue, qwen_ai, "user", qwen=True),
        eleven=_audio_key(scenario_id, dialogue, ai_voice, user_voice),
        qwen_legacy=_audio_key(scenario_id, dialogue, "ai", "user", qwen=True),
        eleven_legacy=_audio_key(scenario_id, dialogue, legacy_ai, pick_example_user_voice(legacy_ai)),
    )


def _cache_hit(
    settings: Settings,
    row: ScenarioORM | None,
    scenario_id: int,
    dialogue: list[dict],
    keys: _PlainKeys | None = None,
) -> str | None:
    """저장된 키가 ns 키거나 예전 굽기 후보면 그 키"""
    cached = row.example_audio_url if row is not None else None
    if not cached:
        return None
    hit = _ns_cached(scenario_id, dialogue, cached)
    if hit:
        return hit
    keys = keys or _plain_keys(settings, row, scenario_id, dialogue)
    for candidate in dict.fromkeys((keys.qwen, keys.eleven, keys.qwen_legacy, keys.eleven_legacy)):
        if cached.endswith(candidate):
            return candidate
    return None


async def _store_example(
    db: AsyncSession,
    row: ScenarioORM | None,
    storage: RecordingStorageService,
    key: str,
    pcm: bytes,
    after_upload: Callable[[str | None], None] | None = None,
) -> tuple[str | None, str | None]:
    """삭제된 행이면 올리지 않고, 저장하면 (키, 이전 키)"""
    if row is not None:
        try:
            await db.refresh(row, ["deleted_at"])
            gone = is_deleted(row)
        except InvalidRequestError:
            gone = True
        if gone:
            if after_upload is not None:
                after_upload(None)
            return None, None
    stored_key = storage.upload_wav(key, pcm)
    if after_upload is not None:
        after_upload(stored_key)
    prev_key = row.example_audio_url if row is not None else None
    if row is not None and stored_key:
        row.example_audio_url = stored_key
        try:
            await db.commit()
        except StaleDataError:
            storage.delete(stored_key)
            raise
    return stored_key, prev_key


async def _bake_locked(
    db: AsyncSession,
    settings: Settings,
    row: ScenarioORM | None,
    scenario_id: int,
    dialogue: list[dict],
    storage: RecordingStorageService,
    *,
    trigger: str,
    ns_fallback_reason: str | None = None,
) -> str | None:
    """캐시에 없으면 합성해 저장하고 S3 키를 반환"""
    keys = _plain_keys(settings, row, scenario_id, dialogue)
    qwen_ai, ai_voice, user_voice = keys.qwen_ai, keys.ai_voice, keys.user_voice
    key_qwen, key_eleven, key_qwen_legacy = keys.qwen, keys.eleven, keys.qwen_legacy
    hit = _cache_hit(settings, row, scenario_id, dialogue, keys)
    if hit:
        return hit

    qwen_client = QwenTTSClient(settings)
    slot_busy = not has_free_worker(settings)
    use_qwen = False if slot_busy else await qwen_client.healthy()

    started = now_ms()
    engine, reason, pcm = "eleven", None, None
    if use_qwen:
        try:
            pcm = await _synthesize_qwen(qwen_client, dialogue, qwen_ai)
            engine = "qwen"
        except QwenVoiceMissingError as exc:
            reason = "voice_missing"
            logger.warning("Qwen 워커에 보이스 없음: %s", exc)
            if exc.voice == qwen_ai != "ai":
                try:
                    pcm = await _synthesize_qwen(qwen_client, dialogue)
                    engine, qwen_ai, key_qwen = "qwen", "ai", key_qwen_legacy
                except QwenVoiceMissingError as exc2:
                    logger.warning("Qwen 워커에 보이스 없음: %s", exc2)
                except QwenTTSUnavailableError as exc2:
                    reason = "synth_failed"
                    logger.warning("Qwen TTS 합성 실패 — ElevenLabs로 전체 재합성: %s", exc2)
        except QwenTTSUnavailableError as exc:
            reason = "synth_failed"
            logger.warning("Qwen TTS 합성 실패 — ElevenLabs로 전체 재합성: %s", exc)
    elif slot_busy:
        reason = "busy"
    else:
        reason = "unhealthy" if qwen_client.enabled else "disabled"

    eleven_chars: list[int] = []
    if pcm is None:
        engine = "eleven"
        tts_client = ElevenLabsTTSClient(settings)
        pcm = await _synthesize(tts_client, dialogue, ai_voice, user_voice, eleven_chars)

    key = key_qwen if engine == "qwen" else key_eleven

    def _logged(_stored: str | None) -> None:
        log_metric(
            "example_tts",
            scenario_id=scenario_id,
            trigger=trigger,
            engine=engine,
            fallback_reason=reason,
            voice=qwen_ai if engine == "qwen" else ai_voice,
            turns=len(dialogue),
            duration_ms=round(now_ms() - started, 1),
            **({"ns_fallback_reason": ns_fallback_reason} if ns_fallback_reason else {}),
        )
        log_tts_usage(
            engine,
            getattr(settings, "elevenlabs_model", None) if engine == "eleven" else "qwen",
            "example_audio",
            chars=sum(eleven_chars) if engine == "eleven" else sum(len(t["text"]) for t in dialogue),
            pcm_bytes=len(pcm),
            turns=len(dialogue),
            user_id=getattr(row, "user_id", None),
            scenario_id=scenario_id,
            trigger=trigger,
        )

    stored_key, _ = await _store_example(db, row, storage, key, pcm, _logged)
    return stored_key


async def bake_example_audio(scenario_id: int, trigger: str = "prebake") -> None:
    """커스텀 시나리오 생성 직후 예시 오디오를 미리 생성"""
    settings = get_settings()
    if not settings.s3_bucket:
        return
    if _ns_enabled(settings) and scenario_id not in _skip_presets(settings):
        try:
            async with AsyncSessionLocal() as db:
                row = await db.get(ScenarioORM, scenario_id)
                skip = row is None or is_deleted(row) or row.example_audio_url or getattr(row, "is_warmup", False)
            if not skip:
                await asyncio.shield(_ensure_job(scenario_id, trigger).task)
        except Exception:
            logger.exception("시나리오 %s 예시 오디오 미리 굽기 실패", scenario_id)
        return
    try:
        async with _prebake_semaphore(), asyncio.timeout(_PREBAKE_TIMEOUT_SEC), AsyncSessionLocal() as db:
            row = await db.get(ScenarioORM, scenario_id)
            if row is None or is_deleted(row) or row.example_audio_url or getattr(row, "is_warmup", False):
                return
            await _resolve_and_bake(db, settings, row, None, scenario_id,
                                    RecordingStorageService(settings), trigger="prebake")
    except Exception:
        logger.exception("시나리오 %s 예시 오디오 미리 굽기 실패", scenario_id)


_CUSTOM_NS_BUDGET_SEC = 200.0
_PRESET_NS_BUDGET_SEC = 600.0
_FALLBACK_BUDGET_SEC = 180.0
_COOLDOWN_SEC = 600.0
_UNREACHABLE_GRACE_SEC = 45 * 60.0
_PREBAKE_WALL_SEC = 24 * 3600.0
_NS_PRESET_RETRY_SEC = 300.0
_NS_PRESET_ROUNDS = 12


class ExampleOpError(Exception):
    """내부 예시 작업을 지금 할 수 없음"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _ns_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "example_qwen_ns", False))


def _skip_presets(settings: Settings) -> set[int]:
    raw = getattr(settings, "example_ns_skip_presets", "") or ""
    return {int(part) for part in raw.replace(" ", "").split(",") if part.isdigit()}


def _ns_target(settings: Settings, scenario_id: int, row: ScenarioORM | None) -> bool:
    """이 시나리오를 N 굽기 작업으로 다룰지"""
    return _ns_enabled(settings) and row is not None and scenario_id not in _skip_presets(settings)


@dataclass
class _BakeJob:
    dialogue: asyncio.Future
    task: asyncio.Task | None = None
    phase: str = "dialogue"
    turns_total: int | None = None
    turns_done: int = 0
    turn_secs: list[float] | None = None
    fail_reason: str | None = None

    def turn_done(self, seconds: float) -> None:
        self.turns_done += 1
        self.turn_secs = [*(self.turn_secs or []), seconds]


_jobs_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, _BakeJob]] = (
    weakref.WeakKeyDictionary()
)
_cooldown_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, float]] = (
    weakref.WeakKeyDictionary()
)


def _loop_jobs() -> dict[int, _BakeJob]:
    loop = asyncio.get_running_loop()
    jobs = _jobs_by_loop.get(loop)
    if jobs is None:
        jobs = _jobs_by_loop[loop] = {}
    return jobs


def _loop_cooldown() -> dict[int, float]:
    loop = asyncio.get_running_loop()
    cooldown = _cooldown_by_loop.get(loop)
    if cooldown is None:
        cooldown = _cooldown_by_loop[loop] = {}
    return cooldown


def _cooling(scenario_id: int) -> bool:
    until = _loop_cooldown().get(scenario_id)
    return until is not None and time.monotonic() < until


def _resolve_future(future: asyncio.Future, value: object) -> None:
    if not future.done():
        future.set_result(value)


def _ensure_job(
    scenario_id: int,
    trigger: str,
    *,
    expect_refs: dict[str, str] | None = None,
    exclude: Iterable[str] = (),
    force: bool = False,
) -> _BakeJob:
    """시나리오마다 굽기 작업은 하나, 있으면 그것을 쓴다"""
    jobs = _loop_jobs()
    job = jobs.get(scenario_id)
    if job is not None and job.task is not None and not job.task.done():
        return job
    loop = asyncio.get_running_loop()
    job = _BakeJob(dialogue=loop.create_future())
    job.task = loop.create_task(
        _bake_job(scenario_id, trigger, job, expect_refs=expect_refs, exclude=tuple(exclude), force=force)
    )
    jobs[scenario_id] = job

    def _done(task: asyncio.Task, sid: int = scenario_id, mine: _BakeJob = job) -> None:
        if jobs.get(sid) is mine:
            jobs.pop(sid, None)
        _resolve_future(mine.dialogue, [])
        if not task.cancelled() and task.exception() is not None:
            logger.error("시나리오 %s 예시 굽기 작업 실패", sid, exc_info=task.exception())

    job.task.add_done_callback(_done)
    return job


async def cancel_example_jobs() -> None:
    """종료 때 남은 굽기 작업 정리"""
    loop = asyncio.get_running_loop()
    tasks = [job.task for job in _jobs_by_loop.get(loop, {}).values() if job.task is not None]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    _jobs_by_loop.pop(loop, None)


def _ns_voices(settings: Settings, row: ScenarioORM, scenario_id: int, dialogue: list[dict]) -> dict[str, str]:
    return {"ai": _plain_keys(settings, row, scenario_id, dialogue).qwen_ai, "user": "user"}


def _ns_reason(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, QwenNonStreamingMissingError):
        return "no_ns"
    if isinstance(exc, QwenVoiceRefError):
        return "voice_ref"
    if isinstance(exc, QwenVoiceMissingError):
        return "voice_missing"
    return "synth_failed"


async def _bake_job(
    scenario_id: int,
    trigger: str,
    job: _BakeJob,
    *,
    expect_refs: dict[str, str] | None,
    exclude: tuple[str, ...],
    force: bool,
) -> str | None:
    settings = get_settings()
    storage = RecordingStorageService(settings)
    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    async with lock, AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        if row is None or is_deleted(row):
            _resolve_future(job.dialogue, [])
            return None
        seed = None if row.is_custom else PRESET_MAP.get(scenario_id)
        dialogue = await _resolve_dialogue(db, settings, row, seed)
    _resolve_future(job.dialogue, dialogue)
    job.turns_total = len(dialogue)
    job.phase = "queued"
    async with _prebake_semaphore():
        job.phase = "baking"
        try:
            return await _bake_ns(
                settings, storage, scenario_id, dialogue, trigger, job,
                expect_refs=expect_refs, exclude=exclude, force=force,
            )
        finally:
            job.phase = "done"


async def _render_ns(
    settings: Settings,
    dialogue: list[dict],
    voices: dict[str, str],
    levels: dict[str, float],
    job: _BakeJob,
    expect_refs: dict[str, str] | None,
    exclude: tuple[str, ...],
) -> tuple[bytes, dict, dict[str, str]]:
    """연결이 안 되는 워커를 만나면 그 워커를 빼고 한 번 더"""
    skip = set(exclude)
    for attempt in range(2):
        job.turns_done, job.turn_secs = 0, None
        try:
            return await _synthesize_qwen_ns(
                QwenTTSClient(settings), dialogue, voices, levels=levels,
                expect_refs=expect_refs, on_turn=job.turn_done, exclude=skip,
            )
        except QwenUnreachableError as exc:
            if attempt or not exc.worker or exc.worker in skip:
                raise
            skip.add(exc.worker)
    raise QwenTTSUnavailableError("굽기 재시도 소진")


async def _bake_ns(
    settings: Settings,
    storage: RecordingStorageService,
    scenario_id: int,
    dialogue: list[dict],
    trigger: str,
    job: _BakeJob,
    *,
    expect_refs: dict[str, str] | None,
    exclude: tuple[str, ...],
    force: bool,
) -> str | None:
    preset = scenario_id in PRESET_MAP
    levels = _levels(settings)
    async with AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        if row is None or is_deleted(row):
            return None
        voices = _ns_voices(settings, row, scenario_id, dialogue)
        cached = row.example_audio_url
        await db.commit()
    if not force and cached:
        if expect_refs is None:
            hit = _ns_cached(scenario_id, dialogue, cached)
        else:
            want = _ns_key(scenario_id, dialogue, _ns_v8(voices, expect_refs, levels))
            hit = want if cached.endswith(want) else None
        if hit:
            return hit
    started = now_ms()
    try:
        budget = _PRESET_NS_BUDGET_SEC if preset else _CUSTOM_NS_BUDGET_SEC
        async with asyncio.timeout(budget):
            pcm, info, refs = await _render_ns(settings, dialogue, voices, levels, job, expect_refs, exclude)
    except (QwenTTSUnavailableError, TimeoutError) as exc:
        reason = job.fail_reason = _ns_reason(exc)
        logger.warning("시나리오 %s N 굽기 실패(%s): %s", scenario_id, reason, exc)
        if preset:
            log_metric("example_tts_deferred", scenario_id=scenario_id, trigger=trigger, reason=reason)
            return None
        return await _fallback_plain(settings, storage, scenario_id, dialogue, trigger, reason)
    key = _ns_key(scenario_id, dialogue, _ns_v8(voices, refs, levels))
    async with AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        prev_key = row.example_audio_url if row is not None else None

        def _logged(stored: str | None) -> None:
            log_metric(
                "example_tts",
                scenario_id=scenario_id,
                trigger=trigger,
                engine="qwen",
                method=_NS_METHOD,
                v8=key.rsplit("-", 2)[-2],
                voice=voices["ai"],
                turns=len(dialogue),
                duration_ms=round(now_ms() - started, 1),
                stored=stored is not None,
                prev_key=prev_key,
                **info,
            )
            log_tts_usage(
                "qwen",
                f"qwen-{_NS_METHOD}",
                "example_audio",
                chars=info["chars"],
                pcm_bytes=len(pcm),
                turns=len(dialogue),
                user_id=getattr(row, "user_id", None),
                scenario_id=scenario_id,
                trigger=trigger,
                attempts=info["attempts"],
            )

        if row is None:
            _logged(None)
            return None
        stored_key, _ = await _store_example(db, row, storage, key, pcm, _logged)
        return stored_key


async def _fallback_plain(
    settings: Settings,
    storage: RecordingStorageService,
    scenario_id: int,
    dialogue: list[dict],
    trigger: str,
    reason: str,
) -> str | None:
    """커스텀은 예전 굽기로, 그것도 실패하면 잠시 다시 굽지 않는다"""
    key = None
    try:
        async with asyncio.timeout(_FALLBACK_BUDGET_SEC), AsyncSessionLocal() as db:
            row = await db.get(ScenarioORM, scenario_id)
            if row is not None and not is_deleted(row):
                key = await _bake_locked(
                    db, settings, row, scenario_id, dialogue, storage, trigger=trigger, ns_fallback_reason=reason
                )
    except Exception:
        logger.exception("시나리오 %s 예전 굽기 폴백 실패", scenario_id)
    if key is None:
        _loop_cooldown()[scenario_id] = time.monotonic() + _COOLDOWN_SEC
    return key


def _remaining_sec(job: _BakeJob, settings: Settings) -> float:
    """다 구워질 때까지 남은 시간 추정, 모르면 무한"""
    if job.phase in ("dialogue", "queued"):
        return float("inf")
    if job.phase == "done":
        return 0.0
    secs = job.turn_secs or []
    per = sum(secs) / len(secs) if secs else getattr(settings, "example_turn_prior_sec", 6.0)
    total = job.turns_total or 12
    return max(0, total - job.turns_done) * per


async def _request_ns(
    db: AsyncSession,
    settings: Settings,
    row: ScenarioORM,
    seed: dict | None,
    scenario_id: int,
) -> tuple[list[dict], str | None]:
    """굽기 작업을 잠깐 기다리고, 오래 걸리면 대본만 준다"""
    fallback_dialogue = seed["example_dialogue"] if seed is not None else (row.example_dialogue or [])
    if _cooling(scenario_id):
        return fallback_dialogue, None
    await db.commit()
    job = _ensure_job(scenario_id, "request")
    await asyncio.sleep(0)
    wait = float(getattr(settings, "example_request_wait_sec", 20.0))
    started = now_ms()
    key: str | None = None
    early = _remaining_sec(job, settings) > wait
    timed_out = False
    if not early:
        try:
            key = await asyncio.wait_for(asyncio.shield(job.task), wait)
        except TimeoutError:
            timed_out = True
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
    dialogue = job.dialogue.result() if job.dialogue.done() else fallback_dialogue
    log_metric(
        "example_wait",
        scenario_id=scenario_id,
        waited_ms=round(now_ms() - started, 1),
        early_return=early,
        timed_out=timed_out,
        served=key is not None,
    )
    return dialogue or fallback_dialogue, key


def _prebake_gate(settings: Settings) -> bool:
    return _ns_enabled(settings) and bool(settings.s3_bucket) and bool(worker_urls(settings))


@dataclass
class _Driver:
    wake: asyncio.Event
    running: bool = False


_drivers: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Driver] = weakref.WeakKeyDictionary()


def _loop_driver() -> _Driver:
    loop = asyncio.get_running_loop()
    driver = _drivers.get(loop)
    if driver is None:
        driver = _drivers[loop] = _Driver(wake=asyncio.Event())
    return driver


def _rejected_key(key: str) -> str:
    return f"examples/prev/{key.rsplit('/', 1)[-1]}.rejected"


async def _preset_rows(settings: Settings) -> dict[int, tuple[ScenarioORM, list[dict]]]:
    rows: dict[int, tuple[ScenarioORM, list[dict]]] = {}
    async with AsyncSessionLocal() as db:
        for sid, seed in PRESET_MAP.items():
            row = await db.get(ScenarioORM, sid)
            if row is not None and not is_deleted(row):
                rows[sid] = (row, seed["example_dialogue"])
        await db.commit()
    return rows


async def _prebake_round(settings: Settings, state: dict) -> str:
    """한 라운드: done / wait(안 셈) / counted(셈) / end(끝)"""
    storage = RecordingStorageService(settings)
    skip = _skip_presets(settings)
    levels = _levels(settings)
    rows = {sid: v for sid, v in (await _preset_rows(settings)).items() if sid not in skip}
    voices = {sid: _ns_voices(settings, row, sid, dialogue) for sid, (row, dialogue) in rows.items()}
    needed = {name for vs in voices.values() for name in vs.values()}
    refs_state = await QwenTTSClient(settings).health_refs(sorted(needed))
    summary = {"checked": len(rows), "hit": 0, "reused": 0, "baked": 0, "deferred": 0, "state": refs_state.state}
    if refs_state.state in ("down", "rolling"):
        log_metric("example_preset_prebake", waiting=refs_state.state, **summary)
        return "wait"
    if refs_state.state == "no_refs":
        log_metric("example_preset_prebake", reason="no_refs", **summary)
        return "end"
    exclude: tuple[str, ...] = ()
    if refs_state.unreachable:
        since = state.setdefault("unreachable_since", time.monotonic())
        if time.monotonic() - since < _UNREACHABLE_GRACE_SEC:
            log_metric("example_preset_prebake", waiting="unreachable", unreachable=",".join(refs_state.unreachable),
                       **summary)
            return "wait"
        exclude = refs_state.unreachable
    else:
        state.pop("unreachable_since", None)
    stop_round = False
    for sid, (row, dialogue) in rows.items():
        vs = voices[sid]
        if any(name in refs_state.missing for name in vs.values()):
            summary["deferred"] += 1
            continue
        refs = {"ai": refs_state.refs[vs["ai"]], "user": refs_state.refs[vs["user"]]}
        want = _ns_key(sid, dialogue, _ns_v8(vs, refs, levels))
        if row.example_audio_url and row.example_audio_url.endswith(want):
            summary["hit"] += 1
            continue
        if stop_round:
            summary["deferred"] += 1
            continue
        if storage.exists(_rejected_key(want)):
            summary["deferred"] += 1
            continue
        if storage.exists(want):
            await _point_row(sid, want, "preset_prebake")
            summary["reused"] += 1
            continue
        job = _ensure_job(sid, "preset_prebake", expect_refs=refs, exclude=exclude)
        key = await asyncio.shield(job.task)
        if key == want:
            summary["baked"] += 1
        else:
            summary["deferred"] += 1
            stop_round = job.fail_reason in ("no_ns", "voice_ref")
    log_metric(
        "example_preset_prebake",
        refs=",".join(f"{k}:{v}" for k, v in sorted(refs_state.refs.items())),
        unreachable=",".join(refs_state.unreachable) or None,
        **summary,
    )
    return "done" if summary["deferred"] == 0 else "counted"


async def _point_row(scenario_id: int, key: str, trigger: str) -> None:
    """이미 있는 객체로 행만 바꾼다"""
    async with AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        if row is None or is_deleted(row):
            return
        prev_key = row.example_audio_url
        row.example_audio_url = key
        await db.commit()
    log_metric("example_tts", scenario_id=scenario_id, trigger=trigger, engine="qwen", method=_NS_METHOD,
               reused=True, prev_key=prev_key)


async def prebake_presets() -> None:
    """서버 시작 때 프리셋 예시를 지금 ref 지문으로 맞춘다"""
    settings = get_settings()
    if not _prebake_gate(settings):
        return
    driver = _loop_driver()
    if driver.running:
        driver.wake.set()
        return
    driver.running = True
    state: dict = {}
    counted = 0
    began = time.monotonic()
    try:
        while True:
            driver.wake.clear()
            try:
                result = await _prebake_round(settings, state)
            except Exception:
                logger.exception("프리셋 예시 미리 굽기 라운드 실패")
                result = "counted"
            if result in ("done", "end"):
                return
            counted += result == "counted"
            if counted >= _NS_PRESET_ROUNDS or time.monotonic() - began > _PREBAKE_WALL_SEC:
                logger.warning("프리셋 예시 미리 굽기 못 끝냄(라운드 %s)", counted)
                return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(driver.wake.wait(), _NS_PRESET_RETRY_SEC)
    finally:
        driver.running = False


def trigger_prebake() -> bool:
    """내부 API: 미리 굽기 라운드를 지금 돌린다, 이미 돌고 있으면 깨우기만"""
    settings = get_settings()
    if not _prebake_gate(settings):
        raise ExampleOpError(409, "예시 N 굽기가 꺼져 있다")
    driver = _loop_driver()
    if driver.running:
        driver.wake.set()
        return True
    asyncio.get_running_loop().create_task(prebake_presets())
    return False


async def _expected_preset_key(settings: Settings, scenario_id: int) -> tuple[str, dict[str, str], str | None]:
    """프리셋의 지금 기대 키, (키, refs, 저장된 키)"""
    if scenario_id not in PRESET_MAP:
        raise ExampleOpError(404, f"프리셋 아님: {scenario_id}")
    rows = await _preset_rows(settings)
    if scenario_id not in rows:
        raise ExampleOpError(404, f"프리셋 행 없음: {scenario_id}")
    row, dialogue = rows[scenario_id]
    vs = _ns_voices(settings, row, scenario_id, dialogue)
    refs_state = await QwenTTSClient(settings).health_refs(sorted(set(vs.values())))
    if refs_state.state != "ok":
        raise ExampleOpError(409, f"워커 지문 상태 {refs_state.state}")
    refs = {"ai": refs_state.refs[vs["ai"]], "user": refs_state.refs[vs["user"]]}
    return _ns_key(scenario_id, dialogue, _ns_v8(vs, refs, _levels(settings))), refs, row.example_audio_url


def _backup_path(key: str, suffix: str) -> str:
    return f"examples/prev/{key.rsplit('/', 1)[-1]}.{suffix}"


async def rebake_preset(scenario_id: int, *, stamp: str) -> dict:
    """내부 API: 지금 객체를 prev 로 옮겨 두고 새로 굽는다"""
    settings = get_settings()
    if not _prebake_gate(settings):
        raise ExampleOpError(409, "예시 N 굽기가 꺼져 있다")
    want, refs, _ = await _expected_preset_key(settings, scenario_id)
    storage = RecordingStorageService(settings)
    backup = _backup_path(want, stamp)
    try:
        backed_up = storage.copy(want, backup)
    except Exception as exc:
        raise ExampleOpError(503, f"백업 복사 실패: {exc}") from exc
    _ensure_job(scenario_id, "rebake", expect_refs=refs, force=True)
    return {"key": want, "backup": backup if backed_up else None}


async def restore_preset(scenario_id: int, prev: str, *, stamp: str) -> dict:
    """내부 API: prev 객체를 지금 기대 키로 되살린다"""
    settings = get_settings()
    if not _ns_enabled(settings):
        raise ExampleOpError(409, "예시 N 굽기가 꺼져 있다")
    want, _, _ = await _expected_preset_key(settings, scenario_id)
    if not prev.startswith(f"examples/prev/{want.rsplit('/', 1)[-1]}."):
        raise ExampleOpError(409, "prev 가 지금 기대 키의 백업이 아니다")
    storage = RecordingStorageService(settings)
    backed_up = storage.copy(want, _backup_path(want, stamp))
    if not storage.copy(prev, want):
        raise ExampleOpError(404, f"prev 없음: {prev}")
    await _point_row(scenario_id, want, "restore")
    return {"key": want, "backup": _backup_path(want, stamp) if backed_up else None}


async def reject_preset(scenario_id: int, restore_key: str) -> dict:
    """내부 API: 지금 ns 객체를 탈락으로 옮기고 행을 restore_key 로 되돌린다"""
    settings = get_settings()
    if not _ns_enabled(settings):
        raise ExampleOpError(409, "예시 N 굽기가 꺼져 있다")
    if scenario_id not in PRESET_MAP:
        raise ExampleOpError(404, f"프리셋 아님: {scenario_id}")
    storage = RecordingStorageService(settings)
    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    async with lock, AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        if row is None or is_deleted(row):
            raise ExampleOpError(404, f"프리셋 행 없음: {scenario_id}")
        current = row.example_audio_url
        dialogue = PRESET_MAP[scenario_id]["example_dialogue"]
        current_ns = _ns_cached(scenario_id, dialogue, current)
        if current_ns is None:
            raise ExampleOpError(409, "지금 행이 ns 키가 아니다")
        if not storage.exists(restore_key):
            raise ExampleOpError(409, f"되돌릴 객체 없음: {restore_key}")
        storage.copy(current_ns, _rejected_key(current_ns))
        row.example_audio_url = restore_key
        await db.commit()
    storage.delete(current_ns)
    log_metric("example_tts", scenario_id=scenario_id, trigger="reject", prev_key=current, key=restore_key)
    return {"rejected": current_ns, "restored": restore_key}
