from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.services.example_audio import CleanTurn, clean_turn
from app.services.qwen_tts import (
    QwenTTSClient,
    QwenTTSUnavailableError,
    QwenVoiceMissingError,
    has_free_worker,
    worker_urls,
)
from app.services.recording_storage import RecordingStorageService
from app.services.tts import ElevenLabsTTSClient

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_TURN_GAP_MS = 400
_TURN_GAP_PCM = b"\x00" * (_SAMPLE_RATE * _TURN_GAP_MS // 1000 * 2)
_TTS_MAX_CONCURRENCY = 3
# 대본 생성(LLM) + 12턴 clean 합성 최악(턴당 5회 × ~3초 ≈ 180초)에 여유를 둔 상한
_PREBAKE_TIMEOUT_SEC = 300.0
_prebake_semaphore = loop_semaphore(1)
_CLEAN_METHOD = "clean1"  # Qwen 키에 넣는 굽기 방식 표지
_PRESET_PREBAKE_TIMEOUT_SEC = 600.0  # 10턴 clean 합성 최악(~150초)과 워커 대기에 여유
_PRESET_RETRY_SEC = 300.0
_PRESET_PREBAKE_ROUNDS = 12

# 사용 중인 요청(지역변수)만 락을 강참조 → 요청이 끝나면 GC가 엔트리를 자동 제거
_generation_locks: weakref.WeakValueDictionary[int, asyncio.Lock] = weakref.WeakValueDictionary()


class ScenarioNotFoundError(Exception):
    """시나리오가 없거나 요청자가 접근할 수 없는 경우"""


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


async def _synthesize_qwen(
    qwen_client: QwenTTSClient, dialogue: list[dict], ai_voice: str = "ai"
) -> tuple[bytes, list[CleanTurn]]:
    """GPU 서버로 턴마다 끝 잘림 없이 합성"""
    turns: list[CleanTurn] = []
    async with qwen_client.connect() as session:
        for turn in dialogue:
            voice = ai_voice if turn["speaker"] == "ai" else "user"
            turns.append(await clean_turn(lambda text, v=voice: session.synth(v, text), turn["text"]))
    return _TURN_GAP_PCM.join(t.pcm for t in turns), turns


def _audio_key(
    scenario_id: int,
    dialogue: list[dict],
    ai_voice: str,
    user_voice: str,
    *,
    qwen: bool = False,
    method: str | None = None,
) -> str:
    """대본, 보이스, 굽기 방식이 바뀌면 키가 달라져 캐시가 자동 무효화"""
    voices = [f"qwen-{ai_voice}", f"qwen-{user_voice}"] if qwen else [ai_voice, user_voice]
    payload = json.dumps([dialogue, *voices, *([method] if method else [])], ensure_ascii=False)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    suffix = "-q" if qwen else ""
    return f"examples/{scenario_id}-{digest}{suffix}.wav"


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
    qwen_only: bool = False,
) -> tuple[list[dict], str | None]:
    """시나리오 생성 직후 오디오도 만들어 지도록"""
    if not qwen_only and storage is not None and row is not None and row.example_audio_url:
        dialogue = seed["example_dialogue"] if seed is not None else row.example_dialogue
        if dialogue:
            hit = _cache_hit(_keys(settings, row, scenario_id, dialogue), row.example_audio_url, qwen_only=False)
            if hit:
                return dialogue, hit
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
            db, settings, row, scenario_id, dialogue, storage, trigger=trigger, qwen_only=qwen_only
        )
        return dialogue, key


@dataclass(frozen=True)
class _Keys:
    qwen_ai: str
    ai_voice: str
    user_voice: str
    qwen: str
    qwen_fallback: str
    eleven: str
    legacy: tuple[str, str]


def _keys(settings: Settings, row: ScenarioORM | None, scenario_id: int, dialogue: list[dict]) -> _Keys:
    """설정 목소리 후 ai clean, EL, 이전 공식 키"""
    stored = row.tts_voice_id if row is not None else None
    voice = resolve_voice(stored, Difficulty.LOW)
    ai_voice = el_voice_for(voice) or settings.elevenlabs_voice_id
    user_voice = pick_example_user_voice(ai_voice)
    qwen_ai = qwen_voice_for(voice, Difficulty.LOW) if getattr(settings, "qwen_scenario_voices", False) else "ai"
    slot = profile_for(stored)
    legacy_ai = (None if slot is not None and slot.voice_id == stored else stored) or settings.elevenlabs_voice_id
    return _Keys(
        qwen_ai=qwen_ai,
        ai_voice=ai_voice,
        user_voice=user_voice,
        qwen=_audio_key(scenario_id, dialogue, qwen_ai, "user", qwen=True, method=_CLEAN_METHOD),
        qwen_fallback=_audio_key(scenario_id, dialogue, "ai", "user", qwen=True, method=_CLEAN_METHOD),
        eleven=_audio_key(scenario_id, dialogue, ai_voice, user_voice),
        legacy=(
            _audio_key(scenario_id, dialogue, "ai", "user", qwen=True),
            _audio_key(scenario_id, dialogue, legacy_ai, pick_example_user_voice(legacy_ai)),
        ),
    )


def _cache_hit(keys: _Keys, cached: str | None, *, qwen_only: bool) -> str | None:
    """요청은 옛 예시도 바로 주고, 미리 굽기는 설정 목소리 clean 만 인정"""
    candidates = [keys.qwen] if qwen_only else [keys.qwen, keys.qwen_fallback, keys.eleven, *keys.legacy]
    for candidate in dict.fromkeys(candidates):
        if cached and cached.endswith(candidate):
            return candidate
    return None


async def _bake_locked(
    db: AsyncSession,
    settings: Settings,
    row: ScenarioORM | None,
    scenario_id: int,
    dialogue: list[dict],
    storage: RecordingStorageService,
    *,
    trigger: str,
    qwen_only: bool = False,
) -> str | None:
    """캐시에 없으면 합성해 저장하고 S3 키를 반환, qwen_only 면 Qwen 으로 못 구울 때 None"""
    keys = _keys(settings, row, scenario_id, dialogue)
    qwen_ai, ai_voice, user_voice = keys.qwen_ai, keys.ai_voice, keys.user_voice
    key_qwen, key_eleven = keys.qwen, keys.eleven
    cached = row.example_audio_url if row is not None else None
    hit = _cache_hit(keys, cached, qwen_only=qwen_only)
    if hit:
        return hit

    qwen_client = QwenTTSClient(settings)
    state = None
    if qwen_only and cached and cached.endswith(keys.qwen_fallback):
        state = await qwen_client.health_state()
        if qwen_client.voices is not None and qwen_ai not in qwen_client.voices:
            return keys.qwen_fallback
    slot_busy = not has_free_worker(settings)
    if slot_busy:
        use_qwen = False
    elif state is not None:
        use_qwen = state in ("ready", "busy")
    else:
        use_qwen = await qwen_client.healthy()

    started = now_ms()
    engine, reason, pcm, turns = "eleven", None, None, []
    if use_qwen:
        try:
            pcm, turns = await _synthesize_qwen(qwen_client, dialogue, qwen_ai)
            engine = "qwen"
        except QwenVoiceMissingError as exc:
            reason = "voice_missing"
            logger.warning("Qwen 워커에 보이스 없음: %s", exc)
            if exc.voice == qwen_ai != "ai":
                try:
                    pcm, turns = await _synthesize_qwen(qwen_client, dialogue)
                    engine, qwen_ai, key_qwen = "qwen", "ai", keys.qwen_fallback
                except QwenVoiceMissingError as exc2:
                    logger.warning("Qwen 워커에 보이스 없음: %s", exc2)
                except QwenTTSUnavailableError as exc2:
                    reason = "synth_failed"
                    logger.warning("Qwen TTS 합성 실패 — ElevenLabs로 전체 재합성: %s", exc2)
        except QwenTTSUnavailableError as exc:
            reason = "synth_failed"
            logger.warning("Qwen TTS 합성 실패 — ElevenLabs로 전체 재합성: %s", exc)
    elif not qwen_client.enabled:
        reason = "disabled"
    elif slot_busy:
        reason = "busy"
    else:
        reason = "unhealthy"

    if pcm is None and qwen_only:
        log_metric("example_tts_deferred", scenario_id=scenario_id, trigger=trigger, reason=reason)
        return None

    eleven_chars: list[int] = []
    if pcm is None:
        engine = "eleven"
        tts_client = ElevenLabsTTSClient(settings)
        pcm = await _synthesize(tts_client, dialogue, ai_voice, user_voice, eleven_chars)

    key = key_qwen if engine == "qwen" else key_eleven
    stored_key = storage.upload_wav(key, pcm)
    log_metric(
        "example_tts",
        scenario_id=scenario_id,
        trigger=trigger,
        engine=engine,
        fallback_reason=reason,
        voice=qwen_ai if engine == "qwen" else ai_voice,
        turns=len(dialogue),
        clean_attempts=sum(t.attempts for t in turns) if engine == "qwen" else None,
        clean_residual_cuts=sum(t.cut for t in turns) if engine == "qwen" else None,
        duration_ms=round(now_ms() - started, 1),
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

    if row is not None and stored_key:
        row.example_audio_url = stored_key
        await db.commit()
    return stored_key


async def bake_example_audio(scenario_id: int) -> None:
    """커스텀 시나리오 생성 직후 예시 오디오를 미리 생성"""
    settings = get_settings()
    if not settings.s3_bucket:
        return
    try:
        async with _prebake_semaphore(), asyncio.timeout(_PREBAKE_TIMEOUT_SEC), AsyncSessionLocal() as db:
            row = await db.get(ScenarioORM, scenario_id)
            if row is None or is_deleted(row) or row.example_audio_url:
                return
            await _resolve_and_bake(db, settings, row, None, scenario_id,
                                    RecordingStorageService(settings), trigger="prebake")
    except Exception:
        logger.exception("시나리오 %s 예시 오디오 미리 굽기 실패", scenario_id)


async def prebake_presets() -> None:
    """서버 시작 때 프리셋 예시를 지금 목소리,굽기 방식으로 맞춤"""
    settings = get_settings()
    if not settings.s3_bucket or not getattr(settings, "example_preset_prebake", True) or not worker_urls(settings):
        return
    pending = list(PRESET_MAP)
    for round_ in range(_PRESET_PREBAKE_ROUNDS):
        if round_:
            await asyncio.sleep(_PRESET_RETRY_SEC)
        if not await QwenTTSClient(settings).healthy():
            continue
        pending = [sid for sid in pending if not await _prebake_preset(settings, sid)]
        if not pending:
            return
    logger.warning("프리셋 예시 미리 굽기 못 끝냄: %s", pending)


async def _prebake_preset(settings: Settings, scenario_id: int) -> bool:
    """맞췄으면 True, Qwen 으로 못 구워 미뤘으면 False"""
    try:
        async with (
            _prebake_semaphore(),
            asyncio.timeout(_PRESET_PREBAKE_TIMEOUT_SEC),
            AsyncSessionLocal() as db,
        ):
            row = await db.get(ScenarioORM, scenario_id)
            _, key = await _resolve_and_bake(
                db, settings, row, PRESET_MAP[scenario_id], scenario_id,
                RecordingStorageService(settings), trigger="preset_prebake", qwen_only=True,
            )
            return key is not None
    except Exception:
        logger.exception("프리셋 %s 예시 미리 굽기 실패", scenario_id)
        return False
