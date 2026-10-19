from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import weakref
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
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
    QwenVoiceMissingError,
    has_free_worker,
    is_voice_ref,
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
) -> tuple[bytes, dict, dict[str, str]]:
    """non_streaming_mode 로 턴마다 다시 합성해 고르고 다듬어 크기를 맞춘다"""
    parts: list[bytes] = []
    gains: list[float] = []
    refs: dict[str, str] = {}
    info: dict = {"attempts": 0, "retried_turns": 0, "residual_cuts": 0, "length_fails": 0,
                  "tail_after_rms_s_max": 0.0, "worker": None}
    async with qwen_client.connect() as session:
        async for turn in _iter_turns(turns):
            role, text = turn["speaker"], turn["text"]
            renders: list[example_audio.Render] = []
            tried = 0
            for tried in range(1, example_audio.MAX_ATTEMPTS + 1):
                info["attempts"] += 1
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
class _F109Keys:
    qwen_ai: str
    ai_voice: str
    user_voice: str
    qwen: str
    eleven: str
    qwen_legacy: str
    eleven_legacy: str


def _f109_keys(settings: Settings, row: ScenarioORM | None, scenario_id: int, dialogue: list[dict]) -> _F109Keys:
    stored = row.tts_voice_id if row is not None else None
    voice = resolve_voice(stored, Difficulty.LOW)
    ai_voice = el_voice_for(voice) or settings.elevenlabs_voice_id
    user_voice = pick_example_user_voice(ai_voice)
    qwen_ai = qwen_voice_for(voice, Difficulty.LOW) if getattr(settings, "qwen_scenario_voices", False) else "ai"
    slot = profile_for(stored)
    legacy_ai = (None if slot is not None and slot.voice_id == stored else stored) or settings.elevenlabs_voice_id
    return _F109Keys(
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
    keys: _F109Keys | None = None,
) -> str | None:
    """저장된 키가 ns 키거나 F109 후보면 그 키"""
    cached = row.example_audio_url if row is not None else None
    if not cached:
        return None
    hit = _ns_cached(scenario_id, dialogue, cached)
    if hit:
        return hit
    keys = keys or _f109_keys(settings, row, scenario_id, dialogue)
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
        await db.refresh(row, ["deleted_at"])
        if is_deleted(row):
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
) -> str | None:
    """캐시에 없으면 합성해 저장하고 S3 키를 반환"""
    keys = _f109_keys(settings, row, scenario_id, dialogue)
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


async def bake_example_audio(scenario_id: int) -> None:
    """커스텀 시나리오 생성 직후 예시 오디오를 미리 생성"""
    settings = get_settings()
    if not settings.s3_bucket:
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
