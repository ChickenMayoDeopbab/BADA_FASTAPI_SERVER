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
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
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


def _normalize_turn(item: object) -> dict | None:
    """대화 원소 하나를 {speaker, text} 로, 아니면 None"""
    if not isinstance(item, dict):
        return None
    speaker = str(item.get("speaker", "")).strip().lower()
    text = str(item.get("text", "")).strip()
    if speaker not in ("ai", "user") or not text:
        return None
    return {"speaker": speaker, "text": text}


def _normalize_dialogue(raw: object) -> list[dict]:
    """LLM이 생성한 예시 대화를 {speaker, text} 형태로 정규화"""
    if not isinstance(raw, list):
        raise ValueError("예시 대화 응답이 JSON 배열이 아닙니다.")
    turns = [turn for turn in (_normalize_turn(item) for item in raw) if turn is not None]
    if len(turns) < 2:
        raise ValueError("예시 대화 턴이 부족합니다.")
    return turns


def _parse_dialogue_text(raw: str) -> list[dict]:
    """LLM 출력 글 전체를 대화로"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return _normalize_dialogue(json.loads(raw.strip()))


def _dialogue_prompt(row: ScenarioORM) -> str:
    steps = "\n".join(
        f"{t.get('step')}. {t.get('ai_goal', '')} (힌트: {t.get('hint', '')})"
        for t in (row.script or [])
        if isinstance(t, dict)
    )
    return (
        f"Scenario title: {row.title}\n"
        f"Call purpose: {row.call_purpose}\n"
        f"Call target: {row.call_target}\n"
        f"Script steps:\n{steps}"
    )


async def _generate_custom_dialogue(
    settings: Settings, row: ScenarioORM, attempt: int = 1, timeout: float | None = None
) -> list[dict]:
    """커스텀 시나리오의 예시 대화를 LLM으로 생성"""
    user_msg = _dialogue_prompt(row)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    ai_response = await client.messages.create(
        model=settings.llm_analysis_model,
        max_tokens=1500,
        system=_EXAMPLE_GEN_SYSTEM,
        messages=[MessageParam(role="user", content=user_msg)],
        **({"timeout": httpx.Timeout(timeout, connect=5.0)} if timeout is not None else {}),
    )
    log_llm_usage(
        "anthropic",
        settings.llm_analysis_model,
        "example_dialogue",
        usage=getattr(ai_response, "usage", None),
        user_id=getattr(row, "user_id", None),
        scenario_id=getattr(row, "scenario_id", None),
        **({"attempt": attempt} if attempt != 1 else {}),
    )
    return _parse_dialogue_text(ai_response.content[0].text)


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
    ns_attempts_wasted: int = 0,
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
            **({"ns_attempts_wasted": ns_attempts_wasted} if ns_attempts_wasted else {}),
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


async def bake_example_audio(scenario_id: int, trigger: str = "prebake") -> asyncio.Task | None:
    """커스텀 시나리오 생성 직후 예시 오디오를 미리 생성, N 작업이면 기다리지 않고 그 태스크를 돌려준다"""
    settings = get_settings()
    if not settings.s3_bucket:
        return None
    if _ns_enabled(settings) and scenario_id not in _skip_presets(settings):
        try:
            async with AsyncSessionLocal() as db:
                row = await db.get(ScenarioORM, scenario_id)
                skip = row is None or is_deleted(row) or row.example_audio_url or getattr(row, "is_warmup", False)
            if not skip:
                return _ensure_job(scenario_id, trigger).task
        except Exception:
            logger.exception("시나리오 %s 예시 오디오 미리 굽기 실패", scenario_id)
        return None
    try:
        async with _prebake_semaphore(), asyncio.timeout(_PREBAKE_TIMEOUT_SEC), AsyncSessionLocal() as db:
            row = await db.get(ScenarioORM, scenario_id)
            if row is None or is_deleted(row) or row.example_audio_url or getattr(row, "is_warmup", False):
                return
            await _resolve_and_bake(db, settings, row, None, scenario_id,
                                    RecordingStorageService(settings), trigger="prebake")
    except Exception:
        logger.exception("시나리오 %s 예시 오디오 미리 굽기 실패", scenario_id)
    return None


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
    attempts: int = 0
    stream_failed: bool = False

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


def _job_running(scenario_id: int) -> bool:
    job = _loop_jobs().get(scenario_id)
    return job is not None and job.task is not None and not job.task.done()


def _check_op(settings: Settings, scenario_id: int, *, skip_ok: bool = False) -> None:
    """운영 작업을 지금 해도 되는지, 안 되면 409"""
    if not skip_ok and scenario_id in _skip_presets(settings):
        raise ExampleOpError(409, "skip 목록 프리셋, 빼고 재생성한 뒤 다시")
    if _job_running(scenario_id):
        raise ExampleOpError(409, "굽는 중, 끝난 뒤 다시")


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
    """종료 때 미리 굽기 드라이버와 남은 굽기 작업 정리"""
    loop = asyncio.get_running_loop()
    driver = _drivers.get(loop)
    if driver is not None and driver.task is not None and driver.task is not asyncio.current_task():
        driver.task.cancel()
        await asyncio.gather(driver.task, return_exceptions=True)
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
        stream = bool(row.is_custom and not row.example_dialogue)
        if not stream:
            seed = None if row.is_custom else PRESET_MAP.get(scenario_id)
            dialogue = await _resolve_dialogue(db, settings, row, seed)
    if stream:
        return await _bake_streaming(settings, storage, scenario_id, trigger, job, expect_refs=expect_refs,
                                     exclude=exclude)
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


class _CountedSession:
    def __init__(self, session: QwenTTSClient, job: _BakeJob) -> None:
        self._session, self._job = session, job

    async def synth_ns(self, voice: str, text: str):
        self._job.attempts += 1
        return await self._session.synth_ns(voice, text)


class _BakeClient:
    """굽기 작업용: 꺼낸 워커가 ready 인지 보고(probe 일 때), 합성 시도 수를 센다"""

    def __init__(self, settings: Settings, job: _BakeJob, *, probe: bool) -> None:
        self._settings, self._job, self._probe = settings, job, probe

    @asynccontextmanager
    async def connect(self, exclude: Iterable[str] = ()) -> AsyncIterator[_CountedSession]:
        async with QwenTTSClient(self._settings).connect(exclude=exclude) as session:
            url = getattr(session, "_session_url", None)
            if self._probe and url:
                state = await _taken_state(self._settings, url)
                if state != "ready":
                    raise QwenUnreachableError(f"꺼낸 워커 {state}", worker=url)
            yield _CountedSession(session, self._job)


_BUSY_PROBES = 10
_BUSY_PROBE_SEC = 0.3


async def _taken_state(settings: Settings, url: str) -> str:
    """꺼낸 워커 상태, busy 는 잠깐 기다려 본다(통화 끝 drain)"""
    state = "busy"
    for i in range(_BUSY_PROBES):
        state = await QwenTTSClient(settings, base_url=url).health_state()
        if state != "busy":
            return state
        if i + 1 < _BUSY_PROBES:
            await asyncio.sleep(_BUSY_PROBE_SEC)
    return state


async def _render_ns(
    settings: Settings,
    dialogue: list[dict],
    voices: dict[str, str],
    levels: dict[str, float],
    job: _BakeJob,
    expect_refs: dict[str, str] | None,
    exclude: tuple[str, ...],
) -> tuple[bytes, dict, dict[str, str]]:
    """연결이 안 되거나 ready 아닌 워커를 만나면 그 워커를 빼고 한 번 더"""
    skip = set(exclude)
    for attempt in range(2):
        job.turns_done, job.turn_secs = 0, None
        try:
            return await _synthesize_qwen_ns(
                _BakeClient(settings, job, probe=expect_refs is None), dialogue, voices, levels=levels,
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
            if job.attempts:
                _loop_cooldown()[scenario_id] = time.monotonic() + _COOLDOWN_SEC
            log_metric("example_tts_deferred", scenario_id=scenario_id, trigger=trigger, reason=reason,
                       ns_attempts_wasted=job.attempts)
            return None
        return await _fallback_plain(settings, storage, scenario_id, dialogue, trigger, reason, job.attempts)
    return await _store_ns(settings, storage, scenario_id, dialogue, voices, levels, pcm, info, refs, trigger, started)


async def _store_ns(
    settings: Settings,
    storage: RecordingStorageService,
    scenario_id: int,
    dialogue: list[dict],
    voices: dict[str, str],
    levels: dict[str, float],
    pcm: bytes,
    info: dict,
    refs: dict[str, str],
    trigger: str,
    started: float,
) -> str | None:
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
    wasted: int = 0,
) -> str | None:
    """커스텀은 예전 굽기로, 그것도 실패하면 잠시 다시 굽지 않는다"""
    key = None
    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    try:
        async with asyncio.timeout(_FALLBACK_BUDGET_SEC), lock, AsyncSessionLocal() as db:
            row = await db.get(ScenarioORM, scenario_id)
            if row is not None and not is_deleted(row):
                key = await _bake_locked(
                    db, settings, row, scenario_id, dialogue, storage, trigger=trigger, ns_fallback_reason=reason,
                    ns_attempts_wasted=wasted,
                )
    except Exception:
        logger.exception("시나리오 %s 예전 굽기 폴백 실패", scenario_id)
    if key is None:
        _loop_cooldown()[scenario_id] = time.monotonic() + _COOLDOWN_SEC
    return key


def _remaining_sec(job: _BakeJob, settings: Settings) -> float:
    """다 구워질 때까지 남은 시간 추정, 모르면 무한"""
    if job.phase in ("dialogue", "queued") or not job.dialogue.done():
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
    task: asyncio.Task | None = None


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
    summary = {"checked": len(rows), "hit": 0, "reused": 0, "baked": 0, "deferred": 0, "rejected": 0,
               "state": refs_state.state, "rounds": state.get("rounds")}
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
    v8s: list[str] = []
    mismatch: str | None = None
    s3_errors: list[str] = []
    for sid, (row, dialogue) in rows.items():
        vs = voices[sid]
        if any(name in refs_state.missing for name in vs.values()):
            summary["deferred"] += 1
            continue
        refs = {"ai": refs_state.refs[vs["ai"]], "user": refs_state.refs[vs["user"]]}
        want = _ns_key(sid, dialogue, _ns_v8(vs, refs, levels))
        v8s.append(f"{sid}:{want.rsplit('-', 2)[-2]}")
        if row.example_audio_url and row.example_audio_url.endswith(want):
            summary["hit"] += 1
            continue
        if stop_round:
            summary["deferred"] += 1
            continue
        try:
            rejected = storage.strict_exists(_rejected_key(want))
            present = not rejected and storage.strict_exists(want)
        except Exception:
            logger.warning("프리셋 %s S3 확인 실패, 미룸", sid, exc_info=True)
            s3_errors.append(str(sid))
            summary["deferred"] += 1
            continue
        if rejected:
            summary["rejected"] += 1
            continue
        if present:
            try:
                reused = await _reuse(sid, want, storage)
            except Exception:
                logger.warning("프리셋 %s S3 확인 실패, 미룸", sid, exc_info=True)
                s3_errors.append(str(sid))
                summary["deferred"] += 1
                continue
            summary["reused" if reused else "rejected"] += 1
            continue
        job = _ensure_job(sid, "preset_prebake", expect_refs=refs, exclude=exclude)
        key = await asyncio.shield(job.task)
        if key == want:
            summary["baked"] += 1
        else:
            summary["deferred"] += 1
            if job.fail_reason in ("no_ns", "voice_ref"):
                stop_round = True
                mismatch = f"{sid}:{job.fail_reason}"
    log_metric(
        "example_preset_prebake",
        refs=",".join(f"{k}:{v}" for k, v in sorted(refs_state.refs.items())),
        unreachable=",".join(refs_state.unreachable) or None,
        v8=",".join(v8s) or None,
        mismatch=mismatch,
        s3_error=",".join(s3_errors) or None,
        **summary,
    )
    return "done" if summary["deferred"] == 0 else "counted"


async def _reuse(scenario_id: int, want: str, storage: RecordingStorageService) -> bool:
    """이미 있는 기대 키로 행을 바꾼다, 그사이 탈락됐으면 바꾸지 않는다"""
    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    async with lock:
        if storage.strict_exists(_rejected_key(want)):
            return False
        await _point_row(scenario_id, want, "preset_prebake")
    return True


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


async def prebake_presets(max_rounds: int | None = None) -> None:
    """서버 시작 때 프리셋 예시를 지금 ref 지문으로 맞춘다"""
    settings = get_settings()
    if not _prebake_gate(settings):
        return
    driver = _loop_driver()
    if driver.running:
        driver.wake.set()
        return
    driver.running = True
    driver.task = asyncio.current_task()
    limit = _NS_PRESET_ROUNDS if max_rounds is None else max_rounds
    state: dict = {"rounds": 0}
    counted = 0
    began = time.monotonic()
    try:
        while True:
            driver.wake.clear()
            state["rounds"] += 1
            try:
                result = await _prebake_round(settings, state)
            except Exception:
                logger.exception("프리셋 예시 미리 굽기 라운드 실패")
                result = "counted"
            woken = driver.wake.is_set()
            if result in ("done", "end") and not woken:
                return
            counted += result == "counted"
            if not woken and (counted >= limit or time.monotonic() - began > _PREBAKE_WALL_SEC):
                logger.warning("프리셋 예시 미리 굽기 못 끝냄(라운드 %s)", counted)
                return
            if woken:
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(driver.wake.wait(), _NS_PRESET_RETRY_SEC)
    finally:
        driver.running = False
        driver.task = None


def trigger_prebake() -> bool:
    """내부 API: 미리 굽기 라운드를 지금 돌린다, 이미 돌고 있으면 깨우기만"""
    settings = get_settings()
    if not _prebake_gate(settings):
        raise ExampleOpError(409, "예시 N 굽기가 꺼져 있다")
    cooldown = _loop_cooldown()
    for sid in PRESET_MAP:
        cooldown.pop(sid, None)
    driver = _loop_driver()
    if driver.running:
        driver.wake.set()
        return True
    driver.task = asyncio.get_running_loop().create_task(prebake_presets(max_rounds=1))
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
    _check_op(settings, scenario_id)
    want, refs, _ = await _expected_preset_key(settings, scenario_id)
    _check_op(settings, scenario_id)
    storage = RecordingStorageService(settings)
    backup = _backup_path(want, stamp)
    try:
        backed_up = storage.copy(want, backup)
    except Exception as exc:
        raise ExampleOpError(503, f"백업 복사 실패: {exc}") from exc
    _loop_cooldown().pop(scenario_id, None)
    _ensure_job(scenario_id, "rebake", expect_refs=refs, force=True)
    return {"key": want, "backup": backup if backed_up else None}


async def restore_preset(scenario_id: int, prev: str, *, stamp: str) -> dict:
    """내부 API: prev 객체를 지금 기대 키로 되살린다"""
    settings = get_settings()
    if not _ns_enabled(settings):
        raise ExampleOpError(409, "예시 N 굽기가 꺼져 있다")
    _check_op(settings, scenario_id)
    want, _, _ = await _expected_preset_key(settings, scenario_id)
    if not prev.startswith(f"examples/prev/{want.rsplit('/', 1)[-1]}.") or prev.endswith(".rejected"):
        raise ExampleOpError(409, "prev 가 지금 기대 키의 백업이 아니다")
    storage = RecordingStorageService(settings)
    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    async with lock:
        _check_op(settings, scenario_id)
        try:
            backed_up = storage.copy(want, _backup_path(want, stamp))
            restored = storage.copy(prev, want)
        except Exception as exc:
            raise ExampleOpError(503, f"복사 실패: {exc}") from exc
        if not restored:
            raise ExampleOpError(404, f"prev 없음: {prev}")
        await _point_row(scenario_id, want, "restore")
    return {"key": want, "backup": _backup_path(want, stamp) if backed_up else None}


def _own_example_key(scenario_id: int, key: str) -> bool:
    """이 시나리오의 예시 키(plain·-q·-qn) 모양인지"""
    return re.fullmatch(rf"examples/{scenario_id}-[0-9a-f]{{8}}(?:-q|-[0-9a-f]{{8}}-qn)?\.wav", key) is not None


async def reject_preset(scenario_id: int, restore_key: str) -> dict:
    """내부 API: 지금 ns 객체를 탈락으로 옮기고 행을 restore_key 로 되돌린다"""
    settings = get_settings()
    if not _ns_enabled(settings):
        raise ExampleOpError(409, "예시 N 굽기가 꺼져 있다")
    if scenario_id not in PRESET_MAP:
        raise ExampleOpError(404, f"프리셋 아님: {scenario_id}")
    if not _own_example_key(scenario_id, restore_key):
        raise ExampleOpError(400, f"restore_key 가 이 프리셋의 예시 키가 아니다: {restore_key}")
    storage = RecordingStorageService(settings)
    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    async with lock, AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        if row is None or is_deleted(row):
            raise ExampleOpError(404, f"프리셋 행 없음: {scenario_id}")
        _check_op(settings, scenario_id, skip_ok=True)
        current = row.example_audio_url
        dialogue = PRESET_MAP[scenario_id]["example_dialogue"]
        current_ns = _ns_cached(scenario_id, dialogue, current)
        if current_ns is None:
            raise ExampleOpError(409, "지금 행이 ns 키가 아니다")
        if restore_key == current_ns:
            raise ExampleOpError(400, "restore_key 가 지금 탈락시킬 키와 같다")
        if not storage.exists(restore_key):
            raise ExampleOpError(409, f"되돌릴 객체 없음: {restore_key}")
        if not storage.copy(current_ns, _rejected_key(current_ns)):
            storage.put_marker(_rejected_key(current_ns))
        row.example_audio_url = restore_key
        await db.commit()
    try:
        deleted = storage.delete(current_ns)
    except Exception:
        logger.warning("탈락 ns 객체 삭제 실패(행·표지는 반영됨): %s", current_ns, exc_info=True)
        deleted = False
    log_metric("example_tts", scenario_id=scenario_id, trigger="reject", prev_key=current, key=restore_key,
               deleted=deleted)
    return {"rejected": current_ns, "restored": restore_key, "deleted": deleted}


_LLM_TIMEOUT_SEC = 60.0
_STREAM_END = object()


class _StreamFailed:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _StreamAbortedError(Exception):
    """대본 스트림이 끊겨 굽기를 멈춤"""


class _ArrayStreamParser:
    """JSON 배열 출력을 조금씩 받아 최상위 원소가 닫힐 때마다 돌려준다"""

    def __init__(self) -> None:
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._buf: list[str] = []
        self._started = False
        self.closed = False

    def feed(self, text: str) -> list[object]:
        out: list[object] = []
        for ch in text:
            if self.closed:
                continue
            if not self._started:
                if ch == "[":
                    self._started, self._depth = True, 1
                continue
            if self._depth >= 2:
                self._buf.append(ch)
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                continue
            if ch == '"':
                self._in_string = True
            elif ch in "{[":
                if self._depth == 1:
                    self._buf = [ch]
                self._depth += 1
            elif ch in "}]":
                self._depth -= 1
                if self._depth == 1 and self._buf:
                    with contextlib.suppress(ValueError):
                        out.append(json.loads("".join(self._buf)))
                    self._buf = []
                elif self._depth == 0:
                    self.closed = True
        return out


def _snapshot_usage(stream) -> object:
    try:
        return stream.current_message_snapshot.usage
    except Exception:
        return None


async def _stream_dialogue(settings: Settings, row: ScenarioORM, queue: asyncio.Queue) -> list[dict]:
    """대본을 스트리밍으로 받아 완성된 턴을 큐에 넣고, 끝나면 검증한다"""
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    parser = _ArrayStreamParser()
    turns: list[dict] = []
    parts: list[str] = []
    stream = None
    try:
        async with asyncio.timeout(_LLM_TIMEOUT_SEC), client.messages.stream(
            model=settings.llm_analysis_model,
            max_tokens=1500,
            system=_EXAMPLE_GEN_SYSTEM,
            messages=[MessageParam(role="user", content=_dialogue_prompt(row))],
            timeout=httpx.Timeout(_LLM_TIMEOUT_SEC, connect=5.0),
        ) as stream:
            async for text in stream.text_stream:
                parts.append(text)
                for element in parser.feed(text):
                    turn = _normalize_turn(element)
                    if turn is not None:
                        turns.append(turn)
                        queue.put_nowait(turn)
            final = await stream.get_final_message()
    except BaseException:
        log_llm_usage("anthropic", settings.llm_analysis_model, "example_dialogue",
                      usage=_snapshot_usage(stream) if stream is not None else None,
                      user_id=getattr(row, "user_id", None), scenario_id=getattr(row, "scenario_id", None), ok=False)
        raise
    log_llm_usage("anthropic", settings.llm_analysis_model, "example_dialogue", usage=getattr(final, "usage", None),
                  user_id=getattr(row, "user_id", None), scenario_id=getattr(row, "scenario_id", None))
    if getattr(final, "stop_reason", None) != "end_turn" or _parse_dialogue_text("".join(parts)) != turns:
        raise ValueError("스트림 대본 검증 실패")
    return turns


async def _produce_dialogue(
    settings: Settings,
    scenario_id: int,
    queue: asyncio.Queue,
    committed: asyncio.Future,
    job: _BakeJob,
    released: asyncio.Event,
) -> None:
    """대본을 만들어 commit, 스트림이 실패하면 비스트리밍으로 한 번 더"""
    signalled = False
    try:
        lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
        async with lock, AsyncSessionLocal() as db:
            row = await db.get(ScenarioORM, scenario_id)
            if row is None or is_deleted(row):
                raise ScenarioNotFoundError(f"시나리오 {scenario_id} 없음")
            if row.example_dialogue:
                dialogue = row.example_dialogue
                for turn in dialogue:
                    queue.put_nowait(turn)
                queue.put_nowait(_STREAM_END)
                signalled = True
                _ready(job, dialogue)
            else:
                streamed = False
                try:
                    dialogue = await _stream_dialogue(settings, row, queue)
                    queue.put_nowait(_STREAM_END)
                    signalled = streamed = True
                except Exception as exc:
                    logger.warning("시나리오 %s 대본 스트림 실패, 비스트리밍으로 다시: %s", scenario_id, exc)
                    job.stream_failed = True
                    queue.put_nowait(_StreamFailed(exc))
                    signalled = True
                    if job.phase == "baking":
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(released.wait(), _LLM_TIMEOUT_SEC)
                    async with asyncio.timeout(_LLM_TIMEOUT_SEC):
                        dialogue = await _generate_custom_dialogue(
                            settings, row, attempt=2, timeout=_LLM_TIMEOUT_SEC
                        )
                row.example_dialogue = dialogue
                await db.commit()
                if streamed:
                    _ready(job, dialogue)
        _resolve_future(committed, dialogue)
    except BaseException as exc:
        if not signalled:
            job.stream_failed = True
            queue.put_nowait(_StreamFailed(exc))
        if not committed.done():
            committed.set_exception(exc if isinstance(exc, Exception) else RuntimeError("대본 생성 취소"))
        raise


def _ready(job: _BakeJob, dialogue: list[dict]) -> None:
    """대본이 확정되면 요청이 남은 시간을 셀 수 있게 알린다"""
    job.turns_total = len(dialogue)
    _resolve_future(job.dialogue, dialogue)


async def _consume_turns(
    settings: Settings,
    queue: asyncio.Queue,
    job: _BakeJob,
    voices: dict[str, str],
    levels: dict[str, float],
    expect_refs: dict[str, str] | None,
    exclude: tuple[str, ...],
    released: asyncio.Event,
) -> tuple:
    """첫 턴이 오면 굽기 시작, 결과는 예외 대신 값으로 돌려준다"""
    try:
        return await _consume(settings, queue, job, voices, levels, expect_refs, exclude)
    finally:
        released.set()


async def _consume(
    settings: Settings,
    queue: asyncio.Queue,
    job: _BakeJob,
    voices: dict[str, str],
    levels: dict[str, float],
    expect_refs: dict[str, str] | None,
    exclude: tuple[str, ...],
) -> tuple:
    first = await queue.get()
    if first is _STREAM_END or isinstance(first, _StreamFailed):
        return ("aborted",)
    consumed: list[dict] = []

    async def _turns():
        item = first
        while True:
            if job.stream_failed or isinstance(item, _StreamFailed):
                raise _StreamAbortedError()
            if item is _STREAM_END:
                return
            consumed.append(item)
            yield item
            item = await queue.get()

    job.phase = "queued"
    async with _prebake_semaphore():
        if job.stream_failed:
            return ("aborted",)
        job.phase = "baking"
        try:
            async with asyncio.timeout(_CUSTOM_NS_BUDGET_SEC):
                pcm, info, refs = await _synthesize_qwen_ns(
                    _BakeClient(settings, job, probe=expect_refs is None), _turns(), voices, levels=levels,
                    expect_refs=expect_refs, on_turn=job.turn_done, exclude=exclude,
                )
        except _StreamAbortedError:
            return ("aborted",)
        except QwenUnreachableError as exc:
            return ("aborted", exc.worker)
        except (QwenTTSUnavailableError, TimeoutError) as exc:
            return ("fail", _ns_reason(exc))
        except Exception:
            logger.exception("예시 대본 스트리밍 굽기 중 뜻밖의 오류")
            return ("fail", "synth_failed")
    return ("ok", pcm, info, refs, consumed)


async def _bake_streaming(
    settings: Settings,
    storage: RecordingStorageService,
    scenario_id: int,
    trigger: str,
    job: _BakeJob,
    *,
    expect_refs: dict[str, str] | None,
    exclude: tuple[str, ...],
) -> str | None:
    """대본 스트림과 굽기를 겹쳐 돌린다"""
    loop = asyncio.get_running_loop()
    levels = _levels(settings)
    async with AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        if row is None or is_deleted(row):
            _resolve_future(job.dialogue, [])
            return None
        voices = _ns_voices(settings, row, scenario_id, [])
        await db.commit()
    queue: asyncio.Queue = asyncio.Queue()
    committed: asyncio.Future = loop.create_future()
    started = now_ms()
    released = asyncio.Event()
    producer = loop.create_task(_produce_dialogue(settings, scenario_id, queue, committed, job, released))
    consumer = loop.create_task(_consume_turns(settings, queue, job, voices, levels, expect_refs, exclude, released))
    try:
        rendered = await consumer
        dialogue = await committed
    finally:
        for task in (producer, consumer):
            if not task.done():
                task.cancel()
        await asyncio.gather(producer, consumer, return_exceptions=True)
    _resolve_future(job.dialogue, dialogue)
    job.turns_total = len(dialogue)
    if rendered[0] == "ok" and rendered[4] == dialogue:
        _, pcm, info, refs, _ = rendered
        job.phase = "done"
        return await _store_ns(settings, storage, scenario_id, dialogue, voices, levels, pcm, info, refs, trigger,
                               started)
    if rendered[0] == "fail":
        job.fail_reason = rendered[1]
        job.phase = "done"
        return await _fallback_plain(settings, storage, scenario_id, dialogue, trigger, rendered[1], job.attempts)
    skip = (*exclude, *(w for w in rendered[1:] if w))
    job.phase = "queued"
    async with _prebake_semaphore():
        job.phase = "baking"
        try:
            return await _bake_ns(settings, storage, scenario_id, dialogue, trigger, job,
                                  expect_refs=expect_refs, exclude=skip, force=False)
        finally:
            job.phase = "done"
