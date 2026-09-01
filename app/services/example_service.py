from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import weakref
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.concurrency import loop_semaphore
from app.core.config import Settings, get_settings
from app.core.metrics import log_metric, now_ms
from app.core.preset_scenarios import PRESET_MAP
from app.core.tts_voices import pick_example_user_voice
from app.db.base import AsyncSessionLocal
from app.db.models import ScenarioORM, is_deleted
from app.schemas.scenario import ExampleConversationResponse, ExampleTurn
from app.services.qwen_tts import QwenTTSClient, QwenTTSUnavailableError
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
    raw = ai_response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return _normalize_dialogue(json.loads(raw.strip()))


async def _one_text(text: str) -> AsyncIterator[str]:
    yield text


async def _synth_turn(tts_client: ElevenLabsTTSClient, voice_id: str, text: str) -> bytes:
    session = await tts_client.open(voice_id)
    chunks: list[bytes] = []
    try:
        await session.begin()
        async for pcm in session.stream(_one_text(text)):
            chunks.append(pcm)
    finally:
        await session.aclose()
    return b"".join(chunks)


async def _synthesize(
    tts_client: ElevenLabsTTSClient,
    dialogue: list[dict],
    ai_voice: str,
    user_voice: str,
) -> bytes:
    semaphore = asyncio.Semaphore(_TTS_MAX_CONCURRENCY)

    async def _synth_limited(turn: dict) -> bytes:
        voice = ai_voice if turn["speaker"] == "ai" else user_voice
        async with semaphore:
            return await _synth_turn(tts_client, voice, turn["text"])

    parts = await asyncio.gather(*(_synth_limited(turn) for turn in dialogue))
    return _TURN_GAP_PCM.join(parts)


async def _synthesize_qwen(qwen_client: QwenTTSClient, dialogue: list[dict]) -> bytes:
    """GPU 서버로 합성"""
    parts: list[bytes] = []
    async with qwen_client.connect() as session:  # 턴마다 TCP 재연결하지 않게
        for turn in dialogue:
            voice = "ai" if turn["speaker"] == "ai" else "user"
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
    voices = ["qwen-ai", "qwen-user"] if qwen else [ai_voice, user_voice]
    payload = json.dumps([dialogue, *voices], ensure_ascii=False)
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
) -> tuple[list[dict], str | None]:
    """시나리오 생성 직후 오디오도 만들어 지도록"""
    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    async with lock:
        if row is not None:
            await db.refresh(row, ["example_dialogue", "example_audio_url"])
        dialogue = await _resolve_dialogue(db, settings, row, seed)
        if storage is None:
            return dialogue, None
        key = await _bake_locked(
            db, settings, row, scenario_id, dialogue, storage, trigger=trigger
        )
        return dialogue, key


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
    ai_voice = (row.tts_voice_id if row is not None else None) or settings.elevenlabs_voice_id
    user_voice = pick_example_user_voice(ai_voice)
    key_qwen = _audio_key(scenario_id, dialogue, ai_voice, user_voice, qwen=True)
    key_eleven = _audio_key(scenario_id, dialogue, ai_voice, user_voice)

    cached = row.example_audio_url if row is not None else None
    for candidate in (key_qwen, key_eleven):
        if cached and cached.endswith(candidate):
            return candidate

    qwen_client = QwenTTSClient(settings)
    use_qwen = await qwen_client.healthy()

    started = now_ms()
    engine, reason, pcm = "eleven", None, None
    if use_qwen:
        try:
            pcm = await _synthesize_qwen(qwen_client, dialogue)
            engine = "qwen"
        except QwenTTSUnavailableError as exc:
            reason = "synth_failed"
            logger.warning("Qwen TTS 합성 실패 — ElevenLabs로 전체 재합성: %s", exc)
    else:
        reason = "unhealthy" if qwen_client.enabled else "disabled"

    if pcm is None:
        engine = "eleven"
        tts_client = ElevenLabsTTSClient(settings)
        pcm = await _synthesize(tts_client, dialogue, ai_voice, user_voice)

    key = key_qwen if engine == "qwen" else key_eleven
    stored_key = storage.upload_wav(key, pcm)
    log_metric(
        "example_tts",
        scenario_id=scenario_id,
        trigger=trigger,
        engine=engine,
        fallback_reason=reason,
        turns=len(dialogue),
        duration_ms=round(now_ms() - started, 1),
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
        async with (asyncio.timeout(_PREBAKE_TIMEOUT_SEC), _prebake_semaphore(),
                    AsyncSessionLocal() as db):
            row = await db.get(ScenarioORM, scenario_id)
            if row is None or is_deleted(row) or row.example_audio_url:
                return
            await _resolve_and_bake(db, settings, row, None, scenario_id,
                                    RecordingStorageService(settings), trigger="prebake")
    except Exception:
        logger.exception("시나리오 %s 예시 오디오 미리 굽기 실패", scenario_id)
