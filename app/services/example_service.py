from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.preset_scenarios import PRESET_MAP
from app.core.tts_voices import pick_example_user_voice
from app.db.models import ScenarioORM
from app.schemas.scenario import ExampleConversationResponse, ExampleTurn
from app.services.recording_storage import RecordingStorageService
from app.services.tts import ElevenLabsTTSClient

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_TURN_GAP_MS = 400
_TURN_GAP_PCM = b"\x00" * (_SAMPLE_RATE * _TURN_GAP_MS // 1000 * 2)

_generation_locks: dict[int, asyncio.Lock] = {}


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
    parts: list[bytes] = []
    for turn in dialogue:
        voice = ai_voice if turn["speaker"] == "ai" else user_voice
        parts.append(await _synth_turn(tts_client, voice, turn["text"]))
    return _TURN_GAP_PCM.join(parts)


def _audio_key(scenario_id: int, dialogue: list[dict], ai_voice: str, user_voice: str) -> str:
    """대본이나 보이스가 바뀌면 키가 달라져 캐시가 자동 무효화"""
    payload = json.dumps([dialogue, ai_voice, user_voice], ensure_ascii=False)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    return f"examples/{scenario_id}-{digest}.wav"


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
        if row.user_id != user_id:
            raise ScenarioNotFoundError(f"시나리오 {scenario_id} 없음")
        seed = None
    else:
        seed = PRESET_MAP.get(scenario_id)
        if seed is None:
            raise ScenarioNotFoundError(f"시나리오 {scenario_id} 없음")

    dialogue = await _resolve_dialogue(db, settings, row, seed)
    turns = [ExampleTurn(**t) for t in dialogue]

    if not settings.s3_bucket:
        return ExampleConversationResponse(scenario_id=scenario_id, dialogue=turns, audio_url=None)

    ai_voice = (row.tts_voice_id if row is not None else None) or settings.elevenlabs_voice_id
    user_voice = pick_example_user_voice(ai_voice)
    key = _audio_key(scenario_id, dialogue, ai_voice, user_voice)

    lock = _generation_locks.setdefault(scenario_id, asyncio.Lock())
    async with lock:
        if row is not None and row.example_audio_url and row.example_audio_url.endswith(key):
            return ExampleConversationResponse(
                scenario_id=scenario_id, dialogue=turns, audio_url=row.example_audio_url
            )

        tts_client = ElevenLabsTTSClient(settings)
        pcm = await _synthesize(tts_client, dialogue, ai_voice, user_voice)
        storage = RecordingStorageService(settings)
        audio_url = storage.upload_wav(key, pcm)

        if row is not None and audio_url:
            row.example_audio_url = audio_url
            await db.commit()

    return ExampleConversationResponse(scenario_id=scenario_id, dialogue=turns, audio_url=audio_url)
