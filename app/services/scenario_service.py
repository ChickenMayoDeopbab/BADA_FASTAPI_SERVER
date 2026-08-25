from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.enums import ALL_DIFFICULTIES, ALL_PERSONALITIES, ScenarioCategory
from app.core.preset_scenarios import PRESET_MAP, PRESET_SCENARIOS, scenario_to_info
from app.core.prompt_matrix import PERSONALITY_BASE
from app.core.tts_voices import parse_speaker, pick_voice_id
from app.db.models import ScenarioORM, is_deleted
from app.schemas.scenario import (
    CustomScenarioResponse,
    CustomSessionRequest,
    GenerateDetailScenario,
    ScenarioInfo,
    ScenarioListResponse,
    ScriptTurnContext,
)
from app.services.recording_storage import RecordingStorageService

logger = logging.getLogger(__name__)

_IMAGE_URL_TTL_SEC = 3600


def _image_storage(rows: Sequence[ScenarioORM]) -> RecordingStorageService | None:
    if not any(row.scenario_image for row in rows):
        return None
    return RecordingStorageService(get_settings())


def _image_url(storage: RecordingStorageService | None, key: str | None) -> str | None:
    if storage is None or not key:
        return None
    try:
        return storage.presigned_url(key, expires_in=_IMAGE_URL_TTL_SEC)
    except Exception as e:
        logger.warning(
            "썸네일 URL 서명 실패, 이미지 없이 응답: %s: %s",
            type(e).__name__,
            e,
            extra={"s3_key": key},
        )
        return None


async def get_scenarios(
        db: AsyncSession,
        category: ScenarioCategory | None,
        user_id: int) -> ScenarioListResponse:
    stmt = select(ScenarioORM)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        result_list = PRESET_SCENARIOS
        if category is not None:
            result_list = [scenario for scenario in result_list if scenario["category"] == category]
        return ScenarioListResponse(scenarios=[scenario_to_info(s) for s in result_list])

    preset_rows: list[tuple[ScenarioORM, dict]] = []
    custom_rows: list[tuple[ScenarioORM, ScenarioCategory]] = []

    for row in rows:
        if row.is_custom:
            if is_deleted(row):
                continue
            if row.is_warmup:
                continue
            if row.user_id != user_id:
                continue
            row_category = ScenarioCategory(row.category)
            if category is not None and row_category != category:
                continue

            custom_rows.append((row, row_category))
            continue
        seed = PRESET_MAP.get(row.scenario_id)
        if seed is None:
            continue
        if category is not None and seed["category"] != category:
            continue
        preset_rows.append((row, seed))

    preset_rows.sort(key=lambda item: item[0].scenario_id)
    custom_rows.sort(key=lambda item:
        (
            getattr(item[0], "created_at", datetime.min),
            item[0].scenario_id,
        )
    )

    storage = _image_storage(rows)
    infos = [
        ScenarioInfo(
            scenario_id=row.scenario_id,
            title=row.title,
            content=row.content,
            category=seed["category"],
            difficulties=seed["difficulties"],
            personalities=seed["personalities"],
            scenario_image=_image_url(storage, row.scenario_image),
            tts_voice_id=row.tts_voice_id,
            ai_prompt=row.ai_prompt,
            is_custom=False,
        )
        for row, seed in preset_rows
    ]
    infos.extend(
        ScenarioInfo(
            scenario_id=row.scenario_id,
            title=row.title,
            content=row.content,
            category=row_category,
            difficulties=ALL_DIFFICULTIES,
            personalities=ALL_PERSONALITIES,
            scenario_image=_image_url(storage, row.scenario_image),
            tts_voice_id=row.tts_voice_id,
            ai_prompt=row.ai_prompt,
            is_custom=True,
        )
        for row in custom_rows
    )
    return ScenarioListResponse(scenarios=infos)


async def delete_custom_scenario(
        db: AsyncSession,
        scenario_id: int,
        user_id: int,
) -> bool:
    """본인 소유 커스텀 시나리오를 삭제"""
    row = await db.get(ScenarioORM, scenario_id, with_for_update=True)
    if row is None or not row.is_custom or row.user_id != user_id:
        return False
    if is_deleted(row):
        return False
    row.deleted_at = datetime.now(UTC).replace(tzinfo=None)
    await db.commit()
    return True


_SCENARIO_GEN_SYSTEM = """You are an expert scenario designer for a Korean phone call training application.
The user provides the scenario title, category, call purpose, call target, difficulty, and AI personality.
Generate a realistic phone call training scenario that matches them.
You must output ONLY valid JSON.
Do not output markdown, explanations, code fences, comments, or any additional text.
JSON schema:
{
  "content": "Short one-line scenario description in Korean (max 40 Korean characters)",
  "ai_prompt": "Pure role instruction in English for the AI character (max 150 characters, \
excluding personality or difficulty instructions)",
  "script": [
    {"step": 1, "ai_goal": "이 단계에서 상대역(AI)이 달성할 목표 (Korean)", \
"hint": "사용자가 무엇을 말해야 하는지 힌트 (Korean)"}
  ],
  "speaker": {"gender": "male|female", "age": "young|middle|old", "tone": "soft|neutral|rough"}
}
Create 3 to 5 steps ordered from the call opening to the closing, matching the call purpose.
Higher difficulty means the AI gives less guidance and demands more from the user.
"speaker" is the voice profile of the AI character answering the phone: gender/age inferred
from the call target (e.g. hospital desk staff -> female/middle), tone from the role,
personality and difficulty combined (e.g. rude complaint handler -> rough).
"""


def _normalize_script(raw: object) -> list[dict]:
    """AI가 생성한 script를 {step, ai_goal, hint} 형태로 정규화한다."""
    if not isinstance(raw, list):
        return []
    turns: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            turns.append({
                "step": int(item["step"]),
                "ai_goal": str(item["ai_goal"]),
                "hint": str(item.get("hint", "")),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(turns, key=lambda t: t["step"])


async def create_custom_scenario(
        db: AsyncSession,
        request: CustomSessionRequest,
        user_id: int,
) -> CustomScenarioResponse:
    """사용자 입력(제목/목적/상대)으로 AI가 단계별 script를 생성해 저장한다. 세션 생성은 Spring 소유."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY가 설정되어 있지 않습니다.")

    user_msg = (
        f"Scenario title: {request.title}\n"
        f"Category: {request.category.value}\n"
        f"Call purpose: {request.call_purpose}\n"
        f"Call target: {request.call_target}\n"
        f"Difficulty: {request.difficulty.value}\n"
        f"Personality: {request.personality.value} — {PERSONALITY_BASE[request.personality]}"
    )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    ai_response = await client.messages.create(
        model=settings.llm_analysis_model,
        max_tokens=1000,
        system=_SCENARIO_GEN_SYSTEM,
        messages=[MessageParam(role="user", content=user_msg)],
    )

    raw = ai_response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data: dict = json.loads(raw.strip())

    gender, age, tone = parse_speaker(data.get("speaker"))
    voice_id = pick_voice_id(gender, age, tone)

    script = _normalize_script(data.get("script"))
    now = datetime.now(UTC).replace(tzinfo=None)

    scenario_orm = ScenarioORM(
        title=request.title,
        content=data["content"],
        category=request.category.value,
        scenario_image=None,
        tts_voice_id=voice_id,
        ai_prompt=data["ai_prompt"],
        call_target=request.call_target,
        call_purpose=request.call_purpose,
        user_id=user_id,
        is_custom=True,
        is_warmup=request.is_warmup,
        script=script,
        created_at=now,
    )
    db.add(scenario_orm)
    await db.commit()
    await db.refresh(scenario_orm)

    return CustomScenarioResponse(
        scenario=GenerateDetailScenario(
            scenario_id=scenario_orm.scenario_id,
            title=request.title,
            content=data["content"],
            ai_prompt=data["ai_prompt"],
            tts_voice_id=voice_id,
            script=[ScriptTurnContext(**turn) for turn in script],
        ),
        created_at=now,
    )
