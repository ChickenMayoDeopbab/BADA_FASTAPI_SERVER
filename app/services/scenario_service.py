from __future__ import annotations

import json
from datetime import datetime

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.enums import Difficulty, Personality, ScenarioCategory
from app.core.preset_scenarios import PRESET_MAP, PRESET_SCENARIOS, scenario_to_info
from app.core.prompt_matrix import PERSONALITY_BASE
from app.db.models import ScenarioORM
from app.schemas.scenario import (
    CustomScenarioResponse,
    CustomSessionRequest,
    GenerateDetailScenario,
    ScenarioInfo,
    ScenarioListResponse,
)


async def get_scenarios(
        db: AsyncSession,
        category: ScenarioCategory | None) -> ScenarioListResponse:
    stmt = select(ScenarioORM)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        result_list = PRESET_SCENARIOS
        if category:
            result_list = [s for s in result_list if s["category"] == category]
        return ScenarioListResponse(scenarios=[scenario_to_info(s) for s in result_list])

    infos = []
    for row in rows:
        if row.is_custom:
            if category and category != ScenarioCategory.CUSTOM:
                continue
            infos.append(ScenarioInfo(
                scenario_id=row.scenario_id,
                title=row.title,
                content=row.content,
                category=ScenarioCategory.CUSTOM,
                difficulties=[Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH],
                personalities=[
                    Personality.KIND,
                    Personality.NEUTRAL,
                    Personality.TOUGH,
                    Personality.RUDE,
                ],
                scenario_image=row.scenario_image,
                tts_voice_id=row.tts_voice_id,
                ai_prompt=row.ai_prompt,
                is_custom=row.is_custom,
            ))
            continue

        seed = PRESET_MAP.get(row.scenario_id)
        if seed is None:
            continue
        if category and seed["category"] != category:
            continue
        infos.append(ScenarioInfo(
            scenario_id=row.scenario_id,
            title=row.title,
            content=row.content,
            category=seed["category"],
            difficulties=seed["difficulties"],
            personalities=seed["personalities"],
            scenario_image=row.scenario_image,
            tts_voice_id=row.tts_voice_id,
            ai_prompt=row.ai_prompt,
            is_custom=row.is_custom,
        ))
    return ScenarioListResponse(scenarios=infos)


_SCENARIO_GEN_SYSTEM = """You are an expert scenario designer for a Korean phone call training application.
Your task is to generate a realistic phone call training scenario based on the user's input.
You must output ONLY valid JSON.
Do not output markdown, explanations, code fences, comments, or any additional text.
JSON schema:
{
  "title": "Scenario title (max 15 Korean characters)",
  "content": "Short one-line scenario description (max 40 Korean characters)",
  "ai_prompt": "Pure role instruction for the AI character (max 150 characters, \
excluding personality or difficulty instructions)"
}
"""


async def create_custom_scenario(
        db: AsyncSession,
        request: CustomSessionRequest,
        user_id: int,
) -> CustomScenarioResponse:
    """AI가 커스텀 시나리오를 생성해 저장한다. 세션 생성은 Spring 소유."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY가 설정되어 있지 않습니다.")

    user_msg = (
        f"Call purpose: {request.call_purpose}\n"
        f"Call target: {request.call_target}\n"
        f"Personality: {request.personality.value} — {PERSONALITY_BASE[request.personality]}"
    )

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    ai_response = await client.messages.create(
        model=settings.llm_analysis_model,
        max_tokens=500,
        system=_SCENARIO_GEN_SYSTEM,
        messages=[MessageParam(role="user", content=user_msg)],
    )

    raw = ai_response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data: dict = json.loads(raw.strip())

    now = datetime.utcnow()

    scenario_orm = ScenarioORM(
        title=data["title"],
        content=data["content"],
        scenario_image=None,
        tts_voice_id=None,
        ai_prompt=data["ai_prompt"],
        call_target=request.call_target,
        call_purpose=request.call_purpose,
        user_id=user_id,
        is_custom=True,
        created_at=now,
    )
    db.add(scenario_orm)
    await db.commit()
    await db.refresh(scenario_orm)

    return CustomScenarioResponse(
        scenario=GenerateDetailScenario(
            scenario_id=scenario_orm.scenario_id,
            title=data["title"],
            content=data["content"],
            ai_prompt=data["ai_prompt"],
            tts_voice_id=None,
        ),
        created_at=now,
    )
