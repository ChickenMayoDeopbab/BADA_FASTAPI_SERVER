from __future__ import annotations
import json
from datetime import datetime, timezone
from anthropic import AsyncAnthropic
from app.core.enums import ScenarioCategory, Difficulty, SessionType
from app.core.preset_scenarios import PRESET_MAP, PRESET_SCENARIOS, scenario_to_info
from app.core.prompt_matrix import PERSONALITY_BASE, build_system_prompt
from app.schemas.scenario import (
    CallSession, CreateSessionRequest, CreateSessionResponse,
    CustomSessionRequest, CustomSessionResponse,
    GenerateDetailScenario, Scenario, ScenarioListResponse,
)
from anthropic.types import MessageParam

_client = AsyncAnthropic()

_id_counter = 1000
def _next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter

def get_scenarios(category: ScenarioCategory | None,
                  difficulty: Difficulty | None) -> ScenarioListResponse:
    result = PRESET_SCENARIOS
    if category:
        result = [s for s in result if s["category"] == category]
    if difficulty:
        result = [s for s in result if difficulty in s["difficulties"]]

    return ScenarioListResponse(scenarios=[scenario_to_info(s) for s in result])

def create_session(
        request: CreateSessionRequest,
        user_id: int,
) -> tuple[CreateSessionResponse, CallSession]:
    scenario = PRESET_MAP.get(request.scenario_id)
    if scenario is None:
        raise ValueError("시나리오를 찾을 수 없습니다.")

    system_prompt = build_system_prompt(
        personality=request.personality,
        call_target=scenario["call_target"],
        call_purpose=scenario["call_purpose"],
        base_prompt=scenario["ai_prompt"],
    )
    now = datetime.now(timezone.utc)
    session_id = _next_id()

    db_record = CallSession(
        session_id=session_id,
        scenario_id=request.scenario_id,
        user_id=user_id,
        session_type=request.session_type,
        personality=request.personality,
        difficulty=request.difficulty,
        created_at=now,
    )

    response = CreateSessionResponse(
        session_id=session_id,
        scenario_id=request.scenario_id,
        session_type=request.session_type,
        personality=request.personality,
        difficulty=request.difficulty,
        ai_prompt=system_prompt,
        tts_voice_id=scenario["tts_voice_id"],
        created_at=now,
    )
    return response, db_record

_SCENARIO_GEN_SYSTEM = """You are an expert scenario designer for a Korean phone call training application.
Your task is to generate a realistic phone call training scenario based on the user's input.
You must output ONLY valid JSON.
Do not output markdown, explanations, code fences, comments, or any additional text.
JSON schema:
{
  "title": "Scenario title (max 15 Korean characters)",
  "content": "Short one-line scenario description (max 40 Korean characters)",
  "ai_prompt": "Pure role instruction for the AI character (max 150 characters, excluding personality or difficulty instructions)"
}
"""


async def create_custom_session(
        req: CustomSessionRequest,
        user_id: int,
) -> tuple[CustomSessionResponse, Scenario, CallSession]:
    user_msg = (
        f"Call purpose: {req.call_purpose}\n"
        f"Call target: {req.call_target}\n"
        f"Personality: {req.personality.value} — {PERSONALITY_BASE[req.personality]}"
    )

    ai_response = await _client.messages.create(
        model="claude-sonnet-4-20250514",
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

    now = datetime.now(timezone.utc)
    scenario_id = _next_id()
    session_id = _next_id()

    scenario_record = Scenario(
        scenario_id=scenario_id,
        title=data["title"],
        content=data["content"],
        scenario_image=None,
        tts_voice_id=None,
        ai_prompt=data["ai_prompt"],
        call_target=req.call_target,
        call_purpose=req.call_purpose,
        user_id=user_id,
        is_custom=True,
        created_at=now,
    )

    system_prompt = build_system_prompt(
        personality=req.personality,
        call_target=req.call_target,
        call_purpose=req.call_purpose,
        base_prompt=data["ai_prompt"],
    )

    # TODO: DB → INSERT INTO call_session (...) VALUES (...)
    db_session = CallSession(
        session_id=session_id,
        scenario_id=scenario_id,
        user_id=user_id,
        session_type=SessionType.TRAINING,
        personality=req.personality,
        difficulty=req.difficulty,
        created_at=now,
    )

    response = CustomSessionResponse(
        session_id=session_id,
        scenario=GenerateDetailScenario(
            scenario_id=scenario_id,
            title=data["title"],
            content=data["content"],
            ai_prompt=system_prompt,
            tts_voice_id=None,
        ),
        personality=req.personality,
        difficulty=req.difficulty,
        created_at=now,
    )
    return response, scenario_record, db_session