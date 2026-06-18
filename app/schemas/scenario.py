from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import Difficulty, Personality, ScenarioCategory


class ScenarioInfo(BaseModel):
    scenario_id: int
    title: str
    content: str
    category: ScenarioCategory
    difficulties: list[Difficulty]
    personalities: list[Personality]
    scenario_image: str | None = None
    tts_voice_id: str | None = None
    ai_prompt: str
    is_custom: bool

class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioInfo]


# --- 내부용: Spring이 세션 생성 시 가져가는 시나리오 컨텍스트 ---
# JSON 키는 Spring ScenarioContext record와 정확히 일치해야 한다(aiRole, aiGoal).

class ScriptTurnContext(BaseModel):
    step: int
    ai_goal: str = Field(serialization_alias="aiGoal")
    hint: str = ""
    model_config = ConfigDict(populate_by_name=True)

class ScenarioContextResponse(BaseModel):
    title: str
    ai_role: str = Field(serialization_alias="aiRole")
    script: list[ScriptTurnContext]
    model_config = ConfigDict(populate_by_name=True)


# --- 커스텀 시나리오 생성 ---
# 세션 생성은 Spring 소유. 여기서는 AI가 시나리오만 만들어 저장하고 반환한다.

class CustomSessionRequest(BaseModel):
    call_target: str = Field(..., min_length=2, max_length=100)
    call_purpose: str = Field(..., min_length=5, max_length=200)
    personality: Personality = Personality.NEUTRAL
    difficulty: Difficulty = Difficulty.MEDIUM

class GenerateDetailScenario(BaseModel):
    scenario_id: int
    title: str
    content: str
    ai_prompt: str
    tts_voice_id: str | None

class CustomScenarioResponse(BaseModel):
    scenario: GenerateDetailScenario
    created_at: datetime
    message: str = "커스텀 시나리오가 생성되었습니다."
