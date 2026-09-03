from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import AiPersonality, Difficulty, ScenarioCategory
from app.core.timeutil import KstDatetime


class ScenarioInfo(BaseModel):
    scenario_id: int
    title: str
    content: str
    category: ScenarioCategory
    difficulties: list[Difficulty]
    personalities: list[AiPersonality]
    scenario_image: str | None = Field(
        default=None,
        description="시나리오별 고유 이미지의 임시 접근 URL",
    )
    tts_voice_id: str | None = None
    ai_prompt: str
    is_custom: bool
    is_copied: bool = False
    practice_count: int = 0

class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioInfo]


ScenarioRecommendationReason = Literal[
    "CUSTOM_NOT_PRACTICED",
    "NOT_PRACTICED",
    "LONGEST_ABSENT",
]


class ScenarioRecommendationResponse(BaseModel):
    scenario: ScenarioInfo
    reason: ScenarioRecommendationReason
    category_icon_url: str | None = Field(
        default=None,
        description="시나리오 카테고리에 대응하는 공통 아이콘의 임시 접근 URL",
    )


class ExampleTurn(BaseModel):
    speaker: Literal["ai", "user"]
    text: str


class ExampleConversationResponse(BaseModel):
    scenario_id: int
    dialogue: list[ExampleTurn]
    audio_url: str | None = None


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
    ai_prompt: str = Field(serialization_alias="aiPrompt")
    script: list[ScriptTurnContext]
    tts_voice_id: str | None = Field(default=None, serialization_alias="ttsVoiceId")
    model_config = ConfigDict(populate_by_name=True)

class CustomSessionRequest(BaseModel):
    """커스텀 시나리오 생성, 생성은 Spring이 여기선 AI가 시나리오만 만들고 저장 후 반환"""
    title: str = Field(..., min_length=1, max_length=50)
    category: ScenarioCategory = ScenarioCategory.OTHER
    call_target: str = Field(..., min_length=2, max_length=100)
    call_purpose: str = Field(..., min_length=5, max_length=200)
    personality: AiPersonality = AiPersonality.NORMAL
    difficulty: Difficulty = Difficulty.MEDIUM
    is_warmup: bool = False  # 실전 워밍업용

class GenerateDetailScenario(BaseModel):
    scenario_id: int
    title: str
    content: str
    ai_prompt: str
    tts_voice_id: str | None
    script: list[ScriptTurnContext]

class CustomScenarioResponse(BaseModel):
    scenario: GenerateDetailScenario
    created_at: KstDatetime
    message: str = "커스텀 시나리오가 생성되었습니다."
