from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from app.core.enums import Difficulty, Personality, ScenarioCategory, SessionType

class CallSession(BaseModel):
    session_id: int
    scenario_id: int
    user_id: int
    session_type: SessionType
    personality: Personality
    difficulty: Difficulty
    created_at: datetime

class Scenario(BaseModel):
    scenario_id: int
    title: str = Field(..., max_length=50)
    content: str
    scenario_image: Optional[str] = None
    tts_voice_id: Optional[str] = None
    ai_prompt: str = None
    user_id: Optional[int] = None
    is_custom: bool = False
    call_target: str = Field(..., max_length=100)
    call_purpose: str = Field(..., max_length=200)
    created_at: datetime


class ScenarioInfo(BaseModel):
    scenario_id: int
    title: str
    content: str
    category: ScenarioCategory
    difficulties: list[Difficulty]
    personalities: list[Personality]
    scenario_image: Optional[str] = None
    tts_voice_id: Optional[str] = None
    ai_prompt: str
    is_custom: bool

class ScenarioListResponse(BaseModel):
    scenarios: List[ScenarioInfo]


class CreateSessionRequest(BaseModel):
    scenario_id: int
    session_type: SessionType = SessionType.TRAINING
    personality: Personality = Personality.NEUTRAL
    difficulty: Difficulty = Difficulty.MEDIUM

class CreateSessionResponse(BaseModel):
    session_id: int
    scenario_id: int
    session_type: SessionType
    personality: Personality
    difficulty: Difficulty
    ai_prompt: str
    tts_voice_id: Optional[str]
    created_at: datetime
    message: str = "훈련 세선이 생성되었습니다."


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
    tts_voice_id: Optional[str]

class CustomSessionResponse(BaseModel):
    session_id: int
    scenario: GenerateDetailScenario
    personality: Personality
    difficulty: Difficulty
    created_at: datetime
    message: str = "커스텀 훈련 세션이 생성되었습니다."