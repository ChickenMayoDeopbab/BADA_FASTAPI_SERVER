from dataclasses import dataclass
from enum import StrEnum

from app.core.enums import AiPersonality

__all__ = [
    "AiPersonality",
    "AiEmotion",
    "LLMEventType",
    "LLMEvent",
    "ScenarioTurn",
    "TurnContext",
]


class AiEmotion(StrEnum):
    """AI 감정 상태"""

    NEUTRAL = "NEUTRAL"         # 평상시
    FRIENDLY = "FRIENDLY"       # 긍정
    ANNOYED = "ANNOYED"         # 약간 짜증냄
    ANGRY = "ANGRY"             # 화남
    APOLOGETIC = "APOLOGETIC"   # 미안함

class LLMEventType(StrEnum):
    """LLM 이벤트 상태"""

    EMOTION_RESOLVED = "EMOTION_RESOLVED"   # 감정 확정
    TEXT_DELTA = "TEXT_DELTA"               # 텍스트 스트리밍
    STEP_DONE = "STEP_DONE"                 # 현재 시나리오 step 완료 신호
    END_CALL = "END_CALL"                   # 통화 종료 신호
    TURN_END = "TURN_END"                   # 응답 완료
    SAFETY_BLOCK = "SAFETY_BLOCK"           # 부적절한 응답
    ERROR = "ERROR"                         # 서버/네트워크 에러

@dataclass
class LLMEvent:
    """오케스트레이터/websocket 프레임과의 계약 객체."""

    type: LLMEventType
    text: str = ""
    emotion: AiEmotion | None = None

@dataclass
class ScenarioTurn:
    """시나리오 턴별 스크립트의 한 턴"""

    step: int # 1부터
    ai_goal: str # 이 턴에서 AI가 말할거
    hint: str = "" # 사용자에게 힌트 보여주는거

@dataclass
class TurnContext:
    """LLM 호출에 필요한 입력"""

    personality: AiPersonality  # AI 성격
    scenario_title: str         # 시나리오 제목
    scenario_role: str          # AI가 연기할 역할
    script: list[ScenarioTurn]
    current_step: int           # 몇 번째
    history: list[dict]         # 나눴던 대화
    user_utterance: str         # 사용자가 말한 내용
    scenario_prompt: str = ""   # 시나리오별 역할 지시
