from enum import StrEnum


class SessionType(StrEnum):
    WARMUP = "WARMUP"
    SCENARIO = "SCENARIO"
    CUSTOM = "CUSTOM"

class AiPersonality(StrEnum):
    """AI 성격"""

    KIND = "KIND"       # 친절
    NORMAL = "NORMAL"   # 보통
    TOUGH = "TOUGH"     # 까다로움
    RUDE = "RUDE"       # 진상

class Difficulty(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ScenarioCategory(StrEnum):
    RESTAURANT = "restaurant"
    HOSPITAL = "hospital"
    COMPLAINT = "complaint"
    DELIVERY = "delivery"
    BANK = "bank"
    CUSTOM = "custom"

ALL_DIFFICULTIES: list[Difficulty] = [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH]
ALL_PERSONALITIES: list[AiPersonality] = [
    AiPersonality.KIND,
    AiPersonality.NORMAL,
    AiPersonality.TOUGH,
    AiPersonality.RUDE,
]
