from enum import StrEnum


class SessionType(StrEnum):
    WARMUP = "WARMUP"
    SCENARIO = "SCENARIO"
    CUSTOM = "CUSTOM"

class FileType(StrEnum):
    PROFILE = "PROFILE"
    SCENARIO_PROFILE = "SCENARIO_PROFILE"
    CALL_FILE = "CALL_FILE"

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
    WORK = "work"
    DAILY = "daily"
    SCHOOL = "school"
    OTHER = "other"

ALL_DIFFICULTIES: list[Difficulty] = [Difficulty.LOW, Difficulty.MEDIUM, Difficulty.HIGH]
ALL_PERSONALITIES: list[AiPersonality] = [
    AiPersonality.KIND,
    AiPersonality.NORMAL,
    AiPersonality.TOUGH,
    AiPersonality.RUDE,
]

class SpeakerGender(StrEnum):
    MALE = "male"
    FEMALE = "female"

class SpeakerAge(StrEnum):
    YOUNG = "young"
    MIDDLE = "middle"
    OLD = "old"

class SpeakerTone(StrEnum):
    SOFT = "soft"
    NEUTRAL = "neutral"
    ROUGH = "rough"
