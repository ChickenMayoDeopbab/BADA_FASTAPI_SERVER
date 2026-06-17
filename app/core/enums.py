from enum import StrEnum


class SessionType(StrEnum):
    WARMUP = "warmup"
    TRAINING = "training"

class Personality(StrEnum):
    KIND = "kind"
    NEUTRAL = "neutral"
    TOUGH = "tough"
    RUDE = "rude"

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
