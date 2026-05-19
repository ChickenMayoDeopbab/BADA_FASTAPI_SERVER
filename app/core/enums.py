from enum import Enum

class SessionType(str, Enum):
    WARMUP = "warmup"
    TRAINING = "training"

class Personality(str, Enum):
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    PICKY = "picky"
    RUDE = "rude"

class Difficulty(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ScenarioCategory(str, Enum):
    RESTAURANT = "restaurant"
    HOSPITAL = "hospital"
    COMPLAINT = "complaint"
    DELIVERY = "delivery"
    BANK = "bank"
    CUSTOM = "custom"