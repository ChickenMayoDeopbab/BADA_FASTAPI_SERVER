import random
from contextlib import suppress
from dataclasses import dataclass

from app.core.enums import Difficulty, SpeakerAge, SpeakerGender, SpeakerTone


@dataclass(frozen=True)
class VoiceProfile:
    voice_id: str
    gender: SpeakerGender
    age: SpeakerAge
    tone: SpeakerTone
    label: str             # 인간용 메모
    el_voice_id: str | None = None  # ElevenLabs 폴백 보이스


_F, _M = SpeakerGender.FEMALE, SpeakerGender.MALE
_Y, _MID, _O = SpeakerAge.YOUNG, SpeakerAge.MIDDLE, SpeakerAge.OLD
_SOFT, _ROUGH = SpeakerTone.SOFT, SpeakerTone.ROUGH

# voice_id = Qwen 워커, voices.json 키 = DB tts_voice_id
VOICE_REGISTRY: list[VoiceProfile] = [
    VoiceProfile("female_young_soft", _F, _Y, _SOFT, "부드러운 젊은 여성", "uyVNoMrnUku1dZyVEXwD"),
    VoiceProfile("female_young_rough", _F, _Y, _ROUGH, "거친 젊은 여성", "uyVNoMrnUku1dZyVEXwD"),
    VoiceProfile("female_middle_soft", _F, _MID, _SOFT, "부드러운 중년 여성", "ETPP7D0aZVdEj12Aa7ho"),
    VoiceProfile("female_middle_rough", _F, _MID, _ROUGH, "거친 중년 여성", "ETPP7D0aZVdEj12Aa7ho"),
    VoiceProfile("female_old_soft", _F, _O, _SOFT, "부드러운 노년 여성", "fNmw8sukfGuvWVOp33Ge"),
    VoiceProfile("female_old_rough", _F, _O, _ROUGH, "거친 노년 여성", "fNmw8sukfGuvWVOp33Ge"),
    VoiceProfile("male_young_soft", _M, _Y, _SOFT, "부드러운 젊은 남성", "yUhh6u2RvOjeCsWPEaEg"),
    VoiceProfile("male_young_rough", _M, _Y, _ROUGH, "거친 젊은 남성", "4JJwo477JUAx3HV0T7n7"),
    VoiceProfile("male_middle_soft", _M, _MID, _SOFT, "부드러운 중년 남성", "Ctl3Pjfp5c2p3ThSA8Hn"),
    VoiceProfile("male_middle_rough", _M, _MID, _ROUGH, "거친 중년 남성", "Ctl3Pjfp5c2p3ThSA8Hn"),
    VoiceProfile("male_old_soft", _M, _O, _SOFT, "부드러운 노년 남성", "Ctl3Pjfp5c2p3ThSA8Hn"),
    VoiceProfile("male_old_rough", _M, _O, _ROUGH, "거친 노년 남성", "Ctl3Pjfp5c2p3ThSA8Hn"),
]

_LEGACY_EL_SLOT: dict[str, str] = {
    "uyVNoMrnUku1dZyVEXwD": "female_young_soft",
    "ETPP7D0aZVdEj12Aa7ho": "female_middle_soft",
    "fNmw8sukfGuvWVOp33Ge": "female_old_soft",
    "yUhh6u2RvOjeCsWPEaEg": "male_young_soft",
    "4JJwo477JUAx3HV0T7n7": "male_young_soft",
    "Ctl3Pjfp5c2p3ThSA8Hn": "male_middle_soft",
}

# 화자 정보 없는 시나리오의 칸
DEFAULT_VOICE = "male_young_soft"


# 난이도따라서 톤 다르게, 하·중은 부드러운 칸
_DIFFICULTY_TONE: dict[Difficulty, SpeakerTone | None] = {
    Difficulty.LOW: SpeakerTone.SOFT,
    Difficulty.MEDIUM: SpeakerTone.SOFT,
    Difficulty.HIGH: SpeakerTone.ROUGH,
}


# 예시 대화에서 '나(사용자 역할)' 대사에 쓰는 고정 보이스
EXAMPLE_USER_VOICE_ID = "4JJwo477JUAx3HV0T7n7"  # 평범한 젊은 남성


def pick_example_user_voice(ai_voice_id: str | None) -> str:
    """예시 대화 user 턴 ElevenLabs 보이스"""
    if ai_voice_id != EXAMPLE_USER_VOICE_ID:
        return EXAMPLE_USER_VOICE_ID
    for profile in VOICE_REGISTRY:
        el = profile.el_voice_id or profile.voice_id
        if el != ai_voice_id:
            return el
    return EXAMPLE_USER_VOICE_ID


def profile_for(voice_id: str | None) -> VoiceProfile | None:
    """레지스트리 보이스, 옛 ElevenLabs id 면 그 칸"""
    if not voice_id:
        return None
    for key in (voice_id, _LEGACY_EL_SLOT.get(voice_id)):
        for profile in VOICE_REGISTRY:
            if profile.voice_id == key:
                return profile
    return None


def resolve_voice(voice_id: str | None, difficulty: Difficulty | None) -> str | None:
    """난이도에 맞는 톤으로 보이스 바꾸기, 옛 id 는 칸으로"""
    profile = profile_for(voice_id)
    if profile is None:
        return voice_id
    target_tone = _DIFFICULTY_TONE.get(difficulty) if difficulty is not None else None
    if target_tone is None:
        return profile.voice_id

    candidates = [
        v
        for v in VOICE_REGISTRY
        if v.gender == profile.gender and v.age == profile.age and v.tone == target_tone
    ]
    if not candidates:
        return profile.voice_id
    return min(c.voice_id for c in candidates)


def qwen_voice_for(voice_id: str | None, difficulty: Difficulty | None) -> str:
    """Qwen 워커 보이스 이름, 모르는 보이스면 기본 칸"""
    profile = profile_for(voice_id)
    if profile is not None:
        return profile.voice_id
    return resolve_voice(DEFAULT_VOICE, difficulty) or DEFAULT_VOICE


def el_voice_for(voice_id: str | None) -> str | None:
    """ElevenLabs 보이스 id, 모르는 값은 그대로"""
    profile = profile_for(voice_id)
    if profile is None:
        return voice_id
    return profile.el_voice_id or profile.voice_id


def el_voice_for_call(stored: str | None, difficulty: Difficulty | None) -> str | None:
    """통화 ElevenLabs 보이스, 옛 EL id 는 하 난이도 말고는 저장값 그대로"""
    if stored in _LEGACY_EL_SLOT and difficulty is not Difficulty.LOW:
        return stored
    return el_voice_for(resolve_voice(stored, difficulty))


def parse_speaker(
    raw: object,
) -> tuple[SpeakerGender | None, SpeakerAge | None, SpeakerTone | None]:
    """LLM speaker 필드 방어 파싱. dict 아님/키 없음/enum 밖 값 → 해당 축 None."""
    if not isinstance(raw, dict):
        return None, None, None
    gender: SpeakerGender | None = None
    age: SpeakerAge | None = None
    tone: SpeakerTone | None = None
    with suppress(ValueError):
        gender = SpeakerGender(str(raw.get("gender", "")).strip().lower())
    with suppress(ValueError):
        age = SpeakerAge(str(raw.get("age", "")).strip().lower())
    with suppress(ValueError):
        tone = SpeakerTone(str(raw.get("tone", "")).strip().lower())
    return gender, age, tone


def pick_voice_id(
    gender: SpeakerGender | None,
    age: SpeakerAge | None,
    tone: SpeakerTone | None = None,
    *,
    rng: random.Random | None = None,
) -> str | None:
    """(성별, 연령) 필수 매칭 후 tone 은 정확 일치 후보가 있을 때만 좁힌다(완화 매칭)."""
    if gender is None or age is None:
        return None
    pool = [v for v in VOICE_REGISTRY if v.gender == gender and v.age == age]
    if not pool:
        return None
    if tone is not None:
        exact = [v for v in pool if v.tone == tone]
        if exact:
            pool = exact
    return (rng or random).choice(pool).voice_id
