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


# 같은 gender나 age면 랜덤 선택
# 괜찮은 보이스 찾으면 계속 업데이트 해줘야함
VOICE_REGISTRY: list[VoiceProfile] = [
    # VoiceProfile("<voice_id>", SpeakerGender.FEMALE, SpeakerAge.MIDDLE, SpeakerTone.SOFT, "차분한 중년 여성"),
    VoiceProfile("uyVNoMrnUku1dZyVEXwD", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.SOFT, "차분한 젊은 여성"),
    VoiceProfile("ETPP7D0aZVdEj12Aa7ho", SpeakerGender.FEMALE, SpeakerAge.MIDDLE, SpeakerTone.SOFT, "차분한 중년 여성"),
    VoiceProfile("fNmw8sukfGuvWVOp33Ge", SpeakerGender.FEMALE, SpeakerAge.OLD, SpeakerTone.SOFT, "차분한 노년 여성"),
    VoiceProfile("yUhh6u2RvOjeCsWPEaEg", SpeakerGender.MALE, SpeakerAge.YOUNG, SpeakerTone.SOFT, "차분한 젊은 남성"),
    VoiceProfile("4JJwo477JUAx3HV0T7n7", SpeakerGender.MALE, SpeakerAge.YOUNG, SpeakerTone.NEUTRAL, "평범한 젊은 남성"),
    VoiceProfile("Ctl3Pjfp5c2p3ThSA8Hn", SpeakerGender.MALE, SpeakerAge.MIDDLE, SpeakerTone.SOFT, "차분한 중년 남성"),
]


# 난이도따라서 톤 다르게
_DIFFICULTY_TONE: dict[Difficulty, SpeakerTone | None] = {
    Difficulty.LOW: SpeakerTone.SOFT,
    Difficulty.MEDIUM: None,
    Difficulty.HIGH: SpeakerTone.ROUGH,
}


# 예시 대화에서 '나(사용자 역할)' 대사에 쓰는 고정 보이스
EXAMPLE_USER_VOICE_ID = "4JJwo477JUAx3HV0T7n7"  # 평범한 젊은 남성


def pick_example_user_voice(ai_voice_id: str | None) -> str:
    """예시 대화 user 턴 보이스"""
    if ai_voice_id != EXAMPLE_USER_VOICE_ID:
        return EXAMPLE_USER_VOICE_ID
    for profile in VOICE_REGISTRY:
        if profile.voice_id != ai_voice_id:
            return profile.voice_id
    return EXAMPLE_USER_VOICE_ID


def profile_for(voice_id: str | None) -> VoiceProfile | None:
    """받은 voice_id가 VOICE_REGISTRY에 있으면 반환"""
    if not voice_id:
        return None
    for profile in VOICE_REGISTRY:
        if profile.voice_id == voice_id:
            return profile
    return None


def resolve_voice(voice_id: str | None, difficulty: Difficulty | None) -> str | None:
    """난이도에 맞는 톤으로 보이스 바꾸기"""
    if difficulty is None:
        return voice_id
    target_tone = _DIFFICULTY_TONE.get(difficulty)
    if target_tone is None:
        return voice_id

    profile = profile_for(voice_id)
    if profile is None:
        return voice_id

    candidates = [
        v
        for v in VOICE_REGISTRY
        if v.gender == profile.gender and v.age == profile.age and v.tone == target_tone
    ]
    if not candidates:
        return voice_id
    return min(c.voice_id for c in candidates)


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
