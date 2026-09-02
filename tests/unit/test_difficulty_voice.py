from types import SimpleNamespace

import pytest

import app.core.tts_voices as tts_voices
import app.services.pipeline as pipeline_module
import app.services.tts as tts
from app.core.config import get_settings
from app.core.enums import Difficulty, SpeakerAge, SpeakerGender, SpeakerTone
from app.core.tts_voices import VoiceProfile, resolve_voice
from app.schemas.llm import AiEmotion
from app.services.pipeline import VoicePipeline
from app.services.session import parse_difficulty
from app.services.tts import ElevenLabsTTSClient, TTSSession

_FEMALE_YOUNG_SOFT = VoiceProfile(
    "v-fy-soft", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.SOFT, "차분한 젊은 여성"
)
_FEMALE_YOUNG_NEUTRAL = VoiceProfile(
    "v-fy-neutral", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.NEUTRAL, "평범한 젊은 여성"
)
_FEMALE_YOUNG_ROUGH = VoiceProfile(
    "v-fy-rough", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.ROUGH, "거친 젊은 여성"
)
_MALE_OLD_ROUGH = VoiceProfile(
    "v-mo-rough", SpeakerGender.MALE, SpeakerAge.OLD, SpeakerTone.ROUGH, "거친 노년 남성"
)

_FULL_REGISTRY = [
    _FEMALE_YOUNG_SOFT,
    _FEMALE_YOUNG_NEUTRAL,
    _FEMALE_YOUNG_ROUGH,
    _MALE_OLD_ROUGH,
]


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", _FULL_REGISTRY)
    return _FULL_REGISTRY


@pytest.mark.parametrize("difficulty", [None, Difficulty.MEDIUM])
def test_no_difficulty_keeps_assigned_voice(registry, difficulty) -> None:
    assert resolve_voice("v-fy-neutral", difficulty) == "v-fy-neutral"


def test_low_difficulty_picks_soft_voice_of_same_speaker(registry) -> None:
    assert resolve_voice("v-fy-neutral", Difficulty.LOW) == "v-fy-soft"


def test_high_difficulty_picks_rough_voice_of_same_speaker(registry) -> None:
    assert resolve_voice("v-fy-neutral", Difficulty.HIGH) == "v-fy-rough"


def test_never_crosses_speaker_identity(monkeypatch) -> None:
    same_gender_other_age = VoiceProfile(
        "v-fo-rough", SpeakerGender.FEMALE, SpeakerAge.OLD, SpeakerTone.ROUGH, "거친 노년 여성"
    )
    same_age_other_gender = VoiceProfile(
        "v-my-rough", SpeakerGender.MALE, SpeakerAge.YOUNG, SpeakerTone.ROUGH, "거친 젊은 남성"
    )
    monkeypatch.setattr(
        tts_voices,
        "VOICE_REGISTRY",
        [_FEMALE_YOUNG_NEUTRAL, same_gender_other_age, same_age_other_gender, _MALE_OLD_ROUGH],
    )

    assert resolve_voice("v-fy-neutral", Difficulty.HIGH) == "v-fy-neutral"


def test_unregistered_voice_is_left_alone(registry) -> None:
    assert resolve_voice("v-hand-written", Difficulty.LOW) == "v-hand-written"


def test_no_assigned_voice_stays_none(registry) -> None:
    assert resolve_voice(None, Difficulty.HIGH) is None


def test_selection_is_deterministic(monkeypatch) -> None:
    second_soft = VoiceProfile(
        "v-fy-soft-2", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.SOFT, "또 다른 차분한"
    )
    monkeypatch.setattr(
        tts_voices, "VOICE_REGISTRY", [_FEMALE_YOUNG_NEUTRAL, second_soft, _FEMALE_YOUNG_SOFT]
    )

    results = {resolve_voice("v-fy-neutral", Difficulty.LOW) for _ in range(20)}

    assert results == {"v-fy-soft"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("low", Difficulty.LOW),
        ("LOW", Difficulty.LOW),          # Spring 이 대문자로 보낼 수 있다
        ("  High  ", Difficulty.HIGH),
        ("medium", Difficulty.MEDIUM),
    ],
)
def test_parse_difficulty_accepts_case_and_whitespace(raw, expected) -> None:
    assert parse_difficulty({"difficulty": raw}) == expected


@pytest.mark.parametrize(
    "raw",
    [None, "", "  ", "하", "easy", 1, 3.5, [], {}, True],
)
def test_parse_difficulty_rejects_unknown_values(raw) -> None:
    assert parse_difficulty({"difficulty": raw}) is None


def test_parse_difficulty_missing_key() -> None:
    assert parse_difficulty({}) is None


_BASE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": False,
    "speed": 1.0,
}


def _settings_for(emotion, difficulty=None) -> dict:
    session = TTSSession(None, _BASE_SETTINGS, difficulty=difficulty)
    return session._voice_settings_for(emotion)


@pytest.mark.parametrize("emotion", [AiEmotion.ANGRY])
def test_low_difficulty_calms_negative_emotions(emotion) -> None:
    normal = _settings_for(emotion)
    calmed = _settings_for(emotion, Difficulty.LOW)

    assert calmed["stability"] > normal["stability"]
    assert calmed["style"] < normal["style"]
    assert calmed["speed"] <= normal["speed"]


@pytest.mark.parametrize(
    "emotion", [AiEmotion.NEUTRAL, AiEmotion.FRIENDLY, AiEmotion.APOLOGETIC]
)
def test_low_difficulty_leaves_other_emotions_untouched(emotion) -> None:
    assert _settings_for(emotion, Difficulty.LOW) == _settings_for(emotion)


@pytest.mark.parametrize("difficulty", [None, Difficulty.MEDIUM, Difficulty.HIGH])
def test_other_difficulties_keep_current_emotion_settings(difficulty) -> None:
    for emotion in AiEmotion:
        assert _settings_for(emotion, difficulty) == _settings_for(emotion)


def test_voice_settings_are_clamped_to_valid_range() -> None:
    session = TTSSession(
        None, {"stability": 3.0, "style": -1.0, "speed": 9.9}, difficulty=None
    )

    result = session._voice_settings_for(AiEmotion.NEUTRAL)

    assert result["stability"] == 1.0
    assert result["style"] == 0.0
    assert result["speed"] == 1.2


def test_emotion_table_is_not_mutated_by_composition() -> None:
    before = dict(tts._EMOTION_VOICE_OVERRIDES[AiEmotion.ANGRY])

    _settings_for(AiEmotion.ANGRY, Difficulty.LOW)
    _settings_for(AiEmotion.ANGRY)

    assert tts._EMOTION_VOICE_OVERRIDES[AiEmotion.ANGRY] == before



def test_pipeline_wires_difficulty_into_voice_and_tts_client(registry, monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "GoogleSTTClient", lambda **kw: object())
    monkeypatch.setattr(pipeline_module, "LLMClient", lambda: object())
    monkeypatch.setattr(pipeline_module, "RecordingStorageService", lambda s: object())

    p = VoicePipeline(
        ws=object(),
        session_id="s-1",
        session={
            "difficulty": "LOW",
            "scenario": {"ttsVoiceId": "v-fy-neutral"},
        },
        settings=get_settings(),
        spring=object(),
    )

    assert p._difficulty is Difficulty.LOW
    assert p._voice_id_override == "v-fy-soft"
    assert p._tts._difficulty is Difficulty.LOW


def test_pipeline_without_difficulty_keeps_assigned_voice(registry, monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "GoogleSTTClient", lambda **kw: object())
    monkeypatch.setattr(pipeline_module, "LLMClient", lambda: object())
    monkeypatch.setattr(pipeline_module, "RecordingStorageService", lambda s: object())

    p = VoicePipeline(
        ws=object(),
        session_id="s-1",
        session={"scenario": {"ttsVoiceId": "v-fy-neutral"}},
        settings=get_settings(),
        spring=object(),
    )

    assert p._difficulty is None
    assert p._voice_id_override == "v-fy-neutral"


@pytest.mark.asyncio
async def test_client_open_hands_difficulty_to_session(monkeypatch) -> None:

    async def _fake_connect(*args, **kwargs):
        return SimpleNamespace(uri=args[0] if args else None)

    monkeypatch.setattr(tts.websockets, "connect", _fake_connect)
    client = ElevenLabsTTSClient(get_settings(), Difficulty.LOW)

    session = await client.open("v-abc")

    assert session._difficulty is Difficulty.LOW


def test_real_registry_never_changes_speaker_identity() -> None:
    for profile in tts_voices.VOICE_REGISTRY:
        for difficulty in Difficulty:
            resolved = resolve_voice(profile.voice_id, difficulty)
            picked = tts_voices.profile_for(resolved)
            assert picked is not None, f"{profile.voice_id} → 미등록 보이스 {resolved}"
            assert (picked.gender, picked.age) == (profile.gender, profile.age)
