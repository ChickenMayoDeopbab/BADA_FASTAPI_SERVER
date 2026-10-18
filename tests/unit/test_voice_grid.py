import pytest

from app.core import tts_voices
from app.core.config import Settings
from app.core.enums import Difficulty, SpeakerAge, SpeakerGender, SpeakerTone
from app.core.preset_scenarios import PRESET_SCENARIOS
from app.core.tts_voices import (
    DEFAULT_VOICE,
    EXAMPLE_USER_VOICE_ID,
    VOICE_REGISTRY,
    el_voice_for,
    el_voice_for_call,
    pick_example_user_voice,
    profile_for,
    qwen_voice_for,
    resolve_voice,
)

_LEGACY = {
    "uyVNoMrnUku1dZyVEXwD": "female_young_soft",
    "ETPP7D0aZVdEj12Aa7ho": "female_middle_soft",
    "fNmw8sukfGuvWVOp33Ge": "female_old_soft",
    "yUhh6u2RvOjeCsWPEaEg": "male_young_soft",
    "4JJwo477JUAx3HV0T7n7": "male_young_soft",
    "Ctl3Pjfp5c2p3ThSA8Hn": "male_middle_soft",
}


def test_registry_is_the_gender_age_tone_grid_named_by_slot():
    grid = {(g, a, t) for g in SpeakerGender for a in SpeakerAge for t in (SpeakerTone.SOFT, SpeakerTone.ROUGH)}
    assert {(v.gender, v.age, v.tone) for v in VOICE_REGISTRY} == grid and len(VOICE_REGISTRY) == 12
    for v in VOICE_REGISTRY:
        assert v.voice_id == f"{v.gender.value}_{v.age.value}_{v.tone.value}" and len(v.voice_id) <= 50
        assert v.el_voice_id and v.el_voice_id in _LEGACY
    assert DEFAULT_VOICE == "male_young_soft"


@pytest.mark.parametrize("legacy,slot", _LEGACY.items())
@pytest.mark.parametrize("difficulty", [None, Difficulty.MEDIUM, Difficulty.LOW, Difficulty.HIGH])
def test_legacy_elevenlabs_ids_resolve_to_the_same_speaker_slot(legacy, slot, difficulty):
    resolved = resolve_voice(legacy, difficulty)
    expected_tone = {Difficulty.HIGH: "rough", Difficulty.LOW: "soft"}.get(difficulty, "soft")
    assert resolved == slot.rsplit("_", 1)[0] + "_" + expected_tone


@pytest.mark.parametrize("slot", [v.voice_id for v in VOICE_REGISTRY])
def test_difficulty_picks_tone_and_medium_keeps_stored_tone(slot):
    base = slot.rsplit("_", 1)[0]
    assert resolve_voice(slot, Difficulty.LOW) == f"{base}_soft"
    assert resolve_voice(slot, Difficulty.HIGH) == f"{base}_rough"
    assert resolve_voice(slot, Difficulty.MEDIUM) == slot and resolve_voice(slot, None) == slot


def test_missing_tone_slot_keeps_the_normalized_slot(monkeypatch):
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", [v for v in VOICE_REGISTRY if v.voice_id != "male_middle_rough"])
    assert resolve_voice("Ctl3Pjfp5c2p3ThSA8Hn", Difficulty.HIGH) == "male_middle_soft", "옛 EL id 로 새지 않는다"


def test_unknown_and_missing_voices():
    assert resolve_voice("hand-made-el-id", Difficulty.HIGH) == "hand-made-el-id"
    assert resolve_voice(None, Difficulty.HIGH) is None
    assert profile_for("hand-made-el-id") is None and profile_for(None) is None and profile_for("") is None


def test_qwen_voice_name_falls_back_to_default_slot_with_difficulty():
    assert qwen_voice_for("male_old_rough", Difficulty.LOW) == "male_old_rough"
    assert qwen_voice_for(None, Difficulty.LOW) == "male_young_soft"
    assert qwen_voice_for(None, Difficulty.HIGH) == "male_young_rough"
    assert qwen_voice_for("hand-made-el-id", None) == "male_young_soft"
    assert qwen_voice_for("4JJwo477JUAx3HV0T7n7", None) == "male_young_soft"


def test_elevenlabs_id_for_slots_and_passthrough_for_others():
    assert el_voice_for("male_old_rough") == "Ctl3Pjfp5c2p3ThSA8Hn"
    assert el_voice_for("male_young_rough") == "4JJwo477JUAx3HV0T7n7"
    assert el_voice_for("hand-made-el-id") == "hand-made-el-id"
    assert el_voice_for(None) is None
    assert el_voice_for("uyVNoMrnUku1dZyVEXwD") == "uyVNoMrnUku1dZyVEXwD"


def test_example_user_voice_is_an_elevenlabs_id_and_avoids_the_ai_voice(monkeypatch):
    assert pick_example_user_voice("Ctl3Pjfp5c2p3ThSA8Hn") == EXAMPLE_USER_VOICE_ID
    picked = pick_example_user_voice(EXAMPLE_USER_VOICE_ID)
    assert picked != EXAMPLE_USER_VOICE_ID and picked in _LEGACY
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", list(reversed(VOICE_REGISTRY)))
    picked = pick_example_user_voice(EXAMPLE_USER_VOICE_ID)
    assert picked != EXAMPLE_USER_VOICE_ID and picked in _LEGACY and "_" not in picked


def test_presets_are_cast_to_soft_slots():
    cast = {p["title"]: p["tts_voice_id"] for p in PRESET_SCENARIOS}
    assert cast == {
        "음식점 예약": "female_young_soft",
        "음식점 예약 변경/취소": "male_young_soft",
        "병원 진료 예약": "female_middle_soft",
        "병원 진료 결과 문의": "female_young_soft",
        "배달 지연 문의": "male_young_soft",
        "제품 환불·교환 요청": "female_young_soft",
        "택배 배송 조회 및 변경": "male_middle_soft",
        "은행 업무 문의": "male_middle_soft",
    }


def test_scenario_voice_switch_defaults_off():
    assert Settings.model_fields["qwen_scenario_voices"].default is False


def test_slot_elevenlabs_fallback_table():
    assert {v.voice_id: v.el_voice_id for v in VOICE_REGISTRY} == {
        "female_young_soft": "uyVNoMrnUku1dZyVEXwD",
        "female_young_rough": "uyVNoMrnUku1dZyVEXwD",
        "female_middle_soft": "ETPP7D0aZVdEj12Aa7ho",
        "female_middle_rough": "ETPP7D0aZVdEj12Aa7ho",
        "female_old_soft": "fNmw8sukfGuvWVOp33Ge",
        "female_old_rough": "fNmw8sukfGuvWVOp33Ge",
        "male_young_soft": "yUhh6u2RvOjeCsWPEaEg",
        "male_young_rough": "4JJwo477JUAx3HV0T7n7",
        "male_middle_soft": "Ctl3Pjfp5c2p3ThSA8Hn",
        "male_middle_rough": "Ctl3Pjfp5c2p3ThSA8Hn",
        "male_old_soft": "Ctl3Pjfp5c2p3ThSA8Hn",
        "male_old_rough": "Ctl3Pjfp5c2p3ThSA8Hn",
    }


@pytest.mark.parametrize("legacy", _LEGACY)
@pytest.mark.parametrize("difficulty", [None, Difficulty.MEDIUM, Difficulty.LOW, Difficulty.HIGH])
def test_call_elevenlabs_voice_for_legacy_rows_matches_before_slots(legacy, difficulty):
    before = "yUhh6u2RvOjeCsWPEaEg" if (legacy, difficulty) == ("4JJwo477JUAx3HV0T7n7", Difficulty.LOW) else legacy
    assert el_voice_for_call(legacy, difficulty) == before


@pytest.mark.parametrize("stored", [*(v.voice_id for v in VOICE_REGISTRY), None, "hand-made-el-id"])
@pytest.mark.parametrize("difficulty", [None, Difficulty.MEDIUM, Difficulty.LOW, Difficulty.HIGH])
def test_call_elevenlabs_voice_for_slots_follows_difficulty(stored, difficulty):
    assert el_voice_for_call(stored, difficulty) == el_voice_for(resolve_voice(stored, difficulty))
