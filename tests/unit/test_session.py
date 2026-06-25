from app.schemas.llm import AiPersonality
from app.services.llm_prompt import build_system_prompt
from app.services.session import (
    _parse_script,
    build_turn_context,
    is_session_owner,
)


def test_owner_match_int() -> None:
    assert is_session_owner({"userId": 7}, 7) is True

def test_owner_match_when_stored_as_str() -> None:
    assert is_session_owner({"userId": "7"}, 7) is True

def test_owner_mismatch() -> None:
    assert is_session_owner({"userId": 7}, 8) is False

def test_owner_missing_field() -> None:
    assert is_session_owner({}, 7) is False

def test_owner_unparsable_type() -> None:
    assert is_session_owner({"userId": "abc"}, 7) is False


def test_parse_script_sorts_by_step() -> None:
    raw = [
        {"step": 2, "aiGoal": "둘째", "hint": "h2"},
        {"step": 1, "aiGoal": "첫째", "hint": "h1"},
    ]
    turns = _parse_script(raw)
    assert [t.step for t in turns] == [1, 2]
    assert turns[0].ai_goal == "첫째"

def test_parse_script_skips_malformed_items() -> None:
    raw = [
        {"step": 1, "aiGoal": "정상"},
        {"aiGoal": "step 없음"},
        "문자열이라 dict 아님",
        {"step": "x", "aiGoal": "숫자아님"},
    ]
    turns = _parse_script(raw)
    assert len(turns) == 1
    assert turns[0].ai_goal == "정상"

def test_parse_script_non_list_returns_empty() -> None:
    assert _parse_script(None) == []
    assert _parse_script({"step": 1}) == []

def test_parse_script_hint_defaults_empty() -> None:
    turns = _parse_script([{"step": 1, "aiGoal": "목표"}])
    assert turns[0].hint == ""


def _session() -> dict:
    return {
        "userId": 1,
        "aiPersonality": "tough",
        "scenario": {
            "title": "병원 예약",
            "aiRole": "접수원",
            "aiPrompt": "You are a hospital receptionist.",
            "script": [{"step": 1, "aiGoal": "용건 확인"}],
        },
    }


def test_build_turn_context_maps_fields() -> None:
    ctx = build_turn_context(
        _session(),
        current_step=1,
        history=[{"role": "user", "text": "안녕"}],
        user_utterance="예약하려고요",
    )
    assert ctx.personality == AiPersonality.TOUGH
    assert ctx.scenario_title == "병원 예약"
    assert ctx.scenario_role == "접수원"
    assert ctx.scenario_prompt == "You are a hospital receptionist."
    assert ctx.current_step == 1
    assert ctx.user_utterance == "예약하려고요"
    assert len(ctx.script) == 1


def test_build_system_prompt_includes_scenario_prompt() -> None:
    ctx = build_turn_context(
        _session(),
        current_step=1,
        history=[],
        user_utterance="예약하려고요",
    )
    prompt = build_system_prompt(ctx)

    assert "병원 예약" in prompt
    assert "접수원" in prompt
    assert "You are a hospital receptionist." in prompt
    assert "용건 확인 <- 지금 이 단계" in prompt

def test_build_turn_context_unknown_personality_falls_back_normal() -> None:
    session = _session()
    session["aiPersonality"] = "WHATEVER"
    ctx = build_turn_context(session, current_step=1, history=[], user_utterance="x")
    assert ctx.personality == AiPersonality.NORMAL

def test_build_turn_context_missing_scenario_is_safe() -> None:
    ctx = build_turn_context(
        {"userId": 1}, current_step=1, history=[], user_utterance="x"
    )
    assert ctx.scenario_title == ""
    assert ctx.scenario_role == ""
    assert ctx.script == []

def test_build_turn_context_non_dict_scenario_is_safe() -> None:
    ctx = build_turn_context(
        {"scenario": "깨진값"}, current_step=1, history=[], user_utterance="x"
    )
    assert ctx.script == []
