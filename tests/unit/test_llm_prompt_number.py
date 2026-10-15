from app.schemas.llm import AiPersonality, ScenarioTurn, TurnContext
from app.services.llm_prompt import build_system_prompt


def _ctx(step: int = 1, script_level: int | None = None) -> TurnContext:
    return TurnContext(
        personality=AiPersonality.NORMAL,
        scenario_title="음식점 예약",
        scenario_role="레스토랑 예약 담당 직원",
        script=[
            ScenarioTurn(step=1, ai_goal="희망 날짜와 시간을 묻는다"),
            ScenarioTurn(step=2, ai_goal="예약 인원수를 확인한다"),
        ],
        current_step=step,
        history=[],
        user_utterance="10명이요",
        script_level=script_level,
    )


def _rules() -> str:
    prompt = build_system_prompt(_ctx())
    return prompt[prompt.index("[표현 규칙]"):]


def test_rule_keeps_arabic_numerals_by_default() -> None:
    rules = _rules()
    assert "아라비아 숫자" in rules
    assert "3시 30분" in rules


def test_rule_writes_honorific_people_counter_in_hangul() -> None:
    """'10분'은 TTS 가 시간 '십 분'으로 읽으므로 사람 수 '분'은 한글로 쓴다."""
    rules = _rules()
    assert "네 대사에서 사람 수" in rules
    assert "열 분" in rules
    assert "네 분" in rules


def test_rule_covers_user_digits_copied_into_bun() -> None:
    assert "숫자로 말해도" in _rules()


def test_rule_does_not_break_step_invariance() -> None:
    assert len({build_system_prompt(_ctx(step=s)) for s in (1, 2)}) == 1


def test_hint_counts_user_party_with_myeong() -> None:
    prompt = build_system_prompt(_ctx(script_level=1))
    hint = prompt[prompt.index("[힌트 지시]"):]
    assert "'명'으로 쓴다" in hint
