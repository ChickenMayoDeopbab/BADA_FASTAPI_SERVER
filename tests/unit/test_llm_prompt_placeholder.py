from app.schemas.llm import AiPersonality, ScenarioTurn, TurnContext
from app.services.llm_prompt import build_system_prompt


def _ctx(step: int = 1) -> TurnContext:
    return TurnContext(
        personality=AiPersonality.NORMAL,
        scenario_title="음식점 예약",
        scenario_role="레스토랑 예약 담당 직원",
        script=[
            ScenarioTurn(step=1, ai_goal="전화를 응대하고 예약 의사를 확인한다"),
            ScenarioTurn(step=2, ai_goal="희망 날짜와 시간을 묻는다"),
        ],
        current_step=step,
        history=[],
        user_utterance="여보세요",
    )


def test_prompt_forbids_placeholder_output() -> None:
    assert "자리표시자" in build_system_prompt(_ctx())


def test_prompt_tells_model_to_invent_unknown_proper_nouns() -> None:
    prompt = build_system_prompt(_ctx())
    assert "지어내" in prompt
    assert "상호명" in prompt


def test_prompt_requires_invented_names_stay_consistent() -> None:
    assert "일관" in build_system_prompt(_ctx())


def test_placeholder_rule_does_not_break_step_invariance() -> None:
    assert len({build_system_prompt(_ctx(step=s)) for s in (1, 2)}) == 1
