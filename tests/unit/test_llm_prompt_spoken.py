import re

import pytest

from app.schemas.llm import AiPersonality, ScenarioTurn, TurnContext
from app.services.llm_prompt import build_system_prompt

_NO_HINT_LEVELS = (None, 0, 3)


def _ctx(
    personality: AiPersonality = AiPersonality.NORMAL,
    step: int = 1,
    script_level: int | None = None,
    utterance: str = "여보세요",
    history: list[dict] | None = None,
) -> TurnContext:
    return TurnContext(
        personality=personality,
        scenario_title="음식점 예약",
        scenario_role="레스토랑 예약 담당 직원",
        script=[
            ScenarioTurn(step=1, ai_goal="전화를 응대하고 예약 의사를 확인한다"),
            ScenarioTurn(step=2, ai_goal="희망 날짜와 시간을 묻는다"),
            ScenarioTurn(step=3, ai_goal="예약을 확정한다"),
        ],
        current_step=step,
        history=history or [],
        user_utterance=utterance,
        script_level=script_level,
    )


def _section(prompt: str, start: str, end: str | None = None) -> str:
    body = prompt[prompt.index(start):]
    if end is not None and end in body:
        body = body[: body.index(end)]
    return body


def _rules(prompt: str) -> str:
    return _section(prompt, "[표현 규칙]", "[힌트 지시]")


def test_spoken_rule_is_in_expression_rules() -> None:
    rules = _rules(build_system_prompt(_ctx()))
    assert "글이 아니라 전화로 하는 말" in rules
    assert "상담원 서면 공식" in rules
    assert "그대로 되풀이해 확인하지 않는다" in rules
    assert "필요한 것 하나만 짧게 묻는다" in rules


def test_spoken_rule_allows_reaction_word_without_repeating() -> None:
    rules = _rules(build_system_prompt(_ctx()))
    assert "짧은 반응어로 시작해도" in rules
    assert "같은 반응어를 매 턴 되풀이하지 않는다" in rules


def test_spoken_rule_requires_question_mark() -> None:
    """조기 종료 방어(awaiting_answer)가 대사 끝 '?' 만 본다."""
    assert "질문은 반드시 물음표(?)로 끝낸다" in _rules(build_system_prompt(_ctx()))


def test_spoken_rule_scoped_to_own_lines() -> None:
    assert "말투 규칙은 네 대사에만 적용한다" in _rules(build_system_prompt(_ctx()))


def test_reaction_word_goes_after_emotion_tag_line() -> None:
    """반응어가 감정 태그 앞에 오면 잘리거나, 따로 온 청크면 감정이 NEUTRAL 이 되고 태그가 TTS 로 읽힌다."""
    prompt = build_system_prompt(_ctx())
    assert "반응어도 대사이므로 감정 태그 줄 다음 줄부터 쓴다" in _rules(prompt)


def test_emotion_tag_instruction_kept_before_rules() -> None:
    prompt = build_system_prompt(_ctx())
    head = _section(prompt, "[출력 형식]", "[진행/종료 제어]")
    assert "응답의 맨 첫 줄에 반드시" in head
    assert "[EMOTION:VALUE]" in head
    assert "그 다음 줄부터 실제 대사를 쓴다" in head
    assert prompt.index("[출력 형식]") < prompt.index("[표현 규칙]")


def test_number_rule_wording_kept_and_not_overridden() -> None:
    rules = _rules(build_system_prompt(_ctx()))
    assert rules.startswith("[표현 규칙]\n숫자는 아라비아 숫자 그대로 쓴다")
    assert "네 대사에서 사람 수를 높여 세는 '분'은 반드시 한글로 쓴다" in rules
    assert "숫자로 말해도" in rules


@pytest.mark.parametrize("personality", list(AiPersonality))
@pytest.mark.parametrize("level", (*_NO_HINT_LEVELS, 1, 2))
def test_no_rule_precedence_override(personality: AiPersonality, level: int | None) -> None:
    """규칙 간 우선순위 문구는 어느 블록에 있든 숫자 규칙을 덮는다."""
    prompt = build_system_prompt(_ctx(personality=personality, script_level=level))
    for phrase in ("보다 우선", "우선한다"):
        assert phrase not in prompt, f"{personality.value} level={level}: {phrase}"


@pytest.mark.parametrize("personality", list(AiPersonality))
@pytest.mark.parametrize("level", (*_NO_HINT_LEVELS, 1, 2))
def test_no_disallowed_spoken_devices(personality: AiPersonality, level: int | None) -> None:
    """허용 장치는 반응어 시작뿐, 필러·말 고치기·미완결 문장은 금지 예시로도 쓰지 않는다."""
    prompt = build_system_prompt(_ctx(personality=personality, script_level=level))
    for phrase in ("필러", "'음", "말 고치", "고쳐 말", "끝맺지", "미완결"):
        assert phrase not in prompt, f"{personality.value} level={level}: {phrase}"


def test_placeholder_rule_still_in_rules() -> None:
    assert "자리표시자" in _rules(build_system_prompt(_ctx()))


@pytest.mark.parametrize("personality", list(AiPersonality))
@pytest.mark.parametrize("level", (*_NO_HINT_LEVELS, 1, 2))
def test_no_sigh_or_laugh_literals(personality: AiPersonality, level: int | None) -> None:
    prompt = build_system_prompt(_ctx(personality=personality, script_level=level))
    for literal in ("하아", "어휴", "하하"):
        assert literal not in prompt, f"{personality.value} level={level}: {literal}"


def test_rude_persona_sighs_rarely() -> None:
    prompt = build_system_prompt(_ctx(personality=AiPersonality.RUDE))
    persona = _section(prompt, "[성격 설정]", "[대화 진행 스크립트]")
    assert "진상" in persona
    assert "드물게" in persona


def test_rude_persona_shows_irritation_in_words_without_examples() -> None:
    """짜증은 한숨 대신 퉁명스러운 말로, 예시 문구는 따라 쓰므로 넣지 않는다."""
    prompt = build_system_prompt(_ctx(personality=AiPersonality.RUDE))
    persona = _section(prompt, "[성격 설정]", "[대화 진행 스크립트]")
    assert "한숨 대신 짧고 퉁명스러운 말" in persona
    assert "요청을 들어줄 때도 그 말투를 유지한다" in persona
    assert re.findall(r"'([^']+)'", persona) == ["연습용 연기"]


def test_style_rule_sounds_rarely_and_bans_stage_directions() -> None:
    rules = _rules(build_system_prompt(_ctx()))
    style = rules[rules.index("전화 통화이므로"):].split("\n", 1)[0]
    assert "괄호 안 행동 묘사" in style
    assert "드물게" in style


@pytest.mark.parametrize("personality", list(AiPersonality))
@pytest.mark.parametrize("level", _NO_HINT_LEVELS)
def test_no_suggest_tag_without_hint_level(personality: AiPersonality, level: int | None) -> None:
    prompt = build_system_prompt(_ctx(personality=personality, script_level=level))
    assert "SUGGEST" not in prompt
    assert "[힌트 지시]" not in prompt


def test_hint_l1_keeps_spoken_rule_before_hint_block() -> None:
    prompt = build_system_prompt(_ctx(script_level=1))
    assert prompt.index("말투 규칙은 네 대사에만") < prompt.index("[힌트 지시]")


@pytest.mark.parametrize("personality", list(AiPersonality))
@pytest.mark.parametrize("level", (*_NO_HINT_LEVELS, 1, 2))
def test_prompt_invariant_across_turns(personality: AiPersonality, level: int | None) -> None:
    history = [
        {"role": "user", "text": "여보세요"},
        {"role": "assistant", "text": "네, 바다레스토랑입니다. 무엇을 도와드릴까요?"},
    ]
    prompts = {
        build_system_prompt(_ctx(personality=personality, script_level=level, step=s, utterance=u, history=h))
        for s, u, h in ((1, "여보세요", None), (2, "예약이요", history), (3, "10명이요", history))
    }
    assert len(prompts) == 1, f"{personality.value} level={level} 프롬프트가 턴에 따라 변함"
