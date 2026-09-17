import pytest

from app.schemas.frames import EndReason
from app.services.pipeline import _State, _TurnTimings
from app.services.session import build_turn_context
from tests.unit.test_transcript_frames import _make_pipeline


async def _finalize(text, *, step=4, script_len=4, end_call=True, step_done=True, error=False):
    p = _make_pipeline()
    p._state = _State.SPEAKING
    p._current_step = step
    p._script_len = script_len
    ctx = build_turn_context(p._session, current_step=step, history=[], user_utterance="보상은 없나요?")
    await p._finalize_turn(
        "보상은 없나요?", [text],
        {"end_call": end_call, "step_done": step_done, "error": error},
        _TurnTimings(final_at=0), {"prompt": 0, "cached": 0, "output": 0}, ctx=ctx,
    )
    return p


@pytest.mark.parametrize("text", [
    "다시 보내드리거나 전액 환불해 드리겠습니다. 어떤 방법으로 도와드릴까요?",
    "어떤 방법으로 도와드릴까요？",
    '“어떤 방법으로 도와드릴까요?”',
])
@pytest.mark.parametrize("end_call,step_done", [(True, True), (True, False), (False, True)])
async def test_question_at_last_step_waits_for_user_instead_of_ending(text, end_call, step_done):
    p = await _finalize(text, end_call=end_call, step_done=step_done)
    assert not p._closing.is_set()
    assert p._state == _State.LISTENING
    assert p._current_step == 4
    assert p._completed_script_steps == 0


async def test_question_before_last_step_can_advance_without_ending_call():
    p = await _finalize("어떤 방법으로 도와드릴까요?", script_len=6)
    assert not p._closing.is_set()
    assert p._current_step == 5
    assert p._completed_script_steps == 1


@pytest.mark.parametrize("end_call,reason", [(True, EndReason.END_CALL), (False, EndReason.SCENARIO_DONE)])
async def test_final_greeting_still_closes(end_call, reason):
    p = await _finalize("이용해 주셔서 감사합니다. 안녕히 계세요.", end_call=end_call)
    assert p._closing.is_set()
    assert p._end_reason == reason


async def test_crisis_style_early_closing_statement_still_closes():
    p = await _finalize("이번 통화는 여기서 마치겠습니다.", step=2, script_len=6, step_done=False)
    assert p._end_reason == EndReason.END_CALL


async def test_pipeline_error_is_not_hidden_by_question_guard():
    p = await _finalize("어떤 방법으로 도와드릴까요?", error=True)
    assert p._end_reason == EndReason.ERROR


async def test_free_conversation_question_does_not_end_call():
    p = await _finalize("더 궁금한 점이 있으신가요?", step=1, script_len=0, step_done=False)
    assert not p._closing.is_set()
    assert p._state == _State.LISTENING
