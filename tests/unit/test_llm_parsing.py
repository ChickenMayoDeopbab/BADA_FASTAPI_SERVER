from app.schemas.llm import AiEmotion, LLMEventType
from app.services.llm import (
    LLMClient,
    _drain_pending,
    _find_first_tag,
    _sanitize_speech,
    _strip_step_echo,
    _trailing_partial_len,
)


def test_find_first_tag_none_when_absent() -> None:
    assert _find_first_tag("그냥 평범한 대사입니다") is None

def test_find_first_tag_returns_earliest() -> None:
    buf = "안녕 [STEP_DONE] 어쩌고 [END_CALL]"
    idx, tag = _find_first_tag(buf)
    assert tag == "[STEP_DONE]"
    assert buf[idx:].startswith("[STEP_DONE]")


def test_trailing_partial_len_holds_incomplete_tag() -> None:
    assert _trailing_partial_len("대사 [STEP_") == len("[STEP_")

def test_trailing_partial_len_zero_for_plain_text() -> None:
    assert _trailing_partial_len("그냥 텍스트") == 0

def test_trailing_partial_len_zero_when_bracket_not_tag_prefix() -> None:
    assert _trailing_partial_len("대사 [") == 1


def test_drain_pending_plain_text_passthrough() -> None:
    emit, controls, pending, suggest = _drain_pending("순수 대사")
    assert emit == "순수 대사"
    assert controls == []
    assert pending == ""
    assert suggest is False

def test_drain_pending_extracts_control_tag() -> None:
    emit, controls, pending, suggest = _drain_pending("대사 끝[STEP_DONE]")
    assert emit == "대사 끝"
    assert controls == [LLMEventType.STEP_DONE]
    assert pending == ""
    assert suggest is False

def test_drain_pending_holds_partial_trailing_tag() -> None:
    emit, controls, pending, suggest = _drain_pending("말하는중 [END")
    assert emit == "말하는중 "
    assert controls == []
    assert pending == "[END"
    assert suggest is False

def test_drain_pending_reassembles_split_tag() -> None:
    emit1, controls1, pending1, _ = _drain_pending("좋아요 [STEP_")
    assert controls1 == []
    assert pending1 == "[STEP_"
    emit2, controls2, pending2, _ = _drain_pending(pending1 + "DONE] 뒷말")
    assert controls2 == [LLMEventType.STEP_DONE]
    assert pending2 == ""
    assert (emit1 + emit2).strip() == "좋아요  뒷말".strip()

def test_drain_pending_two_tags_in_order() -> None:
    emit, controls, pending, suggest = _drain_pending("마무리[STEP_DONE][END_CALL]")
    assert emit == "마무리"
    assert controls == [LLMEventType.STEP_DONE, LLMEventType.END_CALL]
    assert pending == ""
    assert suggest is False


def test_drain_pending_strips_complete_step_echo() -> None:
    emit, controls, pending, suggest = _drain_pending("네, 예약 도와드릴게요.\n(지금 단계: 1)")
    assert emit == "네, 예약 도와드릴게요."
    assert controls == []
    assert pending == ""
    assert suggest is False

def test_drain_pending_holds_partial_step_echo() -> None:
    emit, controls, pending, _ = _drain_pending("확인했습니다. (지금 단")
    assert emit == "확인했습니다. "
    assert controls == []
    assert pending == "(지금 단"

def test_drain_pending_reassembles_split_step_echo() -> None:
    emit1, _, pending1, _ = _drain_pending("확인했습니다. (지금 단")
    assert pending1 == "(지금 단"
    emit2, _, pending2, _ = _drain_pending(pending1 + "계: 2)")
    assert emit2 == ""
    assert pending2 == ""
    assert (emit1 + emit2).rstrip() == "확인했습니다."

def test_drain_pending_strips_step_echo_before_control_tag() -> None:
    emit, controls, pending, _ = _drain_pending("예약 완료됐습니다. (지금 단계: 3)\n[STEP_DONE]")
    assert emit.strip() == "예약 완료됐습니다."
    assert controls == [LLMEventType.STEP_DONE]
    assert pending == ""

def test_drain_pending_keeps_normal_parentheses() -> None:
    emit, _, pending, _ = _drain_pending("3시 30분에 2명 (창가 자리) 맞으시죠?")
    assert emit == "3시 30분에 2명 (창가 자리) 맞으시죠?"
    assert pending == ""

def test_drain_pending_strips_stage_direction_sigh() -> None:
    emit, controls, pending, suggest = _drain_pending("하... (길게 한숨) 그러니까 뭐가 문제야.")
    assert emit == "하... 그러니까 뭐가 문제야."
    assert controls == []
    assert pending == ""
    assert suggest is False

def test_drain_pending_strips_stage_direction_variants() -> None:
    assert _drain_pending("알았어. (한숨)")[0].rstrip() == "알았어."
    assert _drain_pending("(웃으며) 어서오세요.")[0].strip() == "어서오세요."
    assert _drain_pending("그래. (잠시 침묵) 알았어.")[0] == "그래. 알았어."
    assert _drain_pending("뭐? (어이없다는 목소리로) 다시 말해봐.")[0] == "뭐? 다시 말해봐."

def test_drain_pending_holds_and_strips_split_stage_direction() -> None:
    emit1, controls1, pending1, _ = _drain_pending("하... (길게 한")
    assert emit1 == "하... "
    assert controls1 == []
    assert pending1 == "(길게 한"
    emit2, _, pending2, _ = _drain_pending(pending1 + "숨) 그러니까.")
    assert emit2.strip() == "그러니까."
    assert pending2 == ""

def test_sanitize_speech_removes_stage_direction_and_step_echo() -> None:
    assert _sanitize_speech("네. (한숨) (지금 단계: 2)") == "네."
    assert _sanitize_speech("알겠습니다. (창가 자리) 예약할게요.") == "알겠습니다. (창가 자리) 예약할게요."


def test_strip_step_echo_spacing_variants() -> None:
    assert _strip_step_echo("네.(지금단계:2)") == "네."
    assert _strip_step_echo("네. ( 지금 단계 3 )") == "네."
    assert _strip_step_echo("(지금 단계: 1) 어서오세요").strip() == "어서오세요"


def test_emit_emotion_head_parses_known_emotion() -> None:
    result = LLMClient._emit_emotion_head("[EMOTION:ANGRY]\n뭐라고요?")
    assert result is not None
    emotion, remainder = result
    assert emotion == AiEmotion.ANGRY
    assert remainder == "뭐라고요?"

def test_emit_emotion_head_unknown_falls_back_to_neutral() -> None:
    emotion, _ = LLMClient._emit_emotion_head("[EMOTION:WEIRD]\n대사")
    assert emotion == AiEmotion.NEUTRAL

def test_emit_emotion_head_none_when_no_tag() -> None:
    assert LLMClient._emit_emotion_head("그냥 대사") is None


def test_emit_emotion_head_case_insensitive() -> None:
    emotion, remainder = LLMClient._emit_emotion_head("[emotion: friendly ]\n안녕")
    assert emotion == AiEmotion.FRIENDLY
    assert remainder == "안녕"


def test_looks_like_partial_head_true_for_incomplete() -> None:
    assert LLMClient._looks_like_partial_head("[EMOT") is True
    assert LLMClient._looks_like_partial_head("[EMOTION:ANG") is True

def test_looks_like_partial_head_false_when_closed() -> None:
    assert LLMClient._looks_like_partial_head("[EMOTION:ANGRY]") is False

def test_looks_like_partial_head_false_for_plain_text() -> None:
    assert LLMClient._looks_like_partial_head("안녕하세요") is False
