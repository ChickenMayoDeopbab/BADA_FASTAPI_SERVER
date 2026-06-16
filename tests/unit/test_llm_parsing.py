from app.schemas.llm import AiEmotion, LLMEventType
from app.services.llm import (
    LLMClient,
    _drain_pending,
    _find_first_tag,
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
    emit, controls, pending = _drain_pending("순수 대사")
    assert emit == "순수 대사"
    assert controls == []
    assert pending == ""

def test_drain_pending_extracts_control_tag() -> None:
    emit, controls, pending = _drain_pending("대사 끝[STEP_DONE]")
    assert emit == "대사 끝"
    assert controls == [LLMEventType.STEP_DONE]
    assert pending == ""

def test_drain_pending_holds_partial_trailing_tag() -> None:
    emit, controls, pending = _drain_pending("말하는중 [END")
    assert emit == "말하는중 "
    assert controls == []
    assert pending == "[END"

def test_drain_pending_reassembles_split_tag() -> None:
    emit1, controls1, pending1 = _drain_pending("좋아요 [STEP_")
    assert controls1 == []
    assert pending1 == "[STEP_"
    emit2, controls2, pending2 = _drain_pending(pending1 + "DONE] 뒷말")
    assert controls2 == [LLMEventType.STEP_DONE]
    assert pending2 == ""
    assert (emit1 + emit2).strip() == "좋아요  뒷말".strip()

def test_drain_pending_two_tags_in_order() -> None:
    emit, controls, pending = _drain_pending("마무리[STEP_DONE][END_CALL]")
    assert emit == "마무리"
    assert controls == [LLMEventType.STEP_DONE, LLMEventType.END_CALL]
    assert pending == ""


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
