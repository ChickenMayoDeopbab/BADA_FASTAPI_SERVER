"""구간별 피드백 — AVTI 로 칭찬/조언 구간을 가르는 규칙."""

from app.services.feedback_points import is_safe, polish, split_title
from app.services.llm import LLMClient
from app.services.pipeline import VoicePipeline


def _pipeline(
    turns: list[tuple[float, float]],
    texts: list[str] | None = None,
) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "test-session"
    p._user_turn_intervals = turns
    p._user_turn_texts = (
        texts if texts is not None else [f"발화{i}" for i in range(len(turns))]
    )
    return p


# --- 구간 선택 -----------------------------------------------------------


def test_bad_avti_turn_becomes_improve_segment() -> None:
    """떨림이 큰 턴은 짚어줄 구간이 된다."""
    p = _pipeline([(0.0, 3.0), (5.0, 12.0)])
    segs = p._pick_segments([(5.5, 11.0)], {2: 8.3})
    assert [s["type"] for s in segs] == ["IMPROVE"]
    assert segs[0]["turn"] == 2
    assert segs[0]["avti"] == 8.3


def test_good_avti_turn_becomes_good_segment() -> None:
    p = _pipeline([(0.0, 3.0), (5.0, 12.0)])
    segs = p._pick_segments([(5.5, 11.0)], {2: 1.8})
    assert [s["type"] for s in segs] == ["GOOD"]
    assert segs[0]["reason"] == "voice"


def test_middle_avti_turn_is_skipped() -> None:
    """애매한 값에서는 그 턴을 아예 안 고른다."""
    p = _pipeline([(0.0, 3.0), (5.0, 12.0)])
    segs = p._pick_segments([], {2: 4.0})
    # 근거가 없으니 그 턴을 근거로 고른 구간은 없다(폴백만 남는다)
    assert all(s.get("reason") == "fallback" for s in segs)


def test_turn_without_avti_falls_back_to_tremor_free_span() -> None:
    """열 턴에 아홉은 AVTI 가 없다. 그때는 떨림 없는 구간인지로 판단한다."""
    p = _pipeline([(0.0, 3.0), (5.0, 12.0)])
    segs = p._pick_segments([(5.5, 11.0)], {})
    assert [s["type"] for s in segs] == ["GOOD"]
    assert segs[0]["reason"] == "clean"


def test_short_clean_overlap_is_not_picked() -> None:
    p = _pipeline([(5.0, 12.0)])
    segs = p._pick_segments([(5.5, 5.9)], {})
    assert all(s["reason"] == "fallback" for s in segs)


def test_improve_segments_survive_the_cap() -> None:
    """자리가 모자라면 아쉬운 구간을 먼저 남긴다."""
    turns = [(float(i * 5), float(i * 5 + 4)) for i in range(6)]
    p = _pipeline(turns)
    clean = [(t[0] + 0.5, t[1] - 0.5) for t in turns]
    segs = p._pick_segments(clean, {2: 8.5, 5: 9.0})
    assert len(segs) == 3
    assert sum(1 for s in segs if s["type"] == "IMPROVE") == 2


def test_segments_are_sorted_by_time() -> None:
    turns = [(0.0, 4.0), (10.0, 14.0), (20.0, 24.0)]
    p = _pipeline(turns)
    segs = p._pick_segments([(t[0] + 0.5, t[1]) for t in turns], {3: 8.5})
    assert [s["start"] for s in segs] == sorted(s["start"] for s in segs)


def test_never_returns_empty_when_turns_exist() -> None:
    """고를 근거가 없어도 구간이 비지 않는다."""
    p = _pipeline([(0.0, 4.0)])
    assert p._pick_segments([], {1: 4.0})


# --- 발화 없는 구간 제외 --------------------------------------------------


def test_turn_without_transcript_is_not_picked() -> None:
    """마이크만 열려 있던 구간 — 받아적힌 말이 없으면 안 고른다."""
    p = _pipeline([(0.0, 10.82)], texts=[""])
    assert p._pick_segments([], {}) == []


def test_turn_without_transcript_is_not_rescued_by_fallback() -> None:
    """떨림 없는 구간이 겹쳐도 발화가 없으면 폴백까지 막는다."""
    p = _pipeline([(0.0, 10.82)], texts=[""])
    assert p._pick_segments([(0.0, 10.82)], {}) == []


def test_whitespace_only_transcript_counts_as_silence() -> None:
    p = _pipeline([(0.0, 10.0)], texts=["   "])
    assert p._pick_segments([(0.0, 10.0)], {}) == []


def test_turn_without_transcript_survives_when_avti_measured() -> None:
    """AVTI 는 3초 넘게 이어진 목소리에서만 나온다 — 말은 했는데 문장이 못 넘어온 턴."""
    p = _pipeline([(0.0, 10.0)], texts=[""])
    segs = p._pick_segments([(0.5, 9.0)], {1: 8.3})
    assert [s["type"] for s in segs] == ["IMPROVE"]


def test_silent_turn_dropped_but_spoken_turn_keeps_its_number() -> None:
    """빈 턴을 빼도 남은 턴의 번호는 그대로 — AVTI 키와 어긋나지 않는다."""
    p = _pipeline([(0.0, 4.0), (5.0, 12.0)], texts=["", "네 맞아요"])
    segs = p._pick_segments([(5.5, 11.0)], {2: 8.3})
    assert [s["turn"] for s in segs] == [2]
    assert segs[0]["type"] == "IMPROVE"
    assert segs[0]["avti"] == 8.3


def test_all_turns_silent_yields_no_segments() -> None:
    """한마디도 인식되지 않은 통화는 구간 피드백 자체가 없다."""
    p = _pipeline([(0.0, 4.0), (5.0, 12.0)], texts=["", ""])
    assert p._pick_segments([(0.5, 3.5), (5.5, 11.0)], {}) == []


# --- 구간 문구 파싱 ------------------------------------------------------


def test_parse_numbered_pairs_keeps_order_and_count() -> None:
    raw = (
        "1. 용건을 먼저 말했어요 | 무슨 일로 걸었는지 밝혀서 상대가 바로 알아들었어요.\n"
        "2. 목소리가 흔들렸어요 | 다음엔 한 호흡 쉬고 말해봐요."
    )
    pairs = LLMClient._parse_numbered_pairs(raw, 2)
    assert pairs[0] == ("용건을 먼저 말했어요", "무슨 일로 걸었는지 밝혀서 상대가 바로 알아들었어요.")
    assert pairs[1][0] == "목소리가 흔들렸어요"


def test_parse_numbered_pairs_pads_when_model_returns_too_few() -> None:
    pairs = LLMClient._parse_numbered_pairs("1. 하나 | 내용이에요.", 3)
    assert len(pairs) == 3
    assert pairs[1] == ("", "")


def test_parse_numbered_pairs_truncates_extra_lines() -> None:
    raw = "1. a | 가.\n2. b | 나.\n3. c | 다."
    assert len(LLMClient._parse_numbered_pairs(raw, 2)) == 2


def test_parse_numbered_pairs_recovers_without_delimiter() -> None:
    pairs = LLMClient._parse_numbered_pairs("1. 용건을 말했어요. 바로 알아들었어요.", 1)
    assert pairs[0] == ("용건을 말했어요", "바로 알아들었어요.")


# --- 안전장치 ------------------------------------------------------------


def test_measurement_leak_is_filtered() -> None:
    assert not is_safe("떨림이 있어요", "떨림 점수가 8.4점이라 아쉬워요.")


def test_call_content_words_pass() -> None:
    assert is_safe("증상을 잘 말했어요", "아픈 곳을 순서대로 설명해서 접수가 빨랐어요.")


def test_split_title_helper() -> None:
    assert split_title("앞 문장이에요. 뒷 문장이에요.") == ("앞 문장이에요", "뒷 문장이에요.")


# --- 문구 다듬기 ---------------------------------------------------------


def test_polish_fixes_broken_reduplication() -> None:
    """'또또박' 같은 없는 말은 제 모양으로 되돌린다."""
    assert polish("다음엔 조금 더 큰 목소리로 또또박 말해봐요.") == (
        "다음엔 조금 더 큰 목소리로 또박또박 말해봐요."
    )
    assert polish("또박박 말했어요") == "또박또박 말했어요"
    assert polish("차차근 설명해봐요") == "차근차근 설명해봐요"


def test_polish_keeps_correct_reduplication() -> None:
    """멀쩡한 첩어는 건드리지 않는다."""
    assert polish("또박또박 말했어요") == "또박또박 말했어요"
    assert polish("차근차근 설명해서 좋았어요") == "차근차근 설명해서 좋았어요"


def test_polish_rewrites_metaphor_to_plain_wording() -> None:
    """'닿지 않았어요' 같은 비유는 들린 그대로의 말로 바꾼다."""
    assert polish("소리가 작아서 상대방에게 닿지 않았어요.") == (
        "소리가 작아서 상대방에게 잘 들리지 않았어요."
    )
    assert polish("상대방에게 잘 닿지 않았어요.") == "상대방에게 잘 들리지 않았어요."
    assert polish("말이 전달되지 못했어요.") == "말이 잘 들리지 않았어요."


def test_polish_applied_to_parsed_pairs() -> None:
    """LLM 응답을 파싱하는 길목에서 자동으로 다듬어진다."""
    pairs = LLMClient._parse_numbered_pairs(
        "1. 소리가 작았어요 | 상대방에게 닿지 않았어요. 다음엔 또또박 말해봐요.", 1
    )
    assert pairs[0] == (
        "소리가 작았어요",
        "상대방에게 잘 들리지 않았어요. 다음엔 또박또박 말해봐요.",
    )


def test_parse_strips_segment_label_from_title() -> None:
    """입력 표시가 제목에 새어 나오면 벗겨낸다."""
    pairs = LLMClient._parse_numbered_pairs(
        "1. [칭찬할 구간] 용건을 먼저 말했어요 | 상대방이 바로 알아들었어요.", 1
    )
    assert pairs[0][0] == "용건을 먼저 말했어요"


def test_polish_leaves_ordinary_text_untouched() -> None:
    """멀쩡한 문장은 한 글자도 바뀌지 않는다."""
    for text in (
        "또 한 번 물어봤어요",
        "박수를 쳐주고 싶어요",
        "차분하게 근처 지점을 물었어요",
        "손이 닿는 곳에 두고 말했어요",
        "마음이 잘 전달됐어요",
    ):
        assert polish(text) == text


def test_polish_handles_empty_and_whitespace() -> None:
    assert polish("") == ""
    assert polish("   ") == ""
    assert polish("  또또박   말해봐요  ") == "또박또박 말해봐요"


def test_polish_fixes_multiple_problems_in_one_sentence() -> None:
    """한 문장에 여러 군데 섞여 있어도 전부 잡는다."""
    assert polish("차차근 말했지만 상대방에게 닿지 않았어요. 다음엔 또박박 말해봐요.") == (
        "차근차근 말했지만 상대방에게 잘 들리지 않았어요. 다음엔 또박또박 말해봐요."
    )


def test_polish_does_not_stack_the_word_jal() -> None:
    """'잘 닿지 않았어요' 가 '잘 잘 들리지' 로 겹치지 않는다."""
    assert "잘 잘" not in polish("목소리가 잘 닿지 않았어요.")
    assert "잘 잘" not in polish("말이 잘 전달되지 않았어요.")


def test_polish_covers_other_metaphor_verbs() -> None:
    assert polish("말이 전해지지 않았어요.") == "말이 잘 들리지 않았어요."
    assert polish("목소리가 닿지 않아요.") == "목소리가 잘 들리지 않아요."


def test_polished_text_still_passes_safety_check() -> None:
    """다듬은 뒤에도 검증을 통과해서 폴백으로 안 떨어진다."""
    title = polish("소리가 작았어요")
    content = polish("상대방에게 닿지 않았어요. 다음엔 또또박 말해봐요.")
    assert is_safe(title, content)


def test_polish_keeps_measurement_leak_visible_to_safety_check() -> None:
    """다듬기는 걸러내는 일을 대신하지 않는다 — 지표 유출은 그대로 남아 검증에 걸린다."""
    content = polish("떨림 점수가 8.4점이었어요.")
    assert not is_safe("떨림이 있었어요", content)


def test_parse_strips_label_and_polishes_together() -> None:
    """라벨 제거와 문구 다듬기가 한 번에 적용된다."""
    pairs = LLMClient._parse_numbered_pairs(
        "1. [짚어줄 구간] 소리가 작았어요 | 상대방에게 닿지 않았어요. 다음엔 또또박 말해봐요.\n"
        "2. [칭찬할 구간] 차차근 설명했어요 | 순서대로 말해서 좋았어요.",
        2,
    )
    assert pairs[0] == (
        "소리가 작았어요",
        "상대방에게 잘 들리지 않았어요. 다음엔 또박또박 말해봐요.",
    )
    assert pairs[1] == ("차근차근 설명했어요", "순서대로 말해서 좋았어요.")


def test_parse_pads_and_trims_to_requested_count() -> None:
    """구간 수보다 적게/많이 와도 개수가 맞는다."""
    assert len(LLMClient._parse_numbered_pairs("1. 가 | 나", 3)) == 3
    assert LLMClient._parse_numbered_pairs("1. 가 | 나", 3)[2] == ("", "")
    assert len(LLMClient._parse_numbered_pairs(
        "1. 가 | 나\n2. 다 | 라\n3. 마 | 바\n4. 사 | 아", 2
    )) == 2


def test_parse_empty_response_yields_blank_pairs() -> None:
    """응답이 비면 전부 빈 쌍 → 파이프라인이 폴백 문구로 채운다."""
    pairs = LLMClient._parse_numbered_pairs("", 2)
    assert pairs == [("", ""), ("", "")]
    assert all(not is_safe(t, c) for t, c in pairs)


# --- 마음 상태 단정 -------------------------------------------------------


def test_mind_state_assertion_is_filtered() -> None:
    """사용자를 규정하는 말은 내보내지 않는다."""
    assert not is_safe("자신감이 부족했어요", "다음엔 조금 더 크게 말해봐요.")
    assert not is_safe("말끝이 흐려졌어요", "긴장해서 목소리가 작아졌어요.")
    assert not is_safe("되묻는 말이 작았어요", "자신감이 없어 보였어요.")


def test_mind_state_with_listener_view_passes() -> None:
    """상대가 어떻게 들었을지로 돌리면 통과한다."""
    assert is_safe(
        "질문할 때 흔들렸어요",
        "상대방에게 자신 없게 들렸을 수 있어요. 다음엔 한 호흡 쉬고 물어봐요.",
    )
    assert is_safe(
        "첫마디가 작았어요",
        "듣는 사람이 불안하게 느꼈을 수 있어요. 다음엔 또박또박 말해봐요.",
    )


def test_mind_state_with_hedge_passes() -> None:
    """추측을 달면 통과한다."""
    assert is_safe("첫마디가 흐렸어요", "긴장해서 목소리가 잘 안 나온 것 같아요.")
    assert is_safe("말끝이 흐려졌어요", "긴장하면 말이 잘 안 나올 수 있어요.")


def test_positive_mind_state_words_pass() -> None:
    """'긴장하지 않고' 같은 칭찬은 걸리지 않는다."""
    assert is_safe("끝까지 차분했어요", "긴장하지 않고 끝까지 이어서 말했어요.")
    assert is_safe("또렷하게 말했어요", "긴장 없이 용건을 한 번에 말했어요.")
    assert is_safe("자신감 있게 말했어요", "목소리가 흔들리지 않아 편하게 들렸어요.")


def test_mind_state_checked_per_sentence() -> None:
    """뒤 문장에 추측이 있어도 앞 문장의 단정은 걸러진다."""
    assert not is_safe(
        "말이 흔들렸어요",
        "긴장해서 목소리가 작아졌어요. 다음엔 상대방이 잘 들을 수 있게 말해봐요.",
    )

