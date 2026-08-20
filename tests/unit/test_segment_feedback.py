"""구간별 피드백 — AVTI 로 칭찬/조언 구간을 가르는 규칙."""

from app.services.feedback_points import is_safe, split_title
from app.services.llm import LLMClient
from app.services.pipeline import VoicePipeline


def _pipeline(turns: list[tuple[float, float]]) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._user_turn_intervals = turns
    p._user_turn_texts = [f"발화{i}" for i in range(len(turns))]
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
