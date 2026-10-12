"""말 중간의 긴 멈춤 — 세는 규칙, 구간 선택, AI 에 넘기는 근거."""

import pytest

from app.services.feedback_points import LONG_PAUSE_SEC, long_pauses
from app.services.llm import LLMClient
from app.services.pipeline import VoicePipeline, _segment_fallback
from app.services.training_analysis import _LONG_PAUSE_SECONDS


def _pipeline(turns, texts=None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "test-session"
    p._session = {"type": "SCENARIO", "scenario": {"title": "식당 예약"}}
    p._user_turn_intervals = turns
    p._user_turn_texts = texts if texts is not None else [f"발화{i}" for i in range(len(turns))]
    return p


# --- 세는 규칙 -------------------------------------------------------------


def test_long_gap_between_voice_is_a_pause() -> None:
    assert long_pauses((0.0, 10.0), [(1.0, 3.0), (5.2, 8.0)]) == [2.2]


def test_gap_shorter_than_the_threshold_is_not_counted() -> None:
    """숨 고르는 정도의 짧은 끊김은 세지 않는다. 딱 기준값부터 센다."""
    assert long_pauses((0.0, 10.0), [(1.0, 3.0), (4.4, 8.0)]) == []
    assert long_pauses((0.0, 10.0), [(1.0, 3.0), (4.5, 8.0)]) == [1.5]


def test_wait_before_and_after_speaking_is_not_a_pause() -> None:
    """말을 꺼내기 전과 마친 뒤의 빈 시간은 말 중간의 멈춤이 아니다."""
    assert long_pauses((0.0, 12.0), [(4.0, 6.0)]) == []


def test_voice_outside_the_turn_is_ignored() -> None:
    assert long_pauses((5.0, 12.0), [(0.0, 2.0), (6.0, 7.0), (10.0, 11.0), (20.0, 21.0)]) == [3.0]


def test_every_long_pause_in_the_turn_is_listed() -> None:
    assert long_pauses((0.0, 20.0), [(8.0, 9.0), (1.0, 2.0), (4.0, 5.0)]) == [2.0, 3.0]


def test_turn_without_voice_has_no_pause() -> None:
    assert long_pauses((0.0, 5.0), []) == []


def test_threshold_matches_the_score_calculation() -> None:
    """글이 짚는 멈춤과 대화 점수가 깎는 멈춤이 같은 기준이어야 서로 어긋나지 않는다."""
    assert LONG_PAUSE_SEC == _LONG_PAUSE_SECONDS


# --- 구간 선택 -------------------------------------------------------------


def test_turn_with_a_long_pause_becomes_an_improve_segment() -> None:
    p = _pipeline([(0.0, 4.0), (8.0, 16.0)])

    segs = p._pick_segments([(0.2, 3.8), (8.2, 15.8)], {}, {2: [2.1]})

    by_turn = {s["turn"]: s for s in segs}
    assert (by_turn[2]["type"], by_turn[2]["reason"], by_turn[2]["pauses"]) == ("IMPROVE", "pause", [2.1])
    assert (by_turn[1]["type"], by_turn[1]["reason"]) == ("GOOD", "clean")


def test_stable_voice_does_not_hide_a_long_pause() -> None:
    """목소리가 안정적이었어도 말이 길게 끊긴 턴은 칭찬하지 않는다."""
    p = _pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: 2.0}, {1: [1.8]})

    assert (seg["type"], seg["reason"]) == ("IMPROVE", "pause")


def test_shaky_voice_stays_the_reason_and_keeps_the_pause() -> None:
    p = _pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: 8.0}, {1: [1.8]})

    assert (seg["type"], seg["reason"], seg["pauses"]) == ("IMPROVE", "voice", [1.8])


def test_ambiguous_voice_with_a_long_pause_is_still_picked() -> None:
    p = _pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: 4.0}, {1: [1.8]})

    assert (seg["type"], seg["reason"]) == ("IMPROVE", "pause")


def test_pause_in_a_turn_without_transcript_is_ignored() -> None:
    """받아적힌 말이 없는 턴은 멈춤이 있어도 고르지 않는다."""
    p = _pipeline([(0.0, 8.0), (10.0, 14.0)], ["", "네 맞아요"])

    segs = p._pick_segments([(10.2, 13.8)], {}, {1: [3.0]})

    assert [s["turn"] for s in segs] == [2]


def test_without_pause_data_selection_is_unchanged() -> None:
    p = _pipeline([(0.0, 4.0), (8.0, 16.0)])
    candidates, avti = [(0.2, 3.8), (8.2, 15.8)], {1: 2.0}

    with_none = p._pick_segments(candidates, avti)
    with_empty = p._pick_segments(candidates, avti, {})

    assert [(s["turn"], s["type"], s["reason"]) for s in with_none] == [
        (1, "GOOD", "voice"), (2, "GOOD", "clean"),
    ]
    assert with_none == with_empty


def test_pause_segments_survive_the_cap_like_other_improve_segments() -> None:
    p = _pipeline([(0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0)])

    segs = p._pick_segments([(0.2, 3.8), (5.2, 8.8), (10.2, 13.8)], {}, {4: [2.0]})

    assert len(segs) == 3
    assert ("IMPROVE", "pause") in [(s["type"], s["reason"]) for s in segs]


# --- 떨림 분석 결과에서 턴별 멈춤 뽑기 -------------------------------------


class _Tremor:
    def __init__(self, voiced_spans) -> None:
        self.voiced_spans = voiced_spans


def test_pauses_are_keyed_by_turn_number() -> None:
    p = _pipeline([(0.0, 4.0), (8.0, 16.0)])

    pauses = p._long_pauses_by_turn(_Tremor([(0.5, 3.5), (8.5, 10.0), (12.2, 15.0)]))

    assert pauses == {2: [2.2]}


def test_missing_analysis_means_no_pauses() -> None:
    p = _pipeline([(0.0, 4.0)])

    assert p._long_pauses_by_turn(None) == {}
    assert p._long_pauses_by_turn(_Tremor([])) == {}


# --- AI 에 넘기는 근거 ------------------------------------------------------


def test_items_carry_pause_count_and_longest() -> None:
    p = _pipeline([(0.0, 8.0)], ["토요일 저녁 여섯 시에 네 명이요"])
    seg = {"turn": 1, "start": 0.0, "end": 8.0, "type": "IMPROVE",
           "reason": "pause", "avti": None, "pauses": [2.1, 1.6]}

    (item,) = p._feedback_items([seg])

    assert (item["pause_count"], item["pause_longest_sec"]) == (2, 2.1)


def test_items_without_pauses_report_zero() -> None:
    p = _pipeline([(0.0, 8.0)])
    seg = {"turn": 1, "start": 0.0, "end": 8.0, "type": "GOOD", "reason": "clean", "avti": None}

    (item,) = p._feedback_items([seg])

    assert (item["pause_count"], item["pause_longest_sec"]) == (0, 0.0)


@pytest.mark.parametrize(
    ("item", "line"),
    [
        ({"reason": "voice"}, "   짚을 점: 목소리가 흔들림"),
        ({"reason": "pause", "pause_count": 2, "pause_longest_sec": 2.1},
         "   짚을 점: 말 중간에 2번 길게 멈춤 (가장 길게 약 2초)"),
        ({"reason": "voice", "pause_count": 1, "pause_longest_sec": 1.6},
         "   짚을 점: 목소리가 흔들림, 말 중간에 1번 길게 멈춤 (가장 길게 약 2초)"),
        ({}, "   짚을 점: 목소리가 흔들림"),
    ],
)
def test_improve_segment_states_why_it_was_picked(item: dict, line: str) -> None:
    lines = LLMClient._segment_lines([{"type": "IMPROVE", "utterance": "네", "avti": None, **item}])

    assert lines == ["1. [짚어줄 구간]", line, '   내가 한 말: "네"']


def test_pause_length_is_given_without_decimals() -> None:
    """소수는 측정값 유출로 보고 안전 검사가 막는다. AI 에는 어림한 정수만 준다."""
    (_, line, _) = LLMClient._segment_lines(
        [{"type": "IMPROVE", "utterance": "네", "avti": None, "reason": "pause",
          "pause_count": 1, "pause_longest_sec": 1.5}]
    )

    assert "." not in line


def test_praise_segment_has_no_reason_line() -> None:
    lines = LLMClient._segment_lines([{"type": "GOOD", "utterance": "네", "avti": None}])

    assert lines == ["1. [칭찬할 구간]", '   내가 한 말: "네"']


# --- 기본 문구와 밖으로 나가는 모양 ----------------------------------------


def test_pause_segment_has_its_own_fallback_text() -> None:
    """멈춤으로 고른 구간에 '말이 흔들린 부분' 이라고 쓰면 고른 이유와 어긋난다."""
    title, content = _segment_fallback("IMPROVE", "pause")

    assert "멈춘" in content
    assert "흔들" not in title + content
    assert _segment_fallback("IMPROVE", "voice") == _segment_fallback("IMPROVE")


class _SilentLLM:
    async def segment_feedback(self, items, **kwargs):
        self.items = items
        return []


async def test_failed_writing_falls_back_by_reason_and_hides_pause_data() -> None:
    p = _pipeline([(0.0, 8.0)], ["토요일 저녁 여섯 시에 네 명이요"])
    p._llm = _SilentLLM()
    segments = p._pick_segments([], {}, {1: [2.1]})

    await p._write_segment_feedback(segments)

    (seg,) = segments
    assert seg["title"] == "말이 잠깐 끊겼어요"
    assert set(seg) == {"start", "end", "type", "title", "content", "good_point"}


async def test_teardown_picks_the_paused_turn_and_tells_the_writer(monkeypatch) -> None:
    """떨림 분석이 준 소리 구간에서 멈춤을 세고, 그 턴을 짚어줄 구간으로 올려 AI 에 근거를 넘긴다."""
    from tests.unit.test_avti_save import _pipeline as teardown_pipeline

    async def _no_avti(**_kwargs) -> dict:
        return {}

    spring: dict = {}

    async def _capture(session_id, **kwargs):
        spring.update(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.run_avti", _no_avti)

    p = teardown_pipeline(
        seconds=20,
        turns=[(1.0, 5.0), (8.0, 16.0)],
        spans=[(1.2, 4.8), (8.5, 10.0), (12.2, 15.0)],     # 2번 턴: 10.0 → 12.2 사이 2.2초 멈춤
    )
    p._user_turn_texts = ["예약하려고 전화드렸어요", "토요일 저녁 여섯 시에 네 명이요"]
    p._spring.notify_session_closed = _capture

    await p._teardown()

    # 이 가짜 분석은 '떨림 없는 구간'을 주지 않아 1번 턴은 고를 근거가 없고, 멈춘 2번 턴만 뽑힌다.
    (paused,) = p._llm.last_items
    assert (paused["type"], paused["reason"]) == ("IMPROVE", "pause")
    assert paused["utterance"] == "토요일 저녁 여섯 시에 네 명이요"
    assert (paused["pause_count"], paused["pause_longest_sec"]) == (1, 2.2)
    assert [seg["type"] for seg in spring["good_segments"]] == ["IMPROVE"]
    for seg in spring["good_segments"]:
        assert set(seg) == {"start", "end", "type", "title", "content", "good_point"}
